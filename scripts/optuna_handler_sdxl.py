
import os
import re
import json
import yaml
import subprocess
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import shutil
import toml
from trainer.utils.training_paths import get_checkpoints_output_path
import trainer.utils.training_paths as train_paths
import trainer.constants as train_cst
from scripts.config_builder import create_config

def objective(trial, task_id, model_path, model_name, model_type, expected_repo_name, trigger_word, train_data_dir=None):
    # Only optimizing LR for SDXL as requested
    # lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    
    # Simple logic: scale text_encoder_lr based on unet_lr ratio (1/10) or just use same?
    # Usually text_encoder_lr is smaller. Let's assume the user wants to search the main LR.
    # We will use lr for unet_lr and lr/10 for text_encoder_lr which is a common heuristic.
    unet_lr = 1.0
    text_encoder_lr = 1.0
    d_coef = trial.suggest_float("d_coef", 0.5, 2.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    prodigy_args = [
        f"d_coef={d_coef}",
        f"weight_decay={weight_decay}",
        "decouple=True",
        "use_bias_correction=True",
        "safeguard_warmup=True",
        "betas=(0.9, 0.999)"
    ]

    # print(f"Trial {trial.number}: Searching with unet_lr={unet_lr}, text_encoder_lr={text_encoder_lr}", flush=True)

    repo_name = expected_repo_name or "output"
    trial_repo_name = f"{repo_name}_trial_{trial.number}"
    
    output_dir = train_paths.get_checkpoints_output_path(task_id, trial_repo_name)
    if os.path.exists(output_dir):
        print(f"Trial {trial.number}: Cleaning up existing output directory to ensure fresh start...", flush=True)
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # config_builder's create_config handles basic config creation. We need to override it.
    # We will let create_config create the default config, then we load it, modify it, and save it back.
    # Or config_builder.create_config can accept overrides, but simpler to modify file.
    # Wait, create_config returns the path. 
    
    # We pass trial_number to create_config to ensure unique filenames
    base_config_path = create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word, trial_number=trial.number, train_data_dir=train_data_dir)
    
    # Load and modify config
    with open(base_config_path, 'r') as f:
        config_data = toml.load(f)
    
    config_data['optimizer_type'] = "prodigy"
    config_data['unet_lr'] = unet_lr
    config_data['text_encoder_lr'] = text_encoder_lr
    config_data['optimizer_args'] = prodigy_args
    config_data["max_train_epochs"] = 10
    config_data["save_every_n_epochs"] = 1

    
    # Ensure epochs/steps are small for optimization if needed? 
    # The user didn't specify to reduce epochs, but typically HPO trials are shorter.
    # However, "fokus pada pencarian lr saja" implies we want to find the best LR for the full training or a representative subset.
    # If we change epochs, we might check a different optimal point.
    # Let's keep epochs as is (from config) or maybe user wants to rely on defaults.
    # For now, we only change LRs.
    
    # Save modified config (overwriting the trial config created by create_config)
    with open(base_config_path, 'w') as f:
        toml.dump(config_data, f)
        
    print(f"Trial {trial.number}: Starting training with unet_lr={unet_lr}", flush=True)
    
    # Construct training command (SDXL)
    # Copied from image_trainer.py logic
    training_command = [
        "accelerate", "launch",
        "--dynamo_backend", "no",
        "--dynamo_mode", "default",
        "--mixed_precision", "bf16",
        "--num_processes", "1",
        "--num_machines", "1",
        "--num_cpu_threads_per_process", "2",
        f"/app/sd-scripts/sdxl_train_network.py",
        "--config_file", base_config_path
    ]
    
    try:
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        final_loss = float('inf')
        # sd-scripts usually outputs "steps: ... loss: ..." 
        # But for accurate pruning we need to parse it. 
        # Since we are focusing on simple implementation first, primarily for result evaluation,
        # we might skip extensive pruning logic if parsing is brittle, but let's try.
        loss_pattern = re.compile(r"loss:?\s*([\d\.e\-\+]+)", re.IGNORECASE)
        
        current_step = 0

        for line in process.stdout:
            print(line, end="", flush=True)
            match = loss_pattern.search(line)
            if match:
                try:
                    current_loss = float(match.group(1))
                    
                    # Update final_loss (this might be noisy, usually we want average or validation loss)
                    # But for now we track training loss for pruning
                    final_loss = current_loss
                    
                    current_step += 1 
                    trial.report(current_loss, current_step)
                    if trial.should_prune():
                        print(f"Trial {trial.number} pruned at step {current_step}.", flush=True)
                        process.terminate()
                        raise optuna.exceptions.TrialPruned()

                except ValueError:
                    pass
        
        return_code = process.wait()
        if return_code != 0:
             print(f"Trial {trial.number} failed with exit code {return_code}", flush=True)
             return float('inf')

        # Evaluation Step
        print(f"Trial {trial.number}: Starting evaluation...", flush=True)
        # Using eval_handler.py
        eval_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_handler.py")
        eval_output_file = os.path.join(output_dir, f"eval_results_trial_{trial.number}.json")
        
        # We need to find the checkpoint. sd-scripts output structure:
        # output_dir/last.safetensors or output_dir/checkpoint-epoch-X.safetensors
        # image_trainer.py looks for .safetensors in output_dir
        
        checkpoints = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".safetensors") and "optimizer" not in file:
                     full_path = os.path.join(root, file)
                     checkpoints.append(full_path)
        
        if not checkpoints:
            print(f"Trial {trial.number}: No checkpoints found to evaluate.", flush=True)
            return float('inf')
            
        # Evaluate the 'last' or most recent checkpoint
        # Heuristic: pick the one with 'last' in name, or else the last alphanumerically
        ckpt_to_eval = sorted(checkpoints)[-1] # Default to last
        for ckpt in checkpoints:
            if "last" in os.path.basename(ckpt):
                ckpt_to_eval = ckpt
                break
        
        print(f"Trial {trial.number}: Evaluating {ckpt_to_eval}", flush=True)

        eval_cmd = [
            "python3",
            eval_script_path,
            "--dataset", train_cst.EVAL_IMAGE,
            "--base-model", model_path,
            "--model-type", model_type,
            "--checkpoint-path", ckpt_to_eval,
            "--output-file", eval_output_file
        ]
        
        try:
            eval_process = subprocess.Popen(
                eval_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output in realtime
            for line in eval_process.stdout:
                print(line, end="", flush=True)
            
            return_code = eval_process.wait()
            
            if return_code != 0:
                print(f"Trial {trial.number}: Evaluation subprocess failed with exit code {return_code}.", flush=True)
                return float('inf')
            
            # Read results
            if os.path.exists(eval_output_file):
                with open(eval_output_file, 'r') as f:
                    res = json.load(f)
                    eval_loss = res.get("weighted_loss", float('inf'))
                    print(f"Trial {trial.number} Eval Loss: {eval_loss:.6f}", flush=True)
                    return eval_loss
            else:
                 print(f"Trial {trial.number}: Evaluation produced no output file.", flush=True)
                 return float('inf')
                 
        except Exception as e:
            print(f"Trial {trial.number}: Evaluation encountered exception: {e}", flush=True)
            return float('inf')
             
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}", flush=True)
        return float('inf')

def optimize_hyperparameters(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word, n_trials=10, train_data_dir=None):
    storage_url = f"sqlite:///{train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH}/optuna_{task_id}.db"
    
    study = optuna.create_study(
        direction="minimize",
        storage=storage_url,
        study_name=f"study_{task_id}",
        load_if_exists=True,
        sampler=TPESampler(multivariate=True, constant_liar=True),
        pruner=HyperbandPruner(min_resource=10, max_resource=300, reduction_factor=3)
    )
    
    func = lambda trial: objective(trial, task_id, model_path, model_name, model_type, expected_repo_name, trigger_word, train_data_dir=train_data_dir)
    
    study.optimize(func, n_trials=n_trials)
    
    print("Best trials:", flush=True)
    trial = study.best_trial
    print(f"  Value: {trial.value}", flush=True)
    print("  Params: ", flush=True)
    for key, value in trial.params.items():
        print(f"    {key}: {value}", flush=True)
        
    best_params_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}_best_params.json")
    with open(best_params_path, 'w') as f:
        json.dump(trial.params, f, indent=4)
        
    return trial.params
