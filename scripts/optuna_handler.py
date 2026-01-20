
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
from trainer.utils.training_paths import get_checkpoints_output_path
import trainer.utils.training_paths as train_paths

import trainer.constants as train_cst
from scripts.config_builder import create_config

def objective(trial, task_id, model_path, model_name, model_type, expected_repo_name, trigger_word, train_data_dir=None):
    lr = trial.suggest_float("lr", 5e-7, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    lr_warmup_steps = trial.suggest_int("lr_warmup_steps", 0, 35)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])

    if model_type == "sdxl":
        rank = trial.suggest_categorical("rank", [16, 32, 64, 128])
        alpha = trial.suggest_categorical("alpha", [16, 32, 64, 128])
        optimizer = trial.suggest_categorical("optimizer", ["adamw8bit"])
        lr_scheduler = trial.suggest_categorical("lr_scheduler", ["cosine"])
    elif model_type in ["flux", "qwen-image", "z-image"]:
        rank = trial.suggest_categorical("rank", [16, 32, 64])
        alpha = trial.suggest_categorical("alpha", [16, 32, 64])
        optimizer = "adamw8bit"
        lr_scheduler = "flowmatch" if model_type == "flux" else "cosine"
    else:
        rank = 32
        alpha = 32
        optimizer = "adamw8bit"
        lr_scheduler = "cosine"
    repo_name = expected_repo_name or "output"
    trial_repo_name = f"{repo_name}_trial_{trial.number}"
    
    output_dir = train_paths.get_checkpoints_output_path(task_id, trial_repo_name)
    if os.path.exists(output_dir):
        print(f"Trial {trial.number}: Cleaning up existing output directory to ensure fresh start...", flush=True)
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    base_config_path = create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word, trial_number=trial.number, train_data_dir=train_data_dir)
    
    with open(base_config_path, 'r') as f:
        config_data = yaml.safe_load(f)
        
    process = config_data['config']['process'][0]
    
    if 'train' in process:
        process['train']['lr'] = lr
        process['train']['steps'] = 20 
        process['train']['lr_warmup_steps'] = lr_warmup_steps
        process['train']['batch_size'] = batch_size
        process['train']['gradient_accumulation_steps'] = gradient_accumulation_steps
        process['train']['optimizer'] = optimizer
        process['train']['lr_scheduler'] = lr_scheduler
        
        if optimizer.lower() == 'prodigy':
            if 'optimizer_params' in process['train']:
                 process['train']['optimizer_params']['weight_decay'] = weight_decay
            else:
                 process['train']['optimizer_params'] = {'weight_decay': weight_decay}
        else:
            process['train']['optimizer_params'] = {'weight_decay': weight_decay}

    if 'network' in process:
        process['network']['linear'] = rank
        process['network']['linear_alpha'] = alpha

    trial_config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}_trial_{trial.number}.yaml")
    with open(trial_config_path, 'w') as f:
        yaml.dump(config_data, f)
    print(f"Trial {trial.number}: Starting training with lr={lr}, opt={optimizer}, rank={rank}", flush=True)
    
    command = ["python3", "/app/ai-toolkit/run.py", trial_config_path]
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        final_loss = float('inf')
        loss_pattern = re.compile(r"loss:\s*([\d\.e\-\+]+)", re.IGNORECASE)
        
        current_step = 0

        for line in process.stdout:
            print(line, end="", flush=True)
            match = loss_pattern.search(line)
            if match:
                try:
                    current_loss = float(match.group(1))
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
        eval_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_handler.py")
        eval_output_file = os.path.join(output_dir, f"eval_results_trial_{trial.number}.json")
        
        eval_cmd = [
            "python3",
            eval_script_path,
            "--dataset", train_cst.EVAL_IMAGE,
            "--base-model", model_path,
            "--model-type", model_type,
            "--checkpoint-path", output_dir,
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
            
            # Read results after subprocess completes successfully
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
             
        # Fallback if evaluation didn't return
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
