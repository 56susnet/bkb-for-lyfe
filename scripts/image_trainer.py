#!/usr/bin/env python3
"""
image-yaya
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import re
import time
import yaml
import toml
import shutil
import random
import glob

# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType
from scripts.config_builder import create_config, get_model_path
import scripts.optuna_handler_sdxl as optuna_handler_sdxl
import scripts.optuna_handler as optuna_handler_generic

def split_dataset(train_dir, eval_dir):
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    os.makedirs(eval_dir, exist_ok=True)
    
    extensions = ('.png', '.jpg', '.jpeg')
    all_files = [f for f in os.listdir(train_dir) if f.lower().endswith(extensions)]
    all_files.sort()
    
    total_files = len(all_files)
    if total_files == 0:
        print("Warning: No images found to split.", flush=True)
        return 0, 0

    if total_files > 20:
        sample_size = 2
    else:
        sample_size = 2
    
    random.seed(42) 
    eval_files = random.sample(all_files, sample_size)
    
    print(f"Splitting dataset: {total_files} total images. Moving {len(eval_files)} to evaluation set ({eval_dir}).", flush=True)
    
    for filename in eval_files:
        src_img = os.path.join(train_dir, filename)
        dst_img = os.path.join(eval_dir, filename)
        shutil.move(src_img, dst_img)
        
        base_name = os.path.splitext(filename)[0]
        txt_name = f"{base_name}.txt"
        src_txt = os.path.join(train_dir, txt_name)
        if os.path.exists(src_txt):
            dst_txt = os.path.join(eval_dir, txt_name)
            shutil.move(src_txt, dst_txt)
            
    num_train = len(all_files) - len(eval_files)
    num_eval = len(eval_files)
    return num_train, num_eval

def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        training_command = [
            "python3",
            "/app/ai-toolkit/run.py",
            config_path
        ]
    else:
        if model_type == "sdxl":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-script/{model_type}_train_network.py",
                "--config_file", config_path
        ]
        elif model_type == "flux":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-scripts/{model_type}_train_network.py",
                "--config_file", config_path
            ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

 

async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of optimization trials")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    base_images_dir = train_cst.IMAGE_CONTAINER_IMAGES_PATH
    train_data_dir = base_images_dir
    
    image_dir = None
    extensions = ('.png', '.jpg', '.jpeg')
    
    if os.path.exists(base_images_dir):
        for root, dirs, files in os.walk(base_images_dir):
            if any(f.lower().endswith(extensions) for f in files):
                image_dir = root
                print(f"Detected images in: {image_dir}", flush=True)
                break
    
    if image_dir:
        train_data_dir = image_dir
        print(f"Setting training config root to: {train_data_dir}", flush=True)
    else:
        print("Warning: Could not find any images in dataset path!", flush=True)
        train_data_dir = base_images_dir
        image_dir = base_images_dir

    num_train_images, num_eval_images = split_dataset(image_dir, train_cst.EVAL_IMAGE)
    num_images = num_train_images
    print(f"Found {num_images} training images (and {num_eval_images} eval images) in {train_data_dir}", flush=True)

    best_params = None
    if args.optimize:
        if args.model_type == "sdxl":
            print(f"Starting Optuna hyperparameter optimization ({args.n_trials} trials)...", flush=True)
            best_params = optuna_handler_sdxl.optimize_hyperparameters(
                args.task_id,
                model_path,
                args.model,
                args.model_type,
                args.expected_repo_name,
                args.trigger_word,
                n_trials=args.n_trials,
                train_data_dir=train_data_dir
            )
            print(f"Optimization complete. Best params: {best_params}", flush=True)
        elif args.model_type in ["qwen-image", "z-image"]:
            print(f"Starting Optuna hyperparameter optimization ({args.n_trials} trials) for {args.model_type}...", flush=True)
            best_params = optuna_handler_generic.optimize_hyperparameters(
                args.task_id,
                model_path,
                args.model,
                args.model_type,
                args.expected_repo_name,
                args.trigger_word,
                n_trials=args.n_trials,
                train_data_dir=train_data_dir
            )
            print(f"Optimization complete. Best params: {best_params}", flush=True)
        else:
            print(f"Optimization not supported for {args.model_type}. Skipping...", flush=True)

    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        args.trigger_word,
        num_images=num_images,
        train_data_dir=train_data_dir
    )
    
    if best_params:
        if args.model_type == "sdxl" and "lr" in best_params:
            lr = best_params["lr"]
            unet_lr = lr
            text_encoder_lr = lr * 0.1
            print(f"Applying best LRs: unet_lr={unet_lr}, text_encoder_lr={text_encoder_lr}", flush=True)
            
            with open(config_path, 'r') as f:
                final_config = toml.load(f)
            
            final_config["unet_lr"] = unet_lr
            final_config["text_encoder_lr"] = text_encoder_lr
            
            with open(config_path, 'w') as f:
                toml.dump(final_config, f)

    run_training(args.model_type, config_path)

    print("\n" + "="*50, flush=True)

    if args.model_type not in ["sdxl", "qwen-image", "z-image"]:
        print(f"Skipping evaluation for model type: {args.model_type}", flush=True)
        return

    print("Starting POST-TRAINING EVALUATION...", flush=True)
    # Determine output directory (mirroring create_config logic)
    output_dir = train_paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name or "output")
    
    if not os.path.isdir(output_dir):
         print(f"Warning: Output directory {output_dir} not found. Skipping evaluation.", flush=True)
         return

    checkpoints = []

    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".safetensors") and "optimizer" not in file:
                 full_path = os.path.join(root, file)
                 checkpoints.append(full_path)
    
    if not checkpoints:
        print("No checkpoints found for evaluation.", flush=True)
    else:
        print(f"Found {len(checkpoints)} checkpoints to evaluate.", flush=True)
        
        best_loss = float('inf')
        best_checkpoint = None
        
        eval_script = os.path.join(script_dir, "eval_handler.py")
        
        for ckpt in checkpoints:
            print(f"Evaluating checkpoint: {ckpt}", flush=True)
            
            eval_output_file = os.path.join(os.path.dirname(ckpt), f"eval_results_{os.path.basename(ckpt)}.json")
            
            eval_cmd = [
                "python3",
                eval_script,
                "--dataset", train_cst.EVAL_IMAGE,
                "--base-model", model_path, 
                "--model-type", args.model_type,
                "--checkpoint-path", ckpt,
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
                
                for line in eval_process.stdout:
                    print(f"  {line}", end="", flush=True)
                
                return_code = eval_process.wait()
                
                if return_code != 0:
                    print(f"  Evaluation failed with exit code {return_code}", flush=True)
                    continue
                
                if os.path.exists(eval_output_file):
                    with open(eval_output_file, 'r') as f:
                        res = json.load(f)
                        avg_loss = res.get("weighted_loss", float('inf'))
                        print(f"  Weighted Loss: {avg_loss:.6f}", flush=True)
                        
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            best_checkpoint = ckpt
                else:
                     print("  Warning: Evaluation produced no output file.", flush=True)

            except Exception as e:
                print(f"  An error occurred evaluating {ckpt}: {e}", flush=True)

        print("-" * 30, flush=True)
        if best_checkpoint:
            print(f"BEST MODEL FOUND: {best_checkpoint} (Loss: {best_loss})", flush=True)
            
            targets = [
                os.path.join(output_dir, "last.safetensors"),
                os.path.join(output_dir, "last", "last.safetensors")
            ]
            
            promoted = False
            for target in targets:
                target_dir = os.path.dirname(target)
                if os.path.isdir(target_dir):
                    try:
                        shutil.copy2(best_checkpoint, target)
                        print(f"Promoted best model to: {target}", flush=True)
                        promoted = True
                    except Exception as e:
                         print(f"Failed to copy to {target}: {e}", flush=True)
            
            if not promoted:
                 target = os.path.join(output_dir, "last.safetensors")
                 try:
                    shutil.copy2(best_checkpoint, target)
                    print(f"Promoted best model to: {target}", flush=True)
                 except Exception as e:
                    print(f"Failed to copy to {target}: {e}", flush=True)

            best_marker_path = os.path.join(output_dir, "best_model_info.json")
            with open(best_marker_path, 'w') as f:
                 json.dump({"best_checkpoint": best_checkpoint, "loss": best_loss}, f)
            
        else:
            print("Could not determine best model.", flush=True)
    
    print("="*50 + "\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
