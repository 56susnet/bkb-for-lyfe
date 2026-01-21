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

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path
def merge_model_config(default_config: dict, model_config: dict) -> dict:
    merged = {}

    if isinstance(default_config, dict):
        merged.update(default_config)

    if isinstance(model_config, dict):
        merged.update(model_config)

    return merged if merged else None

def count_images_in_directory(directory_path: str) -> int:
    image_extensions = {'.jpg', '.jpeg', '.png'}
    count = 0
    
    try:
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}", flush=True)
            return 0
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.startswith('.'):
                    continue
                
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    count += 1
    except Exception as e:
        print(f"Error counting images in directory: {e}", flush=True)
        return 0
    
    return count

def load_size_based_config(model_type: str, is_style: bool, dataset_size: int) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")
    
    if model_type == "flux":
        return None
    elif is_style:
        config_file = os.path.join(config_dir, "size_style.json")
    else:
        config_file = os.path.join(config_dir, "size_person.json")
    
    try:
        with open(config_file, 'r') as f:
            size_config = json.load(f)
        
        size_ranges = size_config.get("size_ranges", [])
        for size_range in size_ranges:
            min_size = size_range.get("min", 0)
            max_size = size_range.get("max", float('inf'))
            
            if min_size <= dataset_size <= max_size:
                print(f"Using size-based config for {dataset_size} images (range: {min_size}-{max_size})", flush=True)
                return size_range.get("config", {})
        
        default_config = size_config.get("default", {})
        if default_config:
            print(f"Using default size-based config for {dataset_size} images", flush=True)
        return default_config
        
    except Exception as e:
        print(f"Warning: Could not load size-based config from {config_file}: {e}", flush=True)
        return None

def get_config_for_model(lrs_config: dict, model_name: str) -> dict:
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        return merge_model_config(default_config, data.get(model_name))

    if default_config:
        return default_config

    return None

def load_lrs_config(model_type: str, is_style: bool) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None



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
        sample_size = 1
    
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

def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word: str | None = None, num_images: int = 0):

    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model_path
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
        return config_path
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        lrs_config = load_lrs_config(model_type, is_style)

        if lrs_config:
            model_hash = hash_model(model_name)
            lrs_settings = get_config_for_model(lrs_config, model_hash)

            if lrs_settings:
                for optional_key in [
                    "max_grad_norm",
                    "prior_loss_weight",
                    "max_train_epochs",
                    "train_batch_size",
                    "max_train_steps",
                    "network_alpha",
                    "optimizer_args",
                    "unet_lr",
                    "text_encoder_lr",
                    "lr_warmup_steps",
                    "network_dropout",
                    "min_snr_gamma",
                    "seed",
                    "noise_offset",
                    "lr_scheduler",
                    "save_every_n_epochs",
                ]:
                    if optional_key in lrs_settings:
                        config[optional_key] = lrs_settings[optional_key]
            else:
                print(f"Warning: No LRS configuration found for model '{model_name}'", flush=True)
        else:
            print("Warning: Could not load LRS configuration, using default values", flush=True)

        network_config_person = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 467,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 467,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 699,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 467,
            "dataautogpt3/TempestV0.1": 456,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 467,
            "fluently/Fluently-XL-Final": 228,
            "mann-e/Mann-E_Dreams": 456,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 228,
            "recoilme/colorfulxl": 228,
            "zenless-lab/sdxl-aam-xl-anime-mix": 456,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
            "zenless-lab/sdxl-anything-xl": 228,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
            "Corcelio/mobius": 228,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 228
        }

        network_config_style = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 235,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 235,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 235,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 235,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 235,
            "dataautogpt3/TempestV0.1": 228,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 235,
            "fluently/Fluently-XL-Final": 235,
            "mann-e/Mann-E_Dreams": 235,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 235,
            "recoilme/colorfulxl": 235,
            "zenless-lab/sdxl-aam-xl-anime-mix": 235,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 235,
            "zenless-lab/sdxl-anything-xl": 235,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 235,
            "Corcelio/mobius": 235,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 235
        }

        config_mapping = {
            228: {
                "network_dim": 32,
                "network_alpha": 32,
                "network_args": []
            },
            235: {
                "network_dim": 32,
                "network_alpha": 32,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
            456: {
                "network_dim": 64,
                "network_alpha": 64,
                "network_args": []
            },
            467: {
                "network_dim": 64,
                "network_alpha": 64,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
            699: {
                "network_dim": 96,
                "network_alpha": 96,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
        }

        config["pretrained_model_name_or_path"] = model_path
        config["train_data_dir"] = train_data_dir
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        if model_type == "sdxl":
            if is_style:
                network_config = config_mapping[network_config_style[model_name]]
            else:
                network_config = config_mapping[network_config_person[model_name]]

            config["network_dim"] = network_config["network_dim"]
            config["network_alpha"] = network_config["network_alpha"]
            config["network_args"] = network_config["network_args"]


        dataset_size = 0
        if os.path.exists(train_data_dir):
            dataset_size = count_images_in_directory(train_data_dir)
            if dataset_size > 0:
                print(f"Counted {dataset_size} images in training directory", flush=True)

        if dataset_size > 0:
            size_config = load_size_based_config(model_type, is_style, dataset_size)
            if size_config:
                print(f"Applying size-based config for {dataset_size} images", flush=True)
                for key, value in size_config.items():
                    config[key] = value
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"config is {config}", flush=True)
        print(f"Created config at {config_path}", flush=True)
        return config_path


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

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 

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

    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        args.trigger_word,
        num_images=num_images
    )

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
