import argparse
import json
import os
import random
import sys
from glob import glob
from io import BytesIO

import numpy as np
import torch
from diffusers import (
    FluxImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    FluxPipeline,
    DiffusionPipeline
)
from PIL import Image


CONSTANTS = {
    "sdxl": {"steps": 20, "cfg": 8, "denoise": 0.9},
    "flux": {"steps": 35, "cfg": 100, "denoise": 0.75},
    "z-image": {"steps": 10, "cfg": 1, "denoise": 0.90},
    "qwen-image": {"steps": 20, "cfg": 8, "denoise": 0.93},
}

# Weighted loss formula matching G.O.D validator (25% text-guided, 75% no-text)
DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT = 0.25


def generate_reproducible_seeds(master_seed: int, n: int = 10) -> list[int]:
    """Generate n reproducible seeds from a master seed."""
    random.seed(master_seed)
    return [random.randint(0, 2**32 - 1) for _ in range(n)]


def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    return image

def adjust_image_size(image: Image.Image) -> Image.Image:
    """
    Resizes and crops image to 1024x1024 without degradation (noise/blur),
    matching validator logic.
    """
    width, height = image.size

    if width > height:
        new_width = 1024
        new_height = int((height / width) * 1024)
    else:
        new_height = 1024
        new_width = int((width / height) * 1024)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    width, height = image.size
    crop_width = min(width, new_width)
    crop_height = min(height, new_height)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    image = image.crop((left, top, right, bottom))

    return image


def calculate_l2_loss(test_image: Image.Image, generated_image: Image.Image) -> float:
    test_image = np.array(test_image.convert("RGB")) / 255.0
    generated_image = np.array(generated_image.convert("RGB")) / 255.0
    
    if test_image.shape != generated_image.shape:
        generated_image_pil = Image.fromarray((generated_image * 255).astype(np.uint8))
        generated_image_pil = generated_image_pil.resize((test_image.shape[1], test_image.shape[0]), Image.Resampling.LANCZOS)
        generated_image = np.array(generated_image_pil) / 255.0

    l2_loss = np.mean((test_image - generated_image) ** 2)
    return l2_loss


def get_dataset_files(dataset_path):
    extensions = ['*.png', '*.jpg', '*.jpeg']
    all_files = []
    for ext in extensions:
        all_files.extend(glob(os.path.join(dataset_path, ext)))
    
    all_files.sort()
    
    return all_files

def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--base-model", required=True, help="Base model path or repo ID")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"], help="Model type")
    parser.add_argument("--checkpoint-path", required=True, help="Path to trained LoRA safetensors")
    parser.add_argument("--output-file", default="evaluation_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype: {dtype}")
    
    model_consts = CONSTANTS[args.model_type]
    print(f"Using Constants for {args.model_type}: {model_consts}")

    print(f"Loading base model: {args.base_model} ({args.model_type})")
    try:
        if args.model_type == "sdxl":
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                args.base_model,
                torch_dtype=dtype,
                use_safetensors=True
            )
        elif args.model_type == "flux":
            try:
                pipe = FluxImg2ImgPipeline.from_pretrained(
                    args.base_model,
                    torch_dtype=dtype
                )
            except (ImportError, ValueError, EnvironmentError):
                print("FluxImg2ImgPipeline not found, trying FluxPipeline...")
                pipe = FluxPipeline.from_pretrained(
                    args.base_model,
                    torch_dtype=dtype
                )
        else:
            print(f"Attempting to load {args.model_type} using DiffusionPipeline...")
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                args.base_model,
                torch_dtype=dtype,
                trust_remote_code=True
            )
            
        pipe.to(device)
        
        print(f"Loading LoRA from: {args.checkpoint_path}")
        
        lora_path = args.checkpoint_path
        if os.path.isdir(lora_path):
            # Check standard subfolder "last/last.safetensors" first (ai-toolkit structure)
            if os.path.exists(os.path.join(lora_path, "last", "last.safetensors")):
                lora_path = os.path.join(lora_path, "last", "last.safetensors")
                print(f"  Auto-resolved to nested checkpoint: {lora_path}")
            else:
                candidates = [f for f in os.listdir(lora_path) if f.endswith(".safetensors")]
                
                if not candidates:
                    # Recursive fallback
                    print(f"  No checkpoints in root, searching recursively...")
                    for root, dirs, files in os.walk(lora_path):
                        for file in files:
                            if file.endswith(".safetensors") and "optimizer" not in file:
                                lora_path = os.path.join(root, file)
                                print(f"  Found recursive candidate: {lora_path}")
                                candidates = [file] # Signal that we found something
                                break
                        if candidates: break
                
                if not candidates and not os.path.isfile(lora_path): # If still not found and not resolved to a file
                    raise ValueError(f"No .safetensors found in {args.checkpoint_path}")

                # If lora_path is still a directory, we need to pick from candidates
                if os.path.isdir(lora_path):
                    if "last.safetensors" in candidates:
                        selected_filename = "last.safetensors"
                    else:
                        epoch_files = []
                        selected_filename = None
                        
                        import re
                        for file in candidates:
                            epoch = None
                            match = re.search(r'[-_](\\d+)\\.safetensors$', file)
                            if match:
                                try:
                                    epoch = int(match.group(1))
                                except ValueError:
                                    pass
                            else:
                                match = re.search(r'(\\d+)\\.safetensors$', file)
                                if match:
                                    try:
                                        epoch = int(match.group(1))
                                    except ValueError:
                                        pass
                            
                            if epoch is None:
                                selected_filename = file
                                break
                            else:
                                epoch_files.append((epoch, file))
                        
                        if selected_filename is None and epoch_files:
                            epoch_files.sort(reverse=True, key=lambda x: x[0])
                            selected_filename = epoch_files[0][1]
                        elif selected_filename is None:
                            selected_filename = candidates[0]

                    lora_path = os.path.join(lora_path, selected_filename)
                    print(f"  Auto-resolved directory to: {lora_path}")
            
        pipe.load_lora_weights(lora_path)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    dataset_files = get_dataset_files(args.dataset)
    if not dataset_files:
        print("No images found in dataset path!")
        sys.exit(1)
        
    print(f"\\nFound {len(dataset_files)} images for evaluation")
    print(f"Using 10 seeds per image for robust evaluation\\n")
    
    total_text_guided_losses = []
    total_no_text_losses = []
    detailed_results = {}
    
    for idx, image_path in enumerate(dataset_files, 1):
        try:
            filename = os.path.basename(image_path)
            print(f"Processing [{idx}/{len(dataset_files)}] {filename}")
            
            original_image = load_image(image_path)
            original_image = adjust_image_size(original_image)
            
            base_name = os.path.splitext(image_path)[0]
            txt_path = f"{base_name}.txt"
            prompt = ""
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
            
            # Generate 10 reproducible seeds for this image
            seeds = generate_reproducible_seeds(master_seed=42, n=10)
            
            text_guided_losses = []
            no_text_losses = []
            
            for seed in seeds:
                generator = torch.Generator(device=device).manual_seed(seed)
                
                # Test 1: Text-guided inference
                kwargs_text = {
                    "prompt": prompt,
                    "num_inference_steps": model_consts["steps"],
                    "guidance_scale": model_consts["cfg"],
                    "generator": generator
                }
                
                # Add image usage only for models that support it (img2img)
                if args.model_type not in ["qwen-image", "z-image"]:
                    kwargs_text["image"] = original_image
                    kwargs_text["strength"] = model_consts["denoise"]
                
                try:
                    generated_image_text = pipe(**kwargs_text).images[0]
                except RuntimeError as e:
                    if "expected scalar type" in str(e) and "but found" in str(e) and "image" in kwargs_text:
                        print(f"  Dtype mismatch detected, converting image to {dtype}...")
                        import torchvision.transforms as T
                        to_tensor = T.ToTensor()
                        to_pil = T.ToPILImage()
                        
                        img_tensor = to_tensor(original_image).to(device)
                        if dtype == torch.bfloat16:
                            img_tensor = img_tensor.bfloat16()
                        else:
                            img_tensor = img_tensor.half()
                        
                        original_image_converted = to_pil(img_tensor.cpu().float())
                        kwargs_text["image"] = original_image_converted
                        generated_image_text = pipe(**kwargs_text).images[0]
                    else:
                        raise
                
                loss_text = calculate_l2_loss(original_image, generated_image_text)
                text_guided_losses.append(loss_text)
                
                # Test 2: No-text inference (empty prompt)
                generator_no_text = torch.Generator(device=device).manual_seed(seed)
                kwargs_no_text = {
                    "prompt": "",
                    "num_inference_steps": model_consts["steps"],
                    "guidance_scale": model_consts["cfg"],
                    "generator": generator_no_text
                }
                
                if args.model_type not in ["qwen-image", "z-image"]:
                    kwargs_no_text["image"] = original_image
                    kwargs_no_text["strength"] = model_consts["denoise"]
                
                try:
                    generated_image_no_text = pipe(**kwargs_no_text).images[0]
                except RuntimeError as e:
                    if "expected scalar type" in str(e) and "but found" in str(e) and "image" in kwargs_no_text:
                        if 'original_image_converted' not in locals():
                            import torchvision.transforms as T
                            to_tensor = T.ToTensor()
                            to_pil = T.ToPILImage()
                            img_tensor = to_tensor(original_image).to(device)
                            if dtype == torch.bfloat16:
                                img_tensor = img_tensor.bfloat16()
                            else:
                                img_tensor = img_tensor.half()
                            original_image_converted = to_pil(img_tensor.cpu().float())
                        kwargs_no_text["image"] = original_image_converted
                        generated_image_no_text = pipe(**kwargs_no_text).images[0]
                    else:
                        raise
                
                loss_no_text = calculate_l2_loss(original_image, generated_image_no_text)
                no_text_losses.append(loss_no_text)
            
            # Calculate per-image averages
            avg_text_loss = np.mean(text_guided_losses)
            avg_no_text_loss = np.mean(no_text_losses)
            
            total_text_guided_losses.append(avg_text_loss)
            total_no_text_losses.append(avg_no_text_loss)
            
            # Log per-image average losses
            print(f"  AVG_TEXT_LOSS:    {avg_text_loss:.6f}")
            print(f"  AVG_NO_TEXT_LOSS: {avg_no_text_loss:.6f}")
            
            detailed_results[filename] = {
                "avg_text_loss": float(avg_text_loss),
                "avg_no_text_loss": float(avg_no_text_loss)
            }
            
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            detailed_results[filename] = {"error": str(e)}

    print(f"\\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total files found: {len(dataset_files)}")
    print(f"Successfully processed: {len(total_text_guided_losses)}")
    
    if not total_text_guided_losses:
        print("ERROR: No images were successfully processed for evaluation!")
        sys.exit(1)
    
    # Calculate overall averages
    overall_avg_text_loss = np.mean(total_text_guided_losses)
    overall_avg_no_text_loss = np.mean(total_no_text_losses)
    
    # Calculate weighted loss (matching G.O.D validator formula)
    weighted_loss = (
        DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT * overall_avg_text_loss + 
        (1 - DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT) * overall_avg_no_text_loss
    )
    
    print(f"\\nOverall AVG_TEXT_LOSS:    {overall_avg_text_loss:.6f}")
    print(f"Overall AVG_NO_TEXT_LOSS: {overall_avg_no_text_loss:.6f}")
    print(f"Weighted Loss (25% text + 75% no-text): {weighted_loss:.6f}")
    print(f"{'='*60}\\n")
    
    final_output = {
        "model_type": args.model_type,
        "base_model": args.base_model,
        "lora_path": args.checkpoint_path,
        "overall_avg_text_loss": float(overall_avg_text_loss),
        "overall_avg_no_text_loss": float(overall_avg_no_text_loss),
        "weighted_loss": float(weighted_loss),
        "text_guided_losses": [float(x) for x in total_text_guided_losses],
        "no_text_losses": [float(x) for x in total_no_text_losses],
        "detailed_results": detailed_results
    }
    
    with open(args.output_file, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
