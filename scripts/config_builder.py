
import os
import json
import yaml
import toml
import hashlib
import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.models.utility_models import ImageModelType

def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path

def merge_model_config(default_config: dict, model_config: dict) -> dict:
    """Merge default config with model-specific overrides."""
    merged = {}

    if isinstance(default_config, dict):
        merged.update(default_config)

    if isinstance(model_config, dict):
        merged.update(model_config)

    return merged if merged else None

def get_config_for_model(lrs_config: dict, model_name: str) -> dict:
    """Get configuration overrides based on model name."""
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
    """Load the appropriate LRS configuration based on model type and training type"""
    # Assuming this script is in scripts/
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

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 

def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word: str | None = None, num_images: int = 0, trial_number: int | None = None, train_data_dir: str | None = None):
    # If train_data_dir is not provided, try to find it
    if train_data_dir is None:
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
                        repo_name = expected_repo_name or "output"
                        if trial_number is not None:
                            repo_name = f"{repo_name}_trial_{trial_number}"
                        
                        output_dir = train_paths.get_checkpoints_output_path(task_id, repo_name)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        filename = f"{task_id}.yaml" if trial_number is None else f"{task_id}_trial_{trial_number}.yaml"
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, filename)
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
                    "optimizer_args",
                    "unet_lr",
                    "text_encoder_lr",
                    "noise_offset",
                    "min_snr_gamma",
                    "seed",
                    "lr_warmup_steps",
                    "loss_type",
                    "huber_c",
                    "huber_schedule",
                ]:
                    if optional_key in lrs_settings:
                        config[optional_key] = lrs_settings[optional_key]
            else:
                print(f"Warning: No LRS configuration found for model '{model_name}'", flush=True)
        else:
            print("Warning: Could not load LRS configuration, using default values", flush=True)

        # Update config
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
        
        repo_name = expected_repo_name or "output"
        if trial_number is not None:
             repo_name = f"{repo_name}_trial_{trial_number}"

        output_dir = train_paths.get_checkpoints_output_path(task_id, repo_name)
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

        filename = f"{task_id}.toml" if trial_number is None else f"{task_id}_trial_{trial_number}.toml"
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, filename)
        save_config_toml(config, config_path)
        print(f"config is {config}", flush=True)
        print(f"Created config at {config_path}", flush=True)
        return config_path
