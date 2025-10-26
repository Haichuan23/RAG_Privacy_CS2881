"""
Model downloader utility for automatically downloading HuggingFace models
when they don't exist locally.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model_if_needed(model_path: str, model_name: str = None) -> str:
    """
    Download a model from HuggingFace if it doesn't exist locally.

    Args:
        model_path: Local path where the model should be stored
        model_name: HuggingFace model name (if different from model_path)

    Returns:
        str: Path to the model (local if exists, downloaded if needed)
    """
    # Check if this is an absolute path to an existing directory (e.g., fine-tuned model)
    if os.path.isabs(model_path) and os.path.exists(model_path):
        # This is a local absolute path that exists
        logger.info(f"Using local model at: {model_path}")
        return model_path

    # If model_path is already a HuggingFace model name, use it directly
    if "/" in model_path and not os.path.exists(model_path):
        # This looks like a HuggingFace model name, not a local path
        hf_model_name = model_path
        local_model_path = os.path.join(
            "./local_models", hf_model_name.replace("/", "--")
        )
    else:
        # This is a local path
        local_model_path = model_path
        hf_model_name = model_name or model_path

    # Check if model already exists locally
    if os.path.exists(local_model_path) and os.path.exists(
        os.path.join(local_model_path, "config.json")
    ):
        logger.info(f"Model already exists locally at: {local_model_path}")
        return local_model_path

    # Create directory if it doesn't exist
    os.makedirs(local_model_path, exist_ok=True)

    logger.info(f"Downloading model '{hf_model_name}' to '{local_model_path}'...")

    try:
        # Download the model using snapshot_download for better control
        downloaded_path = snapshot_download(
            repo_id=hf_model_name,
            local_dir=local_model_path,
            local_dir_use_symlinks=False,  # Don't use symlinks for better portability
            resume_download=True,  # Resume if download was interrupted
        )

        logger.info(f"Successfully downloaded model to: {downloaded_path}")
        return downloaded_path

    except Exception as e:
        logger.error(f"Failed to download model '{hf_model_name}': {str(e)}")
        raise


def verify_model_integrity(model_path: str) -> bool:
    """
    Verify that a downloaded model has all necessary files.

    Args:
        model_path: Path to the model directory

    Returns:
        bool: True if model is complete, False otherwise
    """
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]

    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            logger.warning(f"Missing required file: {file}")
            return False

    # Check for model weights (pytorch_model.bin or model.safetensors)
    model_files = [
        f
        for f in os.listdir(model_path)
        if f.startswith("pytorch_model") or f.startswith("model.safetensors")
    ]
    if not model_files:
        logger.warning("No model weight files found")
        return False

    logger.info("Model integrity verified")
    return True


def load_model_safely(model_path: str, model_name: str = None, **kwargs):
    """
    Safely load a model, downloading it if necessary.

    Args:
        model_path: Local path or HuggingFace model name
        model_name: HuggingFace model name (if different from model_path)
        **kwargs: Additional arguments for model loading

    Returns:
        tuple: (model, tokenizer, actual_model_path)
    """
    # Download model if needed
    actual_model_path = download_model_if_needed(model_path, model_name)

    # Verify model integrity
    if not verify_model_integrity(actual_model_path):
        raise ValueError(f"Model at {actual_model_path} is incomplete or corrupted")

    # Set default loading parameters
    default_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else "cpu",
        "trust_remote_code": True,  # Required for some models like Qwen
    }

    # Update with user-provided kwargs
    default_kwargs.update(kwargs)

    logger.info(f"Loading model from: {actual_model_path}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            actual_model_path,
            trust_remote_code=default_kwargs.get("trust_remote_code", True),
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            actual_model_path, **default_kwargs
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Model loaded successfully")
        return model, tokenizer, actual_model_path

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the downloader
    test_model = "microsoft/DialoGPT-small"  # Small model for testing
    local_path = "./test_model"

    try:
        model, tokenizer, actual_path = load_model_safely(test_model)
        print(f"Successfully loaded model at: {actual_path}")
    except Exception as e:
        print(f"Test failed: {e}")
