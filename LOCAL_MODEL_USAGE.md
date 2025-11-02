# Local Model Usage Guide

This guide explains how to use the RAG Privacy project with local models instead of the TogetherAI API, specifically helpful for evaluating fine-tuned models.

## Files Created

1. **`main_local.py`** - Main script for local model usage
2. **`utils/argparser_local.py`** - Argument parser optimized for local models
3. **`main_local.sh`** - Shell script for easy execution
4. **`LOCAL_MODEL_USAGE.md`** - This documentation

## Prerequisites

1. **CUDA Support**: Your system must have CUDA available for GPU acceleration
2. **Dependencies**: Ensure all required packages are installed (transformers, torch, huggingface_hub, etc.)
3. **Model**: Either a local model directory OR a HuggingFace model name (will be auto-downloaded)

## Usage

1. Edit `main_local.sh` and update the following variables:
   ```bash
   # Option 1: Use a HuggingFace model name (will be auto-downloaded)
   LOCAL_MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.3"
   
   # Option 2: Use a local model directory
   LOCAL_MODEL_PATH="/path/to/your/local/model"
   
   # Data paths
   IO_INPUT_PATH="eval_data/Wikipedia/io_input.json"  # Your input file
   IO_OUTPUT_ROOT="eval_data/Wikipedia/io_output"  # Output directory
   ```

2. Run the script:
   ```bash
   ./main_local.sh
   ```

## Automatic Model Download

The script automatically downloads models from HuggingFace if they don't exist locally:

### Supported Input Formats:
- **HuggingFace Model Name**: `"mistralai/Mistral-7B-Instruct-v0.3"`
- **Local Directory Path**: `"/path/to/your/local/model"`

### Download Location:
- HuggingFace models are downloaded to: `./local_models/{model_name_with_dashes}/`
- Example: `mistralai/Mistral-7B-Instruct-v0.3` → `./local_models/mistralai--Mistral-7B-Instruct-v0.3/`

## Model Requirements

The local model should be:
- A HuggingFace-compatible model (AutoModelForCausalLM)
- Compatible with the tokenizer specified in the model directory

## Configuration Options

### Model Parameters
- `hf_ckpt`: Path to local model directory
- `is_chat_model`: Whether the model is a chat model (true/false)
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (0.0-1.0)
- `top_k`, `top_p`: Sampling parameters

### RIC Parameters
- `k_for_ric`: Number of retrieved documents
- `max_retrieval_seq_length`: Maximum length of retrieved sequences
- `ric_stride`: Stride for retrieval

## Example Workflow

1. **Download a model**:
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   model_name = "mistralai/Mistral-7B-Instruct-v0.3"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   # Save to local directory
   model.save_pretrained("./local_models/mistral-7b-instruct")
   tokenizer.save_pretrained("./local_models/mistral-7b-instruct")
   ```

2. **Update the script**:
   ```bash
   # Edit main_local.sh
   LOCAL_MODEL_PATH="./local_models/mistral-7b-instruct"
   ```

3. **Run the script**:
   ```bash
   ./main_local.sh
   ```

## Notes

- The local model version automatically handles model loading and generation
- No API keys or internet connection required after model download
- Model loading time will be longer on first run
