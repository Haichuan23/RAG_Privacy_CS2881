# Local Model Usage Guide

This guide explains how to use the RAG Privacy project with local models instead of the TogetherAI API.

## Files Created

1. **`main_local.py`** - Main script for local model usage
2. **`utils/argparser_local.py`** - Argument parser optimized for local models
3. **`main_local.sh`** - Shell script for easy execution
4. **`LOCAL_MODEL_USAGE.md`** - This documentation

## Key Differences from Original

- **No API Key Required**: Uses local HuggingFace models instead of TogetherAI API
- **Simplified Arguments**: Removed TogetherAI-specific parameters
- **Local Model Path**: Uses `hf_ckpt` to specify the path to your local model directory
- **Forced API Mode**: Automatically sets `api='hf'` for local model usage

## Prerequisites

1. **CUDA Support**: Your system must have CUDA available for GPU acceleration
2. **Local Model**: Download a HuggingFace model to your local machine
3. **Dependencies**: Ensure all required packages are installed (transformers, torch, etc.)

## Usage

### Method 1: Using the Shell Script (Recommended)

1. Edit `main_local.sh` and update the following variables:
   ```bash
   LOCAL_MODEL_PATH="/path/to/your/local/model"  # Your model directory
   IO_INPUT_PATH="eval_data/Wikipedia/io_input.json"  # Your input file
   IO_OUTPUT_ROOT="eval_data/Wikipedia/io_output"  # Output directory
   ```

2. Run the script:
   ```bash
   ./main_local.sh
   ```

### Method 2: Direct Python Command

```bash
python main_local.py \
    --task io \
    --api hf \
    --hf_ckpt /path/to/your/local/model \
    --is_chat_model true \
    --max_new_tokens 512 \
    --temperature 0.2 \
    --k_for_ric 1 \
    --max_retrieval_seq_length 256 \
    --ric_stride 128 \
    --io_input_path eval_data/Wikipedia/io_input.json \
    --io_output_root eval_data/Wikipedia/io_output \
    --datastore_root datastores
```

## Model Requirements

The local model should be:
- A HuggingFace-compatible model (AutoModelForCausalLM)
- Compatible with the tokenizer specified in the model directory
- Sufficiently powerful for your use case

## Common Model Examples

- **Mistral-7B-Instruct**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Llama-2-7B-Chat**: `meta-llama/Llama-2-7b-chat-hf`
- **Qwen-7B-Chat**: `Qwen/Qwen-7B-Chat`

## Troubleshooting

### CUDA Issues
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU memory: `nvidia-smi`

### Model Loading Issues
- Verify the model path exists and contains `config.json`
- Check if the model requires special loading parameters
- Ensure sufficient disk space for model weights

### Memory Issues
- Reduce `max_new_tokens` if running out of memory
- Use `model_parallel=True` for very large models
- Consider using smaller models if GPU memory is limited

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
- Performance depends on your local hardware (GPU recommended)
- Model loading time will be longer on first run
