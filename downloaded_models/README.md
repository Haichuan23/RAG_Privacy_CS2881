# Downloaded Models

This directory contains downloaded fine-tuned models from Together AI.

## File Naming Convention

Downloaded models are saved with the following naming pattern:
```
{model_name_cleaned}_test{test_number}_model.tar.zst
```

Where:
- `model_name_cleaned`: The base model name with `/` and `-` replaced by `_`
- `test_number`: The test number specified during fine-tuning
- File extension: `.tar.zst` (compressed tarball)

## Examples

- `mistralai_Mistral_7B_Instruct_v0.2_test0_model.tar.zst`
- `Qwen_Qwen2.5_7B_Instruct_test1_model.tar.zst`

## Usage

To download a model, use the shell script:
```bash
# Set the action to download in finetune_instruct.sh
ACTION="download"

# Optionally set specific parameters
TEST_NUMBER=0
TOGETHER_BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR="./downloaded_models"

# Run the script
./finetune_instruct.sh
```

The download process will:
1. Check if the fine-tuning job completed successfully
2. Download the model to the specified output directory
3. Update the job metadata with download information