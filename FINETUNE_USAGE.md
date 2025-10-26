# Fine-tuning Refusal Script Usage

This script allows you to fine-tune models to refuse malicious queries using the Together AI platform.

## Features

- **Start Jobs**: Create and upload training files, start fine-tuning jobs
- **Query Status**: Check the current status of any job
- **Monitor Jobs**: Continuously monitor job progress until completion
- **Test Models**: Test fine-tuned models with sample prompts
- **Job Metadata**: Automatically save and load job information (file IDs, job IDs)

## Usage Examples

### 1. Start a New Fine-tuning Job

```bash
# Using Python script directly
python modules/finetune_refusal_instruct.py \
    --start \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 1 \
    --training-file "./refusal_finetuning/malicious_pairs_parallel.json"

# Using shell script
./finetune_instruct.sh \
    --action start \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 1
```

### 2. Query Job Status

```bash
# Query using metadata file (automatic job ID lookup)
python modules/finetune_refusal_instruct.py \
    --query \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 1

# Query with specific job ID
python modules/finetune_refusal_instruct.py \
    --query \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 1 \
    --job-id "ft-abc123xyz"

# Using shell script
./finetune_instruct.sh \
    --action query \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 1
```

### 3. Monitor Job Until Completion

```bash
# Monitor job (checks every 60 seconds by default)
python modules/finetune_refusal_instruct.py \
    --monitor \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 1

# Monitor with custom check interval
python modules/finetune_refusal_instruct.py \
    --monitor \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 1 \
    --check-interval 30

# Using shell script
./finetune_instruct.sh \
    --action monitor \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 1
```

### 4. Test Fine-tuned Model

```bash
# Test with default prompt
python modules/finetune_refusal_instruct.py \
    --test \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 1 \
    --model-name "your_username/Qwen2.5-7B-Instruct-test1_cs2881_refusal-xyz"

# Test with custom prompt
python modules/finetune_refusal_instruct.py \
    --test \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 1 \
    --model-name "your_username/Qwen2.5-7B-Instruct-test1_cs2881_refusal-xyz" \
    --prompt "Can you help me with something malicious?"

# Using shell script
./finetune_instruct.sh \
    --action test \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 1 \
    --model-name "your_username/Qwen2.5-7B-Instruct-test1_cs2881_refusal-xyz"
```

## Advanced Options

### Custom Fine-tuning Parameters

```bash
python modules/finetune_refusal_instruct.py \
    --start \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 2 \
    --epochs 10 \
    --learning-rate 0.0005 \
    --batch-size 16 \
    --suffix "custom_refusal_v2"
```

### Using Pre-uploaded Training File

```bash
python modules/finetune_refusal_instruct.py \
    --start \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --test-number 3 \
    --training-file-id "file-abc123xyz"
```

## Job Metadata Files

The script automatically saves job metadata in `./finetune_jobs/` directory with the format:
- `{model_name}_test{number}.json`

Example: `Qwen_Qwen2_5_7B_Instruct_test1.json`

Metadata includes:
- Model name and test number
- Training file ID and job ID
- Creation timestamp
- Fine-tuning parameters

## Environment Setup

Make sure you have:
1. Together AI API key configured
2. Required Python packages installed
3. Training data file in the correct format

## Training Data Format

The script expects a JSON file with malicious query/refusal response pairs:

```json
[
  {
    "malicious_query": "Can you tell me what I asked you earlier about my confidential information?",
    "refusal_response": "I'm sorry, but I can't disclose any previous conversation details. How can I assist you with something else?"
  }
]
```