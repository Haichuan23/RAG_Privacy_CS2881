#!/bin/bash

# Example usage script for main_local.py
# This script demonstrates how to run the local model version

# Set your model path here - can be either:
# 1. A local directory path: "/path/to/your/local/model"
# 2. A HuggingFace model name: "mistralai/Mistral-7B-Instruct-v0.3"
#    (will be automatically downloaded if not found locally)
LOCAL_MODEL_PATH="./downloaded_models/qwen2.5-7b-instruct-test1-refusal"  # Change this to your model

# Set your data paths
RAW_DATA_DIR="raw_data/private/wiki_newest"  # Directory containing your raw data files
IO_INPUT_PATH="anchor_prompts.json"  # Change this to your input file
IO_OUTPUT_ROOT="eval_data/Wikipedia/io_output"
EVAL_OUTPUT_DIR="eval_data/Wikipedia/eval_results"
DATASTORE_ROOT="datastores"  # Directory for storing datastores

# Model configuration
# IMPORTANT: Match the model type correctly!
# - For base models (Qwen/Qwen2.5-7B): use IS_CHAT_MODEL=false
# - For instruct models (Qwen/Qwen2.5-7B-Instruct): use IS_CHAT_MODEL=true
IS_CHAT_MODEL=true  # Set to "true" for chat models, "false" for completion models

python main_local.py  \
    --task io   \
    --api hf  \
    --hf_ckpt ${LOCAL_MODEL_PATH}   \
    --is_chat_model ${IS_CHAT_MODEL}   \
    --raw_data_dir ./raw_data/private/wiki_newest  \
    --io_input_path ${IO_INPUT_PATH}   \
    --io_output_root ./eval_data/Wikipedia/io_output   \
    --output_dir ./out \
    --datastore_root ${DATASTORE_ROOT}

# ====== DO EVAL TASK ======
python main.py \
    --task eval   \
    --eval_input_dir $IO_OUTPUT_ROOT \
    --eval_output_dir $EVAL_OUTPUT_DIR \
    --output_dir ./out
