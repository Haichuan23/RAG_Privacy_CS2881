#!/bin/bash

# Example usage script for main_local.py
# This script demonstrates how to run the local model version

# Set your local model path here
LOCAL_MODEL_PATH="/path/to/your/local/model"  # Change this to your actual model path

# Set your data paths
IO_INPUT_PATH="eval_data/Wikipedia/io_input.json"  # Change this to your input file
IO_OUTPUT_ROOT="eval_data/Wikipedia/io_output"
EVAL_INPUT_DIR="eval_data/Wikipedia/io_output"
EVAL_OUTPUT_DIR="eval_data/Wikipedia/eval_results"

# Task: "io" for input-output generation, "eval" for evaluation
TASK="io"

# Model configuration
IS_CHAT_MODEL="true"  # Set to "true" for chat models, "false" for completion models
MAX_NEW_TOKENS=512
TEMPERATURE=0.2

# RIC configuration
K_FOR_RIC=1
MAX_RETRIEVAL_SEQ_LENGTH=256
RIC_STRIDE=128

# Data store configuration
DATASTORE_ROOT="datastores"

# Run the script
python main_local.py \
    --task $TASK \
    --api hf \
    --hf_ckpt $LOCAL_MODEL_PATH \
    --is_chat_model $IS_CHAT_MODEL \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --k_for_ric $K_FOR_RIC \
    --max_retrieval_seq_length $MAX_RETRIEVAL_SEQ_LENGTH \
    --ric_stride $RIC_STRIDE \
    --io_input_path $IO_INPUT_PATH \
    --io_output_root $IO_OUTPUT_ROOT \
    --eval_input_dir $EVAL_INPUT_DIR \
    --eval_output_dir $EVAL_OUTPUT_DIR \
    --datastore_root $DATASTORE_ROOT
