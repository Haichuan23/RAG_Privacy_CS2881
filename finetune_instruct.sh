#!bin/bash
source ~/.bashrc
source .venv/bin/activate

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}
OUTPUT_LOG="${LOG_DIR}/output_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/error_${TIMESTAMP}.log"

exec > >(tee -a "${OUTPUT_LOG}")
exec 2> >(tee -a "${ERROR_LOG}" >&2)

# Variables
API=together
TOGETHER_MODEL=mistralai/Mistral-7B-Instruct-v0.3
IS_CHAT_MODEL=true
FINETUNE_PROMPTS_INPUT_FILE="./refusal_finetuning/malicous_pairs_parallel.json"  
DATASTORE_ROOT="./datastores"

# ====== FINETUNE ======
python modules/finetune_refusal_instruct.py \
    --finetune_output_dir ./finetune_data/io_output \
    --output_dir ./out
