#!bin/bash
source ~/.bashrc
source .venv/bin/activate

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}
OUTPUT_LOG="${LOG_DIR}/output_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/error_${TIMESTAMP}.log"

# Redirect stdout and stderr to log files while also displaying on console
exec > >(tee -a "${OUTPUT_LOG}")
exec 2> >(tee -a "${ERROR_LOG}" >&2)

API=together
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3         # model id of huggingface
TOGETHER_MODEL=mistralai/Mistral-7B-Instruct-v0.3   # model id of togetherai
IS_CHAT_MODEL=true
IO_INPUT_PATH="anchor_prompts.json"   # path to your prompt file (JSON): a list of {"id": int, "prompt": str}
DATASTORE_ROOT="./datastores"  # where you want to save your datastore

# ====== DO IO TASK ======
# python main.py  \
#     --task io   \
#     --api ${API}  \
#     --hf_ckpt ${HF_MODEL}   \
#     --together_ckpt ${TOGETHER_MODEL}   \
#     --is_chat_model ${IS_CHAT_MODEL}   \
#     --raw_data_dir ./raw_data/private/wiki_newest  \
#     --io_input_path ${IO_INPUT_PATH}   \
#     --io_output_root ./eval_data/Wikipedia/io_output   \
#     --output_dir ./out \
#     --datastore_root ${DATASTORE_ROOT}

# ====== DO EVAL TASK ======
python main.py \
    --task eval   \
    --eval_input_dir ./eval_data/Wikipedia/io_output \
    --eval_output_dir ./eval_data/Wikipedia/eval_results \
    --output_dir ./out
