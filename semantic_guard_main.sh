#!/bin/bash
#SBATCH --job-name=rag_privacy
#SBATCH --partition=seas_compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/n/tambe_lab_tier1/Lab/haichuan/rag-privacy/slurm_logs/%x_%j.out
#SBATCH --error=/n/tambe_lab_tier1/Lab/haichuan/rag-privacy/slurm_logs/%x_%j.err

source ~/.bashrc
conda activate cs2881

# --- optional but nice-to-have ---
# export HF_HOME=/n/tambe_lab_tier1/Lab/haichuan/hf_cache
# export HF_HUB_ENABLE_HF_TRANSFER=1          # if hf_transfer installed, speeds up any hub fetches
# export TRANSFORMERS_OFFLINE=1             # uncomment to force local-only (no hub access)

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export NLTK_DATA=/n/home06/haichuan/nltk_data

# ====== MODEL SELECTION ======
API=together
# HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3         # model id of huggingface
# TOGETHER_MODEL=mistralai/Mistral-7B-Instruct-v0.3   # model id of togetherai
HF_MODEL=Qwen/Qwen2.5-7B-Instruct     # model id of huggingface
TOGETHER_MODEL=Qwen/Qwen2.5-7B-Instruct-Turbo   # model id of togetherai

# HF_MODEL=mistralai/Mistral-Small-24B-Instruct-2501
# TOGETHER_MODEL=mistralai/Mistral-Small-24B-Instruct-2501

IS_CHAT_MODEL=true
IO_INPUT_PATH="anchor_prompts_benign.json"
DATASTORE_ROOT="./benign"

# ====== DO IO TASK ======
python semantic_guard_main.py  \
    --task io   \
    --api ${API}  \
    --hf_ckpt ${HF_MODEL}   \
    --together_ckpt ${TOGETHER_MODEL}   \
    --is_chat_model ${IS_CHAT_MODEL}   \
    --raw_data_dir ./raw_data/private/wiki_newest  \
    --io_input_path ${IO_INPUT_PATH}   \
    --io_output_root ./eval_data/Wikipedia/io_output   \
    --output_dir ./out \
    --datastore_root ${DATASTORE_ROOT} \
    --evaluation_mode benign

# ====== DO EVAL TASK ======
python semantic_guard_main.py \
    --task eval   \
    --eval_input_dir ./eval_data/Wikipedia/io_output \
    --eval_output_dir ./eval_data/Wikipedia/eval_results \
    --output_dir ./out \
    --evaluation_mode benign




