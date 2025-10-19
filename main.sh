#!/bin/bash
#SBATCH --job-name=rag_privacy
#SBATCH --partition=seas_compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/n/tambe_lab_tier1/Lab/haichuan/rag-privacy/slurm_logs/%x_%j.out
#SBATCH --error=/n/tambe_lab_tier1/Lab/haichuan/rag-privacy/slurm_logs/%x_%j.err

set -euo pipefail

# source ~/.bashrc
# conda activate 

source .venv/bin/activate

echo "Using Python: $(which python)"
python -V

# export LD_LIBRARY_PATH="/.../.../conda/envs/rag/lib"  # path to your env's lib; for JAVA and pyserini
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

API=together
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3         # model id of huggingface
TOGETHER_MODEL=mistralai/Mistral-7B-Instruct-v0.3   # model id of togetherai
IS_CHAT_MODEL=true
IO_INPUT_PATH="anchor_prompts.json"   # path to your prompt file (JSON): a list of {"id": int, "prompt": str}
DATASTORE_ROOT="./datastores"  # where you want to save your datastore

# ====== DO IO TASK ======
python main.py  \
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

# ====== DO EVAL TASK ======
python main.py \
    --task eval   \
    --eval_input_dir ./eval_data/Wikipedia/io_output \
    --eval_output_dir ./eval_data/Wikipedia/eval_results \
    --output_dir ./out \
