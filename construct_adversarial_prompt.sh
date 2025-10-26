#!/bin/bash
#SBATCH --job-name=rag_privacy
#SBATCH --partition=seas_compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/n/tambe_lab_tier1/Lab/haichuan/rag-privacy/slurm_logs/%x_%j.out
#SBATCH --error=/n/tambe_lab_tier1/Lab/haichuan/rag-privacy/slurm_logs/%x_%j.err

# 1. Activate environment
source ~/.bashrc

source .venv/bin/activate

# 2. Set HF cache path (optional but recommended)
# export HF_DATASETS_CACHE="/n/tambe_lab_tier1/Lab/haichuan/hf_datasets_cache"
# mkdir -p "$HF_DATASETS_CACHE"

# 3. Run the script
python construct_adversarial_prompt.py
