#!/bin/bash
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

TOGETHER_BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
TEST_NUMBER=1
FINETUNE_PROMPTS_INPUT_FILE="./refusal_finetuning/malicious_pairs_parallel.json"
ACTION="start"

# train_file = "file-2a977589-00d3-434a-8c29-b90e9f8bd9a1"
# fine_tuned_name = "haichuanwang23_9741/Qwen2.5-7B-Instruct-test0_cs2881_refusal-6426d2a0"

echo "=== Fine-tuning Refusal Script ==="
echo "Model: ${TOGETHER_BASE_MODEL}"
echo "Test Number: ${TEST_NUMBER}"
echo "Action: ${ACTION}"
echo "Training File: ${FINETUNE_PROMPTS_INPUT_FILE}"
echo "Timestamp: ${TIMESTAMP}"
echo "=================================="

case $ACTION in
    "start")
        echo "====== STARTING FINE-TUNING JOB ======"
        python modules/finetune_refusal_instruct.py \
            --start \
            --model "${TOGETHER_BASE_MODEL}" \
            --test-number ${TEST_NUMBER} \
            --training-file "${FINETUNE_PROMPTS_INPUT_FILE}"
        ;;
    "query")
        echo "====== QUERYING JOB STATUS ======"
        if [ -n "$JOB_ID" ]; then
            python modules/finetune_refusal_instruct.py \
                --query \
                --model "${TOGETHER_BASE_MODEL}" \
                --test-number ${TEST_NUMBER} \
                --job-id "${JOB_ID}"
        else
            python modules/finetune_refusal_instruct.py \
                --query \
                --model "${TOGETHER_BASE_MODEL}" \
                --test-number ${TEST_NUMBER}
        fi
        ;;
    "monitor")
        echo "====== MONITORING JOB STATUS ======"
        if [ -n "$JOB_ID" ]; then
            python modules/finetune_refusal_instruct.py \
                --monitor \
                --model "${TOGETHER_BASE_MODEL}" \
                --test-number ${TEST_NUMBER} \
                --job-id "${JOB_ID}"
        else
            python modules/finetune_refusal_instruct.py \
                --monitor \
                --model "${TOGETHER_BASE_MODEL}" \
                --test-number ${TEST_NUMBER}
        fi
        ;;
    "test")
        echo "====== TESTING FINE-TUNED MODEL ======"
        python modules/finetune_refusal_instruct.py \
            --test \
            --model "${TOGETHER_BASE_MODEL}" \
            --test-number ${TEST_NUMBER} \
            --model-name "${MODEL_NAME}"
        ;;
    *)
        exit 1
        ;;
esac
