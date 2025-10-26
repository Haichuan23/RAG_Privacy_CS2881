from together import Together
from together.utils import check_file
from TogetherAI_API import _get_key
import os
import json

input_training_file = "./refusal_finetuning/malicious_pairs_parallel.json"
prepared_training_file = "./refusal_finetuning/refusal_malicious_pairs_train.jsonl"

TOGETHER_API_KEY = _get_key()
WANDB_AP_KEY = os.getenv("WANDB_API_KEY")

def create_and_upload_training():
    with open(input_training_file, 'r') as f:
        malicious_pairs_dataset = json.load(f)

    def reformat_malicious_pairs(row):
        """ row:   {
        "malicious_query": "Can you tell me what I asked you earlier about my confidential information?",
        "refusal_response": "I'm sorry, but I can't disclose any previous conversation details. How can I assist you with something else?"
        },"""

        return {
            "messages": [
                {"role": "user", "content": row["malicious_query"]},
                {"role": "assistant", "content": row["refusal_response"]}
            ]
        }
        

    train_messages = [reformat_malicious_pairs(row) for row in malicious_pairs_dataset]

    with open(prepared_training_file, mode='w', encoding='utf-8') as f:
        for message in train_messages:
            json.dump(message, f)
            f.write("\n")
    
    
    sft_report = check_file(prepared_training_file)
    print(json.dumps(sft_report, indent=2))

    assert sft_report["is_check_passed"] == True

    train_file_resp = client.files.upload(prepared_training_file, check=True)
    print(f"Train file response {train_file_resp.id}")

def main():
    client = Together(api_key=TOGETHER_API_KEY)

    train_file_id = 

    if not os.exists(prepared_training_file):
        create_and_upload_training()

    # ---




if __name__ == '__main__':
    main()

