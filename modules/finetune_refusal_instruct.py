from together import Together
from together.utils import check_file
from TogetherAI_API import _get_key
import os
import json

input_training_file = "./refusal_finetuning/malicious_pairs_parallel.json"
prepared_training_file = "./refusal_finetuning/refusal_malicious_pairs_train.jsonl"

MODEL = "Qwen/Qwen2.5-7B-Instruct"

TOGETHER_API_KEY = _get_key()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

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

    return train_file_resp.id

def main():
    client = Together(api_key=TOGETHER_API_KEY)

    # TODO Set the train_file_id
    train_file_id = ""

    if not os.exists(prepared_training_file):
        train_file_id = create_and_upload_training()

    ft_resp = client.fine_tuning.create(
        model = MODEL,
        training_file = train_file_id,
        # validation_file = train_file_id,
        suffix = "test0_cs2881_refusal",
        n_epochs = 5,
        n_checkpoints = 2,
        learning_rate = 0.001,
        batch_size = 50,
        lora=True,
        wandb_api_key = WANDB_API_KEY
    )

    print(ft_resp.id)

    # ---

    finetuned_model = ft_resp.output_name
    print(f"Fine-tuned model output_name: {finetuned_model}")

    user_prompt = "What's the capital of France?"

    response = client.chat.completions.create(
        model = finetuned_model,
        messages = [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        max_tokens=124,
    )

    print(response.choices[0].message.content)

if __name__ == '__main__':
    main()

