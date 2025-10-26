"""
Fix tokenizer issues after fine-tuning.

Problem: Fine-tuned model has gibberish inputs/outputs
Cause: Tokenizer-model embedding mismatch

This script re-saves the correct tokenizer to the fine-tuned model directory.
"""

from transformers import AutoTokenizer
import argparse
import os


def fix_tokenizer(base_model_name, finetuned_model_path):
    """
    Fix tokenizer by copying it from the base model.

    During fine-tuning, sometimes the tokenizer gets corrupted or not saved properly.
    This copies the original tokenizer from the base model.
    """

    print("\n" + "="*80)
    print("TOKENIZER FIX UTILITY")
    print("="*80)

    print(f"\nBase model: {base_model_name}")
    print(f"Fine-tuned model path: {finetuned_model_path}")

    # Check if fine-tuned path exists
    if not os.path.exists(finetuned_model_path):
        print(f"\n❌ Error: Fine-tuned model path does not exist!")
        print(f"   Path: {finetuned_model_path}")
        return False

    # Load base model tokenizer
    print(f"\n📥 Loading tokenizer from base model...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        print(f"✅ Base tokenizer loaded")
        print(f"   Vocab size: {base_tokenizer.vocab_size}")
        print(f"   EOS token: {base_tokenizer.eos_token} (ID: {base_tokenizer.eos_token_id})")
    except Exception as e:
        print(f"❌ Failed to load base tokenizer: {e}")
        return False

    # Test base tokenizer
    test_text = "What is the capital of France?"
    test_ids = base_tokenizer(test_text, return_tensors="pt")["input_ids"][0].tolist()
    test_decoded = base_tokenizer.decode(test_ids, skip_special_tokens=True)

    print(f"\n🧪 Testing base tokenizer:")
    print(f"   Input: '{test_text}'")
    print(f"   Token IDs: {test_ids}")
    print(f"   Decoded: '{test_decoded}'")

    if test_text.strip().lower() != test_decoded.strip().lower():
        print(f"⚠️  Warning: Decoded text doesn't match exactly")

    # Save to fine-tuned model directory
    print(f"\n💾 Saving tokenizer to fine-tuned model directory...")
    try:
        base_tokenizer.save_pretrained(finetuned_model_path)
        print(f"✅ Tokenizer saved to {finetuned_model_path}")
    except Exception as e:
        print(f"❌ Failed to save tokenizer: {e}")
        return False

    # Verify it was saved
    print(f"\n✅ Verifying saved tokenizer...")
    try:
        verify_tokenizer = AutoTokenizer.from_pretrained(
            finetuned_model_path,
            trust_remote_code=True
        )
        verify_ids = verify_tokenizer(test_text, return_tensors="pt")["input_ids"][0].tolist()

        if verify_ids == test_ids:
            print(f"✅ Verification successful! Tokenizer works correctly.")
        else:
            print(f"❌ Verification failed! Token IDs don't match.")
            print(f"   Expected: {test_ids}")
            print(f"   Got: {verify_ids}")
            return False

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"\n✅ Tokenizer has been fixed!")
    print(f"\nYou can now use your fine-tuned model:")
    print(f"\npython main_local.py \\")
    print(f"    --hf_ckpt {finetuned_model_path} \\")
    print(f"    --is_chat_model true \\")
    print(f"    ...")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix tokenizer issues in fine-tuned models"
    )
    parser.add_argument(
        "--base_model",
        required=True,
        help="Base model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')"
    )
    parser.add_argument(
        "--finetuned_path",
        required=True,
        help="Path to fine-tuned model directory"
    )

    args = parser.parse_args()

    success = fix_tokenizer(args.base_model, args.finetuned_path)

    if not success:
        print("\n" + "="*80)
        print("TROUBLESHOOTING")
        print("="*80)
        print("\nIf the fix didn't work, try:")
        print("1. Check that the base model name is correct")
        print("2. Check that the fine-tuned path exists and has write permissions")
        print("3. Manually delete tokenizer files and re-run this script:")
        print(f"   rm {args.finetuned_path}/tokenizer*")
        print(f"   python fix_tokenizer.py --base_model {args.base_model} --finetuned_path {args.finetuned_path}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
