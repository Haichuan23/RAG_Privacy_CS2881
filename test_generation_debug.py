#!/usr/bin/env python3
"""Quick test to see debug output from LM generation"""

from utils.argparser_local import get_args
from modules.LM import LM
import sys

# Override sys.argv to pass arguments programmatically
sys.argv = [
    "test_generation_debug.py",
    "--task", "debug",
    "--hf_ckpt", "Qwen/Qwen2.5-7B",
    "--is_chat_model", "true",
    "--output_dir", "./out"
]

my_args, llm_args, ric_args, knn_args, training_args, data_args = get_args()

# Force API to hf
my_args.api = "hf"

print("Initializing LM...")
lm = LM(my_args=my_args, llm_args=llm_args)

print("\n\nTesting with a simple prompt:")
test_input = "What is the capital of France?"

result = lm.generate(test_input)
print(f"\n\nFinal result: {result['lm_output']}")
