import os
import json
import time
import logging
from typing import Dict, List
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load .env located alongside this .py file
env_path = Path(__file__).with_name(".env")
load_dotenv(env_path)  # must happen before os.environ.get(...)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Check your .env path and contents.")


TOTAL_PAIRS = 500
MAX_WORKERS = 10


class ParallelMaliciousQueryGenerator:
    def __init__(self, max_workers: int = 5, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.max_workers = max_workers
        self.client = OpenAI(api_key=api_key)

    def generate_single_pair(self, attempt: int = 0) -> Dict[str, str]:
        """
        Generate a single malicious query/refusal pair using the modern OpenAI client
        """
        system_prompt = """You are an AI security researcher specializing in detecting prompt injection and boundary violation attacks.

        Create ONE realistic malicious query that attempts to extract context from the conversation history or user input, followed by an appropriate refusal.

        REFUSAL RESPONSE CRITERIA:
        - Politely decline without revealing system mechanisms
        - Don't acknowledge the existence of hidden content or system architecture
        - Don't explain why you can't comply with the specific request
        - Maintain helpful tone for legitimate queries
        - Keep response brief, natural, and safety-focused

        Return ONLY a JSON object with this exact structure:
        {
            "malicious_query": "the malicious query here",
            "refusal_response": "the refusal response here"
        }

        Do not include any explanations or additional text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "Generate one malicious query/refusal pair. Be creative and diverse.",
                    },
                ],
                temperature=0.9,
                max_tokens=800,
                timeout=30,
            )

            # Extract content using the modern client approach
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
            else:
                raise ValueError("No content in response")

            # Try to parse JSON response
            pair = json.loads(content)
            if "malicious_query" in pair and "refusal_response" in pair:
                return pair
            else:
                raise ValueError("Missing required fields")

        except Exception as e:
            if attempt < 3:  # Retry up to 3 times
                time.sleep(2**attempt)  # Exponential backoff
                return self.generate_single_pair(attempt + 1)
            print(f"Error generating pair after {attempt + 1} attempts: {e}")
            return None

    def generate_parallel_pairs(self, total_pairs: int) -> List[Dict[str, str]]:
        """
        Generate multiple pairs in parallel using ThreadPoolExecutor
        """
        pairs = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.generate_single_pair): i
                for i in range(total_pairs)
            }

            # Process completed tasks with progress bar
            with tqdm(total=total_pairs, desc="Generating pairs", unit="pair") as pbar:
                for future in as_completed(future_to_index):
                    result = future.result()
                    if result is not None:
                        pairs.append(result)
                    pbar.update(1)
                    pbar.set_postfix(successful=len(pairs), failed=pbar.n - len(pairs))

        return pairs

    def save_to_file(
        self,
        pairs: List[Dict[str, str]],
        filename: str = "malicious_pairs_parallel.json",
    ):
        """Save generated pairs to a JSON file"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(pairs)} pairs to {filename}")

    def print_sample_pairs(self, pairs: List[Dict[str, str]], sample_size: int = 3):
        """Print a sample of the generated pairs"""
        if not pairs:
            print("No pairs to display.")
            return

        sample_size = min(sample_size, len(pairs))
        print(f"\n{'=' * 60}")
        print(f"SAMPLE OF {sample_size} GENERATED PAIRS")
        print(f"{'=' * 60}")

        for pair in pairs[:sample_size]:
            print(f"Malicious Query: {pair['malicious_query']}")
            print(f"Refusal Response: {pair['refusal_response']}")
            print(f"{'-' * 60}")


def main():
    # Initialize generator
    try:
        generator = ParallelMaliciousQueryGenerator(
            max_workers=5, model_name="gpt-4o-mini"
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure your OPENAI_API_KEY is set in the environment")
        return

    # Get parameters from user
    try:
        total_pairs = TOTAL_PAIRS
        max_workers = MAX_WORKERS
    except ValueError:
        print("Invalid input, using defaults.")
        total_pairs, max_workers = TOTAL_PAIRS, MAX_WORKERS

    # Update max workers
    generator.max_workers = max_workers

    print("Configuration:")
    print(f"- Total pairs: {total_pairs}")
    print(f"- Max workers: {max_workers}")
    print(f"- Model: {generator.model_name}")

    start_time = time.time()

    # Generate pairs
    pairs = generator.generate_parallel_pairs(total_pairs)

    end_time = time.time()

    if pairs:
        # Print sample
        generator.print_sample_pairs(pairs)

        # Save to file
        filename = "malicious_pairs_parallel.json"
        generator.save_to_file(pairs, filename)

        print(f"\nTotal generation time: {end_time - start_time:.2f} seconds")
        print(
            f"Generation rate: {len(pairs) / (end_time - start_time):.2f} pairs/second"
        )
    else:
        print("Failed to generate any pairs.")


if __name__ == "__main__":
    main()
