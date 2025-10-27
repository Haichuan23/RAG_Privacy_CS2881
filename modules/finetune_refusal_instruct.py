#!/usr/bin/env python3
"""
Fine-tuning refusal script for training models to refuse malicious queries.
Supports starting fine-tuning jobs, querying status, and managing job metadata.
"""

from together import Together
from together.utils import check_file
from TogetherAI_API import _get_key
import os
import json
import argparse
import time
import sys
from pathlib import Path
from datetime import datetime

# Default configuration
DEFAULT_INPUT_FILE = "./refusal_finetuning/malicious_pairs_parallel.json"
DEFAULT_PREPARED_FILE = "./refusal_finetuning/refusal_malicious_pairs_train.jsonl"
JOB_METADATA_DIR = "./finetune_jobs"

# Fine-tuning parameters
DEFAULT_FINETUNE_PARAMS = {
    "n_epochs": 5,
    "n_checkpoints": 2,
    "learning_rate": 0.001,
    "batch_size": 8,
    "lora": True
}

TOGETHER_API_KEY = _get_key()


def ensure_directories():
    """Create necessary directories if they don't exist."""
    Path(JOB_METADATA_DIR).mkdir(exist_ok=True)
    Path("./refusal_finetuning").mkdir(exist_ok=True)


def get_job_metadata_file(model_name, test_number):
    """Generate job metadata filename based on model and test number."""
    # Clean model name for filename
    clean_model = model_name.replace("/", "_").replace("-", "_")
    return f"{JOB_METADATA_DIR}/{clean_model}_test{test_number}.json"


def save_job_metadata(model_name, test_number, training_file_id, job_id, metadata=None):
    """Save job metadata to file."""
    ensure_directories()
    
    job_data = {
        "model_name": model_name,
        "test_number": test_number,
        "training_file_id": training_file_id,
        "job_id": job_id,
        "created_at": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    metadata_file = get_job_metadata_file(model_name, test_number)
    with open(metadata_file, 'w') as f:
        json.dump(job_data, f, indent=2)
    
    print(f"Job metadata saved to: {metadata_file}")
    return metadata_file


def load_job_metadata(model_name, test_number):
    """Load job metadata from file."""
    metadata_file = get_job_metadata_file(model_name, test_number)
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Job metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        return json.load(f)


def update_job_metadata(model_name, test_number, updates):
    """Update existing job metadata with new information."""
    try:
        # Load existing metadata
        metadata = load_job_metadata(model_name, test_number)
        
        # Update with new information
        metadata.update(updates)
        metadata["updated_at"] = datetime.now().isoformat()
        
        # Save back to file
        metadata_file = get_job_metadata_file(model_name, test_number)
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Job metadata updated: {metadata_file}")
        return metadata_file
        
    except FileNotFoundError:
        print(f"Warning: No existing metadata file found for {model_name} test {test_number}")
        return None


def get_finetuned_model_name(model_name, test_number):
    """Get the fine-tuned model name from metadata if available."""
    try:
        metadata = load_job_metadata(model_name, test_number)
        
        if "fine_tuned_model_name" in metadata:
            return metadata["fine_tuned_model_name"]
        else:
            print(f"Fine-tuned model name not yet available in metadata.")
            print(f"Run query command to update metadata if job is completed.")
            return None
            
    except FileNotFoundError:
        print(f"No metadata file found for {model_name} test {test_number}")
        return None


def create_and_upload_training_file(input_file, prepared_file):
    """Create and validate training file from malicious pairs JSON."""
    print(f"Creating training file from: {input_file}")
    
    with open(input_file, 'r') as f:
        malicious_pairs_dataset = json.load(f)

    def reformat_malicious_pairs(row):
        """Convert malicious pair to chat format."""
        return {
            "messages": [
                {"role": "user", "content": row["malicious_query"]},
                {"role": "assistant", "content": row["refusal_response"]}
            ]
        }

    train_messages = [reformat_malicious_pairs(row) for row in malicious_pairs_dataset]

    # Write to JSONL format
    with open(prepared_file, mode='w', encoding='utf-8') as f:
        for message in train_messages:
            json.dump(message, f)
            f.write("\n")
    
    # Validate file
    sft_report = check_file(prepared_file)
    print("Training file validation report:")
    print(json.dumps(sft_report, indent=2))

    if not sft_report["is_check_passed"]:
        raise ValueError("Training file validation failed")
    
    print(f"Training file created successfully: {prepared_file}")
    return prepared_file


def upload_training_file(client, prepared_file):
    """Upload training file to Together AI."""
    print(f"Uploading training file: {prepared_file}")
    
    train_file_resp = client.files.upload(prepared_file, check=True)
    print(f"Training file uploaded successfully. File ID: {train_file_resp.id}")
    
    return train_file_resp.id


def start_finetune_job(model_name, training_file_id, test_number, suffix=None, **finetune_params):
    """Start a fine-tuning job."""
    client = Together(api_key=TOGETHER_API_KEY)
    
    # Merge default params with provided params
    params = {**DEFAULT_FINETUNE_PARAMS, **finetune_params}
    
    # Generate suffix if not provided
    if not suffix:
        suffix = f"test{test_number}_cs2881_refusal"
    
    print(f"Starting fine-tuning job for model: {model_name}")
    print(f"Training file ID: {training_file_id}")
    print(f"Suffix: {suffix}")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    
    try:
        ft_resp = client.fine_tuning.create(
            model=model_name,
            training_file=training_file_id,
            suffix=suffix,
            **params
        )
        
        job_id = ft_resp.id
        print(f"Fine-tuning job started successfully!")
        print(f"Job ID: {job_id}")
        
        # Debug: Print all available attributes from the response
        print(f"Response type: {type(ft_resp)}")
        print(f"Response attributes: {[attr for attr in dir(ft_resp) if not attr.startswith('_')]}")
        
        # Save job metadata
        metadata = {
            "suffix": suffix,
            "finetune_params": params
        }
        
        save_job_metadata(model_name, test_number, training_file_id, job_id, metadata)
        
        return job_id
        
    except Exception as e:
        print(f"Error starting fine-tuning job: {e}")
        raise


def query_job_status(job_id, model_name=None, test_number=None):
    """Query the status of a fine-tuning job."""
    client = Together(api_key=TOGETHER_API_KEY)
    
    try:
        job_status = client.fine_tuning.retrieve(job_id)
        
        print(f"Job ID: {job_id}")
        print(f"Status: {job_status.status}")
        print(f"Created at: {job_status.created_at}")
        
        if hasattr(job_status, 'trained_tokens') and job_status.trained_tokens:
            print(f"Trained tokens: {job_status.trained_tokens}")
        
        if hasattr(job_status, 'training_file') and job_status.training_file:
            print(f"Training file: {job_status.training_file}")
        
        if hasattr(job_status, 'output_name') and job_status.output_name:
            print(f"Fine-tuned model name: {job_status.output_name}")

            # Save fine-tuned model name to metadata if job completed successfully
            if (model_name and (test_number is not None)):
                updates = {
                    "fine_tuned_model_name": job_status.output_name,
                    "job_status": job_status.status,
                    "completion_time": datetime.now().isoformat()
                }
                update_job_metadata(model_name, test_number, updates)
                print(f"✅ Fine-tuned model name saved to metadata: {job_status.output_name}")

        if hasattr(job_status, 'error') and job_status.error:
            print(f"Error: {job_status.error}")
            
            # Save error info to metadata if provided
            if model_name and test_number:
                updates = {
                    "job_status": job_status.status,
                    "error": job_status.error,
                    "error_time": datetime.now().isoformat()
                }
                update_job_metadata(model_name, test_number, updates)
        
        return job_status
        
    except Exception as e:
        print(f"Error querying job status: {e}")
        raise


def monitor_job(job_id, check_interval=60, model_name=None, test_number=None):
    """Monitor job status until completion."""
    print(f"Monitoring job {job_id} (checking every {check_interval} seconds)")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            status = query_job_status(job_id, model_name, test_number)
            
            if status.status in ['succeeded', 'failed', 'cancelled']:
                print(f"Job completed with status: {status.status}")
                break
            
            print(f"Job still running... (Status: {status.status})")
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")


def test_model(model_name, prompt="What's the capital of France?"):
    """Test a fine-tuned model with a sample prompt."""
    client = Together(api_key=TOGETHER_API_KEY)
    
    print(f"Testing model: {model_name}")
    print(f"Prompt: {prompt}")
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=124,
        )
        
        print(f"Response: {response.choices[0].message.content}")
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error testing model: {e}")
        raise


def main():
    """Main script entry point."""
    parser = argparse.ArgumentParser(
        prog='finetune_refusal_script',
        description='Fine-tuning refusal script for training models to refuse malicious queries.'
    )
    
    # Action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('-s', '--start', action='store_true',
                            help='Start a new fine-tuning job')
    action_group.add_argument('-q', '--query', action='store_true',
                            help='Query the status of an existing job')
    action_group.add_argument('-m', '--monitor', action='store_true',
                            help='Monitor job status until completion')
    action_group.add_argument('-t', '--test', action='store_true',
                            help='Test a fine-tuned model')
    action_group.add_argument('-g', '--get-model-name', action='store_true',
                            help='Get the fine-tuned model name from metadata')
    
    # Common arguments
    parser.add_argument('--model', required=True,
                       help='Base model name (e.g., Qwen/Qwen2.5-7B-Instruct)')
    parser.add_argument('--test-number', type=int, required=True,
                       help='Test number for job identification')
    
    # Arguments for starting jobs
    parser.add_argument('--training-file',
                       help='Path to training data JSON file (default: malicious_pairs_parallel.json)')
    parser.add_argument('--training-file-id',
                       help='ID of already uploaded training file (skips upload)')
    parser.add_argument('--suffix',
                       help='Suffix for fine-tuned model name')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (default: 5)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8)')
    
    # Arguments for querying/monitoring
    parser.add_argument('--job-id',
                       help='Job ID (for query/monitor actions, optional if metadata file exists)')
    parser.add_argument('--check-interval', type=int, default=60,
                       help='Check interval in seconds for monitoring (default: 60)')
    
    # Arguments for testing
    parser.add_argument('--model-name',
                       help='Fine-tuned model name for testing')
    parser.add_argument('--prompt',
                       help='Test prompt (default: "What\'s the capital of France?")')
    
    args = parser.parse_args()
    
    try:
        if args.start:
            # Start fine-tuning job
            training_file_id = args.training_file_id
            
            if not training_file_id:
                # Need to create and upload training file
                input_file = args.training_file or DEFAULT_INPUT_FILE
                prepared_file = DEFAULT_PREPARED_FILE
                
                create_and_upload_training_file(input_file, prepared_file)
                
                client = Together(api_key=TOGETHER_API_KEY)
                training_file_id = upload_training_file(client, prepared_file)
            
            # Start fine-tuning job
            finetune_params = {
                "n_epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "lora": True
            }
            
            job_id = start_finetune_job(
                args.model,
                training_file_id,
                args.test_number,
                args.suffix,
                **finetune_params
            )
            
            print(f"\nFine-tuning job started successfully: {job_id}")
            print(f"Use --query --model {args.model} --test-number {args.test_number} to check status")
            
        elif args.query:
            # Query job status
            job_id = args.job_id
            
            if not job_id:
                # Try to load from metadata file
                try:
                    metadata = load_job_metadata(args.model, args.test_number)
                    job_id = metadata["job_id"]
                    print(f"Using job ID from metadata: {job_id}")
                except FileNotFoundError:
                    print("Error: No job ID provided and no metadata file found.")
                    print("Use --job-id to specify the job ID.")
                    sys.exit(1)
            
            query_job_status(job_id, args.model, args.test_number)
            
        elif args.monitor:
            # Monitor job until completion
            job_id = args.job_id
            
            if not job_id:
                # Try to load from metadata file
                try:
                    metadata = load_job_metadata(args.model, args.test_number)
                    job_id = metadata["job_id"]
                    print(f"Using job ID from metadata: {job_id}")
                except FileNotFoundError:
                    print("Error: No job ID provided and no metadata file found.")
                    print("Use --job-id to specify the job ID.")
                    sys.exit(1)
            
            monitor_job(job_id, args.check_interval, args.model, args.test_number)
            
        elif args.test:
            # Test fine-tuned model
            model_name = args.model_name
            
            if not model_name:
                # Try to get model name from metadata
                print("No model name provided, retrieving from metadata...")
                if not args.model or (args.test_number is None):
                    print("Error: --model and --test-number are required when --model-name is not provided.")
                    sys.exit(1)
                
                model_name = get_finetuned_model_name(args.model, args.test_number)
                if not model_name:
                    print("Error: Could not retrieve model name from metadata.")
                    print("Please provide --model-name explicitly or ensure metadata contains the model name.")
                    sys.exit(1)
                
                print(f"Using model from metadata: {model_name}")
            else:
                print(f"Using provided model name: {model_name}")
            
            prompt = args.prompt or "What's the capital of France?"
            test_model(model_name, prompt)
            
        elif args.get_model_name:
            # Get fine-tuned model name from metadata
            model_name = get_finetuned_model_name(args.model, args.test_number)
            if model_name:
                print(f"Fine-tuned model name: {model_name}")
            else:
                print("Fine-tuned model name not available.")
                print("Try running query command if the job has completed.")
                sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

