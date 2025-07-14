import os
import time
import argparse
from typing import List, Dict, Any, Tuple
from mistralai import Mistral
from tqdm import tqdm
import json

client = Mistral(api_key=os.environ["MISTRAL"])


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description="Run batch inference")
    parser.add_argument(
        "--raw_input_file",
        "-i",
        type=str,
        help="Path to raw input file. Not required if processed_input_file is provided. Input file must be in JSONL format with field: patent_desc_trunc",
    )
    parser.add_argument(
        "--processed_input_file",
        "-p",
        type=str,
        help="Path to processed JSONL input file with fields: custom_id, body",
    )
    parser.add_argument(
        "--results_file",
        "-r",
        type=str,
        default="results/batchinf_results.jsonl",
        help="Path to output results file",
    )
    parser.add_argument(
        "--sys_prompt",
        "-s",
        type=str,
        help="System prompt file for processing raw input file",
    )
    parser.add_argument(
        "--user_prompt",
        "-u",
        type=str,
        help="User prompt file for processing raw input file",
    )
    parser.add_argument("--model", "-m", type=str, help="Model name")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if not args.model:
        raise ValueError("Model name is required")

    if not (args.raw_input_file or args.processed_input_file):
        raise ValueError(
            "You must provide either --raw_input_file or --processed_input_file."
        )

    if args.raw_input_file and args.processed_input_file:
        raise ValueError(
            "Provide only one of --raw_input_file or --processed_input_file, not both."
        )

    if args.raw_input_file:
        _validate_raw_input_file(args.raw_input_file)

    if args.processed_input_file:
        _validate_processed_input_file(args.processed_input_file)


def _validate_raw_input_file(file_path: str) -> None:
    """Validate raw input file format."""
    if not os.path.exists(file_path):
        raise ValueError("Raw input file does not exist")

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if "patent_desc_trunc" not in data:
                raise ValueError("Raw input file must contain field: patent_desc_trunc")


def _validate_processed_input_file(file_path: str) -> None:
    """Validate processed input file format."""
    if not os.path.exists(file_path):
        raise ValueError("Processed input file does not exist")

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if "custom_id" not in data or "body" not in data:
                raise ValueError(
                    "Processed input file must contain fields: custom_id, body"
                )


def set_default_prompts(args: argparse.Namespace) -> None:
    """Set default prompt files if not provided."""
    if not args.sys_prompt:
        args.sys_prompt = "prompts/sys_prompt_zero.md"
        print(f"Using default system prompt: {args.sys_prompt}")

    if not args.user_prompt:
        args.user_prompt = "prompts/user_prompt_zero.md"
        print(f"Using default user prompt: {args.user_prompt}")


def load_prompts(sys_prompt_path: str, user_prompt_path: str) -> Tuple[str, str]:
    """Load system and user prompts from files."""
    with open(sys_prompt_path, "r") as f:
        sys_prompt = f.read()

    with open(user_prompt_path, "r") as f:
        user_prompt = f.read()

    return sys_prompt, user_prompt


def process_raw_input(raw_input_file: str, sys_prompt: str, user_prompt: str, sys_prompt_path: str) -> str:
    """Process raw input file into batch inference format."""
    # Extract base filename
    base_filename = raw_input_file.split('/')[-1].split('.')[0]
    
    # Check for special prompt types in sys_prompt filename and modify filename accordingly
    filename_suffix = ""
    sys_prompt_filename = sys_prompt_path.lower()
    if "cot" in sys_prompt_filename and "fewshot" in sys_prompt_filename:
        filename_suffix = "_fewshot_cot"
    elif "cot" in sys_prompt_filename:
        filename_suffix = "_cot"
    elif "fewshot" in sys_prompt_filename:
        filename_suffix = "_fewshot"
    
    inference_file_name = f"data/batchinf_{base_filename}{filename_suffix}.jsonl"

    # Load raw data
    test_lines: List[Dict[str, Any]] = []
    with open(raw_input_file, "r") as f:
        for line in f:
            test_lines.append(json.loads(line))

    # Create batch requests
    batch_req: List[Dict[str, Any]] = []
    total_test_lines: int = len(test_lines)

    for i, row in tqdm(
        enumerate(test_lines),
        total=total_test_lines,
        desc="Preparing batch inference data",
    ):
        patent_description = row["patent_desc_trunc"]
        if "reasoning" in user_prompt:
            max_tokens = 1000
        else:
            max_tokens = 100

        data = {
            "custom_id": str(i),
            "body": {
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt.format(
                            patent_description=patent_description
                        ),
                    },
                ],
            },
        }
        batch_req.append(data)

    # Write batch requests to file
    with open(inference_file_name, "w") as f:
        for req in batch_req:
            f.write(json.dumps(req) + "\n")

    return inference_file_name


def upload_and_run_batch_job(inference_file_name: str, model: str):
    """Upload input file and create batch job."""
    # Upload input file
    batch_data = client.files.upload(
        file={
            "file_name": inference_file_name,
            "content": open(inference_file_name, "rb"),
        },
        purpose="batch",
    )
    print("Uploaded file")

    # Create batch job
    batch_job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model=model,
        endpoint="/v1/chat/completions",
        metadata={"job_type": "testing"},
    )
    print(f"Batch job {batch_job.id} created")

    return batch_job


def wait_for_batch_completion(batch_job):
    """Wait for batch job to complete and return final job status."""
    print("Waiting for batch inference job to complete...")

    while batch_job.status in ["QUEUED", "RUNNING"]:
        batch_job = client.batch.jobs.get(job_id=batch_job.id)
        time.sleep(1)

    print(
        f"Batch inference job {batch_job.id} completed with status: {batch_job.status}"
    )
    return batch_job


def download_results(batch_job, results_path: str) -> None:
    """Download and save batch job results."""
    output_file_id = batch_job.output_file

    if output_file_id is not None:
        results_file = client.files.download(file_id=output_file_id)
        with open(results_path, "w") as f:
            for chunk in results_file.stream:
                f.write(chunk.decode("utf-8"))
        print(f"Downloaded results file to {results_path}")
    else:
        print(f"Batch job failed with status: {batch_job.status}")


def main():
    """Main function to orchestrate batch inference process."""
    # Parse and validate arguments
    args = parse_arguments()
    validate_arguments(args)
    set_default_prompts(args)

    # Determine inference file
    if args.raw_input_file:
        sys_prompt, user_prompt = load_prompts(args.sys_prompt, args.user_prompt)
        inference_file_name = process_raw_input(
            args.raw_input_file, sys_prompt, user_prompt, args.sys_prompt
        )
    else:
        inference_file_name = args.processed_input_file

    # Run batch inference
    batch_job = upload_and_run_batch_job(inference_file_name, args.model)
    batch_job = wait_for_batch_completion(batch_job)
    download_results(batch_job, args.results_file)


if __name__ == "__main__":
    main()
