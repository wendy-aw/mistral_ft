import os
import time
import argparse
from mistralai import Mistral
from tqdm import tqdm
import json

client = Mistral(api_key=os.environ["MISTRAL"])


def main():
    parser = argparse.ArgumentParser(description="Run batch inference")
    parser.add_argument("--raw_input_file", "-i", type=str, help="Path to raw input file. Not required if processed_input_file is provided. Input file must be in JSONL format with field: patent_desc_trunc")
    parser.add_argument("--processed_input_file", "-p", type=str, help="Path to processed JSONL input file with fields: custom_id, body")
    parser.add_argument(
        "--results_file", "-r", type=str, default="results/batchinf_results.jsonl", help="Path to output results file"
    )
    parser.add_argument("--sys_prompt", "-s", type=str, help="System prompt file for processing raw input file")
    parser.add_argument("--user_prompt", "-u", type=str, help="User prompt file for processing raw input file")
    parser.add_argument("--model", "-m", type=str, help="Model name")
    args = parser.parse_args()

    # Validate args
    if not args.model:
        raise ValueError("Model name is required")
    if not (args.raw_input_file or args.processed_input_file):
        raise ValueError("You must provide either --raw_input_file or --processed_input_file.")
    if args.raw_input_file and args.processed_input_file:
        raise ValueError("Provide only one of --raw_input_file or --processed_input_file, not both.")
    if args.raw_input_file:
        if not os.path.exists(args.raw_input_file):
            raise ValueError("Raw input file does not exist")
        with open(args.raw_input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if "patent_desc_trunc" not in data:
                    raise ValueError("Raw input file must contain field: patent_desc_trunc")
    if args.processed_input_file:
        if not os.path.exists(args.processed_input_file):
            raise ValueError("Processed input file does not exist")
        with open(args.processed_input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if "custom_id" not in data or "body" not in data:
                    raise ValueError("Processed input file must contain fields: custom_id, body")

    # If sys prompt and user prompt files are not provided, use default prompts
    if not args.sys_prompt:
        args.sys_prompt = "prompts/sys_prompt.md"
        print(f"Using default system prompt: {args.sys_prompt}")
    if not args.user_prompt:
        args.user_prompt = "prompts/user_prompt.md"
        print(f"Using default user prompt: {args.user_prompt}")

    # Prepare jsonl file for batch inference if raw_input_file is provided
    if args.raw_input_file:
        inference_file_name = f"data/batchinf_{args.raw_input_file.split('/')[-1].split('.')[0]}.jsonl"

        test_lines = []
        with open(args.raw_input_file, "r") as f:
            for line in f:
                test_lines.append(json.loads(line))

        # Read sys and user prompts
        sys_prompt = open(args.sys_prompt, "r").read()
        user_prompt = open(args.user_prompt, "r").read()

        batch_req = []
        total_test_lines = len(test_lines)
        for i, row in tqdm(enumerate(test_lines), total=total_test_lines, desc="Preparing batch inference data"):
            patent_description = row["patent_desc_trunc"]

            data = {
                "custom_id": str(i),
                "body": {
                    "max_tokens": 100,
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

        # Write batch_req to jsonl
        with open(inference_file_name, "w") as f:
            for req in batch_req:
                f.write(json.dumps(req) + "\n")
    else:
        inference_file_name = args.processed_input_file

    # Upload input file to client
    batch_data = client.files.upload(
        file={"file_name": inference_file_name, "content": open(inference_file_name, "rb")},
        purpose="batch",
    )
    print(f"Uploaded file")

    # Run batch inference job on model
    batch_job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model=args.model,
        endpoint="/v1/chat/completions",
        metadata={"job_type": "testing"},
    )
    print(f"Batch job {batch_job.id} created")

    # Wait for batch job to complete
    print("Waiting for batch inference job to complete...")
    while batch_job.status in ["QUEUED", "RUNNING"]:
        batch_job = client.batch.jobs.get(job_id=batch_job.id)
        time.sleep(1)

    print(f"Batch inference job {batch_job.id} completed with status: {batch_job.status}")

    # Download results file
    results_path = args.results_file
    output_file_id = batch_job.output_file
    if output_file_id is not None:
        results_file = client.files.download(file_id=output_file_id)
        with open(results_path, "w") as f:
            for chunk in results_file.stream:
                f.write(chunk.decode("utf-8"))
        print(f"Downloaded results file to {results_path}")
    else:
        print("Batch job failed with status: {}".format(batch_job.status))


if __name__ == "__main__":
    main()
