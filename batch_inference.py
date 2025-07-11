import os
import time
import argparse
from mistralai import Mistral

client = Mistral(api_key=os.environ["MISTRAL"])
DEFAULT_INPUT_FILE_ID = "" # TODO: Add default input file ID


def main():
    parser = argparse.ArgumentParser(description="Run batch inference")
    parser.add_argument("--input_file", "-i", type=str, help="Path to input file")
    parser.add_argument(
        "--results_file", "-r", type=str, help="Path to output results file"
    )
    parser.add_argument("--model", "-m", type=str, help="Model name")
    args = parser.parse_args()

    # Validate args
    if not args.model:
        raise ValueError("Name of model is required")

    # Upload input file to client if provided, retrieve default file from client otherwise
    if args.input_file:
        batch_data = client.files.upload(
            file={"file_name": args.input_file, "content": open(args.input_file, "rb")},
            purpose="batch",
        )
        print(f"Uploaded file")
    else:
        batch_data = client.files.retrieve(file_id=DEFAULT_INPUT_FILE_ID)

    # Run batch inference job on model
    batch_job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model=args.model,
        endpoint="/v1/chat/completions",
        metadata={"job_type": "testing"},
    )
    print(f"Batch job {batch_job.id} created")

    # Wait for batch job to complete
    print("Waiting for batch job to complete...")
    while batch_job.status in ["QUEUED", "RUNNING"]:
        batch_job = client.batch.jobs.get(job_id=batch_job.id)
        time.sleep(1)

    print(f"Batch job {batch_job.id} completed with status: {batch_job.status}")

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
