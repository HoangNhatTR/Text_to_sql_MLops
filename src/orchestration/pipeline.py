from prefect import flow, task
import subprocess
import sys


@task(retries=2, retry_delay_seconds=10)
def preprocess_task():
    subprocess.run(
        [sys.executable, "src/data/preprocess.py"],
        check=True
    )


@task(retries=1)
def train_task():
    subprocess.run(
        [
            sys.executable,
            "src/training/train.py",
            "--train", "data/processed/train.json",
            "--output_dir", "models/t5-small",
            "--epochs", "1",
            "--batch_size", "4"
        ],
        check=True
    )


@flow(name="text-to-sql-training-pipeline")
def training_flow():
    preprocess_task()
    train_task()


if __name__ == "__main__":
    training_flow()
