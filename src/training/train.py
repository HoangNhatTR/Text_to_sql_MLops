"""Baseline training script (t5-small) with MLflow logging.

This is a minimal, reproducible training script intended for small-scale experiments.
Usage example:
  python src/training/train.py --train data/processed/train.jsonl --output_dir models/t5-small --epochs 1
"""
import argparse
import os
from pathlib import Path

import mlflow
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


def load_jsonl_dataset(path):
    ds = load_dataset("json", data_files=str(path))
    return ds["train"]


def preprocess(tokenizer, examples, max_input_len=512, max_target_len=128):
    inputs = examples["input"]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=max_input_len, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_len, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(args):
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params({
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        })

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

        dataset = load_jsonl_dataset(args.train)
        tokenized = dataset.map(lambda ex: preprocess(tokenizer, ex), batched=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            logging_steps=10,
            save_strategy="epoch",
            predict_with_generate=True,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            tokenizer=tokenizer,
        )

        trainer.train()

        # Save and log model artifact
        model_save_path = Path(args.output_dir) / "checkpoint"
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        mlflow.log_artifacts(str(model_save_path), artifact_path="model")

        # Minimal metrics placeholder
        mlflow.log_metric("train_steps", 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--output_dir", default="models/t5-small")
    p.add_argument("--model_name", default="t5-small")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--mlflow_experiment", default="text2sql_baseline")
    args = p.parse_args()
    main(args)
