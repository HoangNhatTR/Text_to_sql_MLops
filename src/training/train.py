"""Baseline training script (t5-small) with MLflow logging.

This is a minimal, reproducible training script intended for small-scale experiments.
Usage example:
  python src/training/train.py --train data/processed/train.json --output_dir models/t5-small --epochs 1
"""
import argparse
import os
from pathlib import Path

import mlflow

mlflow. enable_system_metrics_logging()

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"


def load_json_dataset(path):
    """Load JSON dataset (list of objects)."""
    ds = load_dataset("json", data_files=str(path))
    return ds["train"]


def preprocess(tokenizer, examples, max_input_len=512, max_target_len=128):
    inputs = examples["input"]
    targets = examples["output"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_len, 
        truncation=True,
        padding=False  # Let data collator handle padding
    )
    
    # Tokenize targets
    labels = tokenizer(
        text_target=targets,  # Use text_target instead of deprecated as_target_tokenizer
        max_length=max_target_len, 
        truncation=True,
        padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class MLflowLoggingCallback(TrainerCallback):
    """Custom callback to log metrics to MLflow in real-time."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Log each metric to MLflow
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=state.global_step)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        mlflow.log_metric("epoch", state.epoch, step=state.global_step)


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

        dataset = load_json_dataset(args.train)
        tokenized = dataset.map(
            lambda ex: preprocess(tokenizer, ex), 
            batched=True,
            remove_columns=dataset.column_names  # Remove original columns
        )

        # Data collator handles dynamic padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            logging_steps=10,
            save_strategy="epoch",
            predict_with_generate=True,
            report_to=[],  # Disable default reporters, use custom callback
            logging_first_step=True,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
            processing_class=tokenizer,
            callbacks=[MLflowLoggingCallback()],  # Add custom MLflow callback
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
