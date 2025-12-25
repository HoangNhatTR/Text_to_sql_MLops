"""Evaluation script for Text-to-SQL model with MLflow logging.

Usage:
  python src/training/evaluate.py --model_path models/t5-small/checkpoint --eval_data data/processed/dev.json
"""
import argparse
import json
import os
from pathlib import Path

import mlflow
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"


def load_json_dataset(path):
    """Load JSON dataset."""
    ds = load_dataset("json", data_files=str(path))
    return ds["train"]


def calculate_exact_match(predictions: list, references: list) -> float:
    """Calculate exact match accuracy."""
    correct = sum(1 for pred, ref in zip(predictions, references) if pred.strip().lower() == ref.strip().lower())
    return correct / len(references) if references else 0.0


def calculate_execution_accuracy(predictions: list, references: list) -> float:
    """
    Placeholder for execution accuracy.
    In production, this would execute SQL against actual databases.
    """
    # For now, use exact match as proxy
    return calculate_exact_match(predictions, references)


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    # Basic normalization: lowercase, remove extra whitespace
    sql = sql.lower().strip()
    sql = ' '.join(sql.split())
    return sql


def calculate_normalized_match(predictions: list, references: list) -> float:
    """Calculate match after SQL normalization."""
    correct = sum(
        1 for pred, ref in zip(predictions, references) 
        if normalize_sql(pred) == normalize_sql(ref)
    )
    return correct / len(references) if references else 0.0


def evaluate_model(model, tokenizer, dataset, device, max_length=128, log_every=50):
    """Generate predictions and calculate metrics with real-time logging."""
    model.eval()
    predictions = []
    references = []
    
    # Running metrics
    running_exact_match = 0
    running_normalized_match = 0
    
    print(f"Evaluating on {len(dataset)} examples...")
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
            input_text = example["input"]
            reference = example["output"]
            
            # Tokenize
            inputs = tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predictions.append(prediction)
            references.append(reference)
            
            # Update running metrics
            if prediction.strip().lower() == reference.strip().lower():
                running_exact_match += 1
            if normalize_sql(prediction) == normalize_sql(reference):
                running_normalized_match += 1
            
            # Log real-time metrics every N steps
            if (idx + 1) % log_every == 0:
                current_exact_match = running_exact_match / (idx + 1)
                current_normalized_match = running_normalized_match / (idx + 1)
                
                mlflow.log_metric("running_exact_match", current_exact_match, step=idx + 1)
                mlflow.log_metric("running_normalized_match", current_normalized_match, step=idx + 1)
                mlflow.log_metric("examples_processed", idx + 1, step=idx + 1)
                
                print(f"  Step {idx + 1}: exact_match={current_exact_match:.4f}, normalized_match={current_normalized_match:.4f}")
    
    # Calculate final metrics
    metrics = {
        "exact_match": calculate_exact_match(predictions, references),
        "normalized_match": calculate_normalized_match(predictions, references),
        "execution_accuracy": calculate_execution_accuracy(predictions, references),
        "num_examples": len(references),
    }
    
    return metrics, predictions, references


def main(args):
    # Set MLflow experiment
    mlflow.set_experiment(args.mlflow_experiment)
    
    # Check if we should log to existing run or create new one
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_param("model_path", args.model_path)
        mlflow.log_param("eval_data", args.eval_data)
        
        # Load model and tokenizer
        print(f"Loading model from {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)
        
        # Load evaluation dataset
        print(f"Loading evaluation data from {args.eval_data}...")
        eval_dataset = load_json_dataset(args.eval_data)
        
        # Run evaluation
        metrics, predictions, references = evaluate_model(
            model, tokenizer, eval_dataset, device, args.max_length, args.log_every
        )
        
        # Log metrics to MLflow
        print("\n=== Evaluation Results ===")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric_name}: {value:.4f}")
                mlflow.log_metric(metric_name, value)
            else:
                print(f"{metric_name}: {value}")
                mlflow.log_metric(metric_name, value)
        
        # Save predictions to file
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = [
            {"input": eval_dataset[i]["input"], "prediction": pred, "reference": ref}
            for i, (pred, ref) in enumerate(zip(predictions, references))
        ]
        
        results_path = output_dir / "predictions.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Log artifacts
        mlflow.log_artifact(str(results_path))
        
        print(f"\n[OK] Predictions saved to {results_path}")
        print(f"[OK] Metrics logged to MLflow experiment: {args.mlflow_experiment}")
        
        return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    p.add_argument("--eval_data", required=True, help="Path to evaluation data (JSON)")
    p.add_argument("--output_dir", default="eval_results", help="Directory to save results")
    p.add_argument("--max_length", type=int, default=128, help="Max generation length")
    p.add_argument("--log_every", type=int, default=50, help="Log metrics every N examples")
    p.add_argument("--mlflow_experiment", default="text2sql_baseline", help="MLflow experiment name")
    args = p.parse_args()
    main(args)
