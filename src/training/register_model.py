"""
Register trained model to MLflow Model Registry
"""
import os
import sys
import mlflow
from mlflow.models import infer_signature
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def register_model(
    model_path: str = "models/t5-small/checkpoint",
    model_name: str = "text2sql-t5-small",
    run_id: str = None
):
    """
    Register trained model to MLflow Model Registry
    
    Args:
        model_path: Path to saved model checkpoint
        model_name: Name for the registered model
        run_id: Optional run_id to log model under (if None, creates new run)
    """
    print(f"[INFO] Loading model from {model_path}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    print(f"[OK] Model loaded successfully!")
    
    # Create example input for signature
    example_input = "Translate to SQL: Show all employees with salary greater than 50000"
    inputs = tokenizer(example_input, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate example output for signature
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256
        )
    example_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"[INFO] Example input: {example_input}")
    print(f"[INFO] Example output: {example_output}")
    
    # Set experiment
    mlflow.set_experiment("text2sql_baseline")
    
    # If run_id provided, log to existing run; otherwise create new run
    if run_id:
        with mlflow.start_run(run_id=run_id):
            result = _log_and_register_model(model, tokenizer, model_name, example_input, example_output)
    else:
        with mlflow.start_run(run_name="model_registration"):
            result = _log_and_register_model(model, tokenizer, model_name, example_input, example_output)
    
    return result


def _log_and_register_model(model, tokenizer, model_name, example_input, example_output):
    """Helper function to log and register model"""
    
    # Create signature
    signature = infer_signature(
        model_input=example_input,
        model_output=example_output
    )
    
    print(f"[INFO] Logging model to MLflow...")
    
    # Create a wrapper class for the model
    class Text2SQLModel(mlflow.pyfunc.PythonModel):
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            
        def predict(self, context, model_input):
            """
            Generate SQL from natural language input
            
            Args:
                model_input: Can be a string, list of strings, or DataFrame with 'text' column
            """
            # Handle different input types
            if isinstance(model_input, str):
                texts = [model_input]
            elif hasattr(model_input, 'tolist'):  # numpy array or similar
                texts = model_input.tolist()
            elif hasattr(model_input, 'values'):  # DataFrame
                texts = model_input.iloc[:, 0].tolist()
            else:
                texts = list(model_input)
            
            results = []
            for text in texts:
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True
                )
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=256,
                        num_beams=4,
                        early_stopping=True
                    )
                
                sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append(sql)
            
            return results if len(results) > 1 else results[0]
    
    # Create artifacts directory with model files
    artifacts_path = "mlflow_model_artifacts"
    os.makedirs(artifacts_path, exist_ok=True)
    
    # Save model and tokenizer to artifacts
    model.save_pretrained(artifacts_path)
    tokenizer.save_pretrained(artifacts_path)
    
    # Log model using pyfunc
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=Text2SQLModel(model, tokenizer),
        artifacts={"model_path": artifacts_path},
        signature=signature,
        input_example=example_input,
        registered_model_name=model_name,
        pip_requirements=[
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "sentencepiece>=0.1.99"
        ]
    )
    
    print(f"[OK] Model logged successfully!")
    print(f"[INFO] Model URI: {model_info.model_uri}")
    
    # Get model version info
    client = mlflow.MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    if model_versions:
        latest_version = max(model_versions, key=lambda x: int(x.version))
        print(f"\n{'='*50}")
        print(f"[OK] Model registered successfully!")
        print(f"{'='*50}")
        print(f"Model Name: {model_name}")
        print(f"Version: {latest_version.version}")
        print(f"Stage: {latest_version.current_stage}")
        print(f"Run ID: {latest_version.run_id}")
        print(f"{'='*50}")
        
        # Transition to Staging
        print(f"\n[INFO] Transitioning model to 'Staging'...")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Staging"
        )
        print(f"[OK] Model version {latest_version.version} is now in 'Staging'")
        
    return model_name, latest_version.version if model_versions else None


def load_registered_model(model_name: str = "text2sql-t5-small", stage: str = "Staging"):
    """
    Load a registered model from MLflow Model Registry
    
    Args:
        model_name: Name of the registered model
        stage: Stage to load from (None, Staging, Production, Archived)
    """
    model_uri = f"models:/{model_name}/{stage}"
    print(f"[INFO] Loading model from {model_uri}...")
    
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print(f"[OK] Model loaded successfully!")
    
    return loaded_model


def test_registered_model(model_name: str = "text2sql-t5-small"):
    """Test the registered model with sample queries"""
    
    print("\n[INFO] Testing registered model...")
    model = load_registered_model(model_name)
    
    test_queries = [
        "Translate to SQL: Show all products with price less than 100",
        "Translate to SQL: Count the number of customers from New York",
        "Translate to SQL: Find the average salary of employees in IT department"
    ]
    
    print("\n" + "="*60)
    print("Model Inference Test")
    print("="*60)
    
    for query in test_queries:
        result = model.predict(query)
        print(f"\nInput: {query}")
        print(f"Output: {result}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Register model to MLflow")
    parser.add_argument("--model-path", type=str, default="models/t5-small/checkpoint",
                        help="Path to model checkpoint")
    parser.add_argument("--model-name", type=str, default="text2sql-t5-small",
                        help="Name for registered model")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Optional: Run ID to log model under")
    parser.add_argument("--test", action="store_true",
                        help="Test the registered model after registration")
    
    args = parser.parse_args()
    
    # Register model
    model_name, version = register_model(
        model_path=args.model_path,
        model_name=args.model_name,
        run_id=args.run_id
    )
    
    # Test if requested
    if args.test and version:
        test_registered_model(model_name)
