# (minimal) src/data/preprocess.py
# Chuyển Spider-like JSON files sang JSONL với fields: "input", "target"
import json
import os
from pathlib import Path

def load_spider_data(raw_path: str):
    """Load Spider dataset from raw JSON files."""
    train_spider = json.load(open(os.path.join(raw_path, "train_spider.json")))
    train_others = json.load(open(os.path.join(raw_path, "train_others.json")))
    dev = json.load(open(os.path.join(raw_path, "dev.json")))
    tables = json.load(open(os.path.join(raw_path, "tables.json")))
    
    # Combine training data
    train_data = train_spider + train_others
    
    return train_data, dev, tables

def get_schema_string(db_id: str, tables: list) -> str:
    """Convert database schema to string format."""
    for db in tables:
        if db["db_id"] == db_id:
            schema_parts = []
            for i, table_name in enumerate(db["table_names_original"]):
                columns = [
                    col[1] for col in db["column_names_original"] 
                    if col[0] == i
                ]
                schema_parts.append(f"{table_name}({', '.join(columns)})")
            return " | ".join(schema_parts)
    return ""

def preprocess_example(example: dict, tables: list) -> dict:
    """
    Convert Spider example to model-friendly format.
    Input: "translate to SQL:  Question:  {question} Schema: {schema}"
    Output: "{sql}"
    """
    question = example["question"]
    sql = example["query"]
    db_id = example["db_id"]
    schema = get_schema_string(db_id, tables)
    
    input_text = f"translate to SQL: Question: {question} Schema: {schema}"
    output_text = sql
    
    return {
        "input": input_text,
        "output": output_text,
        "db_id": db_id
    }

def preprocess_dataset(raw_path: str, processed_path: str):
    """Main preprocessing function."""
    print("Loading Spider dataset...")
    train_data, dev_data, tables = load_spider_data(raw_path)
    
    print(f"Processing {len(train_data)} training examples...")
    train_processed = [preprocess_example(ex, tables) for ex in train_data]
    
    print(f"Processing {len(dev_data)} dev examples...")
    dev_processed = [preprocess_example(ex, tables) for ex in dev_data]
    
    # Create output directory
    Path(processed_path).mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    train_output = os.path.join(processed_path, "train.json")
    dev_output = os.path.join(processed_path, "dev.json")
    
    with open(train_output, "w", encoding="utf-8") as f:
        json.dump(train_processed, f, indent=2, ensure_ascii=False)
    
    with open(dev_output, "w", encoding="utf-8") as f:
        json.dump(dev_processed, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved {len(train_processed)} training examples to {train_output}")
    print(f"[OK] Saved {len(dev_processed)} dev examples to {dev_output}")

if __name__ == "__main__": 
    RAW_PATH = "data/raw/spider"
    PROCESSED_PATH = "data/processed"
    
    preprocess_dataset(RAW_PATH, PROCESSED_PATH)