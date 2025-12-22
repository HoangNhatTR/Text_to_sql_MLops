"""Simple Airflow DAG skeleton for text2sql pipeline.

Tasks: preprocess -> train -> evaluate -> register_model
This file is a placeholder; adapt to your Airflow environment.
"""
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


def preprocess_fn(**kwargs):
    # call the parse_spider module
    from src.data.parse_spider import parse_spider_dir
    parse_spider_dir('data/raw', 'data/processed/train.jsonl')


def train_fn(**kwargs):
    import subprocess
    subprocess.run(['python', 'src/training/train.py', '--train', 'data/processed/train.jsonl', '--output_dir', 'models/t5-small', '--epochs', '1'])


def evaluate_fn(**kwargs):
    # placeholder for evaluation step
    print('Evaluate model (placeholder)')


def register_fn(**kwargs):
    # placeholder for model registration (MLflow)
    print('Register model (placeholder)')


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
}

dag = DAG(
    dag_id='text2sql_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

with dag:
    preprocess = PythonOperator(task_id='preprocess', python_callable=preprocess_fn)
    train = PythonOperator(task_id='train', python_callable=train_fn)
    evaluate = PythonOperator(task_id='evaluate', python_callable=evaluate_fn)
    register = PythonOperator(task_id='register_model', python_callable=register_fn)

    preprocess >> train >> evaluate >> register
