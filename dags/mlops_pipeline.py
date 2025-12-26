from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='text2sql_mlops_pipeline',
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['mlops']
) as dag:

    pull_and_preprocess = BashOperator(
        task_id='dvc_pull_and_preprocess',
        bash_command='bash /opt/airflow/repo/scripts/run_preprocess.sh',
    )

    train = BashOperator(
        task_id='train_model',
        bash_command='bash /opt/airflow/repo/scripts/run_train.sh',
    )

    evaluate = BashOperator(
        task_id='evaluate_model',
        bash_command='bash /opt/airflow/repo/scripts/run_evaluate.sh',
    )

    register = BashOperator(
        task_id='register_model',
        bash_command='bash /opt/airflow/repo/scripts/run_register.sh',
    )

    pull_and_preprocess >> train >> evaluate >> register
