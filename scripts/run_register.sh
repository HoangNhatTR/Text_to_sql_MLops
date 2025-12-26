#!/bin/bash
set -e
cd /opt/airflow/repo
echo "Registering model to MLflow..."
python src/training/register_model.py || true
