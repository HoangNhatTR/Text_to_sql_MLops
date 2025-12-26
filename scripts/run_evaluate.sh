#!/bin/bash
set -e
cd /opt/airflow/repo
echo "Running evaluation..."
python src/training/evaluate.py || true
