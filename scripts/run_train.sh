#!/bin/bash
set -e
cd /opt/airflow/repo
echo "Starting training..."
python src/training/train.py || true
