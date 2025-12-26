#!/bin/bash
set -e
cd /opt/airflow/repo
echo "Running dvc pull..."
dvc pull -q || true
echo "Running preprocess..."
python src/data/preprocess.py || true
