# Text_to_sql_MLops

Mục tiêu
- Xây dựng pipeline MLOps cho bài toán Text-to-SQL (dựa trên dataset Spider).
- Tập trung vào: chuẩn hóa dữ liệu + data versioning (DVC), training baseline nhỏ (t5-small), experiment tracking (MLflow), orchestration (Airflow/Prefect), serving (FastAPI), containerization & CI/CD.

Dataset
- Sử dụng: Spider dataset (v2 recommended).
- Thư mục dữ liệu:
  - `data/raw/` — dữ liệu gốc (raw JSON/SQL).
  - `data/processed/` — dữ liệu đã chuẩn hóa (model-friendly).
- Chuẩn hóa bắt buộc:
  - Tách `question`, `SQL`, `schema` từ Spider.
  - Chuyển về format input/output đơn giản:
    - Input: `translate to SQL: Question: {question} Schema: {schema}`
    - Output: `{sql}`

Pipeline overview (sơ đồ ngắn)
- data/raw → preprocess → data/processed (DVC) → train (MLflow) → evaluate (MLflow) → register_model (MLflow model registry) → serve (FastAPI)
- Orchestration: Airflow DAG / Prefect flow cho các bước: preprocess → train → evaluate → register

Tech stack
- Model: HuggingFace Transformers (PyTorch) — baseline: `t5-small`
- Data versioning: DVC (remote: local/S3)
- Experiment tracking: MLflow
- Orchestration: Apache Airflow or Prefect (DAGs)
- Serving: FastAPI + gunicorn/uvicorn
- Containerization: Docker
- CI/CD: GitHub Actions
- Monitoring: MLflow metrics + simple input/SQL logging (optional Prometheus/ELK)

Reviewer checklist (nhìn README trước model)
- Phase 1: repo structure + README (đã có)
- Phase 2: data pipeline + DVC (quan trọng nhất) — kiểm tra `data/processed` tracked via DVC
- Phase 3: baseline training (t5-small; 1–2 epochs) — đơn giản, reproducible
- Phase 4: MLflow integration — compare runs
- Phase 5: Airflow DAG hiển thị flow
- Phase 6: FastAPI endpoint load từ MLflow model registry
- Phase 7: Docker + GH Actions CI
- Phase 8: Monitoring & retraining triggers (optional)

How to read repo
1. Open `README.md` (this file) → understand pipeline and tech stack.
2. Check `data/` (DVC-tracked) → verify `dvc.lock` / `.dvc` files.
3. Check `pipelines/` / `dags/` → DAGs and orchestration.
4. Check `src/training/train.py` → training + MLflow logging.
5. Check `api/` → FastAPI app loading model from MLflow.

Contact
- Owner: HoangNhatTR
- Created: v1 — baseline scaffold for MLOps-first project

Phân công :
- ModelAI : Lê Anh Thiên
- FastAPI : Nguyễn Đức Hải
- DVC : Trần Nhật Hoàng 
- MLFlow : Trần Nhật Hoàng  - Hà Vĩnh Phước
- Airflow/Prefect : Trần Nhật Hoàng - Nguyễn Đức Minh - Hà Vĩnh Phước
- CI/CD : Nguyễn Đức Minh - Hà Vĩnh Phước

```mermaid
flowchart TD
  subgraph Data
    RAW["data/raw\n(Spider raw files)"]
    PROCESSED["data/processed\n(model-friendly JSON)"]
    DVC["DVC Remote\n(dvc add / dvc push)"]
    RAW --> DVC
    PROCESSED --> DVC
  end

  subgraph Preprocess
    PARSE["src/data/preprocess.py\n(Parse Spider → model format)"]
    PARSE --> PROCESSED
  end

  subgraph Orchestration
    AIRFLOW["Airflow / Prefect\n(pipelines/text2sql_dag)"]
    AIRFLOW --> PARSE
    AIRFLOW --> TRAIN
    AIRFLOW --> EVAL
    AIRFLOW --> REGISTER
  end

  subgraph Training
    TRAIN["src/training/train.py\n(baseline t5-small)"]
    TRAIN --> MLFLOW_RUN["MLflow run & artifacts\n(mlruns / remote)"]
  end

  subgraph Evaluation
    EVAL["src/training/evaluate.py\n(metrics: exact_match, exec_acc)"]
    TRAIN --> EVAL
    EVAL --> MLFLOW_RUN
  end

  subgraph ModelRegistry
    REGISTER["src/training/register_model.py\n(mlflow register)"]
    MLFLOW_RUN --> REGISTER
    REGISTER --> MODEL_REG["MLflow Model Registry\nmodels:/text2sql-model"]
  end

  subgraph Serving
    API["src/api/app.py\n(FastAPI)"]
    DOCKER["docker/Dockerfile.api\n(container)"]
    MODEL_REG --> API
    API --> DOCKER
    DOCKER --> DEPLOY["Deploy\n(k8s / Docker host / local)"]
  end

  subgraph CI_CD
    GHA["GitHub Actions\n(lint, test, build image)"]
    GHA --> AIRFLOW
    GHA --> DOCKER
  end

  %% Trigger node thay cho edge label (tránh lỗi parser)
  TRIGGER["data change\n(dvc hash)"]
  DVC --> TRIGGER --> AIRFLOW

  %% Code-change trigger
  GHA --> AIRFLOW
