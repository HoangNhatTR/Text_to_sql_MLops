"""
FastAPI server for Text-to-SQL model inference
Supports loading from:
1. Local checkpoint path (MODEL_PATH)
2. MLflow Model Registry (MLFLOW_MODEL_NAME)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text2SQL API",
    description="Convert natural language questions to SQL queries using T5 model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "models/t5-small/checkpoint")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "text2sql-t5-small")
MLFLOW_MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "Staging")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
USE_MLFLOW = os.environ.get("USE_MLFLOW", "false").lower() == "true"


# Request/Response Models
class Text2SQLRequest(BaseModel):
    question: str = Field(..., description="Natural language question", example="Show all employees with salary greater than 50000")
    schema: Optional[str] = Field(None, description="Database schema context", example="employees(id, name, salary, department)")


class Text2SQLResponse(BaseModel):
    sql: str = Field(..., description="Generated SQL query")
    question: str = Field(..., description="Original question")


class BatchText2SQLRequest(BaseModel):
    questions: List[str] = Field(..., description="List of natural language questions")
    schema: Optional[str] = Field(None, description="Database schema context")


class BatchText2SQLResponse(BaseModel):
    results: List[Text2SQLResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str
    model_name: Optional[str] = None


# Global model variables
tokenizer = None
model = None
model_source = "none"


def load_from_checkpoint(path: str):
    """Load model from local checkpoint"""
    global tokenizer, model
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    logger.info(f"Loading model from checkpoint: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    logger.info("Model loaded successfully from checkpoint!")
    return True


def load_from_mlflow(model_name: str, stage: str):
    """Load model from MLflow Model Registry"""
    global tokenizer, model
    import mlflow
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/{stage}"
    
    logger.info(f"Loading model from MLflow: {model_uri}")
    
    # Load as pyfunc model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # For pyfunc models, we'll use it directly
    model = loaded_model
    tokenizer = None  # pyfunc model handles tokenization internally
    
    logger.info("Model loaded successfully from MLflow!")
    return True


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model_source
    
    try:
        if USE_MLFLOW:
            load_from_mlflow(MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE)
            model_source = f"mlflow:{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
        else:
            load_from_checkpoint(MODEL_PATH)
            model_source = f"checkpoint:{MODEL_PATH}"
        
        app.state.model_loaded = True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        app.state.model_loaded = False
        model_source = f"error: {str(e)}"


def generate_sql(question: str, schema: Optional[str] = None) -> str:
    """Generate SQL from natural language question"""
    global tokenizer, model
    
    # Build input text
    if schema:
        input_text = f"Translate to SQL: {question} | Schema: {schema}"
    else:
        input_text = f"Translate to SQL: {question}"
    
    # Check if using MLflow pyfunc model or HuggingFace model
    if tokenizer is None:
        # MLflow pyfunc model
        sql = model.predict(input_text)
        if isinstance(sql, list):
            sql = sql[0]
    else:
        # HuggingFace model
        import torch
        
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
        
        sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return sql


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - returns API status"""
    return HealthResponse(
        status="running",
        model_loaded=getattr(app.state, "model_loaded", False),
        model_source=model_source,
        model_name=MLFLOW_MODEL_NAME if USE_MLFLOW else MODEL_PATH
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if getattr(app.state, "model_loaded", False) else "unhealthy",
        model_loaded=getattr(app.state, "model_loaded", False),
        model_source=model_source,
        model_name=MLFLOW_MODEL_NAME if USE_MLFLOW else MODEL_PATH
    )


@app.post("/text2sql", response_model=Text2SQLResponse)
async def text2sql(request: Text2SQLRequest):
    """
    Convert natural language question to SQL query
    
    - **question**: The natural language question to convert
    - **schema**: Optional database schema for context
    """
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Check server logs for details."
        )
    
    try:
        sql = generate_sql(request.question, request.schema)
        return Text2SQLResponse(sql=sql, question=request.question)
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text2sql/batch", response_model=BatchText2SQLResponse)
async def text2sql_batch(request: BatchText2SQLRequest):
    """
    Convert multiple natural language questions to SQL queries
    
    - **questions**: List of natural language questions
    - **schema**: Optional database schema for context (applies to all questions)
    """
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Check server logs for details."
        )
    
    try:
        results = []
        for question in request.questions:
            sql = generate_sql(question, request.schema)
            results.append(Text2SQLResponse(sql=sql, question=question))
        
        return BatchText2SQLResponse(results=results)
    except Exception as e:
        logger.error(f"Error generating SQL batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_loaded": getattr(app.state, "model_loaded", False),
        "model_source": model_source,
        "use_mlflow": USE_MLFLOW,
        "mlflow_model_name": MLFLOW_MODEL_NAME,
        "mlflow_model_stage": MLFLOW_MODEL_STAGE,
        "checkpoint_path": MODEL_PATH,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
