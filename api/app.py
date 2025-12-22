from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI(title="text2sql-api")


class Text2SQLRequest(BaseModel):
    question: str
    schema: str


MODEL_PATH = os.environ.get("MODEL_PATH")  # expected local path to HF checkpoint or model name

# Lazy model loading
tokenizer = None
model = None


@app.on_event("startup")
def load_model():
    global tokenizer, model
    if not MODEL_PATH:
        app.extra["model_loaded"] = False
        return
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        app.extra["model_loaded"] = True
    except Exception as e:
        app.extra["model_loaded"] = False
        print("Failed to load model:", e)


@app.post("/text2sql")
def text2sql(req: Text2SQLRequest):
    if not app.extra.get("model_loaded"):
        raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_PATH to a checkpoint or MLflow model URI.")
    input_text = f"translate to SQL: Question: {req.question} Schema: {req.schema}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=256)
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"sql": sql}
