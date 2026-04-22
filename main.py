from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import huggingface_hub
import os
import shutil
from pathlib import Path

from rag_engine import RAGEngine
from ingest import ingest_documents

# 镜像命令 $env:HF_ENDPOINT="https://hf-mirror.com"
# 强制离线（必须在导入 sentence_transformers 之前） 不要联网检查更新
huggingface_hub.constants.HF_HUB_OFFLINE = True
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# 加载 .env 文件中的密钥
load_dotenv()

engine = RAGEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.initialize()
    yield

app = FastAPI(title="RAG API", lifespan=lifespan)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(data: Question):
    try:
        return engine.ask(data.question)
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """上传文档并自动入库"""
    if not file.filename.endswith(('.txt', '.pdf')):
        raise HTTPException(400, "仅支持 .txt 和 .pdf")
    
    # 保存文件
    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)
    file_path = os.path.join(docs_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 自动入库
    ingest_documents()
    
    return {"status": "success", "filename": file.filename}

@app.get("/")
def root():
    return {"status": "running"}