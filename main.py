from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
import huggingface_hub
import os
import shutil
from pathlib import Path
import uuid
import json

from requests import session

from rag_engine import RAGEngine
from ingest import ingest_documents
from config import settings
from logger import logger


huggingface_hub.constants.HF_HUB_OFFLINE = True
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

load_dotenv()

engine = RAGEngine()

security = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """验证 Token"""
    if not credentials:
        raise HTTPException(status_code=401, detail="缺少认证信息")
    
    if credentials.credentials != settings.API_TOKEN:
        raise HTTPException(status_code=401, detail="Token 无效")
    
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("应用启动中...")
    engine.initialize()
    logger.info("应用启动完成")
    yield
    logger.info("应用关闭")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

class Question(BaseModel):
    question: str
    session_id: Optional[str] = None

@app.post("/ask", dependencies=[Depends(verify_token)])
def ask(data: Question):
    """
    非流式问答接口（需要鉴权）

    :param data: 问题数据
    :return: 答案字典
    """
    try:
        session_id = data.session_id or str(uuid.uuid4())
        logger.info(f"处理非流式问答: session_id={session_id}")
        result = engine.ask(data.question, session_id=session_id)
        logger.info(f"问答完成: from_cache={result.get('from_cache')}")
        return result
    except Exception as e:
        logger.error(f"问答接口异常: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream", dependencies=[Depends(verify_token)])
def ask_stream(data: Question):
    """
    流式输出（需要鉴权）
    :param data: 问题数据
    :return: SSE 流
    """
    try:
        session_id = data.session_id or str(uuid.uuid4())
        logger.info(f"处理流式问答: session_id={session_id}")
        
        def generate():
            try:
                for token in engine.ask_stream(data.question, session_id=session_id):
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'session_id': session_id, 'done': True}, ensure_ascii=False)}\n\n"
                logger.info(f"流式问答完成: session_id={session_id}")
            except Exception as e:
                logger.error(f"流式生成异常: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        logger.error(f"流式接口异常: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/history", dependencies=[Depends(verify_token)])
def get_history(session_id: str):
    """
    获取指定会话的对话历史（需要鉴权）

    :param session_id: 会话ID
    :return: 历史记录列表
    """
    try:
        history = engine.get_session_history(session_id)
        logger.info(f"查询会话历史: session_id={session_id}, 记录数={len(history)}")
        return {"history": history}
    except Exception as e:
        logger.error(f"查询历史失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", dependencies=[Depends(verify_token)])
async def upload(file: UploadFile = File(...)):
    """上传文档并自动入库（需要鉴权）"""
    try:
        if not file.filename.endswith(('.txt', '.pdf', '.md')):
            logger.warning(f"不支持的文件类型: {file.filename}")
            raise HTTPException(400, "仅支持 .txt , .pdf , .md")
        
        docs_dir = settings.DOCS_DIR
        os.makedirs(docs_dir, exist_ok=True)
        file_path = os.path.join(docs_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"文件上传成功: {file.filename}")
        
        ingest_documents()
        
        return {"status": "success", "filename": file.filename}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "running", "version": settings.API_VERSION}
