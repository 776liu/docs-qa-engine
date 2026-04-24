import uuid
import json
from typing import Generator, Optional
from fastapi import HTTPException
from rag_engine import RAGEngine
from logger import logger


class RAGService:
    """"""

    def __init__(self):
        self.rag_engine = RAGEngine()

    def initialize(self):
        """初始化RAG引擎"""
        self.rag_engine.initialize()

    def ask(self,question: str, session_id: Optional[str] = None) -> dict:
       try:
           session_id = session_id or str(uuid.uuid4())
           logger.info(f"处理非流式回答:session_id={session_id}")
           result = self.rag_engine.ask(question, session_id=session_id)
           logger.info(f"问答完成: from_cache={result.get('from_cache')}")
           return result
       except Exception as e:
           logger.error(f"问答处理失败:{e}",exc_info= True)
           raise HTTPException(status_code=500, detail=str(e))

    def ask_stream(self,question: str, session_id: Optional[str] = None) -> Generator[str, None, None]:
        try:
            session_id = session_id or str(uuid.uuid4())
            logger.info(f"处理流式回答:session_id={session_id}")

            for token in self.rag_engine.ask_stream(question, session_id=session_id):
                yield f"data:{json.dump({"token":token}, ensure_ascii=False)}\n\n"

            yield f"data:{json.dump({"session_id":session_id,"done": True}, ensure_ascii=False)}"
            logger.info(f"问答完成:session_id={session_id}")
        except Exception as e:
            logger.error(f"问答处理失败:{e}",exc_info= True)
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        def get_history(session_id: str) -> list:
            try:
                history  = self.rag_engine.get_session_history(session_id)
                logger.infoo(f"查询会话历史: session_id={session_id}, 记录数={len(history)}")
                return history
            except Exception as e:
                logger.error(f"查询会话历史失败:{e}",exc_info= True)
                raise HTTPException(status_code=500, detail=str(e))

rag_service = RAGService()






