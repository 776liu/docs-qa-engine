from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse

from services.rag_service import rag_service
from dependencies.auth import get_current_user

router = APIRouter(prefix="/ask",tags=["问答"])

class Question(BaseModel):
    question: str
    session_id: Optional[str] = None

@router.post("",depandencies=[Depends(get_current_user)])
def ask(data: Question):
    return rag_service.ask(data.question,data.session_id)

@router.post("/stream",dependencies=[Depends(get_current_user)])
def ask_stream(data: Question):
    return StreamingResponse(
        rag_service.ask_stream(data.question,data.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@router.get("/session/{session_id}/history",dependencies=[Depends(get_current_user)])
def get_history(session_id: str):
    return {"history": rag_service.get_history(session_id)}
