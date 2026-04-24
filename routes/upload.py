import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from config import settings
from ingest import ingest_documents
from logger import logger
from dependencies.auth import get_current_user

router = APIRouter(tags=["文件上传"])

@router.post("/upload",dependencies=[Depends(get_current_user)])
async def uupload(file: UploadFile = File(...)):
    """
    上传文件接口
    :param file: 文件对象
    :return: 文件保存路径
    """
    if not file.filename.endswith((".pdf", ".docx", ".txt")):
        logger.warning(f"不支持的文件类型: {file.filename}")
        raise HTTPException(400, "仅支持 .pdf , .docx , .txt")

    docs_dir = settings.DOCS_DIR
    os.makedirs(docs_dir, exist_ok=True)
    file_path = os.path.join(docs_dir, file.filename)

    with open(file_path, "wb")as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"文件上传成功: {file.filename}")
    ingest_documents()

    return {"status": "success", "filename": file.filename}






