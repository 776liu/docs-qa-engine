from fastapi import Header, HTTPException, Depends
from opentelemetry.context import get_current

from config import settings

def verify_token(authorization: str = Header(None)) -> bool:
    """
    验证Token
    :param authorization:
    :return:
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="缺少认证信息，请在请求头添加: Authorization: Bearer <token>"
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        status_code=401,
        detail="认证信息格式错误， Bearer <token>"

    if token != settings.API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="无效Token"
        )

    return True

get_current_user = Depends(verify_token)
