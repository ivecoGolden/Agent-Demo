from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.utils.jwt import decode_access_token
from app.models.user import User

security = HTTPBearer()


def get_db():
    """数据库会话依赖生成器

    Yields:
        Session: SQLAlchemy数据库会话对象

    Note:
        - 使用yield实现依赖注入模式
        - 确保数据库连接在使用后正确关闭
        - 适用于FastAPI的Depends依赖注入系统
    """
    db = SessionLocal()  # 创建新的数据库会话
    try:
        yield db  # 将会话提供给依赖方使用
    finally:
        db.close()  # 确保会话最终被关闭


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """获取当前认证用户

    Args:
        credentials: 从Authorization头获取的Bearer凭证
        db: 数据库会话依赖注入

    Returns:
        User: 认证成功的用户对象

    Raises:
        HTTPException:
            - 401: 当token无效或用户不存在时抛出

    Note:
        - 依赖HTTP Bearer认证方案
        - 自动验证JWT令牌有效性
        - 查询数据库获取完整用户对象
    """
    token = credentials.credentials  # 从凭证中提取JWT令牌
    payload = decode_access_token(token)  # 解码并验证JWT
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="登录状态无效")  # 无效令牌

    user = db.query(User).filter(User.id == int(payload["sub"])).first()  # 查询用户
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")  # 用户不存在

    return user  # 返回认证用户对象
