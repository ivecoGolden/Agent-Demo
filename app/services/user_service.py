from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.user import UserRegisterRequest, UserResponse
from app.utils.security import hash_password
from app.utils.security import verify_password
from app.utils.jwt import create_access_token
from app.schemas.user import UserLoginRequest
from fastapi import HTTPException
import uuid
from datetime import datetime, timedelta, timezone
from app.core.config import settings


def register_user(db: Session, user_data: UserRegisterRequest) -> User:
    """注册新用户

    Args:
        db (Session): 数据库会话对象
        user_data (UserRegisterRequest): 包含用户名、邮箱和密码的注册请求数据

    Returns:
        User: 成功创建并保存到数据库的用户对象

    Raises:
        HTTPException: 当用户名已存在时抛出400错误

    Note:
        - 使用hash_password对密码进行加密存储
        - 自动提交事务并刷新对象状态
        - 返回的用户对象包含数据库生成的ID等字段
    """
    # 检查用户名是否已存在
    existing = db.query(User).filter(User.username == user_data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="用户名已存在")  # 用户名重复

    # 创建新用户并加密密码
    user = User(
        username=user_data.username,
        email=user_data.email,
        userid=str(uuid.uuid4()),  # 添加用户唯一标识
        hashed_password=hash_password(user_data.password),  # 使用bcrypt加密密码
    )
    db.add(user)  # 将用户对象添加到当前数据库会话
    db.commit()  # 提交事务到数据库
    db.refresh(user)  # 从数据库重新加载对象以获取生成的ID等字段
    return user


def authenticate_user(db: Session, data: UserLoginRequest) -> dict:
    """用户认证并生成访问令牌

    Args:
        db (Session): 数据库会话
        data (UserLoginRequest): 包含用户名和密码的登录请求数据

    Returns:
        str: JWT访问令牌字符串

    Raises:
        HTTPException: 当出现以下情况时抛出异常:
            - 用户名不存在(400)
            - 账号被禁用(403)
            - 密码错误(400)

    Note:
        - 使用verify_password验证密码哈希
        - JWT令牌包含用户ID作为subject claim
    """
    user = db.query(User).filter(User.username == data.username).first()
    if not user:
        raise HTTPException(status_code=400, detail="用户名不存在")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="账号已被禁用")

    if not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="密码错误")  # 密码不匹配

    created_at = datetime.now(timezone.utc)
    expire_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = created_at + expire_delta

    token = create_access_token({"sub": str(user.id)}, expires_delta=expire_delta)

    return {
        "access_token": token,
        "created_at": created_at,
        "expires_at": expire,
    }


def get_users(db: Session, skip: int = 0, limit: int = 10):
    """获取分页用户列表

    Args:
        db (Session): 数据库会话
        skip (int): 跳过的记录数，用于分页
        limit (int): 每页返回的记录数

    Returns:
        dict: 包含以下键的字典:
            - total (int): 系统中用户总数
            - users (list[UserResponse]): 当前页的用户列表

    Note:
        - 默认返回前10条记录(skip=0, limit=10)
        - 使用offset和limit实现分页查询
        - 将ORM模型转换为Pydantic响应模型
    """
    total = db.query(User).count()  # 查询用户总数
    users = db.query(User).offset(skip).limit(limit).all()  # 执行分页查询
    return {
        "total": total,
        "users": [UserResponse.model_validate(u) for u in users],  # 序列化用户数据
    }


def get_user_by_uuid(db: Session, user_uuid: str) -> User | None:
    """根据UUID获取用户对象

    Args:
        db (Session): 数据库会话
        user_uuid (str): 用户唯一标识（UUID）

    Returns:
        User | None: 查询到的用户对象，如果不存在则返回None
    """
    return db.query(User).filter(User.userid == user_uuid).first()
