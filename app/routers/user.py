from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.user import (
    TokenResponse,
    UserRegisterRequest,
    UserLoginRequest,
    UserUpdateRequest,
)
from app.services.user_service import get_users, register_user, authenticate_user
from app.core.database import SessionLocal
from app.models.user import User
from app.core.deps import get_current_user
from app.utils.response import success

# 用户相关API路由
router = APIRouter(prefix="/api", tags=["用户"])


# 数据库会话依赖注入
def get_db():
    """生成数据库会话

    Yields:
        Session: SQLAlchemy数据库会话

    Note:
        - 使用yield实现依赖注入
        - 自动关闭数据库连接
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register")
def register(req: UserRegisterRequest, db: Session = Depends(get_db)):
    """用户注册接口

    Args:
        req: 注册请求数据
        db: 数据库会话依赖注入

    Returns:
        注册成功的用户信息
    """
    user = register_user(db, req)
    return success(user)


@router.post("/login")
def login(req: UserLoginRequest, db: Session = Depends(get_db)):
    """用户登录接口

    Args:
        req: 登录请求数据
        db: 数据库会话依赖注入

    Returns:
        包含access_token的响应
    """
    token_info = authenticate_user(db, req)
    return success(
        TokenResponse(
            access_token=token_info["access_token"],
            created_at=str(int(token_info["created_at"].timestamp())),
            expires_at=str(int(token_info["expires_at"].timestamp())),
            token_type="bearer",
        )
    )


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    """获取当前用户信息

    Args:
        current_user: 通过JWT解析的当前用户

    Returns:
        当前用户详细信息
    """
    return success(current_user)


@router.get("/users")
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 10,
):
    """获取用户列表(仅管理员)

    Args:
        db: 数据库会话
        current_user: 当前用户
        skip: 分页偏移量
        limit: 每页数量

    Returns:
        分页用户列表

    Raises:
        HTTPException: 403 非管理员无权访问
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权限")

    data = get_users(db, skip, limit)
    return success(data)


@router.put("/users/{user_id}")
def update_user(
    user_id: int,
    update_data: UserUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """更新用户信息

    Args:
        user_id: 要更新的用户ID
        update_data: 更新数据
        db: 数据库会话
        current_user: 当前用户

    Returns:
        更新后的用户信息

    Raises:
        HTTPException:
            - 404 用户不存在
            - 403 权限不足
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if current_user.id != user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="权限不足")

    if update_data.email is not None:
        user.email = update_data.email
    if update_data.is_active is not None:
        user.is_active = update_data.is_active

    db.commit()
    db.refresh(user)
    return success(user)
