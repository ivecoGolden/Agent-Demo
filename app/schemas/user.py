from pydantic import BaseModel, EmailStr, constr
from datetime import datetime
from typing import List


class UserRegisterRequest(BaseModel):
    # 用户注册请求体
    username: constr(min_length=3, max_length=50)  # 用户名，长度限制3~50
    email: EmailStr | None = None  # 邮箱，可选，自动验证格式
    password: constr(min_length=6, max_length=128)  # 密码，长度限制6~128


class UserResponse(BaseModel):
    # 用户响应模型（返回给前端用）
    id: int  # 用户ID
    username: str  # 用户名
    email: str | None  # 邮箱
    is_active: bool  # 是否启用
    created_at: datetime  # 注册时间

    model_config = {"from_attributes": True}  # 支持从 ORM 对象转换


class UserLoginRequest(BaseModel):
    # 用户登录请求体
    username: str  # 用户名
    password: str  # 密码


class TokenResponse(BaseModel):
    # 登录成功后返回的 JWT token
    access_token: str  # JWT 令牌
    token_type: str = "bearer"  # 令牌类型，默认为 Bearer


class UserListResponse(BaseModel):
    # 用户列表响应结构（分页）
    total: int  # 总用户数
    users: List[UserResponse]  # 用户列表


class UserUpdateRequest(BaseModel):
    # 用户信息更新请求体
    email: EmailStr | None = None  # 修改邮箱（可选）
    is_active: bool | None = None  # 修改启用状态（可选）
