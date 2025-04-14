from sqlalchemy import Column, Integer, String, Boolean, DateTime, func
from app.core.database import Base
from sqlalchemy import Enum as SqlEnum
import enum


class UserRole(str, enum.Enum):
    """用户角色枚举类

    Attributes:
        admin: 管理员角色
        user: 普通用户角色
    """

    admin = "admin"
    user = "user"


class User(Base):
    """用户数据模型

    Attributes:
        __tablename__: 数据库表名
        id: 主键ID
        username: 用户名(唯一)
        email: 邮箱地址(唯一)
        hashed_password: 加密后的密码
        is_active: 账号是否激活
        created_at: 创建时间
        role: 用户角色

    Note:
        - 继承自Base类，使用SQLAlchemy ORM映射
        - 用户名和邮箱字段有唯一约束
        - 密码存储为bcrypt哈希值
    """

    __tablename__ = "users"  # 数据库表名

    id = Column(Integer, primary_key=True, index=True)  # 主键ID
    username = Column(
        String(50), unique=True, nullable=False, index=True
    )  # 用户名(唯一)
    email = Column(String(100), unique=True, nullable=True)  # 邮箱地址(可选)
    hashed_password = Column(String(128), nullable=False)  # 加密密码(必填)
    is_active = Column(Boolean, default=True)  # 账号状态(默认激活)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now()
    )  # 创建时间(自动设置)
    role = Column(SqlEnum(UserRole), default=UserRole.user)  # 用户角色(默认普通用户)
