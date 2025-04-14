from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# 数据库连接URL，从配置中获取
DATABASE_URL = settings.DATABASE_URL

# 创建数据库引擎，配置连接池健康检查
engine = create_engine(
    DATABASE_URL, pool_pre_ping=True  # 启用连接池健康检查，避免使用失效连接
)

# 创建数据库会话工厂
SessionLocal = sessionmaker(
    autocommit=False,  # 禁用自动提交
    autoflush=False,  # 禁用自动flush
    bind=engine,  # 绑定到创建的引擎
)

# SQLAlchemy模型基类，所有模型类都应继承此类
Base = declarative_base()
