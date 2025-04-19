from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey
from datetime import datetime
from app.core.database import Base


class ChatRecord(Base):
    __tablename__ = "chat_record"

    id = Column(Integer, primary_key=True, index=True, comment="主键 ID")

    user_id = Column(
        Integer, ForeignKey("users.id"), index=True, comment="关联的用户 ID"
    )

    uuid = Column(
        String(64), index=True, nullable=False, comment="前端传递的消息唯一标识"
    )

    role = Column(String(32), nullable=False, comment="消息角色，如 user / assistant")

    model = Column(String(128), nullable=True, comment="使用的大模型名称")

    response_start_time = Column(DateTime, nullable=True, comment="回复开始时间")

    response_end_time = Column(DateTime, nullable=True, comment="回复结束时间")

    text = Column(String, nullable=True, comment="文本内容")
    image = Column(JSON, nullable=True, comment="图片列表，字符串数组")
    video = Column(String, nullable=True, comment="视频链接")
