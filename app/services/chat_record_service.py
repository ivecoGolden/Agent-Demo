from sqlalchemy.orm import Session
from app.models.chat_record import ChatRecord  # 确保导入路径正确
from datetime import datetime
from typing import List, Optional
from sqlalchemy import desc
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
)


async def save_chat_record(
    db: Session,  # SQLAlchemy数据库会话
    user_id: str,  # 用户ID
    uuid: str,  # 消息唯一标识符
    role: str,  # 消息角色(user/assistant)
    model: Optional[str],  # 使用的AI模型名称(可为空)
    text: Optional[str],  # 消息文本内容(可为空)
    image: Optional[List[str]] = None,  # 图片URL列表(可选)
    video: Optional[str] = None,  # 视频URL(可选)
    response_start_time: Optional[datetime] = None,  # 响应开始时间(可选)
    response_end_time: Optional[datetime] = None,  # 响应结束时间(可选)
):
    """保存聊天记录到数据库

    Args:
        db: 数据库会话
        user_id: 用户ID
        uuid: 消息唯一ID
        role: 消息发送者角色
        model: 使用的AI模型
        text: 文本内容
        image: 图片URL列表
        video: 视频URL
        response_start_time: 响应开始时间
        response_end_time: 响应结束时间

    Returns:
        ChatRecord: 保存后的聊天记录对象
    """
    # 创建聊天记录对象
    record = ChatRecord(
        user_id=user_id,
        uuid=uuid,
        role=role,
        model=model,
        text=text,
        image=image,
        video=video,
        response_start_time=response_start_time,
        response_end_time=response_end_time,
    )

    # 数据库操作
    db.add(record)  # 添加到会话
    db.commit()  # 提交事务
    db.refresh(record)  # 刷新获取最新状态

    return record  # 返回保存后的记录


def get_recent_chat_history(user_id: str, db: Session, limit: int = 7):
    """获取用户最近的聊天历史记录

    Args:
        user_id: 用户ID
        db: 数据库会话
        limit: 获取的对话轮次数(默认7轮)

    Returns:
        List[Union[ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]]:
            格式化后的聊天历史记录列表
    """
    # 从数据库查询最近的聊天记录(limit*2是因为每轮对话包含user和assistant两条记录)
    records = (
        db.query(ChatRecord)
        .filter(ChatRecord.user_id == user_id)  # 按用户ID过滤
        .filter(ChatRecord.role.in_(["user", "assistant"]))  # 只包含用户和AI助手的消息
        .order_by(desc(ChatRecord.id))  # 按ID降序(最新的在前面)
        .limit(limit * 2)  # 限制查询数量
        .all()
    )

    # 按ID升序排序(恢复时间顺序)
    records = sorted(records, key=lambda r: r.id)

    history = []
    for record in records:
        # 构建文本消息部分
        text_part = ChatCompletionContentPartTextParam(
            type="text", text=record.text or ""
        )

        # 构建图片消息部分(如果有)
        image_parts = []
        if isinstance(record.image, list):
            image_parts = [
                ChatCompletionContentPartImageParam(
                    type="image_url", image_url={"url": url, "detail": "auto"}
                )
                for url in record.image
                if isinstance(url, str)  # 确保URL是字符串类型
            ]

        # 合并文本和图片内容
        content = [text_part] + image_parts

        # 根据角色类型添加到历史记录
        if record.role == "user":
            history.append(ChatCompletionUserMessageParam(role="user", content=content))
        else:
            history.append(
                ChatCompletionAssistantMessageParam(role="assistant", content=content)
            )

    return history
