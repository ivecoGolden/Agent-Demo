from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.security import HTTPAuthorizationCredentials
from app.agents.text_agent import NormalAgent
from app.core.connection_manager import ConnectionManager
from app.core.logger import logger
import json
from app.core.deps import get_current_user
from sqlalchemy.orm import Session
from app.core.deps import get_db
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
)
from app.models.ws_message import StreamMessage
from app.services.chat_record_service import save_chat_record
from datetime import datetime, timezone
from app.services.chat_record_service import get_recent_chat_history
from app.llm.openai_client import get_llm_text_client, get_llm_vl_client
from app.memory.memory_extractor import get_memory_extractor
from app.services.user_service import get_user_by_uuid

router = APIRouter()

manager = ConnectionManager()
agent = NormalAgent()
memory_extractor = get_memory_extractor()


@router.websocket("/ws/{user_uuid}")
async def websocket_endpoint(
    websocket: WebSocket, user_uuid: str, db: Session = Depends(get_db)
):
    user = get_user_by_uuid(db=db, user_uuid=user_uuid)
    user_id = user.id
    await manager.connect(user_uuid, websocket)
    try:
        while True:
            data_raw = await websocket.receive_text()
            history = get_recent_chat_history(user_id=user_id, db=db)
            try:
                payload = json.loads(data_raw)
                uuid = payload.get("uuid", "")
                text = payload.get("text", "")
                image = payload.get("image", [])
                video = payload.get("video", "")
            except json.JSONDecodeError:
                message = StreamMessage(
                    uuid=uuid, content="服务异常，请稍后再试。", status="error"
                )
                await websocket.send_text(message.model_dump_json())
                return

            logger.info(
                f"Received from {user_id}: uuid={uuid}, text={text}, image={image}, video={video}"
            )

            client = get_llm_vl_client() if image else get_llm_text_client()
            content = (
                [ChatCompletionContentPartTextParam(type="text", text=text)]
                + [
                    ChatCompletionContentPartImageParam(
                        type="image_url", image_url={"url": url, "detail": "auto"}
                    )
                    for url in image
                ]
                if image
                else [ChatCompletionContentPartTextParam(type="text", text=text)]
            )
            user_msg = ChatCompletionUserMessageParam(role="user", content=content)
            system = ChatCompletionSystemMessageParam(
                role="system", content="You are a helpful assistant."
            )

            response_start_time = datetime.now(timezone.utc)
            await save_chat_record(
                db=db,
                user_id=int(user_id),
                uuid=uuid,
                role="user",
                model=None,
                text=text,
                image=image,
                video=video,
                response_start_time=response_start_time,
                response_end_time=None,
            )

            first_chunk = True
            response_start_time = datetime.now(timezone.utc)
            full_response = ""
            if image:
                async for chunk in client.stream_chat(system, user_msg, history):
                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason
                    content_piece = delta.content or ""
                    full_response += content_piece
                    message = StreamMessage(
                        uuid=uuid,
                        content=content_piece,
                        status="streaming" if finish_reason is None else "done",
                    )
                    if first_chunk:
                        message = StreamMessage(uuid=uuid, content="", status="start")
                        await websocket.send_text(message.model_dump_json())
                        first_chunk = False
                    await websocket.send_text(message.model_dump_json())
            else:
                async for chunk in agent.run(text, user_uuid, []):
                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason
                    content_piece = delta.content or ""
                    full_response += content_piece
                    message = StreamMessage(
                        uuid=uuid,
                        content=content_piece,
                        status="streaming" if finish_reason is None else "done",
                    )
                    if first_chunk:
                        message = StreamMessage(uuid=uuid, content="", status="start")
                        await websocket.send_text(message.model_dump_json())
                        first_chunk = False
                    await websocket.send_text(message.model_dump_json())

            response_end_time = datetime.now(timezone.utc)
            await save_chat_record(
                db=db,
                user_id=int(user_id),
                uuid=uuid,
                role="assistant",
                model=chunk.model,
                text=full_response,
                image=image,
                video=video,
                response_start_time=response_start_time,
                response_end_time=response_end_time,
            )
            if history and len(history) >= 3:
                await memory_extractor.extract_memory_points(
                    user_id=user_uuid, message=text, reply=history[-3]["content"]
                )
    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        import traceback

        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        message = StreamMessage(
            uuid=uuid, content="服务异常，请稍后再试。", status="error"
        )
        await websocket.send_text(message.model_dump_json())
        manager.disconnect(user_id)


@router.websocket("/ws-auth/{user_id}")
async def websocket_endpoint_auth(
    websocket: WebSocket, user_id: str, db: Session = Depends(get_db)
):
    """带认证的WebSocket聊天端点
    Args:
        websocket: WebSocket连接对象
        user_id: 用户ID路径参数
        db: 数据库会话依赖注入
    """
    try:
        # 从查询参数获取token
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=1008)  # 无token则关闭连接(1008表示策略违规)
            return

        # 验证token并获取当前用户
        scheme = "Bearer"
        credentials = token
        current_user = get_current_user(
            credentials=HTTPAuthorizationCredentials(
                scheme=scheme, credentials=credentials
            ),
            db=db,
        )
        # 验证用户ID是否匹配
        if current_user.id != int(user_id):
            await websocket.close(code=1008)
            return

        # 连接管理
        await manager.connect(user_id, websocket)

        # 主消息循环
        while True:
            # 接收原始消息数据
            data_raw = await websocket.receive_text()
            # 获取最近聊天历史
            history = get_recent_chat_history(user_id=int(user_id), db=db)

            try:
                # 解析JSON消息
                payload = json.loads(data_raw)
                uuid = payload.get("uuid", "")  # 消息唯一ID
                text = payload.get("text", "")  # 文本内容
                image = payload.get("image", [])  # 图片URL列表
                video = payload.get("video", "")  # 视频URL
            except json.JSONDecodeError:
                # JSON解析错误处理
                message = StreamMessage(
                    uuid=uuid, content="服务异常，请稍后再试。", status="error"
                )
                await websocket.send_text(message.model_dump_json())
                return

            # 记录接收日志
            logger.info(
                f"Received from {user_id}: uuid={uuid}, text={text}, image={image}, video={video}"
            )

            # 根据是否有图片选择不同的客户端(VL支持多模态)
            client = get_llm_vl_client() if image else get_llm_text_client()

            # 构建消息内容(支持文本+图片)
            content = (
                [ChatCompletionContentPartTextParam(type="text", text=text)]
                + [
                    ChatCompletionContentPartImageParam(
                        type="image_url", image_url={"url": url, "detail": "auto"}
                    )
                    for url in image
                ]
                if image
                else [ChatCompletionContentPartTextParam(type="text", text=text)]
            )

            # 系统提示词
            system = ChatCompletionSystemMessageParam(
                role="system", content="You are a helpful assistant."
            )
            # 用户消息
            user_msg = ChatCompletionUserMessageParam(role="user", content=content)

            # 保存用户消息记录
            response_start_time = datetime.now(timezone.utc)
            await save_chat_record(
                db=db,
                user_id=int(user_id),
                uuid=uuid,
                role="user",
                model=None,
                text=text,
                image=image,
                video=video,
                response_start_time=response_start_time,
                response_end_time=None,
            )

            # 流式响应处理
            first_chunk = True
            response_start_time = datetime.now(timezone.utc)
            full_response = ""

            # 流式获取AI响应
            async for chunk in client.stream_chat(system, user_msg, history):
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                content_piece = delta.content or ""
                full_response += content_piece

                # 构建流式消息
                message = StreamMessage(
                    uuid=uuid,
                    content=content_piece,
                    status="streaming" if finish_reason is None else "done",
                )
                # 发送开始标记(仅第一次)
                if first_chunk:
                    message = StreamMessage(uuid=uuid, content="", status="start")
                    await websocket.send_text(message.model_dump_json())
                    first_chunk = False
                # 发送内容片段
                await websocket.send_text(message.model_dump_json())

            # 保存AI响应记录
            response_end_time = datetime.now(timezone.utc)
            await save_chat_record(
                db=db,
                user_id=int(user_id),
                uuid=uuid,
                role="assistant",
                model=chunk.model,
                text=full_response,
                image=image,
                video=video,
                response_start_time=response_start_time,
                response_end_time=response_end_time,
            )

    except WebSocketDisconnect:
        # WebSocket断开连接处理
        manager.disconnect(user_id)
    except Exception as e:
        # 其他异常处理
        import traceback

        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        message = StreamMessage(
            uuid=uuid, content="服务异常，请稍后再试。", status="error"
        )
        await websocket.send_text(message.model_dump_json())
        manager.disconnect(user_id)
