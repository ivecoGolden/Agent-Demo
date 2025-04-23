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
from fastapi import WebSocketException, status


async def get_user_from_websocket(websocket: WebSocket, db: Session = Depends(get_db)):
    token = websocket.headers.get("Authorization")
    if token and token.startswith("Bearer "):
        token = token[7:]
    else:
        token = websocket.query_params.get("token")

    if not token:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    return get_current_user(credentials=credentials, db=db)


router = APIRouter()

manager = ConnectionManager()
agent = NormalAgent()
memory_extractor = get_memory_extractor()


@router.websocket("/ws-auth")
async def websocket_endpoint_auth(
    websocket: WebSocket,
    db: Session = Depends(get_db),
    current_user=Depends(get_user_from_websocket),
):
    async def handle_exception(e: Exception, uuid: str = ""):
        import traceback

        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        message = StreamMessage(
            uuid=uuid, content="服务异常，请稍后再试。", status="error"
        )
        await websocket.send_text(message.model_dump_json())
        manager.disconnect(user_uuid)

    try:
        user_uuid = current_user.userid

        # 连接管理
        await manager.connect(user_uuid, websocket)

        # 主消息循环
        while True:
            try:
                # 接收原始消息数据
                data_raw = await websocket.receive_text()
                # 获取最近聊天历史
                history = get_recent_chat_history(user_id=user_uuid, db=db)

                # 解析JSON消息
                payload = json.loads(data_raw)
                uuid = payload.get("uuid", "")  # 消息唯一ID
                text = payload.get("text", "")  # 文本内容
                image = payload.get("image", [])  # 图片URL列表
                video = payload.get("video", "")  # 视频URL
            except json.JSONDecodeError:
                await handle_exception(Exception("JSON解析失败"), uuid)
                continue

            # 记录接收日志
            logger.info(
                f"Received from {uuid}: text={text}, image={image}, video={video}"
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
                user_id=user_uuid,
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
            if image:
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
                        start_msg = StreamMessage(uuid=uuid, content="", status="start")
                        await websocket.send_text(start_msg.model_dump_json())
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
                        start_msg = StreamMessage(uuid=uuid, content="", status="start")
                        await websocket.send_text(start_msg.model_dump_json())
                        first_chunk = False
                    await websocket.send_text(message.model_dump_json())
            # 保存AI响应记录
            response_end_time = datetime.now(timezone.utc)
            await save_chat_record(
                db=db,
                user_id=user_uuid,
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
        manager.disconnect(user_uuid)
    except Exception as e:
        await handle_exception(e)


@router.websocket("/ws-auth-late")
async def websocket_endpoint_auth_late(
    websocket: WebSocket, db: Session = Depends(get_db)
):
    await websocket.accept()

    try:
        auth_data = await websocket.receive_text()
        payload = json.loads(auth_data)
        token = payload.get("token")
        if not token:
            raise ValueError("缺少 token")

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        user = get_current_user(credentials=credentials, db=db)
        user_uuid = user.userid

    except WebSocketDisconnect:
        logger.warning("❌ 客户端在身份验证前断开连接")
        return
    except Exception as e:
        logger.error(f"WebSocket Auth Failed: {e}")
        try:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        except RuntimeError as re:
            logger.warning(f"连接可能已关闭，忽略 websocket.close 错误: {re}")
        return

    async def handle_exception(e: Exception, uuid: str = ""):
        import traceback

        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        message = StreamMessage(
            uuid=uuid, content="服务异常，请稍后再试。", status="error"
        )
        await websocket.send_text(message.model_dump_json())
        manager.disconnect(user_uuid)

    try:
        await manager.connect(user_uuid, websocket)

        while True:
            try:
                data_raw = await websocket.receive_text()
                history = get_recent_chat_history(user_id=user_uuid, db=db)

                payload = json.loads(data_raw)
                uuid = payload.get("uuid", "")
                text = payload.get("text", "")
                image = payload.get("image", [])
                video = payload.get("video", "")
            except json.JSONDecodeError:
                await handle_exception(Exception("JSON解析失败"), uuid)
                continue

            logger.info(
                f"Received from {uuid}: text={text}, image={image}, video={video}"
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

            system = ChatCompletionSystemMessageParam(
                role="system", content="You are a helpful assistant."
            )
            user_msg = ChatCompletionUserMessageParam(role="user", content=content)

            response_start_time = datetime.now(timezone.utc)
            await save_chat_record(
                db=db,
                user_id=user_uuid,
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
                        start_msg = StreamMessage(uuid=uuid, content="", status="start")
                        await websocket.send_text(start_msg.model_dump_json())
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
                        start_msg = StreamMessage(uuid=uuid, content="", status="start")
                        await websocket.send_text(start_msg.model_dump_json())
                        first_chunk = False
                    await websocket.send_text(message.model_dump_json())

            response_end_time = datetime.now(timezone.utc)
            await save_chat_record(
                db=db,
                user_id=user_uuid,
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
        manager.disconnect(user_uuid)
    except Exception as e:
        await handle_exception(e)
