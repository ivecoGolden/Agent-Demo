from fastapi import WebSocket
from typing import Dict
from app.core.logger import logger


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        # await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"User {user_id} connected. Active: {len(self.active_connections)}")

    def disconnect(self, user_id: str):
        self.active_connections.pop(user_id, None)
        logger.info(
            f"User {user_id} disconnected. Active: {len(self.active_connections)}"
        )

    async def send_personal_message(self, message: str, user_id: str):
        websocket = self.active_connections.get(user_id)
        if websocket:
            await websocket.send_text(message)

    async def broadcast(self, message: str):
        for uid, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Send to {uid} failed: {e}")
