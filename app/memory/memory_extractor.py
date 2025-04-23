from typing import Literal
import logging
import json


def log_error(message: str, exception: Exception):
    logging.error(f"{message}: {exception}")


SYSTEM_PROMPT: Literal["str"] = (
    "你是一个用户画像助手，你的任务是从用户与助手的对话中提取可以描述用户特征的信息。要提取你认为有价值的信息"
    "请使用 JSON 数组格式输出，每个对象包含一个分类和对应的内容，例如："
    '[{"兴趣偏好": "我喜欢喝咖啡"}, {"行为习惯": "我每天早上跑步"}]。\n\n'
    "【注意】\n"
    "- 只提取明确表达的事实或倾向，不能编造内容。\n"
    "- 如无可提取信息，请仅返回字符串 “无”（不要返回 JSON 空数组）。\n"
    "- 遇到歧义或不确定内容，请忽略，不要臆测。"
)

from app.llm.openai_client import get_llm_text_client
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from app.memory.memory_service import get_memory_service


class MemoryExtractor:
    def __init__(self):
        self.llm = get_llm_text_client()
        self.service = get_memory_service()

    @staticmethod
    def parse_memory_response(text: str):
        if text == "无":
            return [], []

        try:
            items = json.loads(text)
        except Exception as e:
            log_error("记忆点 JSON 解析失败", e)
            return [], []

        if not isinstance(items, list):
            logging.error("记忆点 JSON 格式错误：不是数组")
            return [], []

        contents = []
        categories = []
        for item in items:
            if isinstance(item, dict):
                for category, content in item.items():
                    categories.append(category)
                    contents.append(content)
        return contents, categories

    async def extract_memory_points(
        self, message: str, reply: str, user_id: str
    ) -> None:
        """提取用户画像相关的记忆点，并保存到向量数据库"""
        system_prompt = ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT,
        )
        user_message = ChatCompletionUserMessageParam(
            role="user",
            content=f"以下是用户与助手的对话，请提取用户画像相关的记忆点：\n助手：{reply}\n用户：{message}",
        )

        response = await self.llm.chat(
            system=system_prompt, prompt=user_message, history=[]
        )
        text = response.choices[0].message.content.strip()

        contents, categories = self.parse_memory_response(text)
        if not contents:
            return

        await self.service.add_user_memories(
            user_id=user_id, contents=contents, categories=categories
        )


_memory_extractor: MemoryExtractor | None = None


def get_memory_extractor() -> MemoryExtractor:
    global _memory_extractor
    if _memory_extractor is None:
        _memory_extractor = MemoryExtractor()
    return _memory_extractor
