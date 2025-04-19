from typing import List, Tuple
from app.llm.openai_client import get_llm_text_client
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from app.memory.memory_service import get_memory_service
import json


class MemoryExtractor:
    def __init__(self):
        self.llm = get_llm_text_client()
        self.service = get_memory_service()

    async def extract_memory_points(
        self, message: str, reply: str, user_id: str
    ) -> None:
        """提取用户画像相关的记忆点，并保存到向量数据库"""
        system_prompt = ChatCompletionSystemMessageParam(
            role="system",
            content=(
                "你是一个用户画像助手，请你从用户和助手的对话中提取出可以描述用户特征、行为、喜好、身份或状态的内容。"
                "输出格式要求是 JSON 数组，每一项是一个对象，包含一个分类和对应内容，例如："
                '[{"兴趣偏好": "我喜欢喝咖啡"}, {"行为习惯": "我每天早上跑步"}]\n\n'
                "请不要虚构内容；如果没有任何可以提取的内容，请只返回字符串 “无”。"
            ),
        )
        user_message = ChatCompletionUserMessageParam(
            role="user",
            content=f"以下是用户与助手的对话，请提取用户画像相关的记忆点：\n\n用户：{message}",
        )

        response = await self.llm.chat(
            system=system_prompt, prompt=user_message, history=[]
        )
        text = response.choices[0].message.content.strip()
        if text == "无":
            return

        try:
            items = json.loads(text)
        except Exception as e:
            print(f"记忆点 JSON 解析失败：{e}")
            return

        if not isinstance(items, list):
            print("记忆点 JSON 格式错误：不是数组")
            return

        contents = []
        categories = []
        for item in items:
            if isinstance(item, dict):
                for category, content in item.items():
                    categories.append(category)
                    contents.append(content)

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
