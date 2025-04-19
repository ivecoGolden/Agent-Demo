from typing import Any, Dict, List, AsyncGenerator
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionChunk,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from app.llm.openai_client import get_llm_text_client
from app.core.config import settings
from app.prompt.systemPrompt import SystemPrompt, build_prompt
from app.services.rag_service import RAGService
import json
from app.memory.memory_service import get_memory_service


# 工具定义（注册给模型）
query_manual_tool_schema: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "query_manual",
        "description": "当用户询问关于你的身份、能力、使用方式或限制等问题时，使用此工具从产品说明文档中检索信息，帮助你更准确地回答。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要查询的用户问题",
                }
            },
            "required": ["query"],
        },
    },
}


# 工具函数（由代码执行）
async def call_query_manual_tool(args: Dict[str, Any]) -> str:
    query = args["query"]
    rag_service = RAGService()
    chunks = await rag_service.query(query)
    return "\n".join(chunks)


memory_service = get_memory_service()


# 智能体类
class NormalAgent:
    def __init__(self):
        self.client = get_llm_text_client()
        self.tools: List[ChatCompletionToolParam] = [
            query_manual_tool_schema,
        ]
        self.tool_map = {
            "query_manual": call_query_manual_tool,
        }

    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]
            if tool_name in self.tool_map:
                result = await self.tool_map[tool_name](tool_args)
                results.append(result)
        return "\n".join(results)

    async def run(
        self, query: str, user_uuid: str, history: list[ChatCompletionMessageParam]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        user_prompt = ChatCompletionUserMessageParam(role="user", content=query)
        parsed_memories = await memory_service.search_user_memory_parsed(
            user_id=user_uuid, query=query
        )
        user_memery = "\n".join(
            f"【{m['category']}】{m['content']}" for m in parsed_memories
        )
        stream = self.client.stream_chat_with_tools(
            system=build_prompt(
                SystemPrompt.BASE_CALL,
                assistant_name="MGAgent",
                user_memory=user_memery,
            ),
            history=history,
            prompt=user_prompt,
            tools=self.tools,
            tool_choice="auto",
        )

        async for chunk in stream:
            choice = chunk.choices[0]

            if chunk.id == "tool_calls_final" and choice.delta.tool_calls:
                tool_calls = []
                for tool_call in choice.delta.tool_calls:
                    tool_info = {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments or "{}"),
                    }
                    tool_calls.append(tool_info)

                tool_result_text = await self._handle_tool_calls(tool_calls)
                async for chunk in self.client.stream_chat(
                    build_prompt(
                        SystemPrompt.TOOL_UESD_CALL,
                        tool_result=tool_result_text,
                        user_memory=user_memery,
                    ),
                    user_prompt,
                    history,
                ):
                    yield chunk
            else:
                yield chunk
