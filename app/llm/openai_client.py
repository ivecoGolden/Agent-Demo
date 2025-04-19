from openai import AsyncOpenAI
import asyncio
from typing import AsyncGenerator, Iterable
from app.llm.base import BaseLLM
from app.core.config import settings
from app.llm import LLMModelConfig
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionToolParam,
)
from app.llm.LLMModelConfig import LLMModelConfig


class OpenAILLMClient(BaseLLM):
    def __init__(self, model_config: LLMModelConfig):
        """初始化OpenAI客户端

        Args:
            model_config: 模型配置对象，包含base_url、model等配置信息
        """
        config = model_config.value  # 获取模型配置值
        self.client = AsyncOpenAI(
            api_key=settings.ALI_LLM_KEY,  # 使用配置的API密钥
            base_url=config.base_url,  # 使用模型配置的基础URL
        )
        self.model = config.model  # 设置模型名称
        self.temperature = config.temperature  # 设置生成温度参数

    async def chat(
        self,
        system: ChatCompletionSystemMessageParam,
        prompt: ChatCompletionMessageParam,
        history: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        messages: list[ChatCompletionMessageParam] = [system]
        messages.extend(history)
        messages.append(prompt)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            temperature=self.temperature,
        )
        return response

    async def stream_chat(
        self,
        system: ChatCompletionSystemMessageParam,
        prompt: ChatCompletionMessageParam,
        history: list[ChatCompletionMessageParam],
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        messages: list[ChatCompletionMessageParam] = [system]
        messages.extend(history)
        messages.append(prompt)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=self.temperature,
        )
        async for chunk in stream:
            yield chunk
            await asyncio.sleep(0)

    async def stream_chat_with_tools(
        self,
        system: ChatCompletionSystemMessageParam,
        prompt: ChatCompletionMessageParam,
        history: list[ChatCompletionMessageParam],
        tools: Iterable[ChatCompletionToolParam],
        tool_choice: str = "auto",
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        messages: list[ChatCompletionMessageParam] = [system]
        messages.extend(history)
        messages.append(prompt)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=self.temperature,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=True,
        )

        tool_calls_complete = False
        tool_call_accumulator = {}

        async for chunk in stream:
            choice = chunk.choices[0]

            # 收集普通响应内容
            if not choice.delta.tool_calls and choice.finish_reason != "tool_calls":
                yield chunk
                continue

            # 收集工具调用片段
            if choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    idx = tool_call.index
                    func = tool_call.function
                    if idx not in tool_call_accumulator:
                        tool_call_accumulator[idx] = {"name": "", "arguments": ""}

                    if func.name:
                        tool_call_accumulator[idx]["name"] = func.name
                    if func.arguments:
                        tool_call_accumulator[idx]["arguments"] += func.arguments

            if choice.finish_reason == "tool_calls":
                tool_calls_complete = True
                break

        if tool_calls_complete:
            from openai.types.chat import ChatCompletionChunk
            from openai.types.chat.chat_completion_chunk import (
                Choice,
                ChoiceDelta,
                ChoiceDeltaToolCall,
                ChoiceDeltaToolCallFunction,
            )

            tool_call_chunks = []
            for idx, call in tool_call_accumulator.items():
                tool_call_chunks.append(
                    ChoiceDeltaToolCall(
                        index=idx,
                        id=f"call_tool_{idx}",
                        function=ChoiceDeltaToolCallFunction(
                            name=call["name"], arguments=call["arguments"]
                        ),
                        type="function",
                    )
                )

            final_chunk = ChatCompletionChunk(
                id="tool_calls_final",
                object="chat.completion.chunk",
                created=0,
                model=self.model,
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(
                            role="assistant",
                            content=None,
                            function_call=None,
                            tool_calls=tool_call_chunks,
                        ),
                        finish_reason="tool_calls",
                    )
                ],
            )
            yield final_chunk


_llm_text_client: OpenAILLMClient | None = None
_llm_vl_client: OpenAILLMClient | None = None


def get_llm_text_client() -> OpenAILLMClient:
    global _llm_text_client
    if _llm_text_client is None:
        _llm_text_client = OpenAILLMClient(LLMModelConfig.QWEN_TURBO)
    return _llm_text_client


def get_llm_vl_client() -> OpenAILLMClient:
    global _llm_vl_client
    if _llm_vl_client is None:
        _llm_vl_client = OpenAILLMClient(LLMModelConfig.QWEN_VL)
    return _llm_vl_client
