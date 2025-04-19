import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import AsyncMock, patch
from app.llm.openai_client import OpenAILLMClient
from app.llm.LLMModelConfig import LLMModelConfig
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionMessageParam,
)


@pytest.mark.asyncio
@patch("app.llm.openai_client.AsyncOpenAI")
async def test_chat(mock_openai_cls):
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value={"mock": "response"})
    mock_openai_cls.return_value = mock_openai

    system = {"role": "system", "content": "You are a helpful assistant."}
    prompt = {"role": "user", "content": "你好"}
    history = []
    client = OpenAILLMClient(model_config=LLMModelConfig.QWEN_TURBO)
    response = await client.chat(system, prompt, history)
    assert response == {"mock": "response"}
    mock_openai.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
@patch("app.llm.openai_client.AsyncOpenAI")
async def test_stream_chat(mock_openai_cls):
    async def mock_stream():
        yield {"mock_chunk": 1}
        yield {"mock_chunk": 2}

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream())
    mock_openai_cls.return_value = mock_openai

    system = {"role": "system", "content": "你是一个AI"}
    prompt = {"role": "user", "content": "讲个笑话"}
    history = []

    client = OpenAILLMClient(model_config=LLMModelConfig.QWEN_TURBO)
    chunks = []
    async for chunk in client.stream_chat(system, prompt, history):
        chunks.append(chunk)

    assert chunks == [{"mock_chunk": 1}, {"mock_chunk": 2}]
    mock_openai.chat.completions.create.assert_called_once()
