from abc import ABC, abstractmethod
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletion,
    ChatCompletionChunk,
)


class BaseLLM(ABC):
    @abstractmethod
    async def chat(
        self,
        system: ChatCompletionSystemMessageParam,
        prompt: ChatCompletionMessageParam,
        history: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        pass

    @abstractmethod
    async def stream_chat(
        self,
        system: ChatCompletionSystemMessageParam,
        prompt: ChatCompletionMessageParam,
        history: list[ChatCompletionMessageParam],
    ) -> ChatCompletionChunk:
        pass
