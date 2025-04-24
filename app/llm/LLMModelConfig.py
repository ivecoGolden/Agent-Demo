from enum import Enum
from dataclasses import dataclass
from app.core.config import settings


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str
    temperature: float
    api_key: str = settings.ALI_LLM_KEY


class LLMModelConfig(Enum):
    QWEN_TURBO = ModelConfig(
        model="qwen-turbo-2025-02-11",
        base_url=settings.ALI_LLM_BASE_URL,
        api_key=settings.ALI_LLM_KEY,
        temperature=0.7,
    )
    QWEN_VL = ModelConfig(
        model="qwen2.5-vl-32b-instruct",
        base_url=settings.ALI_LLM_BASE_URL,
        api_key=settings.ALI_LLM_KEY,
        temperature=0.7,
    )
    QWEN_PLUS_LATEST = ModelConfig(
        model="qwen-plus-latest",
        base_url=settings.ALI_LLM_BASE_URL,
        api_key=settings.ALI_LLM_KEY,
        temperature=0,
    )
    QWEN_TURBO_0 = ModelConfig(
        model="qwen-turbo-2025-02-11",
        base_url=settings.ALI_LLM_BASE_URL,
        api_key=settings.ALI_LLM_KEY,
        temperature=1.0,
    )
    DOUBAO_1_5_LITE = ModelConfig(
        model="doubao-1-5-lite-32k-250115",
        base_url=settings.HUOSHAN_LLM_BASE_URL,
        api_key=settings.HUOSHAN_LLM_KEY,
        temperature=1.0,
    )
