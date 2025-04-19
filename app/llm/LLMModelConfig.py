from enum import Enum
from dataclasses import dataclass
from app.core.config import settings


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str
    temperature: float


class LLMModelConfig(Enum):
    QWEN_TURBO = ModelConfig(
        model="qwen-turbo-2025-02-11",
        base_url=settings.ALI_LLM_BASE_URL,
        temperature=0.7,
    )
    QWEN_VL = ModelConfig(
        model="qwen2.5-vl-32b-instruct",
        base_url=settings.ALI_LLM_BASE_URL,
        temperature=0.7,
    )
