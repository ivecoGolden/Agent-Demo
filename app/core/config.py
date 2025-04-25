from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    APP_NAME: str = "UserApp"
    ENV: Literal["dev", "production"] = "production"
    DATABASE_URL: str

    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    postgres_user: str
    postgres_password: str
    postgres_db: str

    ALI_LLM_KEY: str
    ALI_LLM_BASE_URL: str
    EMBEDDING_MODEL_NAME: str = "text-embedding-v3"

    HUOSHAN_LLM_KEY: str
    HUOSHAN_LLM_BASE_URL: str

    LANGSMITH_TRACING: bool
    LANGSMITH_API_KEY: str
    LANGSMITH_ENDPOINT: str
    LANGSMITH_PROJECT: str

    @property
    def MILVUS_HOST(self) -> str:
        return "localhost" if self.ENV == "dev" else "milvus-standalone"

    class Config:
        env_file = ".env"


settings = Settings()
