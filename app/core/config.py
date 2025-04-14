from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "UserApp"
    ENV: str = "dev"
    DATABASE_URL: str
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    class Config:
        env_file = ".env"


settings = Settings()
