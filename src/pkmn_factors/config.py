from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    env: str = Field(default="dev", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/pkmn",
        alias="DATABASE_URL",
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
