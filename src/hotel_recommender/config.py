from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or ``.env``."""

    app_name: str = "Production Hotel Recommender"
    app_env: str = "development"
    model_path: Path = Path("artifacts/recommender.joblib")
    default_k: int = Field(default=10, ge=1, le=50)
    max_k: int = Field(default=50, ge=1, le=200)
    candidate_pool_size: int = Field(default=100, ge=10, le=2_000)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
