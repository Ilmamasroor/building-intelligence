from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Loaded from the environment and an untracked `.env`. Every credential is optional."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    mongodb_uri: SecretStr | None = None
    voyage_api_key: SecretStr | None = None
    capstone_api_key: SecretStr | None = None
    generation_api_base_url: str | None = None
    generation_api_key: SecretStr | None = None
    rerank_api_key: SecretStr | None = None
    mongodb_db_name: str = "building_with_rag"
    mongodb_test_db_name: str = "building_with_rag_test"
    webui_demo_caller_id: str = "demo-public"
    generation_model_name: str = "gpt-4o-mini"
    rerank_api_base_url: str = "https://api.voyageai.com/v1"
    rerank_model_name: str = "rerank-2.5"
    rerank_request_timeout_seconds: int = 30
    rerank_candidate_limit: int = 20
    rerank_send_limit: int = 10
    rerank_return_limit: int = 5
    # Story 6.1; kept loose so a bad value falls back to 30 in the controller instead of failing.
    agentic_deadline_seconds: int | str = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
