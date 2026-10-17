"""Environment-backed runtime settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    database_url: str = Field(min_length=1)
    ollama_base_url: str = "http://localhost:11434"
    ollama_request_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    ollama_embedding_model: str = "nomic-embed-text"
    academic_document_path: Path = PROJECT_ROOT / "data" / "academic" / "subjects_standards_1_to_10.pdf"
    academic_book_recommendations_document_path: Path = (
        PROJECT_ROOT
        / "data"
        / "academic"
        / "ncert_book_recommendations_standards_1_to_10_2026-27.pdf"
    )
    academic_ingest_api_key: str | None = None
    academic_search_top_k: int = Field(default=4, ge=1, le=10)
    academic_search_min_similarity: float = Field(default=0.30, ge=-1.0, le=1.0)
    academic_agent_ollama_model: str = "qwen3:4b-instruct"
    aws_region: str = "us-east-1"
    supervisor_provider: Literal["ollama", "bedrock"] = "ollama"
    supervisor_ollama_model: str = "qwen3:4b-instruct"
    supervisor_bedrock_model_id: str = "us.amazon.nova-lite-v1:0"
    final_response_provider: Literal["ollama", "bedrock"] = "ollama"
    final_response_ollama_model: str = "qwen3:4b-instruct"
    final_response_bedrock_model_id: str = "us.amazon.nova-lite-v1:0"
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "customer-support-dev"
    langsmith_endpoint: str | None = None
    mcp_transport: str = "stdio"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(default=8000, ge=1, le=65535)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_langsmith_tracing(settings: Settings | None = None) -> None:
    """Configure LangChain's environment-based tracing before building the graph."""
    import os

    active_settings = settings or get_settings()
    if not active_settings.langsmith_tracing:
        # Explicitly disable tracing even if the shell has it enabled.
        os.environ["LANGSMITH_TRACING"] = "false"
        return

    if active_settings.langsmith_api_key is None:
        raise ValueError("LANGSMITH_API_KEY is required when LANGSMITH_TRACING=true.")

    api_key = active_settings.langsmith_api_key.get_secret_value().strip()
    if not api_key:
        raise ValueError("LANGSMITH_API_KEY must not be blank when tracing is enabled.")

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = active_settings.langsmith_project
    if active_settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = active_settings.langsmith_endpoint
