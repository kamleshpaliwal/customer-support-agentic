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
    academic_retrieval_mode: Literal["vector", "hybrid"] = "vector"
    academic_candidate_pool: int = Field(default=20, ge=1, le=100)
    academic_rerank_enabled: bool = False
    academic_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    academic_agent_provider: Literal["ollama", "bedrock"] = "ollama"
    academic_agent_ollama_model: str = "qwen3:4b-instruct"
    # The academic agent calls tools, so the model must support tool use through Converse.
    academic_agent_bedrock_model_id: str = "us.amazon.nova-lite-v1:0"
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
    # Authentication (Amazon Cognito access tokens by default; any OIDC-style JWT issuer works).
    # Fail closed: with auth enabled the app refuses to start without an issuer and client ID.
    auth_enabled: bool = True
    auth_issuer: str | None = None  # https://cognito-idp.<region>.amazonaws.com/<pool-id>
    auth_client_id: str | None = None  # app client ID the tokens were issued to
    auth_jwks_url: str | None = None  # defaults to <issuer>/.well-known/jwks.json
    auth_allowed_groups: str = "staff,admin"  # comma-separated; add "parent" for families
    auth_admin_group: str = "admin"
    auth_cognito_domain: str | None = None  # hosted login domain used by the browser UI
    # Overload protection. Per-user limits are shared by all workers; the model concurrency
    # limit applies per process, so the total is model_max_concurrency x workers.
    rate_limit_requests: int = Field(default=20, ge=0)  # per user per window; 0 disables
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    model_max_concurrency: int = Field(default=4, ge=0)  # simultaneous model calls; 0 = no limit
    model_queue_timeout_seconds: float = Field(default=10.0, gt=0)
    checkpoint_backend: Literal["postgres", "memory"] = "postgres"
    # Keep one MCP server process per API process instead of starting one per tool call.
    mcp_persistent_session: bool = True
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
