"""APP_PROFILE selects .env.<profile> on top of .env."""

from __future__ import annotations

from pathlib import Path

import pytest

from support_agents.mcp.session import _server_parameters
from support_agents.settings import Settings, env_files


def test_without_a_profile_only_env_is_loaded(tmp_path: Path):
    assert env_files(None, tmp_path) == (tmp_path / ".env",)
    assert env_files("", tmp_path) == (tmp_path / ".env",)


def test_a_profile_is_loaded_after_env(tmp_path: Path):
    assert env_files("bedrock", tmp_path) == (tmp_path / ".env", tmp_path / ".env.bedrock")


@pytest.mark.parametrize("bad", ["../x", "a/b", "a b", "bed.rock", "x;rm"])
def test_unsafe_profile_names_are_rejected(bad: str, tmp_path: Path):
    with pytest.raises(ValueError, match="APP_PROFILE"):
        env_files(bad, tmp_path)


def test_the_profile_file_overrides_the_base_file(tmp_path: Path):
    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql://base/base\nSUPERVISOR_PROVIDER=ollama\nRATE_LIMIT_REQUESTS=7\n"
    )
    (tmp_path / ".env.bedrock").write_text(
        "DATABASE_URL=postgresql://bedrock/bedrock\nSUPERVISOR_PROVIDER=bedrock\n"
    )
    loaded = Settings(_env_file=env_files("bedrock", tmp_path))
    assert loaded.database_url == "postgresql://bedrock/bedrock"
    assert loaded.supervisor_provider == "bedrock"
    assert loaded.rate_limit_requests == 7  # only in the base file, still loaded


def test_a_missing_profile_file_is_ignored(tmp_path: Path):
    (tmp_path / ".env").write_text("DATABASE_URL=postgresql://base/base\n")
    loaded = Settings(_env_file=env_files("nope", tmp_path))
    assert loaded.database_url == "postgresql://base/base"


def test_the_mcp_server_receives_the_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APP_PROFILE", "bedrock")
    assert _server_parameters().env["APP_PROFILE"] == "bedrock"
