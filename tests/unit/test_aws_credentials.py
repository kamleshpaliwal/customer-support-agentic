"""Credential rules for Bedrock clients: explicit keys, roles, and no silent CLI fallback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from support_agents import aws as aws_module
from support_agents.aws import AwsCredentialsError, create_bedrock_runtime_client
from support_agents.settings import Settings

CLI_KEY = "AKIACLIFILEKEY"
EXPLICIT_KEY = "AKIAEXPLICITKEY"


@pytest.fixture(autouse=True)
def isolated_aws(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Point boto3 at throwaway files so the real ~/.aws and environment are never touched."""
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_ALLOW_CLI_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    credentials = tmp_path / "credentials"
    credentials.write_text(
        f"[default]\naws_access_key_id = {CLI_KEY}\naws_secret_access_key = cli-secret\n"
        f"[named]\naws_access_key_id = AKIANAMEDKEY\naws_secret_access_key = named-secret\n"
    )
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    return credentials


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, database_url="postgresql://unused/unused", **overrides)


def used_key(client) -> str:
    return client._request_signer._credentials.access_key


def test_explicit_keys_are_used_and_the_cli_profile_is_ignored():
    client = create_bedrock_runtime_client(
        settings(aws_access_key_id=EXPLICIT_KEY, aws_secret_access_key="secret")
    )
    assert used_key(client) == EXPLICIT_KEY


def test_only_one_of_key_and_secret_is_an_error():
    with pytest.raises(AwsCredentialsError, match="both"):
        create_bedrock_runtime_client(settings(aws_access_key_id=EXPLICIT_KEY))
    with pytest.raises(AwsCredentialsError, match="both"):
        create_bedrock_runtime_client(settings(aws_secret_access_key="secret"))


def test_default_cli_credentials_are_refused():
    with pytest.raises(AwsCredentialsError, match="default AWS CLI credentials"):
        create_bedrock_runtime_client(settings())


def test_cli_profile_can_be_allowed_on_purpose():
    client = create_bedrock_runtime_client(settings(aws_allow_cli_profile=True))
    assert used_key(client) == CLI_KEY


def test_a_profile_chosen_through_aws_profile_is_deliberate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AWS_PROFILE", "named")
    client = create_bedrock_runtime_client(settings())
    assert used_key(client) == "AKIANAMEDKEY"


def test_no_credentials_at_all_is_an_error(isolated_aws):
    isolated_aws.write_text("")
    with pytest.raises(AwsCredentialsError, match="No AWS credentials"):
        create_bedrock_runtime_client(settings())


def test_a_role_from_the_compute_service_is_accepted(monkeypatch: pytest.MonkeyPatch):
    class FakeSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_credentials(self):
            return SimpleNamespace(method="container-role")

        def client(self, service, config=None):
            return ("client", service, self.kwargs["region_name"])

    monkeypatch.setattr(aws_module.boto3, "Session", FakeSession)
    client = create_bedrock_runtime_client(settings(aws_region="eu-west-1"))
    assert client == ("client", "bedrock-runtime", "eu-west-1")


def test_secrets_do_not_appear_in_the_printed_settings():
    active = settings(
        aws_access_key_id=EXPLICIT_KEY,
        aws_secret_access_key="super-secret-value",
        aws_session_token="super-token-value",
    )
    printed = f"{active!r} {active}"
    assert "super-secret-value" not in printed
    assert "super-token-value" not in printed
