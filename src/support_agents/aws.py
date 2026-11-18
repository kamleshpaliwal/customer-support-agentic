"""Create AWS (Bedrock) clients with explicit, intentional credentials.

``boto3`` normally searches several places for credentials, and on a developer machine that
search quietly finds whatever the AWS CLI is configured with. This module makes the choice
visible: keys from the settings are used as-is, a role from the compute platform is accepted,
and an accidental fallback to the CLI credentials file is an error.
"""

from __future__ import annotations

import os
from typing import Any

import boto3

from support_agents.settings import Settings, get_settings

# ``Credentials.method`` values that mean "static keys from the AWS CLI files".
_CLI_FILE_METHODS = frozenset({"shared-credentials-file", "config-file"})


class AwsCredentialsError(RuntimeError):
    """Raised when AWS credentials are missing, incomplete, or not chosen deliberately."""


def _build_session(settings: Settings, region_name: str | None) -> boto3.Session:
    region = region_name or settings.aws_region
    key_id = settings.aws_access_key_id
    secret_field, token_field = settings.aws_secret_access_key, settings.aws_session_token
    secret = secret_field.get_secret_value() if secret_field else None

    if bool(key_id) != bool(secret):
        raise AwsCredentialsError(
            "Set both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or neither."
        )
    if key_id and secret:
        token = token_field.get_secret_value() if token_field else None
        return boto3.Session(
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            aws_session_token=token or None,
            region_name=region,
        )

    session = boto3.Session(region_name=region)
    credentials = session.get_credentials()
    if credentials is None:
        raise AwsCredentialsError(
            "No AWS credentials found. For local development set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY in .env; in deployment attach an IAM role to the service."
        )
    deliberate_profile = "AWS_PROFILE" in os.environ
    if (
        credentials.method in _CLI_FILE_METHODS
        and not deliberate_profile
        and not settings.aws_allow_cli_profile
    ):
        raise AwsCredentialsError(
            "Refusing to use the default AWS CLI credentials. Set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY in .env for this project, or set AWS_ALLOW_CLI_PROFILE=true "
            "to use the CLI profile on purpose."
        )
    return session


def create_aws_client(
    service: str,
    settings: Settings | None = None,
    *,
    region_name: str | None = None,
    config: Any | None = None,
) -> Any:
    """Return a client for ``service`` that follows the credential rules above."""
    session = _build_session(settings or get_settings(), region_name)
    return session.client(service, config=config)


def create_bedrock_runtime_client(
    settings: Settings | None = None,
    *,
    region_name: str | None = None,
    config: Any | None = None,
) -> Any:
    """Return a ``bedrock-runtime`` client (model invocation)."""
    return create_aws_client("bedrock-runtime", settings, region_name=region_name, config=config)
