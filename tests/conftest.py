"""Test-wide safety: no test may send a request to AWS, whatever it constructs."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_real_aws_requests(monkeypatch: pytest.MonkeyPatch):
    """Allow tests to build AWS clients around fake keys, but make any real send fail loudly."""
    monkeypatch.setenv("BEDROCK_CALLS_ENABLED", "true")  # only so clients can be constructed
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")  # no credential probing over the network

    def refuse(self, request):
        raise RuntimeError(f"A test tried to send a real AWS request to {request.url}")

    monkeypatch.setattr("botocore.httpsession.URLLib3Session.send", refuse)
