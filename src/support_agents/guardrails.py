"""Input and output guardrails: a free local check, then optionally an Amazon Bedrock guardrail.

Every user query and every final answer goes through ``Guardrails``:

1. The **local** check is plain Python (no AWS, no cost). It blocks obvious prompt-injection
   phrases in a query and masks card numbers, Aadhaar numbers, PAN numbers and AWS access keys in
   both directions. It is on in every profile.
2. The **Bedrock** check (``ApplyGuardrail``) runs after it, only when ``GUARDRAIL_ID`` is set
   *and* the process was started with ``BEDROCK_CALLS_ENABLED=true``. Otherwise it is skipped, so
   the Ollama profile never touches AWS. ``ApplyGuardrail`` works on plain text, so it protects
   Ollama and Bedrock models alike.

If the Bedrock service errors, input fails closed (the query is refused) and output fails open
(the locally checked answer is returned and the error is logged).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from botocore.config import Config

from support_agents.aws import bedrock_calls_enabled, create_aws_client
from support_agents.settings import Settings, get_settings

logger = logging.getLogger(__name__)

BLOCKED_INPUT_MESSAGE = (
    "I can't help with that request. Please ask a question about school fees or academics."
)
UNAVAILABLE_MESSAGE = "The safety check is unavailable right now. Please try again shortly."
BLOCKED_OUTPUT_MESSAGE = (
    "I can't share that answer. Please rephrase your question or contact the school office."
)

Source = Literal["INPUT", "OUTPUT"]


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of a check. ``text`` is the (possibly masked) text to use when ``allowed``."""

    allowed: bool
    text: str
    reason: str | None = None
    message: str | None = None  # what to show the user when not allowed


# --- local check -------------------------------------------------------------------------------

_INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your|any)\b"
        r".{0,30}\b(instructions?|rules?|prompts?|guidelines?)\b",
        r"\b(reveal|show|print|repeat|leak)\b.{0,30}\b(system|hidden|initial)\b.{0,15}\bprompt\b",
        r"\byou are now\b.{0,40}\b(unrestricted|jailbroken|dan|developer mode)\b",
        r"\b(jailbreak|developer mode|do anything now)\b",
    )
)

_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_AADHAAR = re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_AWS_KEY = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")


def _luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value = value * 2 - 9 if value * 2 > 9 else value * 2
        total += value
    return total % 10 == 0


def _mask_cards(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group())
        return (
            "[REDACTED CARD]" if 13 <= len(digits) <= 19 and _luhn_valid(digits) else match.group()
        )

    return _CARD.sub(replace, text)


def mask_sensitive(text: str) -> tuple[str, list[str]]:
    """Mask card, Aadhaar, PAN and AWS key patterns; also return which kinds were found."""
    found: list[str] = []
    for kind, step in (
        ("card", _mask_cards),
        ("aadhaar", lambda t: _AADHAAR.sub("[REDACTED ID]", t)),
        ("pan", lambda t: _PAN.sub("[REDACTED ID]", t)),
        ("aws_key", lambda t: _AWS_KEY.sub("[REDACTED KEY]", t)),
    ):
        masked = step(text)
        if masked != text:
            found.append(kind)
            text = masked
    return text, found


def check_local(text: str, source: Source) -> GuardrailResult:
    """The free, offline check."""
    if source == "INPUT" and any(pattern.search(text) for pattern in _INJECTION_PATTERNS):
        return GuardrailResult(False, text, "prompt_injection", BLOCKED_INPUT_MESSAGE)
    masked, found = mask_sensitive(text)
    if found:
        logger.info("Local guardrail masked %s in %s", ",".join(found), source.lower())
    return GuardrailResult(True, masked, "masked:" + ",".join(found) if found else None)


# --- Bedrock check -----------------------------------------------------------------------------


def _any_blocked(node: Any) -> bool:
    """True if any assessment entry in the ``ApplyGuardrail`` response says BLOCKED."""
    if isinstance(node, dict):
        return node.get("action") == "BLOCKED" or any(_any_blocked(v) for v in node.values())
    if isinstance(node, list):
        return any(_any_blocked(item) for item in node)
    return False


class BedrockGuardrail:
    """Thin wrapper over the ``ApplyGuardrail`` API (creates its AWS client on first use)."""

    def __init__(self, guardrail_id: str, version: str, settings: Settings | None = None) -> None:
        self.guardrail_id = guardrail_id
        self.version = version
        self._settings = settings
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = create_aws_client(
                "bedrock-runtime",
                self._settings,
                config=Config(
                    connect_timeout=3,
                    read_timeout=10,
                    retries={"max_attempts": 2, "mode": "standard"},
                ),
            )
        return self._client

    def _apply(self, text: str, source: Source) -> GuardrailResult:
        response = self._get_client().apply_guardrail(
            guardrailIdentifier=self.guardrail_id,
            guardrailVersion=self.version,
            source=source,
            content=[{"text": {"text": text}}],
        )
        if response.get("action") != "GUARDRAIL_INTERVENED":
            return GuardrailResult(True, text)
        if _any_blocked(response.get("assessments", [])):
            return GuardrailResult(
                False,
                text,
                "bedrock_blocked",
                BLOCKED_INPUT_MESSAGE if source == "INPUT" else BLOCKED_OUTPUT_MESSAGE,
            )
        masked = "".join(
            item.get("text", "") for item in response.get("outputs", []) if isinstance(item, dict)
        )
        return GuardrailResult(True, masked or text, "bedrock_masked")

    async def check(self, text: str, source: Source) -> GuardrailResult:
        return await asyncio.to_thread(self._apply, text, source)


# --- combined ----------------------------------------------------------------------------------


class Guardrails:
    """Local check first, then the Bedrock guardrail when it is configured and enabled."""

    def __init__(
        self,
        *,
        local_enabled: bool = True,
        bedrock: BedrockGuardrail | None = None,
    ) -> None:
        self.local_enabled = local_enabled
        self.bedrock = bedrock

    async def _run(self, text: str, source: Source) -> GuardrailResult:
        result = GuardrailResult(True, text)
        if self.local_enabled:
            result = check_local(text, source)
            if not result.allowed:
                return result
        if self.bedrock is None:
            return result
        try:
            remote = await self.bedrock.check(result.text, source)
        except Exception:
            logger.exception("Bedrock guardrail failed on %s", source.lower())
            if source == "INPUT":
                return GuardrailResult(False, text, "guardrail_unavailable", UNAVAILABLE_MESSAGE)
            return result  # output: fail open with the locally checked text
        return (
            remote
            if not remote.allowed
            else GuardrailResult(True, remote.text, remote.reason or result.reason)
        )

    async def check_input(self, text: str) -> GuardrailResult:
        return await self._run(text, "INPUT")

    async def check_output(self, text: str) -> GuardrailResult:
        return await self._run(text, "OUTPUT")


def create_guardrails(settings: Settings | None = None) -> Guardrails:
    active = settings or get_settings()
    bedrock = None
    if active.guardrail_id:
        if bedrock_calls_enabled():
            bedrock = BedrockGuardrail(active.guardrail_id, active.guardrail_version, active)
        else:
            logger.warning(
                "GUARDRAIL_ID is set but BEDROCK_CALLS_ENABLED is not true; using the local "
                "guardrail only."
            )
    return Guardrails(local_enabled=active.guardrail_local_enabled, bedrock=bedrock)


@lru_cache
def get_guardrails() -> Guardrails:
    return create_guardrails()
