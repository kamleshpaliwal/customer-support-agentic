"""Bridge MCP elicitation prompts to the browser-based support UI."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock

from mcp.types import ElicitResult


@dataclass
class PendingElicitation:
    request_id: str
    message: str
    field_name: str
    options: list[str]
    future: Future[tuple[str, str | None]]


class ElicitationBroker:
    """Hold an MCP tool call until the user answers through the Flask UI."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: dict[str, PendingElicitation] = {}

    async def ask(
        self,
        request_id: str,
        *,
        message: str,
        field_name: str,
        options: list[str],
    ) -> ElicitResult:
        pending = PendingElicitation(
            request_id=request_id,
            message=message,
            field_name=field_name,
            options=options,
            future=Future(),
        )
        with self._lock:
            if request_id in self._pending:
                raise RuntimeError("An elicitation is already pending for this request.")
            self._pending[request_id] = pending

        try:
            action, answer = await asyncio.wrap_future(pending.future)
            if action != "accept" or answer is None:
                return ElicitResult(action=action)
            return ElicitResult(action="accept", content={field_name: answer})
        finally:
            with self._lock:
                if self._pending.get(request_id) is pending:
                    self._pending.pop(request_id, None)

    def get_pending(self, request_id: str) -> PendingElicitation | None:
        with self._lock:
            pending = self._pending.get(request_id)
            return None if pending is None or pending.future.done() else pending

    def submit(self, request_id: str, answer: str) -> str:
        """Resolve a pending prompt with an allowed option or an explicit cancel."""
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None or pending.future.done():
                raise LookupError("No MCP elicitation is pending for this request.")

            normalized = answer.strip()
            if normalized.casefold() in {"cancel", "skip"}:
                pending.future.set_result(("cancel", None))
                return "cancel"

            selected = next(
                (option for option in pending.options if option.casefold() == normalized.casefold()),
                None,
            )
            if selected is None:
                raise ValueError("Choose one of the listed options or cancel the question.")

            pending.future.set_result(("accept", selected))
            return "accepted"


elicitation_broker = ElicitationBroker()
