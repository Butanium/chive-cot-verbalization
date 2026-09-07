"""Pydantic-validated wrappers around llm_client.call_for_tool.

The stage scripts share one pattern for tool-using LLM calls: define a pydantic
model for the tool's input, generate the JSON schema from it
(`pipeline_utils.pydantic_tool`), and validate every returned tool_input through
that same model so flaky LLM output (wrong types, missing fields, out-of-range
ints) is caught at the boundary instead of corrupting downstream data.

`call_for_validated_tool` is a thin wrapper that retries on pydantic
ValidationError in addition to the missing-required-key / no-tool-call / parse
failures that `call_for_tool` already retries on.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from chive.pipeline.llm_client import LLMClient, LLMResponse, call_for_tool


M = TypeVar("M", bound=BaseModel)


def _required_keys(model_cls: type[BaseModel]) -> list[str]:
    """Return field names that have no default — these are what `call_for_tool`'s
    presence check should enforce inside its own retry loop."""
    return [name for name, field in model_cls.model_fields.items() if field.is_required()]


def validate_tool_input(model_cls: type[M], tool_input: dict[str, Any]) -> M | None:
    """Construct `model_cls` from `tool_input`; return None and print on failure.

    For use in code paths that already have a raw tool_input dict (e.g. the batch
    parse path or the agent loop's per-turn handling).
    """
    try:
        return model_cls.model_validate(tool_input)
    except ValidationError as e:
        print(f"[validate_tool_input] {model_cls.__name__} failed: {e}", flush=True)
        return None


async def call_for_validated_tool(
    client: LLMClient,
    *,
    system: str | list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tool: dict[str, Any],
    model_cls: type[M],
    max_tokens: int,
    thinking_budget: int | None,
    temperature: float | None = None,
    max_attempts: int = 3,
    per_attempt_timeout_seconds: float = 240.0,
) -> tuple[M, LLMResponse] | None:
    """Like call_for_tool, but additionally validates the returned tool input
    against `model_cls` and retries on pydantic ValidationError.

    Returns (validated_model, response) on success, or None if every attempt
    either failed call_for_tool's checks or failed pydantic validation. The
    caller skips and counts (same contract as call_for_tool).

    `per_attempt_timeout_seconds` caps the wall time per attempt — the underlying
    httpx client's default timeout is 600s, so a handful of stuck requests at
    small concurrency dominate the run's tail. 240s × 3 retries = ~12 min worst
    case per dropped item.
    """
    for attempt in range(max_attempts):
        # Inner max_attempts=1: this wrapper owns the retry budget so a single
        # validation failure doesn't get amplified 4x by the inner loop.
        try:
            outcome = await asyncio.wait_for(
                call_for_tool(
                    client,
                    system=system,
                    messages=messages,
                    tool=tool,
                    required_keys=_required_keys(model_cls),
                    max_tokens=max_tokens,
                    thinking_budget=thinking_budget,
                    temperature=temperature,
                    max_attempts=1,
                ),
                timeout=per_attempt_timeout_seconds,
            )
        except asyncio.TimeoutError:
            print(
                f"[call_for_validated_tool] {model_cls.__name__} attempt {attempt + 1}/{max_attempts} "
                f"timed out after {per_attempt_timeout_seconds:.0f}s",
                flush=True,
            )
            continue
        if outcome is None:
            continue
        tool_input, resp = outcome
        try:
            return model_cls.model_validate(tool_input), resp
        except ValidationError as e:
            print(
                f"[call_for_validated_tool] {model_cls.__name__} validation failed "
                f"(attempt {attempt + 1}/{max_attempts}): {e}",
                flush=True,
            )
    return None
