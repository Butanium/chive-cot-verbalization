"""Small async LLM client abstraction for model-understanding stages.

The stage scripts use one feature set: chat completion, optional tools,
multi-turn tool-result continuation, usage accounting, and stop reasons.
This module keeps provider-specific message/tool plumbing contained.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import anthropic
import httpx
import openai
from openai import AsyncOpenAI

from chive.pipeline.utils import async_api_call, load_dotenv, load_priority_anthropic_keys


class EmptyResponseError(Exception):
    """An OpenAI-compatible endpoint returned a 200 whose body has no usable choices.

    OpenRouter does this for provider-side failures (provider error, rate-limit passthrough,
    content moderation): the SDK parses the body into a ChatCompletion with choices=None, so
    naive choices[0] access TypeErrors. We raise this instead so callers can retry (call_for_tool
    / call_for_tool_samples) or skip+count (the agent loop) rather than crash a whole run."""


# --- Global per-endpoint concurrency throttle -------------------------------
# Every actual model request in the pipeline funnels through LLMClient.call(),
# so we throttle there with ONE semaphore per endpoint. This makes the
# concurrency number mean the same thing for every stage: "max simultaneous
# in-flight requests to this endpoint." Nested fan-out (e.g. a run_prompt that
# samples 10 completions, each its own .call()) is bounded automatically — each
# call just grabs one of the N slots — so there is no `outer x fan-out` blowup
# and no need to hand-divide concurrency per stage. Clients hitting the same
# endpoint (e.g. the agent and the counterfactual target both on one OpenRouter
# route) share the same semaphore. Off by default (set_max_concurrency(None));
# the pipeline / CLIs set it once at startup.
_MAX_CONCURRENCY: int | None = None
_ENDPOINT_SEMS: dict[str, asyncio.Semaphore] = {}


def set_max_concurrency(n: int | None) -> None:
    """Set the global per-endpoint in-flight cap (None = unthrottled)."""
    global _MAX_CONCURRENCY
    _MAX_CONCURRENCY = n
    _ENDPOINT_SEMS.clear()  # rebuild lazily at the new size


def _endpoint_sem(key: str) -> asyncio.Semaphore | None:
    """Lazily-created semaphore shared by all clients on `key` (an endpoint)."""
    if _MAX_CONCURRENCY is None:
        return None
    sem = _ENDPOINT_SEMS.get(key)
    if sem is None:
        sem = asyncio.Semaphore(_MAX_CONCURRENCY)
        _ENDPOINT_SEMS[key] = sem
    return sem


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "qwen/qwen3.5-397b-a17b"
# Tinker's native SDK is the canonical path for the tinker: backend (see
# TinkerLLMClient below). The OpenAI-compatible endpoint exists but doesn't
# propagate enable_thinking / thinking_budget to the chat template and exposes
# no reasoning_content channel, so it can't be used where thinking control
# matters — kept here as a constant only for the few places that hit it directly.
TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


@dataclass(frozen=True)
class QwenSampling:
    temperature: float
    top_p: float
    top_k: int
    min_p: float


# Qwen's recommended sampling, applied to REASONING calls (judges + investigation
# agent); an explicit temperature from the caller always wins. Behavior sampling
# (Stage 1, Stage 3) passes apply_qwen_profile=False: temperature 1.0 with the
# model's HF generation_config top_p/top_k, applied by the vLLM server.
QWEN_SAMPLING = {
    True: QwenSampling(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0),   # thinking
    False: QwenSampling(temperature=0.7, top_p=0.80, top_k=20, min_p=0.0),  # non-thinking
}


def qwen_sampling(model: str, *, thinking: bool) -> QwenSampling | None:
    """Recommended Qwen sampling for this model, or None if it is not a Qwen model.

    Matching on the model name string is consistent with the fp8 provider routing
    in make_llm_client(); it is the one place that knows "this is a Qwen."
    """
    if "qwen" in model.lower():
        return QWEN_SAMPLING[thinking]
    return None


@dataclass(frozen=True)
class LLMToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    text_blocks: list[str]
    thinking_blocks: list[str]
    tool_calls: list[LLMToolCall]
    usage: LLMUsage
    stop_reason: str
    assistant_message: dict[str, Any]
    # Per-choice finish reasons (one entry per returned choice). For n==1 this is
    # a single-element list; for n>1 (native multi-sample) it is parallel to
    # text_blocks so callers can track truncation per completion.
    finish_reasons: list[str] = field(default_factory=list)
    # Per-choice tool calls (one inner list per returned choice), parallel to
    # text_blocks / finish_reasons. `tool_calls` above is the flattened view
    # across all choices (fine for n==1); n>1 callers that need per-sample
    # alignment (which tool call came from which completion) read this instead.
    tool_calls_by_choice: list[list[LLMToolCall]] = field(default_factory=list)


@dataclass(frozen=True)
class ToolSample:
    """One valid sample from a multi-sample tool call: the parsed tool input plus
    that completion's response text and finish reason."""
    tool_input: dict[str, Any]
    text: str
    finish_reason: str


def reconstruct_completions(response: LLMResponse, *, thinking: bool) -> list[str]:
    """Per-choice completion strings from an OpenAI-compatible (vLLM) response.

    With thinking ON, inline each choice's reasoning as
    ``<think>\\n…\\n</think>\\n\\n{answer}`` so every downstream stage (screener,
    grader, investigator) reads the completion exactly as the model produced it —
    vLLM's reasoning_parser splits the trace into reasoning_content, and this puts
    it back. With thinking OFF, return just the answer text (current behavior).

    Requires thinking_blocks parallel to text_blocks (one entry per choice); the
    OpenAI-compatible client guarantees this, and strict zip fails loudly if they
    ever desync rather than silently misaligning traces to the wrong answers.
    """
    out: list[str] = []
    n_with_reasoning = 0
    if not thinking:
        # No-think: the reasoning trace is never used, so don't require thinking_blocks to be parallel
        # to text_blocks. The Tinker client returns empty thinking_blocks for no-think samples, which
        # would otherwise trip the strict zip below even though we discard reasoning entirely.
        return [t.strip() for t in response.text_blocks]
    for text, reasoning in zip(response.text_blocks, response.thinking_blocks, strict=True):
        text = text.strip()
        if thinking:
            reasoning = reasoning.strip()
            if reasoning:
                n_with_reasoning += 1
            out.append(f"<think>\n{reasoning}\n</think>\n\n{text}" if reasoning else text)
        else:
            out.append(text)
    # If thinking was requested but NOT ONE choice came back with a reasoning trace, the target is
    # misconfigured — the vLLM reasoning parser is missing/mismatched (need --reasoning-parser <model>
    # --reasoning-config '{}') or this model/parser doesn't support thinking_token_budget. Fail loudly
    # instead of silently returning no-think completions labeled as thinking. A single empty trace among
    # several is fine (the model occasionally answers without thinking) — only an ALL-empty batch is a misconfig.
    if thinking and out:
        assert n_with_reasoning > 0, (
            f"thinking was requested (thinking_budget set) but the target returned NO reasoning "
            f"trace across all {len(out)} completion(s) — reasoning_content empty throughout. The "
            f"vLLM target is almost certainly missing/mismatched on its reasoning parser (launch with "
            f"'--reasoning-parser <model> --reasoning-config '{{}}'') or doesn't support "
            f"thinking_token_budget. Refusing to silently return no-think completions labeled as thinking."
        )
    return out


class LLMClient(Protocol):
    model: str
    backend_name: str

    async def call(
        self,
        *,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        thinking_budget: int | None,
        temperature: float | None = None,
        n: int = 1,
        apply_qwen_profile: bool = True,
        effort: str | None = None,
    ) -> LLMResponse: ...

    def tool_result_message(self, tool_results: list[dict[str, Any]]) -> dict[str, Any]: ...


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block["text"]))
            elif hasattr(block, "type") and block.type == "text":
                parts.append(str(block.text))
        return "\n".join(parts)
    raise TypeError(f"Unsupported message content type: {type(content).__name__}")


def _system_to_text(system: str | list[dict[str, Any]]) -> str:
    if isinstance(system, str):
        return system
    return _content_text(system)


def _anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        })
    return openai_tools


def _openai_tools_to_anthropic(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    anthropic_tools = []
    for tool in tools:
        if "input_schema" in tool:
            anthropic_tools.append(tool)
            continue
        fn = tool["function"]
        anthropic_tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn["parameters"],
        })
    return anthropic_tools


class AnthropicLLMClient:
    backend_name = "anthropic"

    def __init__(self, model: str, *, priority_fallback: bool = True):
        self.model = model
        # Default to the LOW-priority key and auto-upgrade to HIGH on a 429/529
        # (handled in async_api_call). Falls back to the plain default client —
        # i.e. exactly today's behavior — if either priority key is unconfigured
        # or priority_fallback is disabled.
        low_key, high_key = load_priority_anthropic_keys() if priority_fallback else (None, None)
        if low_key and high_key:
            self._client = anthropic.AsyncAnthropic(api_key=low_key)
            self._high_priority_client: anthropic.AsyncAnthropic | None = (
                anthropic.AsyncAnthropic(api_key=high_key))
            print("    [AnthropicLLMClient] low-priority key default; auto-upgrade to "
                  "high-priority on 429/529", flush=True)
        else:
            self._client = anthropic.AsyncAnthropic()
            self._high_priority_client = None

    async def call(
        self,
        *,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        thinking_budget: int | None,
        temperature: float | None = None,
        n: int = 1,
        apply_qwen_profile: bool = True,  # accepted for interface parity; Anthropic has no Qwen profile
        effort: str | None = None,
    ) -> LLMResponse:
        assert n == 1, "Anthropic backend does not support n>1; fan out instead"
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        anthropic_tools = _openai_tools_to_anthropic(tools)
        if anthropic_tools is not None:
            # The Anthropic API 400s on tools=null ("Input should be a valid
            # array"); omit the key entirely when there are no tools.
            params["tools"] = anthropic_tools
        # Anthropic has no Qwen profile; None means "use the API default" (which
        # is 1.0, and is also what thinking mode requires).
        if temperature is not None and temperature != 1.0:
            params["temperature"] = temperature
        # Thinking depth: `effort` (adaptive — Opus 4.6+/Sonnet 4.6/Fable) XOR
        # `thinking_budget` (enabled — Sonnet 4.5/Haiku 4.5/older). Mutually exclusive;
        # passing both is a caller bug. (4.7/4.8 also 400 on enabled/budget_tokens.)
        assert not (effort is not None and thinking_budget is not None), (
            "Pass either effort (adaptive) or thinking_budget (enabled), not both"
        )
        if effort is not None:
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": effort}
        elif thinking_budget is not None:
            assert max_tokens > thinking_budget, (
                "Anthropic max_tokens must be greater than thinking_budget"
            )
            params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        # The Anthropic SDK refuses non-streaming requests it estimates may exceed 10 min; empirically
        # max_tokens>20000 trips it. Auto-stream those (transparent — same Message shape back) so large
        # outputs (e.g. per-experiment judges over many experiments) are reliable.
        needs_stream = max_tokens > 20000
        sem = _endpoint_sem(f"anthropic:{self.model}")
        if sem is None:
            response = await async_api_call(
                self._client, high_priority_client=self._high_priority_client, stream=needs_stream, **params)
        else:
            async with sem:
                response = await async_api_call(
                    self._client, high_priority_client=self._high_priority_client, stream=needs_stream, **params)
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0

        text_blocks: list[str] = []
        thinking_blocks: list[str] = []
        tool_calls: list[LLMToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_blocks.append(block.text)
            elif block.type == "thinking":
                thinking_blocks.append(block.thinking)
            elif block.type == "tool_use":
                tool_calls.append(LLMToolCall(
                    id=block.id,
                    name=block.name,
                    input=block.input,
                ))

        return LLMResponse(
            text_blocks=text_blocks,
            thinking_blocks=thinking_blocks,
            tool_calls=tool_calls,
            usage=LLMUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=cache_read,
                cache_create_tokens=cache_create,
            ),
            stop_reason=str(response.stop_reason),
            finish_reasons=[str(response.stop_reason)],
            tool_calls_by_choice=[tool_calls],
            assistant_message={"role": "assistant", "content": response.content},
        )

    def tool_result_message(self, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
        return {"role": "user", "content": tool_results}


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        *,
        backend_name: str,
        model: str,
        base_url: str,
        api_key: str,
        default_extra_body: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        self.backend_name = backend_name
        self.model = model
        # All clients on the same endpoint (base_url) share one throttle, so the
        # agent and the counterfactual target on one OpenRouter route compete for
        # the same in-flight budget.
        self._sem_key = base_url
        # OpenAI SDK's default httpx client caps at 1000 concurrent connections.
        # Override with a custom client to support >1000 in-flight requests, which
        # is needed when the pipeline-side `concurrency` is raised above 1000 to
        # saturate a large GPU pool. Keep this above the pool proxy request
        # timeout so the driver does not abandon requests before the proxy does,
        # which leaves orphaned vLLM work under high-batch long-context load.
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
            # Back off and retry on 429/5xx (the SDK honors Retry-After). Upstream
            # providers (e.g. OpenRouter -> Alibaba) rate-limit, so give generous
            # headroom rather than failing a whole run on a transient throttle.
            max_retries=6,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=16000, max_keepalive_connections=2000),
                timeout=5000.0,
            ),
        )
        self._default_extra_body = default_extra_body or {}
        # Observability: announce the Qwen sampling profile once per client so an
        # auto-applied default is never silent (see qwen_sampling / QWEN_SAMPLING).
        if qwen_sampling(model, thinking=True) is not None:
            think, nothink = QWEN_SAMPLING[True], QWEN_SAMPLING[False]
            print(
                f"[llm_client] {model}: applying Qwen sampling "
                f"(thinking: temp={think.temperature} top_p={think.top_p} "
                f"top_k={think.top_k} min_p={think.min_p}; non-thinking: "
                f"temp={nothink.temperature} top_p={nothink.top_p} "
                f"top_k={nothink.top_k} min_p={nothink.min_p}). "
                f"Explicit temperature overrides.",
                flush=True,
            )

    async def call(
        self,
        *,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        thinking_budget: int | None,
        temperature: float | None = None,
        n: int = 1,
        apply_qwen_profile: bool = True,
        effort: str | None = None,
    ) -> LLMResponse:
        if effort is not None and self.backend_name != "openrouter":
            raise NotImplementedError(
                "effort is only supported on the anthropic and openrouter backends; use "
                "thinking_budget for vllm/openai backends"
            )
        assert not (effort is not None and thinking_budget is not None), (
            "Pass either effort or thinking_budget, not both"
        )
        openai_messages = self._to_openai_messages(system, messages)
        extra_body = dict(self._default_extra_body)
        if self.backend_name == "openrouter":
            # OpenRouter's unified reasoning control. `effort` (low/medium/high) is the
            # portable knob across providers — OpenAI exposes effort buckets natively, Gemini
            # maps it to a thinking budget — so it's the right control for cross-model evals.
            # `max_tokens` is the budget form (Anthropic/Gemini). Always exclude=False: we want
            # to capture whatever reasoning the model produced (read into thinking_blocks, never
            # re-sent in history). With effort=none nothing is generated, so it returns nothing.
            if effort is not None:
                extra_body["reasoning"] = {"effort": effort, "exclude": False}
            elif thinking_budget is None:
                extra_body["reasoning"] = {"effort": "none", "exclude": False}
            else:
                extra_body["reasoning"] = {"max_tokens": thinking_budget, "exclude": False}
        elif self.backend_name == "vllm":
            if thinking_budget is None:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            else:
                # vLLM enforces this as a hard cap: once `thinking_budget` tokens
                # are emitted inside <think>...</think>, the reasoning end token is
                # forced (ThinkingTokenBudgetLogitsProcessor). Requires the server
                # to be launched with `--reasoning-config` (e.g. '{}'); without it
                # vLLM 400s any request that sets thinking_token_budget.
                extra_body["thinking_token_budget"] = thinking_budget

        # Apply Qwen's recommended sampling: top_p/top_k/min_p come from the profile, and
        # temperature falls back to it only when the caller didn't pass one (explicit wins).
        # top_k/min_p are not OpenAI-standard params, so they go through extra_body.
        # apply_qwen_profile=False sends NONE of the four, which does NOT mean "untruncated":
        # vLLM's --generation-config defaults to auto, so the server fills the omitted values
        # in from the model's generation_config.json (Qwen3: top_p 0.95, top_k 20).
        profile = (
            qwen_sampling(self.model, thinking=thinking_budget is not None)
            if apply_qwen_profile else None
        )
        if profile is not None:
            extra_body["top_k"] = profile.top_k
            extra_body["min_p"] = profile.min_p
            if temperature is None:
                temperature = profile.temperature

        create_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=openai_messages,
            tools=_anthropic_tools_to_openai(tools),
            tool_choice="auto" if tools else None,
            max_tokens=max_tokens,
            n=n,
            extra_body=extra_body or None,
        )
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if profile is not None:
            create_kwargs["top_p"] = profile.top_p
        async def _create_once():
            sem = _endpoint_sem(self._sem_key)
            if sem is None:
                return await self._client.chat.completions.create(**create_kwargs)
            async with sem:
                return await self._client.chat.completions.create(**create_kwargs)

        # OpenRouter sometimes returns a 200 whose body is an error (provider failure, rate-limit
        # passthrough, moderation) with choices=None — the SDK parses it but choices[0] then
        # TypeErrors. Treat empty choices as a transient provider error: retry a few times, then
        # raise EmptyResponseError (call_for_tool retries it further; the agent loop skips+counts).
        response = await _create_once()
        for attempt in range(3):
            if response.choices:
                break
            err = getattr(response, "error", None)
            print(f"[llm_client] {self.backend_name}:{self.model} empty choices "
                  f"(error={err}); retry {attempt + 1}/3", flush=True)
            await asyncio.sleep(2 ** attempt)
            response = await _create_once()
        if not response.choices:
            raise EmptyResponseError(
                f"{self.backend_name}:{self.model} returned no choices after retries "
                f"(error={getattr(response, 'error', None)})")

        text_blocks: list[str] = []
        thinking_blocks: list[str] = []
        tool_calls: list[LLMToolCall] = []
        tool_calls_by_choice: list[list[LLMToolCall]] = []
        finish_reasons: list[str] = []
        first_message = response.choices[0].message
        for choice in response.choices:
            finish_reasons.append(str(choice.finish_reason))
            message = choice.message
            # OpenRouter exposes reasoning as `reasoning`; vLLM as `reasoning_content`.
            reasoning = getattr(message, "reasoning", None) or getattr(
                message, "reasoning_content", None
            )
            # Parallel to text_blocks (one entry per choice, "" when a choice
            # emitted no reasoning) so per-choice callers can zip the two with
            # strict=True to reconstruct each completion's <think> trace. See
            # reconstruct_completions.
            thinking_blocks.append(reasoning or "")
            # One text block per choice, kept parallel to finish_reasons (callers
            # like Stage 1 zip the two with strict=True). Empty/None content -> ""
            # so an empty choice among an n>1 batch doesn't desync the lists; the
            # downstream skip-empty logic then drops it instead of crashing.
            text_blocks.append(message.content or "")
            choice_calls: list[LLMToolCall] = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    parsed = LLMToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=json.loads(tool_call.function.arguments),
                    )
                    choice_calls.append(parsed)
                    tool_calls.append(parsed)
            tool_calls_by_choice.append(choice_calls)

        usage = response.usage
        if usage is None:
            llm_usage = LLMUsage(input_tokens=0, output_tokens=0)
        else:
            details = usage.completion_tokens_details
            reasoning_tokens = 0
            if details is not None:
                reasoning_tokens = details.reasoning_tokens or 0
            llm_usage = LLMUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cache_read_tokens=0,
                cache_create_tokens=0,
                reasoning_tokens=reasoning_tokens,
            )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": first_message.content or "",
        }
        if first_message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in first_message.tool_calls
            ]

        return LLMResponse(
            text_blocks=text_blocks,
            thinking_blocks=thinking_blocks,
            tool_calls=tool_calls,
            usage=llm_usage,
            stop_reason=",".join(finish_reasons),
            finish_reasons=finish_reasons,
            tool_calls_by_choice=tool_calls_by_choice,
            assistant_message=assistant_message,
        )

    def _to_openai_messages(
        self,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        system_text = _system_to_text(system)
        if system_text:
            converted.append({"role": "system", "content": system_text})
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "tool":
                converted.append(message)
            elif role == "assistant":
                converted.append(self._assistant_to_openai(message))
            elif isinstance(content, list) and all(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            ):
                for block in content:
                    converted.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block["content"],
                    })
            else:
                converted.append({"role": role, "content": _content_text(content)})
        return converted

    def _assistant_to_openai(self, message: dict[str, Any]) -> dict[str, Any]:
        if "tool_calls" in message:
            return message
        content = message["content"]
        if isinstance(content, str):
            return {"role": "assistant", "content": content}

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block["text"]))
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    })
            elif hasattr(block, "type"):
                if block.type == "text":
                    text_parts.append(str(block.text))
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    })
        out: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
        if tool_calls:
            out["tool_calls"] = tool_calls
        return out

    def tool_result_message(self, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
        return {"role": "user", "content": tool_results}


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _split_qwen_think(text: str) -> tuple[str, str]:
    """Split a Qwen response with <think> tags into (thinking, final_text).

    Qwen's chat template with enable_thinking=True typically PRE-FILLS
    `<think>\\n` after `<|im_start|>assistant`. Tinker's native SDK returns only
    the *generated* tokens, not the pre-fill, so the decoded text usually starts
    directly with the thinking content and contains a `</think>` boundary — no
    opening tag. We anchor on `</think>` for that reason.

    - If `</think>` is present: everything before it is thinking, everything
      after is the final answer. A leading `<think>` (if any) is stripped.
    - Else if `<think>` is present (truncated mid-thought): everything after
      `<think>` is thinking.
    - Else: ("", text) — no thinking content at all.
    """
    close = text.find(_THINK_CLOSE)
    if close != -1:
        thinking = text[:close]
        if thinking.startswith(_THINK_OPEN):
            thinking = thinking[len(_THINK_OPEN):]
        post = text[close + len(_THINK_CLOSE):]
        return thinking.strip(), post.lstrip()
    open_ = text.find(_THINK_OPEN)
    if open_ != -1:
        return text[open_ + len(_THINK_OPEN):].strip(), ""
    return "", text


class TinkerLLMClient:
    """LLMClient over Tinker's native SDK.

    Required because Tinker's OpenAI-compatible endpoint does NOT propagate
    enable_thinking/thinking_budget to the chat template and does NOT expose a
    reasoning_content channel — so the model emits its scratchpad as inline prose
    where the strict response should go (verified by direct probing).

    The native SDK lets us tokenize with the HF chat template (`enable_thinking`
    handled correctly) and sample from a base model or a trained LoRA. We parse
    `<think>...</think>` out of the decoded text manually (Tinker has no separate
    reasoning channel).

    Spec: `tinker:<base_model>` (base) or
          `tinker:<base_model>@tinker://<...>/sampler_weights/<name>` (LoRA on top
          of that base). The `@` syntax mirrors the `vllm:MODEL@URL` convention.
    """
    backend_name = "tinker"

    def __init__(self, *, base_model: str, model_path: str | None = None):
        # Imports are local so the (heavy) tinker + transformers import is only
        # paid when something actually constructs this client.
        import tinker
        from transformers import AutoTokenizer

        self.base_model = base_model
        self.model_path = model_path
        # `model` (LLMClient protocol attr) carries the LoRA path when present,
        # for observability/logging.
        self.model = base_model if model_path is None else f"{base_model}@{model_path}"
        project_id = os.environ.get("TINKER_PROJECT_ID")
        owner = os.environ.get("TINKER_OWNER")
        assert project_id, "TINKER_PROJECT_ID env var is required for tinker: backend"
        assert owner, "TINKER_OWNER env var is required for tinker: backend"

        self._service_client = tinker.ServiceClient(
            project_id=project_id,
            user_metadata={"owner": owner},
        )
        self._sampling_client = self._service_client.create_sampling_client(
            base_model=base_model if model_path is None else None,
            model_path=model_path,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        # One semaphore per (base, lora) endpoint, matching the OAI client's
        # per-endpoint throttle convention.
        self._sem_key = f"tinker:{self.model}"

        # Tinker's SamplingParams accept temperature / max_tokens / top_p only —
        # no top_k / min_p. Announce this once if Qwen sampling would normally
        # add those, so the dropped fields aren't a silent surprise.
        if qwen_sampling(self.base_model, thinking=True) is not None:
            print(
                f"[llm_client] tinker:{self.model}: native SDK applies Qwen "
                f"temperature + top_p only (Tinker SamplingParams has no "
                f"top_k/min_p). Explicit temperature overrides.",
                flush=True,
            )

    async def call(
        self,
        *,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        thinking_budget: int | None,
        temperature: float | None = None,
        n: int = 1,
        apply_qwen_profile: bool = True,
        effort: str | None = None,
    ) -> LLMResponse:
        if effort is not None:
            raise NotImplementedError("effort is not supported on the tinker backend")
        import tinker

        assert tools is None, (
            "TinkerLLMClient does not support tool calls. Route tool-using flows "
            "through anthropic: / openrouter: / vllm: / openai: instead."
        )

        # Build the chat-template input from system + messages.
        chat_messages: list[dict[str, Any]] = []
        system_text = _system_to_text(system)
        if system_text:
            chat_messages.append({"role": "system", "content": system_text})
        for message in messages:
            chat_messages.append({
                "role": message["role"],
                "content": _content_text(message["content"]),
            })

        enable_thinking = thinking_budget is not None
        prompt_text = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompt_ids = self._tokenizer.encode(prompt_text, add_special_tokens=False)

        # Qwen profile: SamplingParams supports top_p (use it) but NOT top_k/min_p.
        profile = (
            qwen_sampling(self.base_model, thinking=enable_thinking)
            if apply_qwen_profile else None
        )
        eff_temperature = (
            temperature if temperature is not None
            else (profile.temperature if profile is not None else 1.0)
        )
        sampling_kwargs: dict[str, Any] = {
            "temperature": eff_temperature,
            "max_tokens": max_tokens,
        }
        if profile is not None:
            sampling_kwargs["top_p"] = profile.top_p
        sampling_params = tinker.SamplingParams(**sampling_kwargs)

        sem = _endpoint_sem(self._sem_key)
        if sem is None:
            response = await self._sampling_client.sample_async(
                prompt=tinker.ModelInput.from_ints(prompt_ids),
                num_samples=n,
                sampling_params=sampling_params,
            )
        else:
            async with sem:
                response = await self._sampling_client.sample_async(
                    prompt=tinker.ModelInput.from_ints(prompt_ids),
                    num_samples=n,
                    sampling_params=sampling_params,
                )

        text_blocks: list[str] = []
        thinking_blocks: list[str] = []
        finish_reasons: list[str] = []
        total_output_tokens = 0
        total_reasoning_tokens = 0
        for sequence in response.sequences:
            tokens = list(sequence.tokens)
            total_output_tokens += len(tokens)
            decoded = self._tokenizer.decode(tokens, skip_special_tokens=True)
            think_text, post_text = _split_qwen_think(decoded)
            if think_text:
                thinking_blocks.append(think_text)
                total_reasoning_tokens += len(
                    self._tokenizer.encode(think_text, add_special_tokens=False)
                )
            text_blocks.append(post_text)
            finish_reasons.append(str(sequence.stop_reason))

        usage = LLMUsage(
            input_tokens=len(prompt_ids),
            output_tokens=total_output_tokens,
            reasoning_tokens=total_reasoning_tokens,
        )
        # The Tinker SDK doesn't return a structured assistant message we can echo
        # back into a chat history (no tool_calls etc.); stash the first sample's
        # final-answer text. Multi-turn assistant continuation isn't supported on
        # the tinker: backend (LLMClient protocol parity only; callers using the
        # eval are single-turn).
        first_text = text_blocks[0] if text_blocks else ""
        return LLMResponse(
            text_blocks=text_blocks,
            thinking_blocks=thinking_blocks,
            tool_calls=[],
            usage=usage,
            stop_reason=",".join(finish_reasons),
            finish_reasons=finish_reasons,
            tool_calls_by_choice=[[] for _ in response.sequences],
            assistant_message={"role": "assistant", "content": first_text},
        )

    def tool_result_message(self, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError("TinkerLLMClient does not support tools")


def _normalize_vllm_base_url(raw_url: str) -> str:
    if raw_url.endswith("/v1"):
        return raw_url
    return raw_url.rstrip("/") + "/v1"


def anthropic_message_provenance(message: Any) -> tuple[str, str, dict]:
    """Extract (response_text, stop_reason, usage_dict) from an Anthropic Message.

    Batch-API results hand back the same `Message` shape that messages.create
    returns, but the batch code paths only read the tool call out of it. This
    rebuilds the exact provenance triple that AnthropicLLMClient.call records for
    async results — joined text blocks, stop reason, and an LLMUsage-shaped usage
    dict — so a stage's records carry one schema regardless of async vs batch.
    """
    text_parts = [block.text for block in message.content if block.type == "text"]
    usage = message.usage
    usage_dict = asdict(LLMUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_create_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    ))
    return "\n\n".join(text_parts), str(message.stop_reason), usage_dict


async def call_for_tool(
    client: LLMClient,
    *,
    system: str | list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tool: dict[str, Any],
    required_keys: list[str],
    max_tokens: int,
    thinking_budget: int | None = None,
    temperature: float | None = None,
    effort: str | None = None,
    max_attempts: int = 4,
    validate: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[dict[str, Any], LLMResponse] | None:
    """Call the model expecting a single tool call whose input has required_keys.

    Open models over OpenRouter emit flaky tool calls (no call at all, prose
    instead, unparseable JSON args, or missing fields) a few percent of the
    time. Each failure is roughly independent, so retrying a handful of times
    drives the effective failure rate to ~0. Returns (tool_input, response) on
    success, or None if every attempt was malformed (caller skips + counts).

    `validate`, when given, must accept the tool input and return True for it
    to count as well-formed; a False verdict is retried like a missing key.
    Use it to enforce value TYPES and enums — the API does not enforce the
    tool's input_schema, and models occasionally emit e.g. a stringified list
    ("[a, b]") for an array field (this shipped 416 malformed `categories`
    rows in the released mechanism_concreteness.jsonl files).
    """
    for _ in range(max_attempts):
        try:
            resp = await client.call(
                system=system,
                messages=messages,
                tools=[tool],
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
                temperature=temperature,
                effort=effort,
            )
        except json.JSONDecodeError:
            continue  # provider returned unparseable tool-call args
        except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError, EmptyResponseError) as e:
            # Transient client-side timeout, network blip, or proxy 503 (the
            # latter happens during preempt-induced backend churn when the
            # proxy briefly has no healthy upstreams). vLLM still healthy;
            # one stuck request shouldn't crash the whole asyncio.gather. Retry
            # within max_attempts; if all fail, fall through to the None return.
            print(f"[call_for_tool] {type(e).__name__}, retrying ({_+1}/{max_attempts})", flush=True)
            continue
        if not resp.tool_calls:
            continue  # model answered in prose instead of calling the tool
        tool_input = resp.tool_calls[0].input
        if not all(k in tool_input for k in required_keys):
            continue
        if validate is not None and not validate(tool_input):
            continue  # schema-shaped but malformed values (e.g. stringified list)
        return tool_input, resp
    return None


def _sum_usages(usages: list[LLMUsage]) -> LLMUsage:
    return LLMUsage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cache_read_tokens=sum(u.cache_read_tokens for u in usages),
        cache_create_tokens=sum(u.cache_create_tokens for u in usages),
        reasoning_tokens=sum(u.reasoning_tokens for u in usages),
    )


async def call_for_tool_samples(
    client: LLMClient,
    *,
    system: str | list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tool: dict[str, Any],
    required_keys: list[str],
    max_tokens: int,
    thinking_budget: int | None,
    n: int,
    temperature: float | None = None,
    max_attempts: int = 4,
) -> tuple[list[ToolSample], LLMUsage] | None:
    """Get `n` independent tool-call samples for the SAME prompt, cheaply.

    On vLLM / OpenAI-compatible backends this issues ONE request with `n` samples
    so the shared prompt prefill is computed once and forked in-batch — far
    cheaper than `n` separate requests that each re-send the prompt and race for
    the prefix cache. Anthropic has no native `n>1`, so we fan out `n` independent
    `call_for_tool` calls there (its own prompt caching keeps the repeated prefix
    cheap).

    Returns `(samples, usage)` where `samples` holds one `ToolSample` per
    completion that produced a valid tool call with all `required_keys`, and
    `usage` is the total token usage across the request(s). Malformed samples are
    dropped (matching `call_for_tool`); the native request is retried (up to
    `max_attempts`) only when NO sample is valid, so a partially-valid batch keeps
    its good subset and may have fewer than `n` members. Returns None if every
    attempt yielded zero valid samples.
    """
    assert n >= 1
    if n == 1 or client.backend_name == "anthropic":
        outcomes = await asyncio.gather(*[
            call_for_tool(
                client,
                system=system,
                messages=messages,
                tool=tool,
                required_keys=required_keys,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
                temperature=temperature,
                max_attempts=max_attempts,
            )
            for _ in range(n)
        ])
        samples: list[ToolSample] = []
        usages: list[LLMUsage] = []
        for outcome in outcomes:
            if outcome is None:
                continue
            tool_input, resp = outcome
            samples.append(ToolSample(
                tool_input=tool_input,
                text="\n\n".join(resp.text_blocks),
                finish_reason=resp.stop_reason,
            ))
            usages.append(resp.usage)
        return (samples, _sum_usages(usages)) if samples else None

    # vLLM / OpenAI-compatible: one request, n samples, shared prefill.
    for attempt in range(max_attempts):
        try:
            resp = await client.call(
                system=system,
                messages=messages,
                tools=[tool],
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
                temperature=temperature,
                n=n,
            )
        except json.JSONDecodeError:
            continue  # a sample returned unparseable tool-call args
        except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError, EmptyResponseError) as e:
            # InternalServerError covers the proxy's transient 503 ("no healthy
            # upstreams" during preempt-induced backend churn) — the SDK already
            # retried 6× internally, give it another `max_attempts` here.
            print(f"[call_for_tool_samples] {type(e).__name__}, retrying ({attempt+1}/{max_attempts})", flush=True)
            continue
        samples = []
        for choice_calls, text, finish in zip(
            resp.tool_calls_by_choice, resp.text_blocks, resp.finish_reasons, strict=True
        ):
            if not choice_calls:
                continue  # this sample answered in prose instead of calling the tool
            tool_input = choice_calls[0].input
            if all(k in tool_input for k in required_keys):
                samples.append(ToolSample(tool_input=tool_input, text=text, finish_reason=finish))
        if samples:
            return samples, resp.usage
    return None


def build_spec(backend: str, model: str) -> str:
    """Build an llm_client spec from a backend routing string and a model name.

    The backend carries routing only (never a model name): "openrouter",
    "tinker", "vllm@http://node:8000", or "openai@https://host/v1". The model is
    supplied separately so it is never specified twice and cannot drift from the
    caller's single source of truth.
    """
    route, sep, url = backend.partition("@")
    assert route, f"Invalid backend routing: {backend!r}"
    if sep:
        return f"{route}:{model}@{url}"
    return f"{route}:{model}"


def make_llm_client(spec: str) -> LLMClient:
    """Create an LLM client from a backend spec.

    Supported specs:
    - anthropic:claude-opus-4-6
    - openrouter:qwen/qwen3.5-397b-a17b
    - tinker:Qwen/Qwen3.5-397B-A17B
    - vllm:Qwen/Qwen3-32B-FP8@http://node:8000
    - openai:MODEL@https://example/v1
    """
    load_dotenv()
    backend, _, rest = spec.partition(":")
    assert backend and rest, f"Invalid LLM spec: {spec}"

    if backend == "anthropic":
        return AnthropicLLMClient(rest)

    if backend == "openrouter":
        model = rest or OPENROUTER_DEFAULT_MODEL
        # The fp8 provider routing below is Qwen-specific (spread load across the
        # fp8 serving stacks, match the planned vLLM-FP8 run). It MUST NOT be
        # applied to other models: e.g. anthropic/* has no fp8 endpoints, so the
        # filter 404s with "No endpoints found ... with quantization: fp8". So
        # only attach it for Qwen; every other model routes with OpenRouter's
        # defaults. This lets the same backend serve Qwen + Claude (judge sweeps).
        extra_body: dict[str, Any] = {}
        if "qwen" in model.lower():
            # Spread load across the fp8 providers for throughput. Alibaba
            # (first-party) is hard-rate-limited on this account (~60% 429 at
            # concurrency 100), whereas fp8-multi sees 0 errors at concurrency
            # 200, load-balanced across ~5 providers. "Morph" is excluded: it
            # returned null content/usage under load. NOTE (experiment
            # semantics): this generates across several third-party serving
            # stacks rather than one model deployment -- fine for iteration;
            # use the vLLM backend for canonical training-data runs.
            extra_body["provider"] = {
                "quantizations": ["fp8"],
                "ignore": ["Morph"],
                "allow_fallbacks": True,
                "require_parameters": True,
            }
        return OpenAICompatibleLLMClient(
            backend_name="openrouter",
            model=model,
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_extra_body=extra_body,
            default_headers={
                "HTTP-Referer": "https://activation-oracles.local",
                "X-Title": "activation-oracles-model-understanding",
            },
        )

    if backend == "tinker":
        # Native SDK (see TinkerLLMClient docstring for why not OAI).
        # Spec: tinker:<base_model>            -> base
        #       tinker:<base_model>@<lora_path>  -> LoRA on top of base
        base_model, sep, model_path = rest.partition("@")
        return TinkerLLMClient(
            base_model=base_model,
            model_path=model_path if sep else None,
        )

    if backend in ("vllm", "openai"):
        model, sep, base_url = rest.partition("@")
        assert sep, f"{backend} spec must be MODEL@BASE_URL: {spec}"
        return OpenAICompatibleLLMClient(
            backend_name=backend,
            model=model,
            base_url=_normalize_vllm_base_url(base_url),
            api_key=os.environ.get("OPENAI_API_KEY", "unused"),
        )

    raise ValueError(f"Unknown LLM backend: {backend}")
