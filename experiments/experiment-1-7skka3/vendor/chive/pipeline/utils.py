"""
Shared utilities for dataset generation pipelines.

Provides:
- model_dir_name: Convert HF model ID to a directory-safe short name
- async_api_call: Anthropic API calls with retry + high-priority key fallback
- parse_json_response: Extract JSON from LLM responses (handles fences, trailing commas, etc.)
- extract_tool_input: Extract structured output from tool_use responses
- run_concurrent: Run async tasks with a concurrency limit and progress tracking
"""

import argparse
import asyncio
import functools
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import anthropic
from anthropic._exceptions import APIStatusError, APIConnectionError, InternalServerError, OverloadedError, RateLimitError
from pydantic import BaseModel

from chive import paths

# Force unbuffered output so we can monitor progress in background
print = functools.partial(print, flush=True)

SKIP_RATE_WARN_THRESHOLD = 0.001  # 0.1% — warn loudly above this
SKIP_RATE_DEFAULT_MAX = 1.0       # default: never fail on skip rate; just warn


def model_dir_name(model_id: str) -> str:
    """Convert a HuggingFace model ID to a directory-safe short name.

    Examples:
        "Qwen/Qwen3-8B" -> "Qwen3-8B"
        "meta-llama/Llama-3-8B" -> "Llama-3-8B"
        "Qwen3-8B" -> "Qwen3-8B"
    """
    return model_id.split("/")[-1]


def add_model_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add a required --model argument to an argparse parser."""
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen3-8B)",
    )
    return parser


def _repo_env_path() -> Path:
    return paths.REPO_ROOT / ".env"


def load_dotenv():
    """Load .env from repo root if API keys aren't already set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_path = _repo_env_path()
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        if key and value:
            # Strip inline comments (e.g. KEY=value # comment)
            if " #" in value:
                value = value[:value.index(" #")]
            os.environ.setdefault(key.strip(), value.strip())


def _parse_env_assignment(line: str) -> tuple[str, str]:
    stripped = line.strip()
    if stripped.startswith("#"):
        stripped = stripped[1:].strip()
    if stripped.startswith("export "):
        stripped = stripped[7:]
    key, _, value = stripped.partition("=")
    if " #" in value:
        value = value[:value.index(" #")]
    value = value.strip().strip('"').strip("'")
    return key.strip(), value


def load_high_priority_anthropic_api_key() -> str:
    """Load the Anthropic high-priority key from env or the labeled .env line."""
    env_key = os.environ.get("ANTHROPIC_API_KEY_HIGH_PRIORITY")
    if env_key:
        return env_key

    env_path = _repo_env_path()
    assert env_path.exists(), f"No .env found at {env_path}"
    for line in env_path.read_text().splitlines():
        if "ANTHROPIC_API_KEY" not in line or "high priority" not in line.lower():
            continue
        key, value = _parse_env_assignment(line)
        assert key == "ANTHROPIC_API_KEY", f"Unexpected high-priority key line: {line}"
        assert value, f"Empty high-priority Anthropic API key in {env_path}"
        return value

    raise AssertionError(
        "No high-priority Anthropic key found. Expected ANTHROPIC_API_KEY_HIGH_PRIORITY "
        "or a .env line labeled '# high priority'."
    )


def make_anthropic_client(*, use_high_priority_key: bool = False) -> anthropic.AsyncAnthropic:
    if use_high_priority_key:
        return anthropic.AsyncAnthropic(api_key=load_high_priority_anthropic_api_key())
    return anthropic.AsyncAnthropic()


def _load_anthropic_key_from_dotenv(var_name: str) -> str | None:
    """Value of an explicit ANTHROPIC_*_PRIORITY var. Checks os.environ first, then
    parses .env directly — load_dotenv() early-returns once ANTHROPIC_API_KEY is set,
    so the priority vars may never reach os.environ."""
    val = os.environ.get(var_name)
    if val:
        return val
    env_path = _repo_env_path()
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[7:]
        key, _, value = s.partition("=")
        if key.strip() == var_name and value.strip():
            v = value.strip()
            if " #" in v:
                v = v[:v.index(" #")]
            return v.strip().strip('"').strip("'")
    return None


def load_priority_anthropic_keys() -> tuple[str | None, str | None]:
    """(low, high) Anthropic API keys, or None for either if not configured."""
    return (
        _load_anthropic_key_from_dotenv("ANTHROPIC_API_KEY_LOW_PRIORITY"),
        _load_anthropic_key_from_dotenv("ANTHROPIC_API_KEY_HIGH_PRIORITY"),
    )


def make_anthropic_client_with_fallback() -> tuple[anthropic.AsyncAnthropic, anthropic.AsyncAnthropic | None]:
    """(default_client, high_priority_client) for use with async_api_call.

    The default client uses the LOW-priority key; pass the returned high_priority_client into
    async_api_call so it auto-upgrades to the HIGH-priority key on a 429/529. If either
    priority key is unconfigured, returns (plain default client, None) — i.e. exactly the
    previous behavior, so this is backward-compatible everywhere.
    """
    low, high = load_priority_anthropic_keys()
    if low and high:
        return anthropic.AsyncAnthropic(api_key=low), anthropic.AsyncAnthropic(api_key=high)
    return anthropic.AsyncAnthropic(), None


load_dotenv()

async def async_api_call(
    client: anthropic.AsyncAnthropic,
    *,
    max_retries: int = 10,
    high_priority_client: anthropic.AsyncAnthropic | None = None,
    stream: bool = False,
    **kwargs,
):
    """Make an Anthropic API call with retry and backoff.

    On transient errors (429/529/500/connection), retries with linear backoff.
    If a high-priority client is supplied, switch to it immediately on a 429 or 529.
    Does NOT manage concurrency — wrap calls in a semaphore at the task level.

    `stream=True` uses the streaming API and returns the accumulated final Message
    (identical shape to a non-streaming response). The SDK REFUSES non-streaming
    requests it estimates may exceed 10 minutes (large max_tokens) — stream those.
    """
    active_client = client
    using_high_priority = False
    for attempt in range(max_retries):
        try:
            if stream:
                async with active_client.messages.stream(**kwargs) as s:
                    return await s.get_final_message()
            return await active_client.messages.create(**kwargs)
        except (OverloadedError, RateLimitError, InternalServerError, APIConnectionError, APIStatusError) as e:
            error_code = getattr(e, 'status_code', type(e).__name__)
            if isinstance(e, APIStatusError) and error_code not in (429, 500, 529):
                raise
            if (
                error_code in (429, 529)
                and high_priority_client is not None
                and not using_high_priority
            ):
                active_client = high_priority_client
                using_high_priority = True
                print(f"    [RATE-LIMIT {error_code}] upgrading to HIGH-PRIORITY Anthropic key", flush=True)
                continue
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"    [RATE-LIMIT/transient {error_code}] retry {attempt+1}/{max_retries} → waiting {wait}s", flush=True)
                await asyncio.sleep(wait)
            else:
                raise


async def async_api_call_with_timeout(
    client: anthropic.AsyncAnthropic,
    *,
    timeout_seconds: float,
    timeout_retries: int,
    **kwargs,
):
    """Make an Anthropic API call with a hard wall-clock timeout.

    This wraps async_api_call, so transient Anthropic errors still get the normal
    retry/backoff behavior. The outer timeout catches hung requests that would
    otherwise block a whole throwaway experiment's gather forever.
    """
    assert timeout_seconds > 0, "timeout_seconds must be positive"
    assert timeout_retries > 0, "timeout_retries must be positive"
    for attempt in range(timeout_retries):
        try:
            return await asyncio.wait_for(
                async_api_call(client, **kwargs),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            if attempt < timeout_retries - 1:
                print(
                    f"    [timeout retry {attempt + 1}/{timeout_retries} "
                    f"after {timeout_seconds:.0f}s]"
                )
            else:
                raise


@dataclass(frozen=True)
class ApiRequest:
    custom_id: str
    params: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class JsonlApiRunResult:
    records: list[dict[str, Any]]
    completed_existing: int
    attempted: int
    succeeded: int
    skipped: int
    output_jsonl_path: str


def _load_completed_jsonl_records(output_jsonl_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    records: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if not output_jsonl_path.exists():
        return records, completed_ids

    with open(output_jsonl_path) as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            custom_id = record["custom_id"]
            assert custom_id not in completed_ids, (
                f"Duplicate custom_id {custom_id} in {output_jsonl_path}:{line_number}"
            )
            completed_ids.add(custom_id)
            records.append(record)
    return records, completed_ids


async def run_jsonl_api_requests(
    *,
    client: anthropic.AsyncAnthropic,
    high_priority_client: anthropic.AsyncAnthropic | None = None,
    requests: list[ApiRequest],
    output_jsonl_path: Path,
    parse_response: Callable[[ApiRequest, Any], dict[str, Any]],
    concurrency: int,
    label: str,
    max_skip_rate: float,
    request_timeout_seconds: float,
    timeout_retries: int,
) -> JsonlApiRunResult:
    """Run Anthropic API requests with JSONL checkpointing and resume.

    The caller owns experiment semantics: build exact API params and provide a
    parser that converts a raw Anthropic response into a JSON-serializable record.
    This helper owns the boring-but-important mechanics: resume by custom_id,
    concurrency, timeout protection, immediate JSONL writes, and loud skip-rate
    failure for malformed or timed-out responses.
    """
    assert requests, "requests must be non-empty"
    assert concurrency > 0, "concurrency must be positive"
    assert 0 <= max_skip_rate <= 1, "max_skip_rate must be in [0, 1]"

    request_ids = [request.custom_id for request in requests]
    assert len(request_ids) == len(set(request_ids)), "ApiRequest custom_id values must be unique"

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    records, completed_ids = _load_completed_jsonl_records(output_jsonl_path)
    if completed_ids:
        print(f"  Resuming {label}: {len(completed_ids)} already completed")

    remaining = [request for request in requests if request.custom_id not in completed_ids]
    if not remaining:
        print(f"  All {label} requests already completed")
        return JsonlApiRunResult(
            records=records,
            completed_existing=len(completed_ids),
            attempted=0,
            succeeded=0,
            skipped=0,
            output_jsonl_path=str(output_jsonl_path),
        )

    write_lock = asyncio.Lock()
    skipped = {"count": 0}
    succeeded = {"count": 0}
    jsonl_file = open(output_jsonl_path, "a")

    async def run_one(i: int, request: ApiRequest) -> None:
        try:
            response = await async_api_call_with_timeout(
                client,
                high_priority_client=high_priority_client,
                timeout_seconds=request_timeout_seconds,
                timeout_retries=timeout_retries,
                **request.params,
            )
            parsed = parse_response(request, response)
            assert "custom_id" not in parsed, "Parsed records must not define custom_id"
            record = {"custom_id": request.custom_id, **parsed}
            async with write_lock:
                records.append(record)
                jsonl_file.write(json.dumps(record) + "\n")
                jsonl_file.flush()
                succeeded["count"] += 1
        except (
            asyncio.TimeoutError,
            KeyError,
            AttributeError,
            TypeError,
            ValueError,
            AssertionError,
        ) as error:
            async with write_lock:
                skipped["count"] += 1
            print(f"  SKIPPING {request.custom_id}: {type(error).__name__}: {error}")

    try:
        await run_concurrent(
            run_one,
            remaining,
            concurrency=concurrency,
            label=label,
            progress_interval=10,
        )
    finally:
        jsonl_file.close()

    skip_rate = skipped["count"] / len(remaining)
    assert skip_rate <= max_skip_rate, (
        f"{label} skip rate {skip_rate:.2%} "
        f"({skipped['count']}/{len(remaining)}) exceeds {max_skip_rate:.2%}"
    )

    return JsonlApiRunResult(
        records=records,
        completed_existing=len(completed_ids),
        attempted=len(remaining),
        succeeded=succeeded["count"],
        skipped=skipped["count"],
        output_jsonl_path=str(output_jsonl_path),
    )


def run_anthropic_batch(
    *,
    requests: list[dict[str, Any]],
    output_path: Path,
    parse_succeeded: Callable[[str, Any], tuple[Any, dict[str, Any]]],
    label: str,
    batch_chunk_size: int,
    max_skip_rate: float = SKIP_RATE_DEFAULT_MAX,
    resubmit_wedged: bool = False,
) -> list[Any]:
    """Submit Anthropic batch requests, poll to completion, and parse results.

    This owns the boring batch machinery that every batch-mode stage shares, so
    the stage scripts don't each re-implement it: chunking to stay under the
    payload limit, resumable submission (batch ids tracked in a sidecar
    `<output>.batch_state.json`), polling, result retrieval, and loud skip-rate
    reporting. The caller owns experiment semantics through two things only:

    - `requests`: the batch request dicts (`{"custom_id", "params"}`, exactly the
      shape `client.messages.batches.create` wants).
    - `parse_succeeded(custom_id, message)`: turns one *succeeded* response into
      `(parsed_obj, jsonl_record)` — the stage's result object plus the dict to
      checkpoint. It must RAISE on a malformed response (e.g. via
      `extract_tool_input`, or a missing required key); the offending request is
      then counted toward the skip rate and dropped, never silently defaulted.

    Batches can take hours to process; the poll loop simply waits until every
    batch ends. Very rarely a batch wedges in "canceling" — an outage leaves a
    cancel that never completes, so it never reaches "ended" and the loop would
    wait forever. That is handled manually, not automatically: re-run with
    `resubmit_wedged=True`, which resubmits each prior-run batch stuck in
    "canceling" as a fresh batch (same chunk payload, so nothing is lost) and then
    polls as usual. Healthy in-progress batches are left alone, so the flag is safe
    to leave on across a requeue. A normal run never resubmits anything.

    Results are rewritten fresh to `<output>.jsonl` on every call: Anthropic keeps
    batch results retrievable server-side, so a re-run re-fetches them rather than
    resuming the jsonl. Returns the parsed objects in retrieval order.
    """
    assert requests, "requests must be non-empty"
    custom_ids = [r["custom_id"] for r in requests]
    assert len(custom_ids) == len(set(custom_ids)), "batch custom_id values must be unique"

    batch_state_path = output_path.with_suffix(".batch_state.json")
    jsonl_path = output_path.with_suffix(".jsonl")

    batch_api_key = os.environ.get("ANTHROPIC_API_KEY_BATCH_API")
    assert batch_api_key, "ANTHROPIC_API_KEY_BATCH_API not set in environment"
    client = anthropic.Anthropic(api_key=batch_api_key)

    chunks = [requests[i:i + batch_chunk_size] for i in range(0, len(requests), batch_chunk_size)]
    print(f"\n{label}: {len(requests)} requests in {len(chunks)} batch chunks "
          f"of up to {batch_chunk_size} (batch API)")

    def save_state() -> None:
        batch_state_path.write_text(json.dumps({
            "batch_ids": batch_ids,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "n_chunks": len(chunks),
            "chunk_size": batch_chunk_size,
            "total_requests": len(requests),
        }, indent=2))

    # Resume: batch ids already submitted in a prior run are kept; only chunks not
    # yet submitted get sent.
    batch_ids: list[str] = []
    if batch_state_path.exists():
        batch_ids = json.loads(batch_state_path.read_text()).get("batch_ids", [])
        if batch_ids:
            print(f"  Resuming {len(batch_ids)} batches")

    # Manual wedge recovery (rare): resubmit any resumed batch stuck in "canceling" (a cancel
    # that never completes) as a fresh batch. Only runs when explicitly asked, only over
    # prior-run batches, and only for "canceling" — healthy in-progress batches are left to
    # finish, so a normal run is untouched and the flag is safe across a requeue.
    if resubmit_wedged and batch_ids:
        for i, bid in enumerate(batch_ids):
            status = client.messages.batches.retrieve(bid).processing_status
            if status != "canceling":
                continue
            fresh = client.messages.batches.create(requests=chunks[i])
            print(f"  RESUBMIT chunk {i + 1}/{len(chunks)}: {bid} ({status}) -> {fresh.id}")
            batch_ids[i] = fresh.id
        save_state()

    while len(batch_ids) < len(chunks):
        chunk = chunks[len(batch_ids)]
        batch = client.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        save_state()
        print(f"  Submitted chunk {len(batch_ids)}/{len(chunks)}: "
              f"{batch.id} ({len(chunk)} requests)")

    # Poll until every batch ends. Batches can take hours; we just wait. If one wedges and
    # never ends, re-run with resubmit_wedged=True to replace it.
    while True:
        statuses: list[str] = []
        for i, bid in enumerate(batch_ids):
            status = client.messages.batches.retrieve(bid)
            counts = status.request_counts
            statuses.append(status.processing_status)
            print(
                f"  [{i + 1}/{len(batch_ids)}] {bid}: {status.processing_status} "
                f"(ok={counts.succeeded} err={counts.errored} "
                f"exp={counts.expired} pending={counts.processing})"
            )
        if all(s == "ended" for s in statuses):
            break
        time.sleep(60)

    results: list[Any] = []
    skipped_ids: list[str] = []
    with open(jsonl_path, "w") as jsonl_file:
        for bid in batch_ids:
            for batch_result in client.messages.batches.results(bid):
                custom_id = batch_result.custom_id
                if batch_result.result.type != "succeeded":
                    skipped_ids.append(custom_id)
                    print(f"  SKIPPING {custom_id}: batch result {batch_result.result.type}")
                    continue
                try:
                    parsed, record = parse_succeeded(custom_id, batch_result.result.message)
                except (KeyError, AttributeError, TypeError, ValueError) as error:
                    skipped_ids.append(custom_id)
                    print(f"  SKIPPING {custom_id}: malformed result "
                          f"({type(error).__name__}: {error})")
                    continue
                results.append(parsed)
                jsonl_file.write(json.dumps(record) + "\n")

    report_skip_rate(
        n_skipped=len(skipped_ids),
        n_total=len(requests),
        label=f"{label} (batch)",
        max_skip_rate=max_skip_rate,
        skipped_ids=skipped_ids,
    )
    return results


def parse_json_response(text: str):
    """Extract JSON from an LLM response, handling markdown fences and common issues."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Fix common LLM JSON issues
    cleaned = re.sub(r',\s*([}\]])', r'\1', text)  # trailing commas
    cleaned = re.sub(r':\s*\'([^\']*)\'\s*([,}\]])', r': "\1"\2', cleaned)  # single quotes to double
    # Fix unescaped backslashes (e.g. \boxed, \text) — escape any \ not followed by valid JSON escape chars
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Aggressive backslash escaping: replace ALL unescaped backslashes
    # (the regex above misses some edge cases with consecutive backslashes)
    aggressive = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu\\])', r'\\\\', cleaned)
    try:
        return json.loads(aggressive)
    except json.JSONDecodeError:
        pass
    # Scan for JSON array or object and use raw_decode to stop at the end
    decoder = json.JSONDecoder()
    for attempt_text in (aggressive, cleaned):
        for start_char in ("[", "{"):
            idx = attempt_text.find(start_char)
            if idx != -1:
                try:
                    obj, _ = decoder.raw_decode(attempt_text, idx)
                    return obj
                except json.JSONDecodeError:
                    continue
    raise json.JSONDecodeError("No JSON found", text, 0)


def extract_tool_input(resp) -> dict:
    """Extract tool use input from a response that used tool_choice.

    Use this when you force a tool call via tool_choice={"type": "tool", "name": "..."}.
    The response is guaranteed to be valid JSON (no parse_json_response needed).
    """
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError("No tool_use block in response")


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline $defs/$ref into a JSON Schema so the result has no internal references.

    Pydantic emits nested models as `$defs` + `$ref`; the previous hand-written tool
    schemas were fully inlined. Keep the inlined form so the tool schema sent to
    every backend stays uniform (Anthropic accepts $defs, but some
    OpenAI-compatible providers are flakier with $ref resolution).
    """
    defs = schema.pop("$defs", None)
    if not defs:
        return schema

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                assert ref.startswith("#/$defs/"), f"Unsupported $ref: {ref}"
                return walk(defs[ref.split("/", 2)[-1]])
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


def pydantic_tool(
    model_cls: type[BaseModel],
    *,
    name: str,
    description: str,
) -> dict[str, Any]:
    """Build an Anthropic-format tool dict from a pydantic model.

    The model class is the single source of truth for the tool input schema and
    its runtime validation: `model_json_schema()` produces what we send to the
    LLM; `model_validate()` checks what comes back. $defs are inlined for parity
    with the previous hand-written schemas.
    """
    schema = _inline_refs(model_cls.model_json_schema())
    return {"name": name, "description": description, "input_schema": schema}


def report_skip_rate(
    n_skipped: int,
    n_total: int,
    label: str,
    max_skip_rate: float = SKIP_RATE_DEFAULT_MAX,
    skipped_ids: list[str] | None = None,
) -> float:
    """Warn loudly if skip rate > 0.1%; raise AssertionError if >= max_skip_rate.

    Default `max_skip_rate=1.0` means "never fail" — the warning is the source of
    truth for exploratory runs. Canonical / production runs should pass a tight
    threshold (e.g. 0.001) to enforce.

    Returns the computed skip_rate so callers can persist it to metadata.
    """
    if n_total == 0:
        return 0.0
    skip_rate = n_skipped / n_total
    if skip_rate > SKIP_RATE_WARN_THRESHOLD:
        bar = "=" * 60
        lines = [
            "",
            bar,
            f"!! WARNING: {label} skip rate {skip_rate:.1%} "
            f"({n_skipped}/{n_total}) > {SKIP_RATE_WARN_THRESHOLD:.1%} threshold",
        ]
        if skipped_ids:
            preview = ", ".join(skipped_ids[:10])
            if len(skipped_ids) > 10:
                preview += f", ... (+{len(skipped_ids) - 10} more)"
            lines.append(f"!! Skipped: {preview}")
        if max_skip_rate >= 1.0:
            lines.append(
                f"!! (Not failing — default max-skip-rate=1.0. "
                f"Pass --max-skip-rate {SKIP_RATE_WARN_THRESHOLD:g} or lower to enforce.)"
            )
        lines += [bar, ""]
        print("\n".join(lines))
    if skip_rate >= max_skip_rate:
        raise AssertionError(
            f"{label} skip rate {skip_rate:.1%} ({n_skipped}/{n_total}) >= "
            f"max-skip-rate threshold {max_skip_rate:.1%}"
        )
    return skip_rate


def experiment_to_messages(entry: dict) -> list[dict]:
    """Return the exact conversation an `experiment_log` entry (from investigate.py) was run on.

    `messages` is the single source of truth: an ordered list (an optional leading
    `system` turn, then user/assistant turns, ending on a user turn). Read every
    experiment's conversation through this helper.

    Backwards compatibility (the reason this exists): a large amount of older run
    data was produced before the schema was unified, when single-turn experiments
    stored only `prompt` (the full user message) + an optional `system_prompt` and
    carried NO `messages`. Those legacy entries are reconstructed here, folding
    `system_prompt` into a leading system turn. New data always has `messages` and
    never takes the legacy branch. Fails loudly if an entry has neither field.
    """
    if "messages" in entry:
        return entry["messages"]
    assert "prompt" in entry, (
        f"experiment_log entry has neither 'messages' nor 'prompt': {sorted(entry)}"
    )
    messages: list[dict] = []
    if entry.get("system_prompt"):
        messages.append({"role": "system", "content": entry["system_prompt"]})
    messages.append({"role": "user", "content": entry["prompt"]})
    return messages


async def run_concurrent(tasks_fn, items, *, concurrency, label="", progress_interval=10):
    """Run async tasks with a concurrency limit and progress tracking.

    Args:
        tasks_fn: async function(semaphore, i, item) -> None. Must handle its own errors.
        items: list of items to process
        concurrency: max concurrent tasks
        label: name for progress prints
        progress_interval: print progress every N completions
    """
    semaphore = asyncio.Semaphore(concurrency)
    progress = {"done": 0}
    total = len(items)

    async def wrapped(i, item):
        async with semaphore:
            await tasks_fn(i, item)
        progress["done"] += 1
        d = progress["done"]
        if d % progress_interval == 0 or d == total or d <= 3:
            print(f"  {label} progress: {d}/{total}")

    await asyncio.gather(*[wrapped(i, item) for i, item in enumerate(items)])
