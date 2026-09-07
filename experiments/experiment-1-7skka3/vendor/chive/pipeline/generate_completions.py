"""
Generate completions from a target model on chat prompts (Stage 1).

Reads a prompts JSONL slice (messages only — produced by run_stage0 from the
canonical pool, see build_prompt_pool.py) and generates multiple temperature-1
completions per prompt with thinking disabled, via a chat-completions backend
selected by an llm_client spec (--llm): a vLLM HTTP server, OpenRouter, etc.

Backend-aware fan-out:
  - vLLM server: one request with n=n_completions (shared prefill).
  - everything else (e.g. OpenRouter, which returns a single choice for the Qwen
    route even when n>1): independent n=1 calls fanned out per completion.

Usage:
    .venv/bin/python chive/pipeline/generate_completions.py \
        --llm vllm:Qwen/Qwen3-32B-FP8@http://node:8000 \
        --input run_dir/prompts.jsonl --concurrency 100 \
        --output run_dir/completions.jsonl
"""

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import openai

from chive import paths
from chive.pipeline.config import DEFAULTS
from chive.pipeline.llm_client import (
    LLMClient,
    make_llm_client,
    reconstruct_completions,
    set_max_concurrency,
)
from chive.pipeline.utils import model_dir_name


async def _call_with_retry(client, **kwargs):
    """Wrap client.call with retry on transient proxy 503s / timeouts.

    The proxy returns 503 ("no healthy upstreams") briefly during preempt
    waves when all backends are momentarily marked unhealthy. The OpenAI SDK
    already retries 6× on 5xx internally; this adds an outer 4× retry so a
    one-time storm of 503s doesn't kill the whole stage's asyncio.gather.
    """
    for attempt in range(4):
        try:
            return await client.call(**kwargs)
        except (openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError) as e:
            if attempt == 3:
                raise
            print(f"[generate_completions] {type(e).__name__}, retrying ({attempt+1}/4)", flush=True)
            await asyncio.sleep(2 ** attempt)

# FIXED generation spec for completions. The completion COUNT is the pipeline
# config's "n_completions" (one knob shared by stage 1 and stage 3's run_prompt
# so the two cannot drift; N_COMPLETIONS below is only the standalone-CLI
# default). Length and temperature stay intentionally constant, NOT config
# knobs, so a run can't silently change them — they determine the per-prompt
# behavior distribution every downstream stage reads.
#
# NOTE on MAX_RESPONSE_TOKENS=500: at this cap a large fraction of completions
# (~58% on Qwen3-27B-class models) hit the length limit and are truncated
# mid-sentence. This is INTENTIONAL and fine, and it is a deliberate exception
# to the general "truncation is a bug" rule in AGENTS.md. Stage-1 completions are
# *behavior samples to be explained* by the investigation pipeline, not answers
# to be graded — a truncated sample is still a faithful sample of how the model
# behaves on that prompt. Generated with temperature=1.0 and thinking DISABLED.
# Do not raise this cap to "fix" the truncation rate; it would change the
# behavior distribution every downstream stage reads.
N_COMPLETIONS = 10
MAX_RESPONSE_TOKENS = 500
# Sampling temperature for the investigated model's behavior completions. Shared
# by stage 1 AND stage 3's run_prompt so the two CANNOT drift — a counterfactual
# completion must be drawn from the same distribution as the original behavior,
# or the comparison is meaningless. (top_p/top_k are the model's HF
# generation_config defaults, applied by the vLLM server — see llm_client.)
# Like N_COMPLETIONS/MAX_RESPONSE_TOKENS, this is a constant, not
# a config knob.
COMPLETION_TEMPERATURE = 1.0

# vLLM HTTP servers support native n>1 (shared prefill); the Tinker SDK takes
# num_samples=n on a single sample_async call (one prefill server-side). Other
# backends get the n=1 fan-out. Keyed on llm_client backend_name.
NATIVE_N_BACKENDS = {"vllm", "tinker"}


@dataclass
class PromptCompletions:
    id: str
    user_message: str
    completions: list[str]
    finish_reasons: list[str]
    prompt_tokens: int
    response_token_counts: list[int]
    messages: list[dict] | None = None
    n_turns: int = 1
    total_chars: int = 0
    conversation_hash: str = ""
    generation_prompts: list[dict] | None = None
    generation_usage: list[dict] | None = None


async def generate_completions_api(
    client: LLMClient,
    candidates: list[dict],
    n_completions: int,
    max_tokens: int,
    concurrency: int,
    jsonl_path: Path | None = None,
    thinking_budget: int | None = None,
) -> list[PromptCompletions]:
    """Generate completions through a chat-completions backend.

    If `jsonl_path` is given, each prompt's record (asdict(PromptCompletions)) is
    appended + flushed to it the moment that prompt finishes, and any records
    already present there are loaded and skipped on startup. This makes a large
    run resumable: re-invoking with the same jsonl_path picks up where it left
    off instead of regenerating everything. The .jsonl is the stage's output
    file; the returned list is assembled from prior + new records.
    """
    native_n = client.backend_name in NATIVE_N_BACKENDS
    semaphore = asyncio.Semaphore(concurrency)
    results_by_id: dict[str, PromptCompletions] = {}
    skipped = {"empty": 0}

    # Resume: load prompts already completed in a prior run, then process only
    # the rest. JSONL split on "\n" only (same Unicode-line-separator caveat as
    # load_prompts) — json.dumps never emits a bare "\n" inside a string.
    if jsonl_path is not None and jsonl_path.exists():
        for line in jsonl_path.read_text().split("\n"):
            if line.strip():
                rec = json.loads(line)
                results_by_id[rec["id"]] = PromptCompletions(**rec)
        print(f"  Stage 1 resume: {len(results_by_id)} prompts already in {jsonl_path.name}")
    remaining = [c for c in candidates if c["id"] not in results_by_id]
    progress = {"done": len(results_by_id)}

    write_lock = asyncio.Lock()
    jsonl_file = open(jsonl_path, "a") if jsonl_path is not None else None

    async def generate_one(candidate: dict) -> None:
        assert candidate["id"], "Every prompt must have a stable prompt id"
        messages = candidate.get("messages")
        if messages is None:
            messages = [{"role": "user", "content": candidate["user_message"]}]
        assert messages[-1]["role"] == "user", (
            f"Prompt {candidate['id']} must end with a user message"
        )
        try:
            await _generate_one_inner(candidate, messages)
        except (openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError) as e:
            # Persistent proxy outage on this prompt even after _call_with_retry's
            # 4 attempts. Don't kill the whole stage — skip this prompt, count it.
            skipped["empty"] += n_completions
            print(f"  SKIPPING {candidate['id']}: {type(e).__name__} after retries", flush=True)
            return

    async def _generate_one_inner(candidate: dict, messages: list[dict]) -> None:
        completions: list[str] = []
        finish_reasons: list[str] = []
        usage: list[dict] = []
        response_token_counts: list[int] = []
        prompt_tokens = 0

        if native_n:
            # One request, n choices (vLLM shares the prefill across them).
            async with semaphore:
                response = await _call_with_retry(
                    client,
                    system="", messages=messages, tools=None,
                    max_tokens=max_tokens, thinking_budget=thinking_budget,
                    temperature=COMPLETION_TEMPERATURE, n=n_completions,
                    apply_qwen_profile=False,  # behavior sampling: temp 1.0, server-default top_p/top_k
                )
            prompt_tokens = response.usage.input_tokens
            usage = [asdict(response.usage)]  # aggregate across the n choices
            # With thinking on, the reasoning trace is inlined as <think>…</think>
            # so every downstream stage reads the completion as the model produced it.
            reconstructed = reconstruct_completions(response, thinking=thinking_budget is not None)
            for text, fr in zip(reconstructed, response.finish_reasons, strict=True):
                if not text.strip():
                    skipped["empty"] += 1
                    continue
                completions.append(text)
                finish_reasons.append(fr)
                # Native-n usage is aggregate, so use a word-count proxy per choice.
                response_token_counts.append(len(text.split()))
        else:
            # Fan out independent n=1 calls; capture real per-call usage.
            async def one_completion() -> None:
                async with semaphore:
                    response = await _call_with_retry(
                        client,
                        system="", messages=messages, tools=None,
                        max_tokens=max_tokens, thinking_budget=thinking_budget,
                        temperature=COMPLETION_TEMPERATURE,
                        apply_qwen_profile=False,  # behavior sampling: temp 1.0, server-default top_p/top_k
                    )
                parts = reconstruct_completions(response, thinking=thinking_budget is not None)
                text = "\n\n".join(parts).strip()
                if not text:
                    # Flaky empty 200-response; drop it (skip rate checked below).
                    skipped["empty"] += 1
                    return
                completions.append(text)
                finish_reasons.append(response.stop_reason)
                usage.append(asdict(response.usage))
                response_token_counts.append(response.usage.output_tokens)

            await asyncio.gather(*[one_completion() for _ in range(n_completions)])
            if usage:
                prompt_tokens = usage[0]["input_tokens"]

        if not completions:
            # Every completion for this prompt was empty; drop the prompt.
            progress["done"] += 1
            return
        record = PromptCompletions(
            id=candidate["id"],
            user_message=candidate["user_message"],
            completions=completions,
            finish_reasons=finish_reasons,
            prompt_tokens=prompt_tokens,
            response_token_counts=response_token_counts,
            messages=candidate.get("messages"),
            n_turns=candidate.get("n_turns", 1),
            total_chars=candidate.get("total_chars", len(candidate["user_message"])),
            conversation_hash=candidate.get("conversation_hash", ""),
            generation_prompts=messages,
            generation_usage=usage,
        )
        results_by_id[candidate["id"]] = record
        if jsonl_file is not None:
            async with write_lock:
                jsonl_file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                jsonl_file.flush()
        progress["done"] += 1
        if progress["done"] <= 3 or progress["done"] % 10 == 0 or progress["done"] == len(candidates):
            print(f"  Stage 1 progress: {progress['done']}/{len(candidates)}")

    try:
        await asyncio.gather(*[generate_one(candidate) for candidate in remaining])
    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    # Skip rate is over prompts actually attempted this run (not resumed ones).
    total_attempted = len(remaining) * n_completions
    skip_rate = skipped["empty"] / total_attempted if total_attempted else 0.0
    print(f"  Empty/flaky completions skipped: {skipped['empty']}/{total_attempted} "
          f"({skip_rate:.2%}); {len(results_by_id)}/{len(candidates)} prompts kept")
    assert skip_rate <= 0.02, (
        f"Empty completion rate {skip_rate:.2%} exceeds 2% threshold -- "
        f"likely a systematic backend problem, not transient flakiness"
    )
    return [results_by_id[c["id"]] for c in candidates if c["id"] in results_by_id]


def load_prompts(input_path: str, n_prompts: int | None, offset: int) -> list[dict]:
    """Load a prompts JSONL slice (one prompt record per line)."""
    # Split ONLY on "\n". str.splitlines() also breaks on other Unicode line
    # separators (VT, FF, FS/GS/RS, NEL, U+2028, U+2029) that appear literally
    # inside some prompt text (WildChat has a few hundred), which would split a
    # JSON record across two "lines". json.dumps never emits a bare "\n" inside
    # a string, so newline-split is the correct delimiter.
    lines = Path(input_path).read_text().split("\n")
    candidates = [json.loads(line) for line in lines if line.strip()]
    if offset > 0:
        candidates = candidates[offset:]
    if n_prompts is not None:
        candidates = candidates[:n_prompts]
    return candidates


def generate_completions(
    llm_spec: str,
    input_path: str,
    n_prompts: int | None = None,
    offset: int = 0,
    output_path: str | None = None,
    *,
    concurrency: int,
) -> str:
    set_max_concurrency(concurrency)
    client = make_llm_client(llm_spec)
    model_name = client.model
    candidates = load_prompts(input_path, n_prompts, offset)
    print(f"Loaded {len(candidates)} prompts from {input_path}")
    print(f"Generating via {llm_spec}: {len(candidates)} prompts x {N_COMPLETIONS} completions "
          f"x {MAX_RESPONSE_TOKENS} tok (native_n={client.backend_name in NATIVE_N_BACKENDS})")

    # Resolve the output .jsonl path up front so the incremental/resume writes
    # are stable (known before the run, independent of how many prompts survive).
    if output_path:
        out_file = Path(output_path)
        assert out_file.suffix == ".jsonl", f"--output must be a .jsonl path, got {out_file}"
    else:
        output_dir = paths.DATA_ROOT / "scratch" / model_dir_name(model_name)
        out_file = output_dir / f"completions_{len(candidates)}p_{N_COMPLETIONS}c.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    prompt_results = asyncio.run(generate_completions_api(
        client=client,
        candidates=candidates,
        n_completions=N_COMPLETIONS,
        max_tokens=MAX_RESPONSE_TOKENS,
        concurrency=concurrency,
        jsonl_path=out_file,
    ))

    total_completions = sum(len(p.completions) for p in prompt_results)
    n_length = sum(1 for p in prompt_results for fr in p.finish_reasons if fr == "length")
    trunc_rate = n_length / total_completions if total_completions else 0.0
    print(f"\nSaved {len(prompt_results)} prompts x {N_COMPLETIONS} completions to {out_file}")

    all_response_lens = [t for p in prompt_results for t in p.response_token_counts]
    print(f"\n--- Summary ---")
    print(f"Total prompts: {len(prompt_results)}")
    print(f"Total completions: {total_completions}")
    print(f"Truncated (finish_reason=length): {n_length}/{total_completions} ({trunc_rate:.1%})")
    if all_response_lens:
        print(f"Response tokens: min={min(all_response_lens)}, max={max(all_response_lens)}, "
              f"mean={sum(all_response_lens)/len(all_response_lens):.0f}")
    return str(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate completions (Stage 1)")
    parser.add_argument(
        "--llm", type=str, required=True,
        help=(
            "Backend spec (make_llm_client form). Examples: "
            "vllm:Qwen/Qwen3-32B-FP8@http://node:8000, "
            "openrouter:Qwen/Qwen3.5-397B-A17B."
        ),
    )
    parser.add_argument("--input", type=str, required=True, help="Prompts JSONL slice (from run_stage0)")
    parser.add_argument("--n-prompts", type=int, default=None, help="Optional cap on prompts (default: all)")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N prompts")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: auto)")
    parser.add_argument("--concurrency", type=int, default=DEFAULTS["concurrency"],
                        help=f"Concurrent API calls (default {DEFAULTS['concurrency']})")
    # n_completions (10) and max_tokens (500) are fixed constants, not flags.
    args = parser.parse_args()
    generate_completions(
        llm_spec=args.llm,
        input_path=args.input,
        n_prompts=args.n_prompts,
        offset=args.offset,
        output_path=args.output,
        concurrency=args.concurrency,
    )
