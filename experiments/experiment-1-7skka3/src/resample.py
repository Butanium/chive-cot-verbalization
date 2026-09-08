"""Step 2: resample the 100 original prompts from Qwen3-8B in thinking mode (GPU job, 1x H100).

Serving: SGLang (the `job-core` image's baked fork; the public vLLM images fail to start on this
fabric). The released run used vLLM with `thinking_token_budget=1024` enforced as a hard cap
(ThinkingTokenBudgetLogitsProcessor forces `</think>` once 1,024 tokens follow `<think>`) and
`max_tokens=2048` overall. SGLang has no equivalent request field, so the cap is reproduced
exactly at the token level in two stages against the raw /generate endpoint:

  stage A: prompt_ids -> sample n=30 continuations, max_new_tokens = 1 + 1024, stop on </think>
           (the model emits `<think>` itself; 1,024 tokens may follow it, matching vLLM's count)
  stage B: for each sample that did not finish: continue from prompt_ids + A_ids (+ forced
           </think> if the cap was hit) with max_new_tokens = 2048 - len(A_ids incl. </think>)

Sampling: temperature 1.0, top_p 0.95, top_k 20 (Qwen3-8B generation_config.json, which is what
vLLM's `--generation-config auto` applied for the paper since chive sends no top_p/top_k).
Chat template: the tokenizer's own template with enable_thinking=True (vLLM's default for the
released run; prior assistant turns lose any <think> block, as in the template).

Output record (one per prompt, chive PromptCompletions-compatible keys plus token-level extras):
  id, user_message, messages, completions (chive format: "<think>\n{reasoning}\n</think>\n\n{answer}"),
  finish_reasons ("stop" | "length"), prompt_tokens, response_token_counts, think_tokens, answer_tokens,
  think_capped (bool), raw_texts.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import httpx
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
N_COMPLETIONS = 30
MAX_TOKENS = 2048
THINKING_BUDGET = 1024
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
THINK_START = 151667
THINK_END = 151668
EOS_IDS = {151645, 151643}


def start_server(port: int, mem_frac: float, context_len: int) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--revision", REVISION,
        "--host", "127.0.0.1", "--port", str(port),
        "--reasoning-parser", "qwen3",
        "--sampling-defaults", "model",
        "--mem-fraction-static", str(mem_frac),
        "--context-length", str(context_len),
        "--log-level", "warning",
    ]
    print("[resample] launching:", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: int = 1500) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    dt = time.time() - t0
                    print(f"[resample] server ready after {dt:.0f}s", flush=True)
                    return dt
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server did not become ready")


async def gen(client: httpx.AsyncClient, url: str, input_ids: list[int], n: int, max_new: int,
              stop_ids: list[int]) -> list[dict]:
    body = {
        "input_ids": input_ids,
        "sampling_params": {
            "n": n, "max_new_tokens": max_new, "temperature": TEMPERATURE,
            "top_p": TOP_P, "top_k": TOP_K, "stop_token_ids": stop_ids,
            "skip_special_tokens": False, "no_stop_trim": True,
        },
    }
    for attempt in range(5):
        try:
            r = await client.post(url, json=body, timeout=3600)
            r.raise_for_status()
            out = r.json()
            return out if isinstance(out, list) else [out]
        except Exception as e:  # noqa: BLE001
            print(f"[resample] request failed ({type(e).__name__}: {e}); retry {attempt+1}/5", flush=True)
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError("generation request failed after retries")


def strip_eos(ids: list[int]) -> list[int]:
    while ids and ids[-1] in EOS_IDS:
        ids = ids[:-1]
    return ids


async def sample_prompt(client, url, tok, cand: dict, n: int, sem: asyncio.Semaphore) -> dict:
    messages = cand["messages"]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=True)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

    async with sem:
        stage_a = await gen(client, url, prompt_ids, n, 1 + THINKING_BUDGET, [THINK_END])
    assert len(stage_a) == n, f"expected {n} samples, got {len(stage_a)}"

    completions, finish_reasons, resp_tokens, think_tokens, answer_tokens, capped, raws = [], [], [], [], [], [], []

    def _pack(all_ids, reasoning, answer, fr, was_capped, n_think, n_answer):
        reasoning_s = reasoning.strip()
        answer_s = answer.strip()
        text = f"<think>\n{reasoning_s}\n</think>\n\n{answer_s}" if reasoning_s else answer_s
        return {
            "text": text, "fr": fr, "n_resp": len(all_ids), "n_think": n_think,
            "n_answer": n_answer, "capped": was_capped,
            "raw": tok.decode(all_ids, skip_special_tokens=False),
        }

    async def finish_one(o: dict):
        a_ids = list(o["output_ids"])
        fr = o["meta_info"]["finish_reason"]
        has_think = bool(a_ids) and a_ids[0] == THINK_START
        if fr["type"] == "stop" and (fr.get("matched") == THINK_END or (a_ids and a_ids[-1] == THINK_END)):
            closed = a_ids if a_ids[-1] == THINK_END else a_ids + [THINK_END]
            was_capped = False
        elif fr["type"] == "stop":
            # EOS without ever closing the think block (model answered without thinking)
            closed = strip_eos(a_ids)
            body = closed[1:] if has_think else closed
            return _pack(closed, "", tok.decode(body, skip_special_tokens=True), "stop", False, 0, len(body))
        else:
            assert fr["type"] == "length", fr
            closed = a_ids + [THINK_END]          # budget exhausted -> force </think>, as vLLM does
            was_capped = True
        think_body = closed[1:-1] if has_think else closed[:-1]
        reasoning = tok.decode(think_body, skip_special_tokens=True)
        remaining = MAX_TOKENS - len(closed)
        if remaining <= 0:
            return _pack(closed, reasoning, "", "length", was_capped, len(think_body), 0)
        async with sem:
            stage_b = await gen(client, url, prompt_ids + closed, 1, remaining, [])
        b = stage_b[0]
        b_ids = strip_eos(list(b["output_ids"]))
        b_fr = b["meta_info"]["finish_reason"]["type"]
        answer = tok.decode(b_ids, skip_special_tokens=True)
        return _pack(closed + b_ids, reasoning, answer, "length" if b_fr == "length" else "stop",
                     was_capped, len(think_body), len(b_ids))

    packed = await asyncio.gather(*[finish_one(o) for o in stage_a])
    for p in packed:
        if not p["text"].strip():
            continue
        completions.append(p["text"]); finish_reasons.append(p["fr"]); resp_tokens.append(p["n_resp"])
        think_tokens.append(p["n_think"]); answer_tokens.append(p["n_answer"]); capped.append(p["capped"])
        raws.append(p["raw"])
    return {
        "id": cand["id"],
        "user_message": cand["user_message"],
        "messages": messages,
        "completions": completions,
        "finish_reasons": finish_reasons,
        "prompt_tokens": len(prompt_ids),
        "response_token_counts": resp_tokens,
        "think_tokens": think_tokens,
        "answer_tokens": answer_tokens,
        "think_capped": capped,
        "raw_texts": raws,
        "n_turns": cand.get("n_turns", 1),
        "n_empty_dropped": n - len(completions),
    }


async def run_all(port, tok, candidates, n, out_file, concurrency):
    url = f"http://127.0.0.1:{port}/generate"
    done = {}
    if out_file.exists():
        for line in out_file.read_text().split("\n"):
            if line.strip():
                rec = json.loads(line); done[rec["id"]] = rec
        print(f"[resample] resume: {len(done)} prompts already done", flush=True)
    remaining = [c for c in candidates if c["id"] not in done]
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    n_done = len(done)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=2000)) as client:
        with open(out_file, "a") as f:
            async def one(c):
                nonlocal n_done
                rec = await sample_prompt(client, url, tok, c, n, sem)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                n_done += 1
                el = time.time() - t0
                rate = (n_done - len(done)) / max(el, 1e-6) * 60
                print(f"[resample] progress {n_done}/{len(candidates)} prompts  elapsed={el:.0f}s  "
                      f"rate={rate:.1f} prompts/min", flush=True)
            await asyncio.gather(*[one(c) for c in remaining])
    return time.time() - t0


def main():
    global THINKING_BUDGET
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--n-completions", type=int, default=N_COMPLETIONS)
    ap.add_argument("--concurrency", type=int, default=24, help="concurrent /generate requests")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--mem-frac", type=float, default=0.85)
    ap.add_argument("--context-len", type=int, default=8192)
    ap.add_argument("--thinking-budget", type=int, default=1024,
                    help="TEST ONLY: override the 1024-token cap to exercise the forced-</think> path")
    args = ap.parse_args()
    THINKING_BUDGET = args.thinking_budget

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "completions.jsonl"

    import sglang  # noqa: F401
    print(f"[resample] sglang={sglang.__version__} python={sys.version.split()[0]}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    assert tok.convert_tokens_to_ids("<think>") == THINK_START and tok.convert_tokens_to_ids("</think>") == THINK_END

    candidates = [json.loads(l) for l in open(args.prompts) if l.strip()]
    if args.n_prompts:
        candidates = candidates[: args.n_prompts]
    print(f"[resample] {len(candidates)} prompts x {args.n_completions} completions; max_tokens={MAX_TOKENS} "
          f"thinking_budget={THINKING_BUDGET} temperature={TEMPERATURE} top_p={TOP_P} top_k={TOP_K}", flush=True)

    proc = start_server(args.port, args.mem_frac, args.context_len)
    try:
        load_s = wait_ready(args.port, proc)
        gen_s = asyncio.run(run_all(args.port, tok, candidates, args.n_completions, out_file, args.concurrency))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()

    recs = [json.loads(l) for l in open(out_file) if l.strip()]
    total = sum(len(r["completions"]) for r in recs)
    n_len = sum(fr == "length" for r in recs for fr in r["finish_reasons"])
    n_cap = sum(c for r in recs for c in r["think_capped"])
    n_nothink = sum(1 for r in recs for c in r["completions"] if not c.startswith("<think>"))
    tt = sorted(t for r in recs for t in r["think_tokens"])
    meta = {
        "model": MODEL, "revision": REVISION, "server": "sglang", "sglang_version": sglang.__version__,
        "n_prompts": len(recs), "n_completions_per_prompt": args.n_completions, "total_completions": total,
        "max_tokens": MAX_TOKENS, "thinking_budget": THINKING_BUDGET, "temperature": TEMPERATURE,
        "top_p": TOP_P, "top_k": TOP_K,
        "finish_length": n_len, "finish_length_rate": n_len / total if total else None,
        "think_capped": n_cap, "think_capped_rate": n_cap / total if total else None,
        "no_think_segment": n_nothink,
        "empty_dropped": sum(r["n_empty_dropped"] for r in recs),
        "think_tokens_quantiles": {q: tt[int(q * (len(tt) - 1))] for q in (0.1, 0.5, 0.9, 0.95, 1.0)} if tt else None,
        "server_load_seconds": load_s, "generation_seconds": gen_s,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("[resample] SUMMARY", json.dumps(meta), flush=True)
    print(f"[resample] wrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
