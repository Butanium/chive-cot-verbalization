"""Step 3: grade behavior presence in the fresh completions with the paper's grader.

The system prompt, tool schema, and user-content builder below are reproduced verbatim from
chive/pipeline/investigate.py (commit 2423952, lines ~957-1076: GRADER_SYSTEM_PROMPT, GRADER_TOOL,
build_grader_user_content, grade_completions), so "behavior present" means the same thing as in the
released data. Each investigation's frozen classifier_question is applied to all 30 fresh completions
in one call (per-completion verdicts via tool call, counted in code). Grader model: the paper's
`claude-sonnet-4-6`, reached through chive's openrouter backend as `anthropic/claude-sonnet-4.6`
with the same 2,048-token thinking budget and 8,192 max_tokens.

Output: results/behavior_grades.jsonl (one row per prompt: verdicts, fresh k/n, original k/n),
cached incrementally so a rerun only grades what is missing.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vendor"))

from chive.pipeline.llm_client import make_llm_client, set_max_concurrency  # noqa: E402
from chive.pipeline.tool_validate import call_for_validated_tool  # noqa: E402
from chive.pipeline.utils import pydantic_tool  # noqa: E402

GRADER_LLM = "openrouter:anthropic/claude-sonnet-4.6"

# ---- verbatim from chive/pipeline/investigate.py ---------------------------------------------
GRADER_MAX_TOKENS = 8192
GRADER_THINKING_BUDGET = 2048

GRADER_SYSTEM_PROMPT = """\
You are a precise behavior grader. You will see a conversation, a yes/no \
question about a single response, and several candidate responses a model \
generated for the final user turn. For EACH response independently, answer \
the question with yes or no.

Rules:
- Judge only the text of each response. Do not reward or penalize quality — \
you are detecting whether a specific behavior is present, not rating.
- Apply the question literally and consistently. If the question names edge \
cases or variants, follow it exactly.
- If the question references content that does not exist in this \
conversation (the conversation may be a modified variant of the one the \
question was written for), answer "no" unless the described response \
behavior is plainly present anyway.

Submit one verdict per response via the `submit_verdicts` tool, covering \
every response number exactly once."""


class CompletionVerdict(BaseModel):
    completion: int = Field(description="1-based response number")
    answer: Literal["yes", "no"]


class GradeCompletionsInput(BaseModel):
    verdicts: list[CompletionVerdict] = Field(
        description=(
            "One verdict per response, in order: response 1..N each exactly once."
        ),
    )


GRADER_TOOL = pydantic_tool(
    GradeCompletionsInput,
    name="submit_verdicts",
    description="Submit your yes/no verdict for every response.",
)


def _render_conversation(messages: list[dict]) -> str:
    return "\n\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in messages)


def build_grader_user_content(
    classifier_question: str, messages: list[dict], completions: list[str]
) -> str:
    parts = [
        "## Conversation\n",
        _render_conversation(messages),
        f"\n\n## Question (answer for EACH response independently)\n\n{classifier_question}\n",
        f"\n## Responses ({len(completions)})\n",
    ]
    for i, comp in enumerate(completions, 1):
        parts.append(f"\n--- Response {i}/{len(completions)} ---\n{comp}\n")
    return "".join(parts)


async def grade_completions(grader_client, classifier_question, messages, completions) -> list[bool]:
    assert classifier_question.strip(), "classifier_question must be non-empty"
    user_content = build_grader_user_content(classifier_question, messages, completions)
    expected = set(range(1, len(completions) + 1))
    for attempt in range(2):
        outcome = await call_for_validated_tool(
            grader_client,
            system=GRADER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            tool=GRADER_TOOL,
            model_cls=GradeCompletionsInput,
            max_tokens=GRADER_MAX_TOKENS,
            thinking_budget=GRADER_THINKING_BUDGET,
        )
        if outcome is None:
            continue
        tool_input, _resp = outcome
        got = [v.completion for v in tool_input.verdicts]
        if set(got) != expected or len(got) != len(expected):
            print(f"    [grader] coverage mismatch (attempt {attempt + 1}/2): "
                  f"expected 1..{len(completions)}, got {sorted(got)}")
            continue
        by_num = {v.completion: v.answer == "yes" for v in tool_input.verdicts}
        return [by_num[i] for i in range(1, len(completions) + 1)]
    raise RuntimeError(
        f"Grader failed to produce full-coverage verdicts for "
        f"{len(completions)} completions after 2 attempts"
    )
# ---- end verbatim ------------------------------------------------------------------------------


async def main_async(args):
    manifest = {r["prompt_id"]: r for r in (json.loads(l) for l in open(args.manifest) if l.strip())}
    comps = {r["id"]: r for r in (json.loads(l) for l in open(args.completions) if l.strip())}
    out = Path(args.out)
    done = {}
    if out.exists():
        for l in open(out):
            if l.strip():
                r = json.loads(l); done[r["prompt_id"]] = r
    todo = [pid for pid in manifest if pid in comps and pid not in done]
    print(f"[grader] {len(manifest)} investigations, {len(comps)} with completions, {len(done)} graded, {len(todo)} to grade")
    set_max_concurrency(args.concurrency)
    client = make_llm_client(GRADER_LLM)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    f = open(out, "a")

    async def one(pid):
        m = manifest[pid]; c = comps[pid]
        async with sem:
            try:
                verdicts = await grade_completions(client, m["classifier_question"], m["messages"], c["completions"])
            except Exception as e:  # noqa: BLE001
                print(f"[grader] FAILED {pid}: {type(e).__name__}: {e}", flush=True)
                return
        row = {
            "prompt_id": pid,
            "grader_llm": GRADER_LLM,
            "verdicts": verdicts,
            "k_fresh": sum(verdicts),
            "n_fresh": len(verdicts),
            "k_orig": m["orig_baseline_yes"],
            "n_orig": m["orig_n_completions"],
        }
        async with lock:
            f.write(json.dumps(row) + "\n"); f.flush()
        print(f"[grader] {pid}: fresh {row['k_fresh']}/{row['n_fresh']} vs original {row['k_orig']}/{row['n_orig']}", flush=True)

    await asyncio.gather(*[one(pid) for pid in todo])
    f.close()
    rows = [json.loads(l) for l in open(out) if l.strip()]
    print(f"[grader] done: {len(rows)} graded; total behavior-exhibiting completions: "
          f"{sum(r['k_fresh'] for r in rows)}/{sum(r['n_fresh'] for r in rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "results/manifest.jsonl"))
    ap.add_argument("--completions", default=os.path.join(os.environ.get("SILICO_EXPERIMENT_ARTIFACTS_DIR", ""), "resample/completions.jsonl"))
    ap.add_argument("--out", default=str(HERE.parent / "results/behavior_grades.jsonl"))
    ap.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main_async(ap.parse_args()))
