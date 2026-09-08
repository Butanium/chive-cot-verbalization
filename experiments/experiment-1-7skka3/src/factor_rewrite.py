"""Rewrite each factor's edit description into a description of the ORIGINAL conversation aspect.

Motivation (checkpoint-010): the v2 judge sometimes read the factor as the hypothetical edited text
and answered "not mentioned" even when the reasoning discussed the original aspect. This step
removes the edit framing before the judge sees the factor. The rewriter sees only the conversation
and the claim's `intervention` text (no effect size, no grounding, no outcome), so causal and inert
factors are treated identically and remain outcome-blind.

Output: results/factor_descriptions.jsonl with one row per (prompt_id, factor_kind):
  {prompt_id, factor_kind, claim_id, intervention, description, model}
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vendor"))
from chive.pipeline.llm_client import make_llm_client, set_max_concurrency  # noqa: E402

from judge import render_conversation  # noqa: E402

SYSTEM = """\
You convert descriptions of hypothetical edits to a conversation into plain descriptions of the \
ORIGINAL, unedited conversation content that the edit would have changed. The edit was never applied. \
Your output must describe ONLY the specific content that the edit would have changed, as it actually \
appears in the original conversation (quote the original wording where it is short enough, and say \
where in the conversation it appears). Do not describe parts of the conversation the edit leaves \
unchanged, do not describe or contrast with the edited wording, and do not mention that an edit, \
modification, replacement, alternative, or hypothetical exists. If the edit would ADD content that the \
original lacks, describe the original wording at that place and state that the original does not \
contain that content. Do not speculate about what the content causes or why it matters. \
Respond with exactly one JSON object: {"description": "<2-4 sentences>"}"""

USER = """\
<conversation>
{conversation}
</conversation>

<edit_description>
{intervention}
</edit_description>

Describe the aspect of the ORIGINAL conversation above that this edit would have changed, as \
instructed. Return the JSON object only."""


def parse(raw):
    s = raw.strip()
    a, b = s.find("{"), s.rfind("}")
    obj = json.loads(s[a:b + 1])
    d = str(obj["description"]).strip()
    if not d:
        raise ValueError("empty")
    return d


async def run(args):
    manifest = [json.loads(l) for l in open(args.manifest) if l.strip()]
    items = []
    for m in manifest:
        for kind in ("causal", "inert"):
            f = m[f"{kind}_factor"]
            if f is None:
                continue
            items.append({"prompt_id": m["prompt_id"], "factor_kind": kind, "claim_id": f["claim_id"],
                          "intervention": f["intervention"],
                          "user": USER.format(conversation=render_conversation(m["messages"]), intervention=f["intervention"])})
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["prompt_id"], r["factor_kind"]) for r in (json.loads(l) for l in open(out) if l.strip())}
    todo = [it for it in items if (it["prompt_id"], it["factor_kind"]) not in done]
    print(f"[rewrite] items={len(items)} todo={len(todo)}", flush=True)
    set_max_concurrency(args.concurrency)
    client = make_llm_client(f"openrouter:{args.model}")
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    f = open(out, "a")

    async def one(it):
        async with sem:
            desc, err = None, None
            for attempt in range(3):
                try:
                    resp = await client.call(system=SYSTEM, messages=[{"role": "user", "content": it["user"]}],
                                             tools=None, max_tokens=500, thinking_budget=None, temperature=0.0)
                    desc = parse("\n".join(resp.text_blocks))
                    break
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:300]
                    await asyncio.sleep(1.5 * (attempt + 1))
        row = {k: v for k, v in it.items() if k != "user"}
        row.update({"description": desc, "error": err, "model": args.model})
        async with lock:
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()

    await asyncio.gather(*[one(it) for it in todo])
    f.close()
    rows = [json.loads(l) for l in open(out) if l.strip()]
    print(f"[rewrite] done: {len(rows)} rows, failed={sum(r['description'] is None for r in rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "results/manifest.jsonl"))
    ap.add_argument("--out", default=str(HERE.parent / "results/factor_descriptions.jsonl"))
    ap.add_argument("--model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--concurrency", type=int, default=20)
    asyncio.run(run(ap.parse_args()))
