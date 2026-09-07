"""Step 1: select the 100 investigations and write the manifest + prompts file.

Selection rule (from the accepted plan; verified in planning to yield 200 candidates):
  * 5-judge verification score >= 9 (verification.jsonl `score`)
  * at least one TRUE claim in the concreteness>=3 eval set whose edit is single-factor
    (edit_cleanliness in {minor_targeted, large_single_factor}) with |effect_pp| >= 50
  * behavior present in >= 15/30 of the original samples (k_baseline / n_baseline >= 0.5)
The causal factor is the largest-|effect| qualifying claim. The control factor is an inert
claim (factor_status == "inert") from the same investigation, chosen deterministically
(smallest |effect_pp|, then lowest claim_index) when several exist.
100 investigations are sampled uniformly at random from the candidates with seed 42.

Outputs (results/):
  manifest.jsonl   one row per selected investigation (metadata + factor texts + messages)
  prompts.jsonl    chive stage-1 prompt records (id, user_message, messages, n_turns) for resampling
  selection_summary.json
"""

import argparse
import collections
import json
import os
import random
from pathlib import Path

SEED = 42
N_SELECT = 100
SINGLE_FACTOR = {"minor_targeted", "large_single_factor"}


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(
        os.environ["SILICO_EXPERIMENT_ARTIFACTS_DIR"],
        "chive_data/datasets/qwen3_8b_wildchat_thinking_eval"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    args = ap.parse_args()
    d = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    claims = list(load_jsonl(d / "eval_set_conc3_n2000.jsonl"))
    ver = {r["prompt_id"]: r["score"] for r in load_jsonl(d / "verification.jsonl")}
    conc = {r["prompt_id"]: r for r in load_jsonl(d / "mechanism_concreteness.jsonl")}
    by_prompt = collections.defaultdict(list)
    for c in claims:
        by_prompt[c["prompt_id"]].append(c)

    candidates = []
    for pid in sorted(by_prompt):
        cl = by_prompt[pid]
        if ver.get(pid, 0.0) < 9:
            continue
        good = [c for c in cl if c["is_true"] and c["edit_cleanliness"] in SINGLE_FACTOR
                and abs(c["effect_pp"]) >= 50 and c["k_baseline"] / c["n_baseline"] >= 0.5]
        if not good:
            continue
        best = max(good, key=lambda c: (abs(c["effect_pp"]), -c["claim_index"]))
        inert = [c for c in cl if c["factor_status"] == "inert"]
        inert_pick = min(inert, key=lambda c: (abs(c["effect_pp"]), c["claim_index"])) if inert else None
        candidates.append((pid, best, inert_pick))
    print(f"candidates: {len(candidates)}; with inert control: {sum(1 for c in candidates if c[2])}")

    rng = random.Random(SEED)
    chosen = rng.sample(candidates, N_SELECT)
    chosen.sort(key=lambda t: t[0])

    # Pull structured_answer, classifier_question, original completions/verdicts from investigations.jsonl
    want = {pid for pid, _, _ in chosen}
    inv = {}
    with open(d / "investigations.jsonl") as f:
        for line in f:
            # cheap prefilter before parsing a ~400KB record
            head = line[:200]
            if not any(pid in head for pid in want):
                continue
            r = json.loads(line)
            if r["prompt_id"] in want:
                inv[r["prompt_id"]] = r
    missing = want - set(inv)
    assert not missing, f"investigations missing for {sorted(missing)}"

    manifest_rows, prompt_rows = [], []
    for pid, best, inert in chosen:
        r = inv[pid]
        messages = r["messages"]
        assert messages == best["messages"], f"messages mismatch for {pid}"
        assert messages[-1]["role"] == "user"
        n_orig = len(r["completions"])
        k_orig = int(r["baseline_grader_yes"])
        row = {
            "prompt_id": pid,
            "claim_id": best["claim_id"],
            "claim_index": best["claim_index"],
            "effect_pp": best["effect_pp"],
            "edit_cleanliness": best["edit_cleanliness"],
            "k_baseline": best["k_baseline"],
            "n_baseline": best["n_baseline"],
            "k_counterfactual": best["k_counterfactual"],
            "n_counterfactual": best["n_counterfactual"],
            "orig_baseline_yes": k_orig,
            "orig_n_completions": n_orig,
            "verification_score": ver[pid],
            "mechanism_concreteness": conc[pid]["mechanism_concreteness"] if pid in conc else None,
            "categories": conc[pid]["categories"] if pid in conc else None,
            "is_mistake": conc[pid]["is_mistake"] if pid in conc else None,
            "classifier_question": r["classifier_question"],
            "behavior_question": best["behavior_question"],
            "investigation_question": r["question"],
            "behavior_summary": r["behavior_summary"],
            "causal_factor": {
                "claim_id": best["claim_id"],
                "intervention": best["intervention"],
                "grounding": best["grounding"],
                "structured_answer": r["structured_findings"]["structured_answer"],
                "effect_pp": best["effect_pp"],
            },
            "inert_factor": None if inert is None else {
                "claim_id": inert["claim_id"],
                "intervention": inert["intervention"],
                "grounding": inert["grounding"],
                "effect_pp": inert["effect_pp"],
                "edit_cleanliness": inert["edit_cleanliness"],
            },
            "messages": messages,
        }
        manifest_rows.append(row)
        prompt_rows.append({
            "id": pid,
            "user_message": r["user_message"],
            "messages": messages,
            "n_turns": r.get("n_turns", 1),
            "total_chars": sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str)),
            "conversation_hash": "",
        })

    with open(out / "manifest.jsonl", "w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out / "prompts.jsonl", "w") as f:
        for row in prompt_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "seed": SEED,
        "n_candidates": len(candidates),
        "n_candidates_with_inert": sum(1 for c in candidates if c[2]),
        "n_selected": len(manifest_rows),
        "n_selected_with_inert": sum(1 for r in manifest_rows if r["inert_factor"]),
        "is_mistake": collections.Counter(str(r["is_mistake"]) for r in manifest_rows),
        "edit_cleanliness": collections.Counter(r["edit_cleanliness"] for r in manifest_rows),
        "effect_pp_quantiles": sorted(r["effect_pp"] for r in manifest_rows)[::10],
        "baseline_rate_quantiles": sorted(r["k_baseline"] / r["n_baseline"] for r in manifest_rows)[::10],
        "n_turns": collections.Counter(r["n_turns"] for r in prompt_rows),
        "has_system": sum(1 for r in prompt_rows if r["messages"][0]["role"] == "system"),
        "categories": collections.Counter(c for r in manifest_rows for c in (r["categories"] or [])),
    }
    with open(out / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
