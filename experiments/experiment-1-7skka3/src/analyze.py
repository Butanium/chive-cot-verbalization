"""Step 6: analysis. Pooled and per-investigation verbalization rates, controls, slices, judge validation,
reproducibility check, and 20 random examples.

Inputs (results/): manifest.jsonl, behavior_grades.jsonl, judge_verdicts.jsonl (think segment, causal+inert),
judge_verdicts_answer.jsonl (answer segment, causal, behavior-exhibiting only),
judge_validation_gpt.jsonl (second-family rejudge), judge_validation_temp1.jsonl (primary at temperature 1).
Outputs (results/): summary.json, per_investigation.csv, examples.json, plot_data.json
"""

import argparse
import collections
import csv
import json
import math
import os
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
THINK_RE = re.compile(r"^<think>\n(.*?)\n</think>\n\n(.*)$", re.S)
SEED = 42
N_BOOT = 5000


def load(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def boot_ci(groups, stat, n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap over investigations. groups: list of per-investigation item lists.
    stat: function(list of items) -> float. Returns (point, lo, hi)."""
    rng = random.Random(seed)
    flat = [x for g in groups for x in g]
    point = stat(flat)
    vals = []
    for _ in range(n_boot):
        sample = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
        vals.append(stat([x for g in sample for x in g]))
    vals.sort()
    return point, vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def rate(items, key):
    items = [x for x in items if x.get(key) is not None]
    return sum(1 for x in items if x[key]) / len(items) if items else float("nan")


def kappa(a, b):
    """Cohen's kappa for two aligned boolean lists."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n; pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--completions", default=os.path.join(os.environ.get("SILICO_EXPERIMENT_ARTIFACTS_DIR", ""), "resample/completions.jsonl"))
    args = ap.parse_args()

    manifest = {r["prompt_id"]: r for r in load(RES / "manifest.jsonl")}
    grades = {r["prompt_id"]: r for r in load(RES / "behavior_grades.jsonl")}
    comps = {r["id"]: r for r in load(args.completions)}
    verdicts = load(RES / "judge_verdicts.jsonl")
    ans_verdicts = load(RES / "judge_verdicts_answer.jsonl")
    val_gpt = load(RES / "judge_validation_gpt.jsonl")
    val_t1 = load(RES / "judge_validation_temp1.jsonl")
    run_meta = json.load(open(Path(args.completions).parent / "run_meta.json")) if (Path(args.completions).parent / "run_meta.json").exists() else {}

    # ---------- index verdicts
    V = {}  # (pid, idx, kind, segment) -> row
    for r in verdicts + ans_verdicts:
        if r.get("valid"):
            V[(r["prompt_id"], r["completion_index"], r["factor_kind"], r["segment"])] = r
    invalid = sum(1 for r in verdicts + ans_verdicts if not r.get("valid"))
    total_verdicts = len(verdicts) + len(ans_verdicts)

    # ---------- per-completion table
    rows = []
    for pid, m in manifest.items():
        if pid not in grades or pid not in comps:
            continue
        g = grades[pid]
        for i, beh in enumerate(g["verdicts"]):
            c = V.get((pid, i, "causal", "think")); n = V.get((pid, i, "inert", "think")); a = V.get((pid, i, "causal", "answer"))
            rows.append({
                "prompt_id": pid, "idx": i, "behavior": bool(beh),
                "c_q1": c["q1"] if c else None, "c_q2": c["q2"] if c else None, "c_q3": c["q3"] if c else None,
                "i_q1": n["q1"] if n else None, "i_q2": n["q2"] if n else None, "i_q3": n["q3"] if n else None,
                "a_q2": a["q2"] if a else None,
                "c_mention_only": (c["q1"] and not c["q2"]) if c else None,
                "c_alt_only": (c["q3"] and not c["q2"]) if c else None,
                "c_none": (not c["q1"] and not c["q2"] and not c["q3"]) if c else None,
                "think_tokens": comps[pid]["think_tokens"][i] if i < len(comps[pid].get("think_tokens", [])) else None,
                "think_capped": comps[pid]["think_capped"][i] if i < len(comps[pid].get("think_capped", [])) else None,
                "finish": comps[pid]["finish_reasons"][i],
            })
    beh_rows = [r for r in rows if r["behavior"] and r["c_q2"] is not None]
    by_pid = collections.defaultdict(list)
    for r in beh_rows:
        by_pid[r["prompt_id"]].append(r)
    groups = list(by_pid.values())

    summary = {"n_investigations_graded": len(grades), "n_completions": len(rows),
               "n_behavior_exhibiting": sum(r["behavior"] for r in rows),
               "n_behavior_exhibiting_judged": len(beh_rows),
               "n_investigations_with_behavior": len(groups),
               "judge_invalid_rate": invalid / total_verdicts if total_verdicts else None,
               "judge_invalid": invalid, "judge_total": total_verdicts,
               "run_meta": run_meta}

    # ---------- headline: pooled rates among behavior-exhibiting completions, cluster bootstrap
    def add_rate(name, key, sub=None):
        grp = groups if sub is None else [[r for r in g if sub(r)] for g in groups]
        grp = [g for g in grp if g]
        if not grp:
            summary[name] = None; return
        p, lo, hi = boot_ci(grp, lambda xs: rate(xs, key))
        summary[name] = {"rate": p, "ci95": [lo, hi], "n": sum(len(g) for g in grp), "n_inv": len(grp)}

    add_rate("causal_q2_attribution", "c_q2")
    add_rate("causal_q1_mention", "c_q1")
    add_rate("causal_q3_alternative", "c_q3")
    add_rate("causal_mention_only", "c_mention_only")
    add_rate("causal_alternative_without_attribution", "c_alt_only")
    add_rate("causal_silent", "c_none")
    add_rate("inert_q2_attribution", "i_q2", sub=lambda r: r["i_q2"] is not None)
    add_rate("inert_q1_mention", "i_q1", sub=lambda r: r["i_q1"] is not None)
    add_rate("inert_q3_alternative", "i_q3", sub=lambda r: r["i_q3"] is not None)
    add_rate("answer_q2_attribution", "a_q2", sub=lambda r: r["a_q2"] is not None)
    # causal Q2 restricted to investigations that have an inert control (paired subset)
    add_rate("causal_q2_attribution_paired_subset", "c_q2", sub=lambda r: r["i_q2"] is not None)
    # non-behavior completions: does the reasoning attribute anyway?
    non_groups = collections.defaultdict(list)
    for r in rows:
        if not r["behavior"] and r["c_q2"] is not None:
            non_groups[r["prompt_id"]].append(r)
    if non_groups:
        p, lo, hi = boot_ci(list(non_groups.values()), lambda xs: rate(xs, "c_q1"))
        summary["nonbehavior_causal_q1_mention"] = {"rate": p, "ci95": [lo, hi], "n": sum(len(g) for g in non_groups.values())}

    # paired difference causal - inert per investigation
    per_inv = []
    for pid, m in manifest.items():
        if pid not in grades:
            continue
        g = by_pid.get(pid, [])
        rec = {"prompt_id": pid, "n_behavior": len(g), "k_fresh": grades[pid]["k_fresh"], "n_fresh": grades[pid]["n_fresh"],
               "k_orig": grades[pid]["k_orig"], "n_orig": grades[pid]["n_orig"],
               "effect_pp": m["effect_pp"], "edit_cleanliness": m["edit_cleanliness"], "is_mistake": m["is_mistake"],
               "categories": "|".join(m["categories"] or []), "has_inert": m["inert_factor"] is not None,
               "causal_q2": rate(g, "c_q2") if g else None, "causal_q1": rate(g, "c_q1") if g else None,
               "causal_q3": rate(g, "c_q3") if g else None,
               "inert_q2": rate([r for r in g if r["i_q2"] is not None], "i_q2") if g and m["inert_factor"] else None,
               "inert_q1": rate([r for r in g if r["i_q1"] is not None], "i_q1") if g and m["inert_factor"] else None,
               "answer_q2": rate([r for r in g if r["a_q2"] is not None], "a_q2") if g else None,
               "question": m["investigation_question"], "causal_intervention": m["causal_factor"]["intervention"],
               "inert_intervention": m["inert_factor"]["intervention"] if m["inert_factor"] else None}
        rec["paired_diff"] = (rec["causal_q2"] - rec["inert_q2"]) if rec["inert_q2"] is not None and rec["causal_q2"] is not None else None
        per_inv.append(rec)
    with open(RES / "per_investigation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_inv[0].keys())); w.writeheader(); w.writerows(per_inv)

    diffs = [r["paired_diff"] for r in per_inv if r["paired_diff"] is not None]
    if diffs:
        rng = random.Random(SEED); bs = []
        for _ in range(N_BOOT):
            s = [diffs[rng.randrange(len(diffs))] for _ in diffs]; bs.append(sum(s) / len(s))
        bs.sort()
        summary["paired_diff_causal_minus_inert"] = {"mean": sum(diffs) / len(diffs), "ci95": [bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT) - 1]],
                                                     "n_inv": len(diffs), "frac_positive": sum(d > 0 for d in diffs) / len(diffs),
                                                     "frac_negative": sum(d < 0 for d in diffs) / len(diffs)}
    cq = [r["causal_q2"] for r in per_inv if r["causal_q2"] is not None]
    summary["per_investigation_causal_q2"] = {"median": sorted(cq)[len(cq) // 2], "frac_ge_0.7": sum(x >= 0.7 for x in cq) / len(cq),
                                              "frac_le_0.3": sum(x <= 0.3 for x in cq) / len(cq), "frac_eq_0": sum(x == 0 for x in cq) / len(cq),
                                              "frac_eq_1": sum(x == 1 for x in cq) / len(cq), "n": len(cq)}

    # ---------- slices
    def slice_rates(name, keyfn):
        out = {}
        buckets = collections.defaultdict(list)
        for g in groups:
            pid = g[0]["prompt_id"]
            for k in keyfn(manifest[pid]):
                buckets[k].append(g)
        for k, grp in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            p, lo, hi = boot_ci(grp, lambda xs: rate(xs, "c_q2"))
            pi = [rate([r for r in g if r["i_q2"] is not None], "i_q2") for g in grp if any(r["i_q2"] is not None for r in g)]
            out[str(k)] = {"causal_q2": p, "ci95": [lo, hi], "n_inv": len(grp), "n": sum(len(g) for g in grp),
                           "inert_q2_mean_per_inv": (sum(pi) / len(pi)) if pi else None}
        summary[name] = out
    slice_rates("slice_is_mistake", lambda m: [m["is_mistake"]])
    slice_rates("slice_edit_cleanliness", lambda m: [m["edit_cleanliness"]])
    slice_rates("slice_category", lambda m: m["categories"] or ["(none)"])
    slice_rates("slice_effect", lambda m: ["50-66" if m["effect_pp"] < 66.7 else ("67-83" if m["effect_pp"] < 83.4 else "84-100")])
    slice_rates("slice_n_turns", lambda m: ["single-turn" if len(m["messages"]) == 1 else "multi-turn"])
    slice_rates("slice_think_capped", lambda m: [None])  # placeholder removed below
    del summary["slice_think_capped"]
    # think-cap slice at completion level
    for label, cond in (("capped", lambda r: r["think_capped"]), ("not_capped", lambda r: r["think_capped"] is False)):
        grp = [[r for r in g if cond(r)] for g in groups]; grp = [g for g in grp if g]
        if grp:
            p, lo, hi = boot_ci(grp, lambda xs: rate(xs, "c_q2"))
            summary.setdefault("slice_think_capped", {})[label] = {"causal_q2": p, "ci95": [lo, hi], "n": sum(len(g) for g in grp)}

    # ---------- reproducibility: fresh vs original behavior rates
    fr = [r["k_fresh"] / r["n_fresh"] for r in per_inv]; orr = [r["k_orig"] / r["n_orig"] for r in per_inv]
    summary["reproducibility"] = {"pearson_r": pearson(fr, orr), "mean_fresh": sum(fr) / len(fr), "mean_orig": sum(orr) / len(orr),
                                  "mean_abs_diff": sum(abs(a - b) for a, b in zip(fr, orr)) / len(fr),
                                  "n_fresh_rate_below_0.5": sum(x < 0.5 for x in fr), "n_fresh_rate_zero": sum(x == 0 for x in fr)}

    # ---------- judge validation
    def align(val_rows):
        pairs = collections.defaultdict(list)
        for v in val_rows:
            if not v.get("valid"):
                continue
            p = V.get((v["prompt_id"], v["completion_index"], v["factor_kind"], v["segment"]))
            if p is None:
                continue
            for q in ("q1", "q2", "q3"):
                pairs[q].append((p[q], v[q]))
        out = {}
        for q, pr in pairs.items():
            a = [x for x, _ in pr]; b = [y for _, y in pr]
            out[q] = {"kappa": kappa(a, b), "agreement": sum(x == y for x, y in pr) / len(pr), "n": len(pr),
                      "primary_yes_rate": sum(a) / len(a), "other_yes_rate": sum(b) / len(b)}
        out["invalid"] = sum(1 for v in val_rows if not v.get("valid"))
        return out
    if val_gpt:
        summary["validation_second_family"] = {"model": val_gpt[0]["model"], **align(val_gpt)}
    if val_t1:
        summary["validation_self_temp1"] = {"model": val_t1[0]["model"], **align(val_t1)}

    # ---------- 20 random behavior-exhibiting examples, chosen without regard to outcome
    rng = random.Random(SEED)
    ex_rows = rng.sample(beh_rows, min(20, len(beh_rows)))
    examples = []
    for r in ex_rows:
        pid, i = r["prompt_id"], r["idx"]
        m = manifest[pid]; text = comps[pid]["completions"][i]
        mt = THINK_RE.match(text)
        think, answer = (mt.group(1), mt.group(2)) if mt else ("", text)
        c = V.get((pid, i, "causal", "think")); n = V.get((pid, i, "inert", "think")); a = V.get((pid, i, "causal", "answer"))
        examples.append({"prompt_id": pid, "completion_index": i, "question": m["investigation_question"],
                         "behavior_question": m["classifier_question"], "messages": m["messages"],
                         "causal_intervention": m["causal_factor"]["intervention"], "causal_grounding": m["causal_factor"]["grounding"],
                         "effect_pp": m["effect_pp"], "inert_intervention": m["inert_factor"]["intervention"] if m["inert_factor"] else None,
                         "think": think, "answer": answer,
                         "causal_verdicts": {k: c[k] for k in c if k.startswith("q")} if c else None,
                         "inert_verdicts": {k: n[k] for k in n if k.startswith("q")} if n else None,
                         "answer_verdicts": {k: a[k] for k in a if k.startswith("q")} if a else None})
    json.dump(examples, open(RES / "examples.json", "w"), indent=1, ensure_ascii=False)

    # ---------- plot-ready data
    plot = {
        "per_investigation": [{"prompt_id": r["prompt_id"], "causal_q2": r["causal_q2"], "inert_q2": r["inert_q2"], "causal_q1": r["causal_q1"],
                               "inert_q1": r["inert_q1"], "answer_q2": r["answer_q2"], "n_behavior": r["n_behavior"], "effect_pp": r["effect_pp"],
                               "edit_cleanliness": r["edit_cleanliness"], "is_mistake": r["is_mistake"], "fresh_rate": r["k_fresh"] / r["n_fresh"],
                               "orig_rate": r["k_orig"] / r["n_orig"]} for r in per_inv],
        "pooled": {k: summary[k] for k in summary if k.startswith(("causal_", "inert_", "answer_")) and isinstance(summary[k], dict)},
    }
    json.dump(plot, open(RES / "plot_data.json", "w"), indent=1)
    json.dump(summary, open(RES / "summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("slice_category", "run_meta")}, indent=1))


if __name__ == "__main__":
    main()
