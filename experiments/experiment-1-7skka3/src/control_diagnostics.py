"""Diagnostics on the inert control and judge validation requested after review.

Writes results/control_diagnostics.json with:
  * zero-effect inert subset: inert vs causal Q2 on the same investigations and the paired difference
  * per-investigation correlation between causal and inert Q2; count of pairs with inert >= causal
  * per-investigation causal-Q2 tail counts alongside the inert distribution; the 1.0/1.0 investigations
  * same-span overlap between causal and inert factor descriptions (v3 rewrites), and the paired difference
    split by overlap
  * judge validation kappa by subset (behavior/causal, behavior/inert, non-behavior), cross-family and self
  * misread-framing rationale counts (v2 vs v3) with the exact fields and regex used
  * slice contrasts restated with their overlapping CIs

Run with:  uv run --no-sync python src/control_diagnostics.py
"""
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import kappa, pearson  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
SEED = 42
N_BOOT = 2000
STOP = set("the a an of to in on at for and or with by as is was were be that this it its from into which not no".split())


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def boot_mean_ci(xs, seed=SEED):
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(N_BOOT))
    return sum(xs) / len(xs), bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT) - 1]


def main():
    per_inv = list(csv.DictReader(open(RES / "per_investigation.csv")))
    manifest = {r["prompt_id"]: r for r in jl(RES / "manifest.jsonl")}
    paired = [r for r in per_inv if r["inert_q2"] not in ("", None)]
    for r in paired:
        r["c"] = float(r["causal_q2"]); r["i"] = float(r["inert_q2"]); r["d"] = r["c"] - r["i"]
    out = {"n_paired_investigations": len(paired)}

    # ---- zero-effect inert subset
    zero = [r for r in paired if abs(float(manifest[r["prompt_id"]]["inert_factor"]["effect_pp"])) == 0]
    m, lo, hi = boot_mean_ci([r["d"] for r in zero])
    out["zero_effect_inert_subset"] = {"n_inv": len(zero), "mean_causal_q2_per_inv": sum(r["c"] for r in zero) / len(zero),
                                       "mean_inert_q2_per_inv": sum(r["i"] for r in zero) / len(zero),
                                       "paired_diff_mean": m, "paired_diff_ci95": [lo, hi]}
    m, lo, hi = boot_mean_ci([r["d"] for r in paired])
    out["paired_diff_all"] = {"n_inv": len(paired), "mean": m, "ci95": [lo, hi]}

    # ---- correlation and ordering
    out["causal_vs_inert_per_inv"] = {"pearson_r": pearson([r["c"] for r in paired], [r["i"] for r in paired]),
                                      "n_inert_ge_causal": sum(r["i"] >= r["c"] for r in paired),
                                      "n_inert_gt_causal": sum(r["i"] > r["c"] for r in paired),
                                      "n_equal": sum(r["i"] == r["c"] for r in paired)}

    # ---- tails
    cq = [float(r["causal_q2"]) for r in per_inv]
    iq = [r["i"] for r in paired]
    out["per_investigation_tails"] = {
        "causal": {"n": len(cq), "frac_eq_0": sum(x == 0 for x in cq) / len(cq), "frac_eq_1": sum(x == 1 for x in cq) / len(cq),
                   "frac_ge_0.7": sum(x >= 0.7 for x in cq) / len(cq), "frac_le_0.3": sum(x <= 0.3 for x in cq) / len(cq), "median": sorted(cq)[len(cq) // 2]},
        "inert": {"n": len(iq), "frac_eq_0": sum(x == 0 for x in iq) / len(iq), "frac_eq_1": sum(x == 1 for x in iq) / len(iq),
                  "frac_ge_0.7": sum(x >= 0.7 for x in iq) / len(iq), "frac_le_0.3": sum(x <= 0.3 for x in iq) / len(iq), "median": sorted(iq)[len(iq) // 2]},
        "causal_eq_1_and_inert_eq_1": [r["prompt_id"] for r in paired if r["c"] == 1 and r["i"] == 1],
        "causal_eq_1_total": [r["prompt_id"] for r in per_inv if float(r["causal_q2"]) == 1],
        "causal_eq_0_and_inert_eq_0": [r["prompt_id"] for r in paired if r["c"] == 0 and r["i"] == 0],
    }

    # ---- same-span overlap of v3 factor descriptions
    desc = {(r["prompt_id"], r["factor_kind"]): r["description"] for r in jl(RES / "factor_descriptions.jsonl")}

    def toks(s):
        return {t for t in re.findall(r"[a-z0-9']+", s.lower()) if t not in STOP and len(t) > 2}

    overlaps = []
    for r in paired:
        a, b = toks(desc[(r["prompt_id"], "causal")]), toks(desc[(r["prompt_id"], "inert")])
        jac = len(a & b) / len(a | b) if a | b else 0.0
        r["jaccard"] = jac
        overlaps.append(jac)
    ov = {}
    for thr in (0.3, 0.4, 0.5):
        hi_ = [r for r in paired if r["jaccard"] >= thr]; lo_ = [r for r in paired if r["jaccard"] < thr]
        ov[str(thr)] = {"n_high_overlap": len(hi_), "paired_diff_high": sum(r["d"] for r in hi_) / len(hi_) if hi_ else None,
                        "paired_diff_rest": sum(r["d"] for r in lo_) / len(lo_) if lo_ else None,
                        "causal_q2_high": sum(r["c"] for r in hi_) / len(hi_) if hi_ else None,
                        "inert_q2_high": sum(r["i"] for r in hi_) / len(hi_) if hi_ else None,
                        "high_overlap_ids": [r["prompt_id"] for r in sorted(hi_, key=lambda r: -r["jaccard"])]}
    out["description_overlap"] = {"metric": "token Jaccard of content words between causal and inert v3 descriptions",
                                  "median_jaccard": sorted(overlaps)[len(overlaps) // 2], "by_threshold": ov,
                                  "top_pairs": [{"prompt_id": r["prompt_id"], "jaccard": round(r["jaccard"], 3), "causal_q2": r["c"], "inert_q2": r["i"],
                                                 "causal_desc": desc[(r["prompt_id"], "causal")], "inert_desc": desc[(r["prompt_id"], "inert")]}
                                                for r in sorted(paired, key=lambda r: -r["jaccard"])[:14]]}

    # ---- judge validation by subset
    prim = {(v["prompt_id"], v["completion_index"], v["factor_kind"], v["segment"]): v for v in jl(RES / "judge_verdicts_v3.jsonl")}
    prim_v2 = {(v["prompt_id"], v["completion_index"], v["factor_kind"], v["segment"]): v for v in jl(RES / "judge_verdicts.jsonl")}

    def by_subset(val_rows, primary):
        groups = defaultdict(lambda: defaultdict(list))
        for v in val_rows:
            if not v.get("valid"):
                continue
            p = primary.get((v["prompt_id"], v["completion_index"], v["factor_kind"], v["segment"]))
            if p is None:
                continue
            sub = ("behavior_" + v["factor_kind"]) if v["behavior_present"] else "non_behavior"
            for q in ("q1", "q2", "q3"):
                groups[sub][q].append((p[q], v[q])); groups["all"][q].append((p[q], v[q]))
        res = {}
        for sub, qs in groups.items():
            res[sub] = {q: {"kappa": kappa([a for a, _ in pr], [b for _, b in pr]), "agreement": sum(a == b for a, b in pr) / len(pr), "n": len(pr),
                            "primary_yes": sum(a for a, _ in pr) / len(pr), "other_yes": sum(b for _, b in pr) / len(pr)} for q, pr in qs.items()}
        return res

    gpt_v3 = jl(RES / "judge_validation_gpt_v3.jsonl"); gpt_v2 = jl(RES / "judge_validation_gpt.jsonl")
    t1_v3 = jl(RES / "judge_validation_temp1_v3.jsonl")
    keyset = lambda rows: {(v["prompt_id"], v["completion_index"], v["factor_kind"], v["segment"]) for v in rows}
    out["validation"] = {"same_300_items_v2_v3": keyset(gpt_v3) == keyset(gpt_v2), "n_items": len(keyset(gpt_v3)),
                         "note": "The 300 validation items were sampled once (seed 42) and reused for v2 and v3; the v2 defect was diagnosed from Q2 disagreements on these items, so the v3 cross-family kappa is not an independent test of the redesigned prompt.",
                         "gpt_5_6_terra_vs_sonnet5_v3": by_subset(gpt_v3, prim),
                         "sonnet5_temp1_vs_temp0_v3": by_subset(t1_v3, prim),
                         "gpt_5_6_terra_vs_sonnet5_v2": by_subset(gpt_v2, prim_v2)}

    # ---- misread framing counts
    pat = re.compile(r"hypothetical|not applied|NOT used|was not used|edited version|modified version", re.I)
    fields = ("q1_rationale", "q2_rationale")
    out["misread_framing"] = {"regex": pat.pattern, "flags": "IGNORECASE", "fields": list(fields),
                              "v2_count": sum(1 for r in prim_v2.values() if any(pat.search(r.get(f) or "") for f in fields)),
                              "v3_count": sum(1 for r in prim.values() if any(pat.search(r.get(f) or "") for f in fields)),
                              "n_items": len(prim_v2)}

    # ---- slice contrasts with CIs
    s = json.load(open(RES / "summary.json"))

    def ci_overlap(a, b):
        return not (a["ci95"][1] < b["ci95"][0] or b["ci95"][1] < a["ci95"][0])
    sl = {}
    for name, ka, kb in (("slice_think_capped", "capped", "not_capped"), ("slice_edit_cleanliness", "large_single_factor", "minor_targeted"),
                         ("slice_is_mistake", "True", "False"), ("slice_n_turns", "multi-turn", "single-turn")):
        blk = s[name]
        if ka in blk and kb in blk:
            sl[name] = {ka: {"causal_q2": blk[ka]["causal_q2"], "ci95": blk[ka]["ci95"], "n_inv": blk[ka].get("n_inv"), "n": blk[ka]["n"]},
                        kb: {"causal_q2": blk[kb]["causal_q2"], "ci95": blk[kb]["ci95"], "n_inv": blk[kb].get("n_inv"), "n": blk[kb]["n"]},
                        "cis_overlap": ci_overlap(blk[ka], blk[kb])}
    out["slice_contrasts"] = sl
    out["reproducibility_n_fresh_rate_below_0.5"] = s["reproducibility"]["n_fresh_rate_below_0.5"]

    json.dump(out, open(RES / "control_diagnostics.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("description_overlap",)}, indent=1)[:6000])
    print(json.dumps(out["description_overlap"]["by_threshold"], indent=1))
    for p in out["description_overlap"]["top_pairs"][:10]:
        print(p["prompt_id"], p["jaccard"], p["causal_q2"], p["inert_q2"])
        print("  C:", p["causal_desc"][:200]); print("  I:", p["inert_desc"][:200])


if __name__ == "__main__":
    main()
