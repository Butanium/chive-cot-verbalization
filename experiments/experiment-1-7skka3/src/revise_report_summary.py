"""Post-review revision of report_summary.json and claims_manifest.json.

Reframes the headline around the paired causal-minus-inert difference, adds the inert-control
diagnostics (zero-effect subset, same-span overlap, per-investigation inert distribution), the
validation-set reuse and subset kappas, and restates slice contrasts with their overlapping CIs.
All numbers come from results/summary.json and results/control_diagnostics.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
P = "experiments/experiment-1-7skka3/results/"
d = json.load(open(RES / "control_diagnostics.json"))
s = json.load(open(RES / "summary.json"))
r = json.load(open(ROOT / "report_summary.json"))

z = d["zero_effect_inert_subset"]; cv = d["causal_vs_inert_per_inv"]; tails = d["per_investigation_tails"]
ov3 = d["description_overlap"]["by_threshold"]["0.3"]; ov4 = d["description_overlap"]["by_threshold"]["0.4"]
vg = d["validation"]["gpt_5_6_terra_vs_sonnet5_v3"]; vt = d["validation"]["sonnet5_temp1_vs_temp0_v3"]
pct = lambda x: f"{100 * x:.1f}"

r["title"] = "Qwen3-8B reasoning names the verified cause of its behavior only 10.7 pp more often than an inert factor; the 42% raw attribution rate is an upper bound"
r["headline"] = (
    "Across 100 Qwen3-8B thinking-mode behaviors whose cause was pinned down by a verified counterfactual (96 of them mistakes), "
    f"the reasoning trace attributes the behavior to that causal factor in {pct(s['causal_q2_attribution']['rate'])}% of behavior-exhibiting samples "
    f"(95% cluster-bootstrap CI {pct(s['causal_q2_attribution']['ci95'][0])}–{pct(s['causal_q2_attribution']['ci95'][1])}%, 2163 completions). "
    f"The same judgment for an inert factor from the same conversation (an edit that did not change the behavior) yields {pct(s['inert_q2_attribution']['rate'])}% "
    f"(CI {pct(s['inert_q2_attribution']['ci95'][0])}–{pct(s['inert_q2_attribution']['ci95'][1])}%, 87 investigations), and {pct(s['inert_by_abs_effect']['0']['inert_q2'])}% "
    f"(CI {pct(s['inert_by_abs_effect']['0']['ci95'][0])}–{pct(s['inert_by_abs_effect']['0']['ci95'][1])}%) on the 26 investigations whose inert factor had exactly 0 pp effect, "
    f"against {pct(s['inert_by_abs_effect']['0']['causal_q2_same_investigations'])}% for the causal factor on those same investigations. "
    f"The supportable claim is therefore the paired one: attribution to the verified cause exceeds attribution to an inert factor by +{100 * s['paired_diff_causal_minus_inert']['mean']:.1f} pp "
    f"(CI {100 * s['paired_diff_causal_minus_inert']['ci95'][0]:.1f}–{100 * s['paired_diff_causal_minus_inert']['ci95'][1]:.1f}) pooled over 87 pairs, and by "
    f"+{100 * z['paired_diff_mean']:.1f} pp (CI {100 * z['paired_diff_ci95'][0]:.1f}–{100 * z['paired_diff_ci95'][1]:.1f}) on the zero-effect subset. "
    f"Causal and inert per-investigation rates correlate (r = {cv['pearson_r']:.2f}) and the inert rate is at least as high as the causal rate in {cv['n_inert_ge_causal']} of 87 pairs, "
    "so the 42% is an upper bound on how often the trace verbalizes the verified cause: much of it would be scored for any salient factor in the conversation. "
    f"Per investigation, the causal rate is spread over the whole range ({pct(tails['causal']['frac_eq_0'])}% of investigations never attribute to the cause, {pct(tails['causal']['frac_eq_1'])}% always do); "
    f"the inert distribution is lower ({pct(tails['inert']['frac_eq_0'])}% at zero, {pct(tails['inert']['frac_eq_1'])}% at one) but 3 of the 15 'always' investigations also attribute to their inert factor in every sample. "
    f"In {pct(s['causal_alternative_without_attribution']['rate'])}% of behavior-exhibiting samples the reasoning gives a different reason and does not name the verified cause."
)
r["assessment_rationale"] = (
    "The primary measurement (pooled Q2 rate with a cluster-bootstrap CI) was obtained with a judge whose pooled cross-family Q2 kappa (0.468) clears the pre-registered 0.4 "
    f"threshold and on the behavior/causal subset alone (kappa {vg['behavior_causal']['q2']['kappa']:.3f}, n = {vg['behavior_causal']['q2']['n']}), and the behavior reproduces "
    "(fresh rate 72.1% vs 76.8% original, r = 0.79). The plan pre-registered that an inert control not near zero makes the absolute rate uninterpretable as a verbalization rate; "
    f"the control came out at {pct(s['inert_q2_attribution']['rate'])}% ({pct(s['inert_by_abs_effect']['0']['inert_q2'])}% for zero-effect inert factors), so the headline is restated "
    "around the paired difference (+10.7 pp, CI 2.4–19.2) and the 42% is reported as an upper bound. The plan's refuting pattern for 'mostly unverbalized' (causal mass above 0.7 with control near zero) did not occur. "
    "Three things inflate the control and are not separable with this design: same-span inert edits whose v3 descriptions overlap the causal description "
    f"({ov3['n_high_overlap']} to {ov4['n_high_overlap']} of 87 pairs by description-overlap threshold; their mean paired difference is +{100 * ov3['paired_diff_high']:.1f} to +{100 * ov4['paired_diff_high']:.1f} pp vs "
    f"+{100 * ov3['paired_diff_rest']:.1f} pp on the rest), judge false positives (cross-family Q2 kappa on the behavior/inert subset is {vg['behavior_inert']['q2']['kappa']:.2f}, below 0.4; "
    f"temperature-1 self-agreement {vt['behavior_inert']['q2']['kappa']:.2f} vs {vt['behavior_causal']['q2']['kappa']:.2f} on causal), and traces that cite a non-causal reason as their reason. "
    "The v3 judge prompt was redesigned after the as-planned v2 prompt failed the agreement check, and the 300-item validation set used to pass v3 is the same set on which the v2 defect was diagnosed, so the v3 kappa is not an independent test. "
    "Verdict: partial_signal. Verbalized reasons discriminate the verified cause from an inert factor, but weakly, and the absolute rate is judge- and control-dependent."
)
r["decision_rule"]["outcome"] = (
    f"Q2 causal {pct(s['causal_q2_attribution']['rate'])}% [{pct(s['causal_q2_attribution']['ci95'][0])}, {pct(s['causal_q2_attribution']['ci95'][1])}]; inert {pct(s['inert_q2_attribution']['rate'])}% "
    f"[{pct(s['inert_q2_attribution']['ci95'][0])}, {pct(s['inert_q2_attribution']['ci95'][1])}]; zero-effect inert {pct(s['inert_by_abs_effect']['0']['inert_q2'])}%. Pooled cross-family Q2 kappa 0.468 (passes on pooled items and on the behavior/causal subset, 0.467; "
    f"fails on the behavior/inert subset, {vg['behavior_inert']['q2']['kappa']:.3f}). The inert control is not near zero, so by the pre-registered rule the absolute 42% is not interpretable as a verbalization rate; "
    "the headline is the paired difference (+10.7 pp [2.4, 19.2]; +2.9 pp [-13.9, 18.5] on the zero-effect subset) with 42% as an upper bound. Q1 vs Q2 gap (65.8% vs 32.6% for inert) shows the judge does separate mention from attribution, "
    "but the elevated control mixes same-span inert edits, judge false positives, and verbalized non-causal reasons. Causal mass above 0.7: 35% of investigations; 'mostly unverbalized' is neither refuted nor confirmed."
)
mf = d["misread_framing"]
r["instrument_correction"]["as_planned_v2"] = r["instrument_correction"]["as_planned_v2"].replace(
    "735/5610 rationales framed", f"{mf['v2_count']}/5610 items had a Q1 or Q2 rationale that framed").replace(", 468 of them with Q1=no/Q2=no", "")
r["instrument_correction"]["corrected_v3"] = r["instrument_correction"]["corrected_v3"].replace(
    "Misread framing 735 → 29 rationales.", f"Misread framing {mf['v2_count']} → {mf['v3_count']} items (regex over Q1 and Q2 rationales).").replace(
    "Cross-family Q2 kappa 0.468, self-agreement 0.750.",
    f"Cross-family Q2 kappa 0.468 pooled (behavior/causal {vg['behavior_causal']['q2']['kappa']:.3f}, behavior/inert {vg['behavior_inert']['q2']['kappa']:.3f}, non-behavior {vg['non_behavior']['q2']['kappa']:.3f}); self-agreement 0.750.")
r["instrument_correction"]["post_hoc"] = (
    "The v3 correction was made after seeing v2 results and the validation failure; the scoring rule (Q2 pooled rate) is unchanged. "
    "The 300 validation items (seed 42) are identical for v2 and v3, and the v2 defect was diagnosed from Q2 disagreements on these items, "
    "so the v3 kappa of 0.468 is a pass on the set the redesign was tuned against, not an independent validation."
)
r["methods"]["validation"] += (
    " The same 300 items were used for the v2 and v3 judges; the v2 defect was diagnosed from Q2 disagreements on them. "
    "Kappas are also reported by subset (behavior/causal n = 124, behavior/inert n = 95, non-behavior n = 81; results/control_diagnostics.json)."
)
r["methods"]["data"] += (
    " The inert claim is chosen by smallest |effect| only; it is not required to target a different span of the prompt than the causal claim, "
    f"and by content-word Jaccard of the v3 descriptions {ov3['n_high_overlap']} pairs overlap at >= 0.3 and {ov4['n_high_overlap']} at >= 0.4."
)
r["methods"]["analysis"] += " Post-review diagnostics (src/control_diagnostics.py): zero-effect inert subset, causal vs inert per-investigation correlation, description overlap, subset kappas, slice CI overlap."

r["results"]["control_diagnostics"] = {
    "zero_effect_inert_subset": {"n_inv": z["n_inv"], "inert_q2_pooled": round(s["inert_by_abs_effect"]["0"]["inert_q2"], 4), "inert_ci95": [round(x, 4) for x in s["inert_by_abs_effect"]["0"]["ci95"]],
                                 "causal_q2_same_inv_pooled": round(s["inert_by_abs_effect"]["0"]["causal_q2_same_investigations"], 4),
                                 "paired_diff_mean": round(z["paired_diff_mean"], 4), "paired_diff_ci95": [round(x, 4) for x in z["paired_diff_ci95"]]},
    "causal_vs_inert_per_inv": {"pearson_r": round(cv["pearson_r"], 3), "n_inert_ge_causal": cv["n_inert_ge_causal"], "n_inert_gt_causal": cv["n_inert_gt_causal"], "n_equal": cv["n_equal"], "n_pairs": 87},
    "per_investigation_inert_q2": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tails["inert"].items()},
    "causal_eq_1_and_inert_eq_1": tails["causal_eq_1_and_inert_eq_1"],
    "description_overlap": {"metric": d["description_overlap"]["metric"],
                            "jaccard_ge_0.3": {"n": ov3["n_high_overlap"], "paired_diff_high": round(ov3["paired_diff_high"], 4), "paired_diff_rest": round(ov3["paired_diff_rest"], 4)},
                            "jaccard_ge_0.4": {"n": ov4["n_high_overlap"], "paired_diff_high": round(ov4["paired_diff_high"], 4), "paired_diff_rest": round(ov4["paired_diff_rest"], 4)}},
    "validation_by_subset_v3": {sub: {"q2_kappa_gpt": round(vg[sub]["q2"]["kappa"], 3), "q2_kappa_temp1": round(vt[sub]["q2"]["kappa"], 3), "n": vg[sub]["q2"]["n"]}
                                for sub in ("behavior_causal", "behavior_inert", "non_behavior", "all")},
    "validation_same_300_items_v2_v3": d["validation"]["same_300_items_v2_v3"],
    "misread_framing": {"v2": mf["v2_count"], "v3": mf["v3_count"], "fields": mf["fields"], "regex": mf["regex"]},
    "slice_cis_overlap": {k: v["cis_overlap"] for k, v in d["slice_contrasts"].items()},
    "n_fresh_rate_below_0.5": d["reproducibility_n_fresh_rate_below_0.5"],
}

r["limitations"] = [
    "The inert control is not a 'nothing' control and is elevated (32.6% pooled; 40.6% for inert factors with exactly 0 pp effect, against 42.1% causal on the same 26 investigations). "
    "Three sources are mixed and cannot be separated with this design: (a) same-span inert edits: the inert claim is selected by smallest |effect| without requiring a different span, and 8 to 14 of 87 pairs have "
    "v3 descriptions that overlap heavily with the causal description (e.g. prompt_002095: first-person pronoun vs the content of the same quoted sentence; both scored 1.0), with mean paired difference +3.7 to +7.3 pp vs +11.3 pp on the rest; "
    "(b) judge false positives: cross-family Q2 kappa on behavior/inert items is 0.36 (below the 0.4 threshold) and temperature-1 self-agreement is 0.60 vs 0.82 on causal items; "
    "(c) traces that present a non-causal aspect of the prompt as their reason (e.g. prompt_004403 citing an 'Nl' token as the reason for answering in Spanish). "
    "The 42% is therefore an upper bound on verbalizing the verified cause, and the inert rate and paired difference rest partly on judgments below the pre-registered agreement threshold.",
    "The judge was redesigned (v2 → v3) after the as-planned prompt failed the cross-family agreement check, and the same 300 validation items were reused to diagnose the defect and to pass v3, so the v3 kappa (0.468 pooled, 0.467 on behavior/causal) is not an independent validation. "
    "Both versions are reported; the headline moved by +2.2 pp. The second-family judge is more liberal (40% vs 35% yes on the validation sample), so the absolute rate carries judge-dependent uncertainty of several points.",
    "Selection favors large, clean, reproducible effects (|effect| ≥ 50 pp, verification ≥ 9, baseline ≥ 50%) and mistakes (96/100); rates may differ for subtler behaviors and non-mistakes (4 investigations only). "
    "21 of the 100 investigations reproduced the behavior in fewer than half of fresh samples.",
    "Slice contrasts are not distinguishable at this run's noise level: think-capped 53.8% [39.4, 67.8] vs uncapped 37.8% [29.2, 46.6] (completion-level split, confounded with investigation identity), "
    "large_single_factor 53.1% [38.6, 67.1] vs minor_targeted 38.5% [29.6, 47.7], mistakes 41.8% [33.7, 50.1] vs non-mistakes 46.9% [13.0, 64.7] on 4 investigations, multi-turn 44.5% [34.3, 54.8] vs single-turn 38.6% [27.0, 51.7]. All intervals overlap.",
    "One model (Qwen3-8B), one thinking budget (1024 tokens, matching the released run; 27% of traces hit the cap). The paper's investigator wrote the claims after reading traces from the same prompts, and SGLang rather than vLLM served the model (token-level settings matched; cap and length profile reproduce).",
]
r["artifacts"]["control_diagnostics"] = P + "control_diagnostics.json"
r["artifacts"]["code"] += ["experiments/experiment-1-7skka3/src/control_diagnostics.py"]
json.dump(r, open(ROOT / "report_summary.json", "w"), indent=1, ensure_ascii=False)

# ---------------- claims manifest
c = json.load(open(ROOT / "claims_manifest.json"))
claims = c["claims"]
CD = P + "control_diagnostics.json"
for cl in claims:
    if cl["claim"].startswith("735 of 5610"):
        cl["claim"] = f"{mf['v2_count']} of 5610 v2 items and {mf['v3_count']} of 5610 v3 items have a Q1 or Q2 rationale matching the edit-misreading regex"
        cl["source"] = CD; cl["key"] = "misread_framing.v2_count, misread_framing.v3_count (regex and fields recorded there)"
        cl["value"] = [mf["v2_count"], mf["v3_count"]]
    if cl["claim"].startswith("Think-capped completions"):
        cl["claim"] = "Think-capped completions 53.8% [39.4, 67.8] vs uncapped 37.8% [29.2, 46.6]; CIs overlap (completion-level split)"
    if cl["claim"].startswith("large_single_factor (23 inv)"):
        cl["claim"] = "large_single_factor (23 inv) 53.1% [38.6, 67.1] vs minor_targeted (77 inv) 38.5% [29.6, 47.7]; CIs overlap"
    if cl["claim"].startswith("Mistakes (96 inv)"):
        cl["claim"] = "Mistakes (96 inv) 41.8% [33.7, 50.1] vs non-mistakes (4 inv) 46.9% [13.0, 64.7]; CIs overlap"
    if cl["claim"].startswith("Multi-turn"):
        cl["claim"] = "Multi-turn 44.5% [34.3, 54.8] vs single-turn 38.6% [27.0, 51.7]; CIs overlap"
new = [
    {"claim": "Zero-effect inert factors (26 inv, 643 completions): inert Q2 40.6% [25.6, 56.3] vs causal Q2 42.1% on the same investigations", "source": P + "summary.json", "key": "inert_by_abs_effect.0", "value": [0.4059, [0.2562, 0.5625], 0.4215, 26, 643]},
    {"claim": "Paired difference on the zero-effect subset +2.9 pp [-13.9, 18.5] (26 inv)", "source": CD, "key": "zero_effect_inert_subset.paired_diff_mean, paired_diff_ci95", "value": [round(z["paired_diff_mean"], 4), [round(x, 4) for x in z["paired_diff_ci95"]]]},
    {"claim": "Causal and inert per-investigation Q2 correlate at r = 0.45; inert >= causal in 43 of 87 pairs (20 strictly greater, 23 equal)", "source": CD, "key": "causal_vs_inert_per_inv", "value": [round(cv["pearson_r"], 3), cv["n_inert_ge_causal"], cv["n_inert_gt_causal"], cv["n_equal"]]},
    {"claim": "Per-investigation inert Q2: median 0.13, 33.3% at 0, 6.9% at 1, 20.7% >= 0.7, 63.2% <= 0.3 (87 inv)", "source": CD, "key": "per_investigation_tails.inert", "value": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tails["inert"].items()}},
    {"claim": "3 of the 15 investigations with causal Q2 = 1.0 also have inert Q2 = 1.0 (prompt_001863, prompt_002095, prompt_004773)", "source": CD, "key": "per_investigation_tails.causal_eq_1_and_inert_eq_1", "value": tails["causal_eq_1_and_inert_eq_1"]},
    {"claim": "Description overlap (content-word Jaccard of v3 descriptions): 14 pairs >= 0.3 with mean paired diff +7.3 pp vs +11.3 pp on the rest; 8 pairs >= 0.4 with +3.7 pp vs +11.4 pp", "source": CD, "key": "description_overlap.by_threshold.0.3, description_overlap.by_threshold.0.4",
     "value": [ov3["n_high_overlap"], round(ov3["paired_diff_high"], 4), round(ov3["paired_diff_rest"], 4), ov4["n_high_overlap"], round(ov4["paired_diff_high"], 4), round(ov4["paired_diff_rest"], 4)]},
    {"claim": "Cross-family Q2 kappa by subset (v3): behavior/causal 0.467 (n=124), behavior/inert 0.360 (n=95), non-behavior 0.529 (n=81)", "source": CD, "key": "validation.gpt_5_6_terra_vs_sonnet5_v3.<subset>.q2.kappa",
     "value": {sub: round(vg[sub]["q2"]["kappa"], 3) for sub in ("behavior_causal", "behavior_inert", "non_behavior")}},
    {"claim": "Temperature-1 self-agreement Q2 kappa by subset (v3): behavior/causal 0.822, behavior/inert 0.598", "source": CD, "key": "validation.sonnet5_temp1_vs_temp0_v3.<subset>.q2.kappa",
     "value": {sub: round(vt[sub]["q2"]["kappa"], 3) for sub in ("behavior_causal", "behavior_inert")}},
    {"claim": "The 300 validation items are identical for the v2 and v3 judges", "source": CD, "key": "validation.same_300_items_v2_v3", "value": True},
    {"claim": "21 of 100 investigations reproduced the behavior in fewer than half of fresh samples", "source": P + "summary.json", "key": "reproducibility.n_fresh_rate_below_0.5", "value": 21},
    {"claim": "All four slice contrasts (think-capped, edit cleanliness, is_mistake, n_turns) have overlapping 95% CIs", "source": CD, "key": "slice_contrasts.<slice>.cis_overlap", "value": {k: v["cis_overlap"] for k, v in d["slice_contrasts"].items()}},
]
claims.extend(new)
json.dump(c, open(ROOT / "claims_manifest.json", "w"), indent=1, ensure_ascii=False)
print("claims:", len(claims))
