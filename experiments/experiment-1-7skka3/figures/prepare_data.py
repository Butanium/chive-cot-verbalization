"""Write plot-ready data.json for each figure bundle from results/plot_data.json and results/summary.json.
Then run each figures/<name>/plot.py (standalone; loads its co-located data.json)."""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
plot = json.load(open(RES / "plot_data.json"))
summary = json.load(open(RES / "summary.json"))
inv = plot["per_investigation"]


def hist_counts(vals, edges):
    counts = [0] * (len(edges) - 1)
    for v in vals:
        for b in range(len(edges) - 1):
            if edges[b] <= v < edges[b + 1] or (b == len(edges) - 2 and v == edges[-1]):
                counts[b] += 1
                break
    return counts


def dump(name, data):
    (HERE / name).mkdir(exist_ok=True)
    json.dump(data, open(HERE / name / "data.json", "w"), indent=1)


# 1 histogram
edges = [i / 10 for i in range(11)]
labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(10)]
causal = [r["causal_q2"] for r in inv if r["causal_q2"] is not None]
inert = [r["inert_q2"] for r in inv if r["inert_q2"] is not None]
dump("per_investigation_attribution_hist", {"bin": labels, "causal_factor": hist_counts(causal, edges), "inert_factor": hist_counts(inert, edges),
                                            "n_causal": len(causal), "n_inert": len(inert)})

# 2 pooled rates
rows = [
    ("Reasoning attributes behavior to causal factor (Q2)", "causal_q2_attribution", "causal factor"),
    ("Reasoning attributes behavior to inert factor (Q2)", "inert_q2_attribution", "inert factor"),
    ("Visible answer attributes behavior to causal factor (Q2)", "answer_q2_attribution", "visible answer"),
    ("Reasoning mentions causal factor (Q1)", "causal_q1_mention", "causal factor"),
    ("Reasoning mentions inert factor (Q1)", "inert_q1_mention", "inert factor"),
    ("Reasoning mentions causal factor but does not attribute", "causal_mention_only", "causal factor"),
    ("Reasoning gives another reason, not the causal factor", "causal_alternative_without_attribution", "causal factor"),
    ("Reasoning gives no reason at all", "causal_silent", "causal factor"),
]
dump("pooled_rates", {"rows": [{"measure": n, "entity": e, "rate": summary[k]["rate"], "lo": summary[k]["ci95"][0], "hi": summary[k]["ci95"][1],
                                "n": summary[k]["n"], "n_inv": summary[k]["n_inv"]} for n, k, e in rows]})

# 3 paired difference
paired = sorted([r for r in inv if r["inert_q2"] is not None and r["causal_q2"] is not None], key=lambda r: r["causal_q2"] - r["inert_q2"])
dump("paired_difference", {"prompt_id": [r["prompt_id"] for r in paired], "diff": [r["causal_q2"] - r["inert_q2"] for r in paired],
                           "causal_q2": [r["causal_q2"] for r in paired], "inert_q2": [r["inert_q2"] for r in paired],
                           "n_behavior": [r["n_behavior"] for r in paired],
                           "mean": summary["paired_diff_causal_minus_inert"]["mean"], "ci95": summary["paired_diff_causal_minus_inert"]["ci95"]})

# 4 slices
slices = []
for key, title in (("slice_edit_cleanliness", "edit type"), ("slice_effect", "effect size (pp)"), ("slice_n_turns", "conversation"), ("slice_is_mistake", "is a mistake")):
    for k, s in summary[key].items():
        nice = {"large_single_factor": "large single-factor edit", "minor_targeted": "minor targeted edit", "True": "mistake", "False": "not a mistake"}.get(k, k)
        slices.append({"group": title, "label": f"{nice} ({s['n_inv']} inv)", "causal_q2": s["causal_q2"], "lo": s["ci95"][0], "hi": s["ci95"][1],
                       "inert_q2_mean_per_inv": s["inert_q2_mean_per_inv"], "n": s["n"], "n_inv": s["n_inv"]})
for k, s in summary["slice_think_capped"].items():
    slices.append({"group": "thinking cap", "label": f"{'thinking hit 1024-token cap' if k == 'capped' else 'thinking ended naturally'} ({s['n']} compl.)",
                   "causal_q2": s["causal_q2"], "lo": s["ci95"][0], "hi": s["ci95"][1], "inert_q2_mean_per_inv": None, "n": s["n"], "n_inv": None})
dump("slices", {"rows": slices, "overall_causal_q2": summary["causal_q2_attribution"]["rate"]})

# 5 reproducibility
dump("fresh_vs_original_behavior_rate", {"prompt_id": [r["prompt_id"] for r in inv], "orig_rate": [r["orig_rate"] for r in inv],
                                         "fresh_rate": [r["fresh_rate"] for r in inv], "pearson_r": summary["reproducibility"]["pearson_r"]})

for name in ("per_investigation_attribution_hist", "pooled_rates", "paired_difference", "slices", "fresh_vs_original_behavior_rate"):
    subprocess.run([sys.executable, str(HERE / name / "plot.py")], check=True)
print("done")
