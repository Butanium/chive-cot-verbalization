"""Standalone figure script: loads the co-located data.json, rebuilds the figure, saves the render."""
import json
from pathlib import Path

import plotly.graph_objects as go
from silico_figures import apply_theme, add_reference_line, save_figure_bundle

HERE = Path(__file__).resolve().parent
COL = json.load(open(HERE.parent / "entity_colors.json"))
C_CAUSAL, C_INERT, C_ANSWER = COL["causal factor"], COL["inert factor"], COL["visible answer"]
C_NEUTRAL = "#B7B1A6"
D = json.load(open(HERE / "data.json"))

slices = D["rows"]
ys = [f"{d['group']}: {d['label']}" for d in slices]
fig = go.Figure()
fig.add_scatter(x=[d["causal_q2"] for d in slices], y=ys, mode="markers", name="causal factor", marker=dict(color=C_CAUSAL, size=10),
                error_x=dict(type="data", symmetric=False, array=[d["hi"] - d["causal_q2"] for d in slices], arrayminus=[d["causal_q2"] - d["lo"] for d in slices], color=C_CAUSAL, thickness=1.5),
                hovertemplate="%{y}<br>causal Q2 %{x:.1%}<extra></extra>")
fig.add_scatter(x=[d["inert_q2_mean_per_inv"] for d in slices], y=ys, mode="markers", name="inert factor (mean of per-investigation rates)",
                marker=dict(color=C_INERT, size=8, symbol="diamond"), hovertemplate="%{y}<br>inert Q2 %{x:.1%}<extra></extra>")
add_reference_line(fig, x=D["overall_causal_q2"], label=f"all: {D['overall_causal_q2']:.0%}")
fig.update_xaxes(title="Attribution rate among behavior-exhibiting completions", range=[0, 1], tickformat=".0%")
fig.update_yaxes(title="Slice", autorange="reversed")
fig.update_layout(legend=dict(orientation="h", y=1.08, x=0))
apply_theme(fig, height=520)
save_figure_bundle(fig, HERE.name, data=D, root=HERE.parent,
                   alt="Dot-and-interval plot of the causal-factor attribution rate by edit type, effect size, conversation length, mistake status and thinking cap, with inert-factor means as diamonds.")
