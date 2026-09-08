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

fig = go.Figure()
fig.add_scatter(x=D["orig_rate"], y=D["fresh_rate"], mode="markers", marker=dict(color=C_CAUSAL, size=8, opacity=0.65),
                text=D["prompt_id"], hovertemplate="%{text}<br>original %{x:.2f}, fresh %{y:.2f}<extra></extra>", showlegend=False)
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C_NEUTRAL, dash="dot", width=1), showlegend=False, hoverinfo="skip")
fig.update_xaxes(title="Behavior rate in the released 30 samples", range=[0.45, 1.02])
fig.update_yaxes(title="Behavior rate in our fresh 30 samples", range=[0, 1.02])
apply_theme(fig, height=420)
save_figure_bundle(fig, HERE.name, data=D, root=HERE.parent,
                   alt="Scatter of fresh versus released per-investigation behavior rates for 100 investigations with the identity line.")
