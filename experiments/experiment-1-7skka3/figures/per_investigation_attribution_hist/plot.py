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
fig.add_bar(x=D["bin"], y=D["causal_factor"], name=f"causal factor (n={D['n_causal']})", marker_color=C_CAUSAL)
fig.add_bar(x=D["bin"], y=D["inert_factor"], name=f"inert factor (n={D['n_inert']})", marker_color=C_INERT, opacity=0.85)
fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.08, x=0))
fig.update_xaxes(title="Fraction of behavior-exhibiting samples whose reasoning attributes the behavior to the factor")
fig.update_yaxes(title="Number of investigations")
apply_theme(fig, height=420)
save_figure_bundle(fig, HERE.name, data=D, root=HERE.parent,
                   alt="Grouped histogram of per-investigation attribution rates for the causal factor and the inert factor; both distributions are spread across the whole range with a mode at 0–0.1.")
