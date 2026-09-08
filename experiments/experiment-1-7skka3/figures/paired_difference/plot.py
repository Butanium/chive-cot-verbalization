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

diffs = D["diff"]
fig = go.Figure()
fig.add_bar(x=list(range(1, len(diffs) + 1)), y=diffs, marker_color=[C_CAUSAL if d > 0 else (C_INERT if d < 0 else C_NEUTRAL) for d in diffs],
            customdata=list(zip(D["prompt_id"], D["causal_q2"], D["inert_q2"], D["n_behavior"])),
            hovertemplate="%{customdata[0]}<br>causal %{customdata[1]:.2f}, inert %{customdata[2]:.2f}<br>Δ = %{y:+.2f} (n=%{customdata[3]})<extra></extra>", showlegend=False)
add_reference_line(fig, y=0, label="no difference")
add_reference_line(fig, y=D["mean"], label=f"mean +{D['mean']:.2f}")
fig.update_xaxes(title=f"Investigations with an inert control, sorted by difference ({len(diffs)})")
fig.update_yaxes(title="Attribution rate: causal factor − inert factor", range=[-1.05, 1.05])
apply_theme(fig, height=400)
save_figure_bundle(fig, HERE.name, data=D, root=HERE.parent,
                   alt="Sorted bar chart of per-investigation differences between causal-factor and inert-factor attribution rates, with zero and mean reference lines.")
