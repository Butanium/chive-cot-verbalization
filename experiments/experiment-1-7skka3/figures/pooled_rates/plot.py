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

rows = D["rows"]
fig = go.Figure()
for entity, col, legend in (("causal factor", C_CAUSAL, "causal factor"), ("inert factor", C_INERT, "inert factor"), ("visible answer", C_ANSWER, "visible answer, causal factor")):
    sel = [d for d in rows if d["entity"] == entity]
    fig.add_scatter(x=[d["rate"] for d in sel], y=[d["measure"] for d in sel], mode="markers", name=legend, marker=dict(color=col, size=11),
                    error_x=dict(type="data", symmetric=False, array=[d["hi"] - d["rate"] for d in sel],
                                 arrayminus=[d["rate"] - d["lo"] for d in sel], color=col, thickness=1.5),
                    customdata=[[d["lo"], d["hi"], d["n"], d["n_inv"]] for d in sel],
                    hovertemplate="%{y}<br>%{x:.1%} [%{customdata[0]:.1%}, %{customdata[1]:.1%}]<br>n=%{customdata[2]} completions, %{customdata[3]} investigations<extra></extra>")
fig.update_layout(legend=dict(orientation="h", y=1.08, x=0))
fig.update_yaxes(title="Judge question", categoryorder="array", categoryarray=[d["measure"] for d in rows], autorange="reversed")
fig.update_xaxes(title="Fraction of behavior-exhibiting completions", range=[0, 1], tickformat=".0%")
apply_theme(fig, height=440)
save_figure_bundle(fig, HERE.name, data=D, root=HERE.parent,
                   alt="Dot-and-interval plot of pooled judge rates among behavior-exhibiting completions with 95% cluster-bootstrap intervals.")
