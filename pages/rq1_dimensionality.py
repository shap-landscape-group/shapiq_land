"""
pages/rq1_dimensionality.py  —  RQ1: Dimensionality
"""
import os
import sys

import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared as S

dash.register_page(
    __name__,
    path="/rq1",
    name="RQ1 — Dimensionality",
    title="RQ1 — Dimensionality",
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV  = os.path.join(_HERE, "results", "rq1_dimensionality.csv")

# ─────────────────────────────────────────────────────────────────────────────
_RQ_HEADER = (
    "RQ1", "Dimensionality",
    "As a user with a dataset that has many features, I want to find the fastest "
    "model-agnostic library for high-dimensional datasets?",
)

_INTERP = (
    "Look for methods whose cost line stays flat as n_features grows (good scalability) "
    "while quality stays high. The failure heatmap reveals where methods break down entirely."
)

# ─────────────────────────────────────────────────────────────────────────────
#  Local helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pill(text, bg="#EEF2FF", color=S.ACCENT) -> html.Span:
    return html.Span(text, style={
        "display": "inline-block", "background": bg, "color": color,
        "fontSize": "11px", "fontWeight": "600",
        "padding": "3px 9px", "borderRadius": "4px",
        "marginRight": "5px", "marginBottom": "5px",
        "border": f"1px solid {color}40",
        "whiteSpace": "nowrap",
    })


def _col(heading, items, bg, color) -> html.Div:
    return html.Div([
        html.Div(heading, style={
            "fontSize": "10px", "fontWeight": "700", "color": S.TEXT2,
            "textTransform": "uppercase", "letterSpacing": "0.07em",
            "marginBottom": "8px",
        }),
        html.Div([_pill(i, bg, color) for i in items]),
    ], style={"flex": "1", "minWidth": "160px"})


def _fig_coverage(df) -> go.Figure:
    """Dataset × n_features heatmap — run count per cell."""
    if df.empty or "n_features" not in df.columns:
        return S.fig_empty("No data")
    counts = df.groupby(["dataset", "n_features"]).size().reset_index(name="n")
    pivot  = counts.pivot(index="dataset", columns="n_features", values="n").fillna(0)
    z      = pivot.values
    text   = [[f"{int(v)}" for v in row] for row in z]
    fig    = go.Figure(go.Heatmap(
        z=z,
        x=[str(int(c)) for c in pivot.columns],
        y=list(pivot.index),
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#EEF2FF"], [1, S.ACCENT]],
        showscale=False,
        hovertemplate=(
            "Dataset: <b>%{y}</b><br>"
            "n_features: <b>%{x}</b><br>"
            "Runs: %{z}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **S._CHART_LAYOUT,
        height=150,
        margin=dict(l=10, r=10, t=4, b=36),
        xaxis=dict(title="n_features", gridcolor="rgba(0,0,0,0)",
                   tickfont=dict(size=10)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True,
                   tickfont=dict(size=10)),
    )
    return fig


def _config_card(df) -> html.Div:
    """Compact benchmark overview — config + experiment coverage heatmap."""
    left  = _col("Swept",
                 ["3 datasets", "4 models", "n_features: 4–256",
                  "4 libraries", "2 approximators", "3 seeds"],
                 "#EEF2FF", S.ACCENT)
    mid   = _col("Fixed",
                 ["budget = 512", "n_background = 100",
                  "n_eval = 10", "imputer = marginal"],
                 "#F1F5F9", S.TEXT2)
    right = _col("Measured",
                 ["runtime_s", "n_model_evals",
                  "relative_mae", "mean_sample_rho"],
                 "#F0FDF4", S.GREEN)

    return html.Div([
        html.Div([
            html.Div("Benchmark at a glance", style={
                "fontSize": "13px", "fontWeight": "700", "color": S.TEXT,
                "marginBottom": "14px",
            }),
            html.Div([left, mid, right],
                     style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
        ], style={"flex": "1.4", "minWidth": "280px"}),
        html.Div([
            html.Div("Experiment coverage  (runs per cell)", style={
                "fontSize": "10px", "fontWeight": "700", "color": S.TEXT2,
                "textTransform": "uppercase", "letterSpacing": "0.07em",
                "marginBottom": "6px",
            }),
            dcc.Graph(
                figure=_fig_coverage(df),
                config={"displayModeBar": False},
                style={"height": "150px"},
            ),
        ], style={"flex": "1", "minWidth": "240px"}),
    ], style={
        "display": "flex", "gap": "32px", "flexWrap": "wrap",
        "background": S.CARD, "borderRadius": "12px",
        "border": f"1px solid {S.BORDER}", "padding": "20px 24px",
        "marginBottom": "20px",
    })


def _col_note(*parts: str) -> html.Div:
    """Compact data-provenance line inside a chart card."""
    children = []
    for i, part in enumerate(parts):
        children.append(html.Span(part, style={"whiteSpace": "pre"}))
        if i < len(parts) - 1:
            children.append(html.Span("  ·  ", style={"color": S.BORDER}))
    return html.Div(children, style={
        "fontSize": "10px", "color": S.TEXT2, "fontFamily": "monospace",
        "padding": "4px 12px 6px",
        "borderBottom": f"1px solid {S.BORDER}",
        "background": S.BG,
        "letterSpacing": "0.01em",
    })


def _axis_toggle(cid: str, options: dict, default: str) -> html.Div:
    """Compact inline axis-selector shown at the top of a chart card."""
    return html.Div([
        html.Span("Axis:", style={
            "fontSize": "11px", "fontWeight": "600", "color": S.TEXT2,
            "marginRight": "10px", "flexShrink": "0",
        }),
        dcc.RadioItems(
            id=cid,
            options=[{"label": v, "value": k} for k, v in options.items()],
            value=default,
            inline=True,
            inputStyle={"marginRight": "4px"},
            labelStyle={"marginRight": "16px", "fontSize": "12px",
                        "cursor": "pointer", "color": S.TEXT},
        ),
    ], style={
        "display": "flex", "alignItems": "center",
        "padding": "8px 12px",
        "borderBottom": f"1px solid {S.BORDER}",
        "background": S.BG,
        "borderRadius": "10px 10px 0 0",
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────────────────────────────

def layout(**kwargs):
    df, _ = S.try_load_data(_CSV)
    return html.Div([
        S.rq_header(*_RQ_HEADER),
        html.Div(id="rq1-kpis", style={"marginBottom": "20px"}),
        _config_card(df),
        S.section(
            "Scaling cost vs n_features",
            "Median cost per library \u00d7 approximator, seeds aggregated. "
            "Note: each dataset covers different n_features ranges \u2014 filter by dataset for a clean view.",
            html.Div([
                _axis_toggle("rq1-cost-metric",
                             {"runtime_s": "runtime (s)",
                              "n_model_evals": "model evaluations"},
                             "runtime_s"),
                _col_note(
                    "y \u2192 runtime_s  or  n_model_evals  (select above)",
                    "agg: median(seed) \u2192 grouped by library \u00d7 approximator \u00d7 n_features",
                ),
                html.Div(id="rq1-cost-chart", style={"padding": "8px"}),
            ]),
            section_id="rq1-cost-section",
        ),
        S.section(
            "Quality vs n_features",
            "Approximation quality vs dimensionality, seeds aggregated. "
            "Note: each dataset covers different n_features ranges \u2014 filter by dataset for a clean view.",
            html.Div([
                _axis_toggle("rq1-qual-metric",
                             {"mean_sample_rho": "Spearman \u03c1",
                              "relative_mae": "relative MAE"},
                             "mean_sample_rho"),
                _col_note(
                    "y \u2192 mean_sample_rho  or  relative_mae  (select above)",
                    "source: pairwise_metrics JSON, vs true_value reference backend",
                    "agg: median(seed) \u2192 grouped by library \u00d7 approximator \u00d7 n_features",
                ),
                html.Div(id="rq1-quality-chart", style={"padding": "8px"}),
            ]),
            section_id="rq1-quality-section",
        ),
        S.section(
            "Failure rate heatmap",
            "Fraction of failed runs per method \u00d7 feature count. "
            "Red cells mark where a method becomes unreliable.",
            html.Div([
                _col_note(
                    "is_failure = relative_mae > 1.0  OR  relative_mae is NaN",
                    "cell = mean(is_failure) over all seeds, models, datasets at that n_features",
                ),
                html.Div(id="rq1-failure-chart", style={"padding": "8px"}),
            ]),
            section_id="rq1-failure-section",
        ),
        S.interpretation_note(_INTERP),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Filter helper
# ─────────────────────────────────────────────────────────────────────────────

def _apply_filters(df, ds, mdl, nf_vals, approxs):
    if ds  != "__all__": df = df[df["dataset"]      == ds]
    if mdl != "__all__": df = df[df["model"]        == mdl]
    if approxs:          df = df[df["approximator"].isin(approxs)]
    if nf_vals:          df = df[df["n_features"].isin([float(v) for v in nf_vals])]
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Callback
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("rq1-kpis",         "children"),
    Output("rq1-cost-chart",    "children"),
    Output("rq1-quality-chart", "children"),
    Output("rq1-failure-chart", "children"),
    Input("rq1-ds",          "value"),
    Input("rq1-mdl",         "value"),
    Input("rq1-nf",          "value"),
    Input("rq1-approx",      "value"),
    Input("rq1-cost-metric", "value"),
    Input("rq1-qual-metric", "value"),
)
def update_rq1(ds, mdl, nf_vals, approxs, cost_metric, qual_metric):
    df, _ = S.try_load_data(_CSV)
    fdf   = _apply_filters(df, ds or "__all__", mdl or "__all__",
                           nf_vals or [], approxs or [])

    cost_metric = cost_metric or "runtime_s"
    qual_metric = qual_metric or "mean_sample_rho"

    # ── KPIs ──────────────────────────────────────────────────────────────
    unique_nf = fdf["n_features"].dropna().nunique() if not fdf.empty else 0
    max_nf    = int(fdf["n_features"].max()) if not fdf.empty and fdf["n_features"].notna().any() else 0
    n_methods = fdf["method"].nunique()      if not fdf.empty else 0
    success   = (1 - fdf["is_failure"].mean()) if not fdf.empty else 0

    kpis = S.kpi_row(
        
    )

    # ── Warnings ──────────────────────────────────────────────────────────
    if unique_nf < 2:
        warn = S.warning_note(
            "Select at least two n_features values to see scaling behaviour."
        )
        empty = dcc.Graph(figure=S.fig_empty(), config={"displayModeBar": False})
        return kpis, warn, empty, empty

    cost_chart = dcc.Graph(
        figure=S.fig_cost_vs_features(fdf, cost_metric),
        config={"displayModeBar": False},
    )
    qual_chart = dcc.Graph(
        figure=S.fig_quality_vs_features(fdf, qual_metric),
        config={"displayModeBar": False},
    )
    fail_chart = dcc.Graph(
        figure=S.fig_failure_heatmap_by_features(fdf),
        config={"displayModeBar": False},
    )

    return kpis, cost_chart, qual_chart, fail_chart


