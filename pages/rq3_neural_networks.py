"""
pages/rq3_neural_networks.py  —  RQ3: Neural Networks

Edit the PAGE CONFIGURATION block below to change what is shown.
The layout / filter / callback logic beneath it should rarely need touching.

Data file: results_config-neural-networks-RQ3.csv (falls back to results_nn.csv).
No fallback to results.csv because the NN benchmark is architecturally distinct.
"""
import os
import sys

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared as S

dash.register_page(
    __name__,
    path="/rq3",
    name="RQ3 — Neural Networks",
    title="RQ3 — Neural Networks",
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV  = os.path.join(_HERE, "results", "rq3_neural_networks.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  Local helpers — config card
# ─────────────────────────────────────────────────────────────────────────────

def _pill(text: str, bg: str, color: str) -> html.Span:
    return html.Span(text, style={
        "fontSize": "11px", "fontWeight": "500", "color": color,
        "background": bg, "borderRadius": "4px",
        "padding": "2px 8px", "marginRight": "4px", "marginBottom": "4px",
        "display": "inline-block",
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
    """Dataset \u00d7 model heatmap \u2014 run count per cell."""
    if df.empty or "model" not in df.columns:
        return S.fig_empty("No data")
    counts = df.groupby(["dataset", "model"]).size().reset_index(name="n")
    pivot  = counts.pivot(index="dataset", columns="model", values="n").fillna(0)
    z      = pivot.values
    text   = [[f"{int(v)}" for v in row] for row in z]
    fig    = go.Figure(go.Heatmap(
        z=z, x=list(pivot.columns), y=list(pivot.index),
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#EEF2FF"], [1, S.ACCENT]],
        showscale=False,
        hovertemplate="Dataset: <b>%{y}</b><br>Model: <b>%{x}</b><br>Runs: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **S._CHART_LAYOUT, height=150,
        margin=dict(l=10, r=10, t=4, b=36),
        xaxis=dict(title="model type", gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True, tickfont=dict(size=10)),
    )
    return fig


def _config_card(df) -> html.Div:
    """Compact benchmark overview \u2014 config + experiment coverage heatmap."""
    left  = _col("Swept",
                 ["3 datasets", "3 model types", "6 libraries",
                  "7 approximators", "3 seeds (10 planned)"],
                 "#EEF2FF", S.ACCENT)
    mid   = _col("Fixed",
                 ["budget = 512", "n_background = 200",
                  "n_eval = 100", "imputer = marginal", "GPU (CUDA)"],
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
            dcc.Graph(figure=_fig_coverage(df), config={"displayModeBar": False},
                      style={"height": "150px"}),
        ], style={"flex": "1", "minWidth": "240px"}),
    ], style={
        "display": "flex", "gap": "32px", "flexWrap": "wrap",
        "background": S.CARD, "borderRadius": "12px",
        "border": f"1px solid {S.BORDER}", "padding": "20px 24px",
        "marginBottom": "20px",
    })


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION — edit here to change what the page shows
# ═════════════════════════════════════════════════════════════════════════════

_RQ_HEADER = (
    "RQ3", "Neural Networks",
    "As a user with a neural network, I want to know which library is the fastest?"
)

# Notes shown on the page directly below the research question.
_REMARKS = [
    "Benchmark: 3 models (MLP, Transformer, CNN-1D) \u00d7 3 datasets \u00d7 6 libraries \u2014 3 seeds (10 planned).",
    "Budget fixed at 512. GPU (CUDA) backend. n_background\u202f=\u202f200, n_eval\u202f=\u202f100.",
    "Open question: ShapIQ supports ProxySHAP \u2014 would this also be interesting?",
]


def _agg(df):
    """Aggregate raw runs to one row per method — used by the Pareto chart."""
    if df.empty:
        return df
    return (
        df.groupby(["method", "library", "approximator"])
        .agg(
            rho_median=    ("mean_sample_rho", "median"),
            mae_median=    ("relative_mae",    "median"),
            runtime_median=("runtime_s",       "median"),
            sign_median=   ("sign_agreement",  "median"),
            failure_rate=  ("is_failure",      "mean"),
        )
        .reset_index()
    )


# Charts rendered in order.
#   title      — section heading shown on the page
#   subtitle   — one-line description below the heading
#   fn         — callable(df) -> Plotly Figure  (use lambdas for extra args or pre-aggregation)
#   section_id — stable DOM id for advisor deep-links and TOC anchors
_CHARTS = [
    dict(
        section_id = "rq3-runtime-section",
        title      = "Runtime Ranking — Fastest to Slowest",
        subtitle   = "Bars sorted by median runtime. "
                     "The shortest bar is the fastest library for your neural network.",
        fn         = S.fig_runtime_ranking,
    ),
    dict(
        section_id = "rq3-boxplot-section",
        title      = "Runtime Distribution per Library",
        subtitle   = "Box plots on a log scale show the spread. "
                     "Wide boxes or high outliers indicate inconsistent performance.",
        fn         = S.fig_runtime_boxplots,
    ),
    dict(
        section_id = "rq3-scatter-section",
        title      = "Speed vs Accuracy",
        subtitle   = "Does being fast come at the cost of accuracy? "
                     "Each dot is one benchmark run. Ideally: fast AND high ρ.",
        fn         = S.fig_rho_vs_runtime,
    ),
    dict(
        section_id = "rq3-pareto-section",
        title      = "Pareto Frontier — Speed vs Accuracy",
        subtitle   = "Colored = Pareto-optimal (no other method is both faster AND more accurate). "
                     "Gray = dominated.",
        fn         = lambda df: S.fig_pareto(_agg(df)),
    ),
    dict(
        section_id = "rq3-failure-section",
        title      = "Failure Rate by Model Type",
        subtitle   = "Fraction of failed runs per method \u00d7 model type. "
                     "Red cells mark where a method becomes unreliable for a specific architecture.",
        fn         = S.fig_failure_vs_complexity,
    ),
]

_INTERP = (
    "How to read this page: if runtime is your only concern, pick the method with the shortest "
    "bar in the ranking chart. Then check the Speed vs Accuracy scatter to confirm it doesn't "
    "sacrifice too much quality for that speed gain."
)


def layout(**kwargs):
    df, src = S.try_load_data(_CSV)

    if src is None:
        return html.Div([
            S.rq_header(*_RQ_HEADER),
            S.missing_data_banner(_CSV),
            _schema_hint(),
        ])

    datasets = [{"label": "All datasets", "value": "__all__"}] + \
               [{"label": d, "value": d} for d in sorted(df["dataset"].dropna().unique())]
    models   = sorted(df["model"].dropna().unique()) if not df.empty else []
    libs     = sorted(df["library"].dropna().unique())

    return html.Div([
        S.rq_header(*_RQ_HEADER),
        _config_card(df),

        # ── Dynamic content ───────────────────────────────────────────────────
        html.Div(id="rq3-kpis"),
        html.Div(id="rq3-charts"),
    ])


def _schema_hint() -> html.Div:
    """Hint shown when data file is missing."""
    rows = [
        ("dataset",    "name of the NN benchmark dataset"),
        ("model",      "neural network type: mlp, resnet50, bert-tiny, etc."),
        ("library",    "explanation library: shap, captum, shapiq, etc."),
        ("approximator", "method used: kernel, gradient, deep, etc."),
        ("runtime_s",  "wall-clock seconds"),
        ("relative_mae", "approximation error vs exact values (optional but recommended)"),
        ("sign_agreement / mean_sample_rho", "accuracy proxies (optional)"),
    ]
    th_s = {"fontSize": "10px", "fontWeight": "600", "color": S.TEXT2,
            "textTransform": "uppercase", "letterSpacing": "0.05em",
            "padding": "8px 12px", "borderBottom": f"2px solid {S.BORDER}",
            "textAlign": "left", "background": S.BG}
    td_s = {"fontSize": "12px", "padding": "8px 12px",
            "borderBottom": f"1px solid {S.BORDER}"}
    return S.section(
        "Expected CSV Schema",
        f"Create results_nn.csv with at least the columns below to populate this page.",
        html.Table(
            [
                html.Thead(html.Tr([html.Th("Column", style=th_s),
                                    html.Th("Description", style=th_s)])),
                html.Tbody([
                    html.Tr([
                        html.Td(col, style={**td_s, "fontFamily": "monospace",
                                            "color": S.ACCENT}),
                        html.Td(desc, style=td_s),
                    ])
                    for col, desc in rows
                ]),
            ],
            style={"width": "100%", "borderCollapse": "collapse"},
        ),
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Callback
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("rq3-kpis",   "children"),
    Output("rq3-charts", "children"),
    Input("rq3-ds",     "value"),
    Input("rq3-model",  "value"),
    Input("rq3-lib",    "value"),
)
def update_rq3(ds, models, libs):
    df, src = S.try_load_data(_CSV)
    if src is None or df.empty:
        return html.Div(), html.Div()

    if ds != "__all__": df = df[df["dataset"] == ds]
    if models:          df = df[df["model"].isin(models)]
    if libs:            df = df[df["library"].isin(libs)]

    lb = S.compute_leaderboard(df)

    # ── KPIs ──────────────────────────────────────────────────────────────
    fastest_m   = lb.loc[lb["runtime_median"].idxmin(), "method"] if not lb.empty else "—"
    fastest_rt  = lb["runtime_median"].min()                       if not lb.empty else float("nan")
    n_libs      = df["library"].dropna().nunique()                 if not df.empty else 0
    fail_of_fast = (
        lb.loc[lb["runtime_median"].idxmin(), "failure_rate"] * 100
        if not lb.empty else float("nan")
    )

    kpis = S.kpi_row(
      )

    warns = []
    if df.empty:
        warns.append(S.warning_note("No data matches the current filter selection."))

    # ── Charts — driven by _CHARTS manifest above ──────────────────────────
    charts = html.Div([
        *warns,
        *[
            S.section(
                c["title"], c["subtitle"],
                dcc.Graph(figure=c["fn"](df),
                          config={"displayModeBar": False}, style={"padding": "8px"}),
                section_id=c["section_id"],
            )
            for c in _CHARTS
        ],
        S.interpretation_note(_INTERP),
    ])

    return kpis, charts
