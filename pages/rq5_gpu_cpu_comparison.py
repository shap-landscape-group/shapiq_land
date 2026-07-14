"""
pages/rq5_gpu_cpu_comparison.py  —  RQ5: GPU vs CPU
"""
import os
import sys

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared as S

dash.register_page(
    __name__,
    path="/rq5",
    name="RQ5 — GPU vs CPU",
    title="RQ5 — GPU vs CPU",
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV  = os.path.join(_HERE, "results", "rq5_gpu_cpu_comparison.csv")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_RQ_HEADER = (
    "RQ5", "GPU vs CPU Comparison",
    "As a user training and running neural network models, how much speedup "
    "does running Shapley explanations on GPU (CUDA) provide compared to CPU?"
)

_REMARKS = [
    "Benchmark compares neural network explanation methods across CPU and GPU devices.",
    "Configurations are identical between both devices to ensure a direct hardware comparison.",
]

_CHARTS = [
    dict(
        section_id = "rq5-runtime-comparison-section",
        title      = "Runtime Comparison — CPU vs GPU",
        subtitle   = "Median runtime for each explanation method on CPU vs GPU. Grouped bars show the direct performance gap.",
        fn         = S.fig_hardware_comparison,
    ),
    dict(
        section_id = "rq5-speedup-section",
        title      = "GPU Speedup Factor",
        subtitle   = "How many times faster is the GPU execution compared to CPU for each method (CPU Runtime / GPU Runtime).",
        fn         = S.fig_hardware_speedup,
    ),
    dict(
        section_id = "rq5-scatter-section",
        title      = "Speed vs Accuracy by Hardware",
        subtitle   = "Scatter plot showing the tradeoff between speed (log scale) and Spearman correlation coefficient, styled by device.",
        fn         = S.fig_rho_vs_runtime_by_hardware,
    ),
]

_INTERP = (
    "How to read this page: check the runtime comparison chart to see the raw seconds saved by moving to a GPU. "
    "The Speedup Factor chart quantifies the relative hardware efficiency improvement, helping you decide whether the GPU "
    "overhead (e.g. data transfer) is justified for smaller datasets or budgets."
)


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


def _config_card(df) -> html.Div:
    """Compact benchmark overview — config card."""
    n_ds = df["dataset"].dropna().nunique() if not df.empty else 0
    n_mdl = df["model"].dropna().nunique() if not df.empty else 0
    n_libs = df["library"].dropna().nunique() if not df.empty else 0
    n_approx = df["approximator"].dropna().nunique() if not df.empty else 0

    left  = _col("Swept",
                 [f"{n_ds} datasets", f"{n_mdl} models", f"{n_libs} libraries", f"{n_approx} approximators", "2 devices (CPU, GPU)"],
                 "#EEF2FF", S.ACCENT)
    mid   = _col("Fixed",
                 ["budget = 512", "n_background = 200", "n_eval = 100", "imputer = marginal"],
                 "#F1F5F9", S.TEXT2)
    right = _col("Measured",
                 ["runtime_s", "n_model_evals", "relative_mae", "mean_sample_rho"],
                 "#F0FDF4", S.GREEN)
    return html.Div([
        html.Div([
            html.Div("Benchmark at a glance", style={
                "fontSize": "13px", "fontWeight": "700", "color": S.TEXT,
                "marginBottom": "14px",
            }),
            html.Div([left, mid, right],
                     style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
        ], style={"flex": "1", "minWidth": "280px"}),
    ], style={
        "display": "flex", "gap": "32px", "flexWrap": "wrap",
        "background": S.CARD, "borderRadius": "12px",
        "border": f"1px solid {S.BORDER}", "padding": "20px 24px",
        "marginBottom": "20px",
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────────────────────────────

def layout(**kwargs):
    df, src = S.try_load_data(_CSV)

    if src is None:
        return html.Div([
            S.rq_header(*_RQ_HEADER),
            *[S.info_note(r) for r in _REMARKS],
            S.missing_data_banner(_CSV),
            _schema_hint(),
        ])

    return html.Div([
        S.rq_header(*_RQ_HEADER),
        *[S.info_note(r) for r in _REMARKS],
        html.Div(
            [html.Span("Data source: ", style={"fontWeight": "600"}),
             html.Code(os.path.basename(src),
                       style={"fontFamily": "monospace", "fontSize": "11px",
                              "background": S.BG, "padding": "2px 6px",
                              "borderRadius": "4px"})],
            style={"fontSize": "12px", "color": S.TEXT2, "marginBottom": "4px"},
        ),
        S.data_summary_card(df),
        _config_card(df),
        S.charts_toc(_CHARTS),

        # ── Dynamic content ───────────────────────────────────────────────────
        html.Div(id="rq5-kpis"),
        html.Div(id="rq5-charts"),
    ])


def _schema_hint() -> html.Div:
    """Hint shown when data file is missing."""
    rows = [
        ("dataset",      "name of the NN benchmark dataset"),
        ("model",        "neural network type: mlp, cnn_1d, transformer, etc."),
        ("library",      "explanation library: shap, captum, shapiq, etc."),
        ("approximator", "method used: kernel, permutation, gradient, etc."),
        ("device",       "hardware device: cpu or cuda (or gpu)"),
        ("runtime_s",    "wall-clock seconds"),
        ("relative_mae", "approximation error vs exact values (optional but recommended)"),
        ("mean_sample_rho", "Spearman rank correlation coefficient (optional)"),
        ("sign_agreement", "fraction of features with matching sign (optional)"),
    ]
    th_s = {"fontSize": "10px", "fontWeight": "600", "color": S.TEXT2,
            "textTransform": "uppercase", "letterSpacing": "0.05em",
            "padding": "8px 12px", "borderBottom": f"2px solid {S.BORDER}",
            "textAlign": "left", "background": S.BG}
    td_s = {"fontSize": "12px", "padding": "8px 12px",
            "borderBottom": f"1px solid {S.BORDER}"}
    return S.section(
        "Expected CSV Schema",
        "Create rq5_gpu_cpu_comparison.csv with at least the columns below to populate this page.",
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


# ─────────────────────────────────────────────────────────────────────────────
#  Callback
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("rq5-kpis",   "children"),
    Output("rq5-charts", "children"),
    Input("rq5-ds",     "value"),
    Input("rq5-mdl",    "value"),
    Input("rq5-device", "value"),
)
def update_rq5(ds, mdl, dev_val):
    df, src = S.try_load_data(_CSV)
    if src is None or df.empty:
        return html.Div(), html.Div()

    # Apply dropdown filters
    if ds != "__all__":
        df = df[df["dataset"] == ds]
    if mdl != "__all__":
        df = df[df["model"] == mdl]
    if dev_val != "__all__":
        df = df[df["device"].astype(str).str.lower().eq(dev_val.lower())]

    # Calculate leaderboard / stats
    # For speedup KPI
    sub = df[df["runtime_s"].notna() & df["device"].notna()].copy()
    sub["device"] = sub["device"].astype(str).str.lower().replace({"cuda": "gpu"})
    
    speedup_text = "—"
    if not sub.empty:
        grp = sub.groupby(["method", "device"])["runtime_s"].median().reset_index()
        pivot = grp.pivot(index="method", columns="device", values="runtime_s").dropna(subset=["cpu", "gpu"])
        if not pivot.empty:
            speedups = pivot["cpu"] / pivot["gpu"]
            med_speedup = speedups.median()
            speedup_text = f"{med_speedup:.1f}x speedup"

    n_models = df["model"].dropna().nunique() if not df.empty else 0
    n_libs = df["library"].dropna().nunique() if not df.empty else 0
    
    # Fastest GPU backend
    fastest_gpu = "—"
    gpu_runs = sub[sub["device"] == "gpu"]
    if not gpu_runs.empty:
        fastest_grp = gpu_runs.groupby("method")["runtime_s"].median()
        if not fastest_grp.empty:
            fastest_gpu = fastest_grp.idxmin()

    kpis = S.kpi_row(
        S.kpi_card(str(n_models), "Models Swept"),
        S.kpi_card(str(n_libs), "Libraries compared"),
        S.kpi_card(speedup_text, "Median GPU Speedup", S.GREEN),
        S.kpi_card(fastest_gpu, "Fastest GPU Method", S.ACCENT),
    )

    warns = []
    if df.empty:
        warns.append(S.warning_note("No data matches the current filter selection."))

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
