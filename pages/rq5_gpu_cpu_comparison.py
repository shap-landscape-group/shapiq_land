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
_AGGREGATED = os.path.join(_HERE, "results", "converted", "rq5_gpu_cpu_comparison_aggregated.csv")
_BY_SEED    = os.path.join(_HERE, "results", "converted", "rq5_gpu_cpu_comparison_by_seed.csv")


def _load(path) -> pd.DataFrame:
    df, src = S.try_load_data(path)
    return df if src is not None else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_RQ_HEADER = (
    "RQ5", "GPU vs CPU Comparison",
    "As a user training and running neural network models, how much speedup "
    "does running Shapley explanations on GPU (CUDA) provide compared to CPU?"
)

_REMARKS = [
    "Benchmark compares neural network and tree explanation methods across CPU and GPU devices.",
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
    n_seeds = int(df["seed_count"].iloc[0]) if not df.empty and "seed_count" in df.columns else 10

    left  = _col("Swept",
                 [f"{n_ds} datasets", f"{n_mdl} models", f"{n_libs} libraries", f"{n_approx} approximators", f"{n_seeds} seeds", "2 devices (CPU, GPU)"],
                 "#EEF2FF", S.ACCENT)
    mid   = _col("Fixed",
                 ["budget = 512 (NNs)", "n_background = 100", "n_eval = 10", "imputer = marginal"],
                 "#F1F5F9", S.TEXT2)
    right = _col("Measured",
                 ["runtime_s", "relative_mae", "mean_sample_rho", "relative_additivity_gap"],
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
    df_agg = _load(_AGGREGATED)

    if df_agg.empty:
        return html.Div([
            S.rq_header(*_RQ_HEADER),
            *[S.info_note(r) for r in _REMARKS],
            S.missing_data_banner(_AGGREGATED),
            _schema_hint(),
        ])

    return html.Div([
        S.rq_header(*_RQ_HEADER),
        *[S.info_note(r) for r in _REMARKS],
        S.info_note([
            "Seed Stability & Variation: ",
            html.Span("All benchmark results are evaluated over 10 independent random seeds. ", style={"fontWeight": "600"}),
            "To represent the central tendency and hardware reliability robustly, we display the ",
            html.Span("Median", style={"fontWeight": "600"}),
            " of the runs with error bars representing the ",
            html.Span("25th and 75th percentiles (Q25–Q75)", style={"fontWeight": "600"}),
            " spread across the seeds."
        ]),
        html.Div(
            [html.Span("Data source: ", style={"fontWeight": "600"}),
             html.Code(os.path.basename(_AGGREGATED),
                       style={"fontFamily": "monospace", "fontSize": "11px",
                              "background": S.BG, "padding": "2px 6px",
                              "borderRadius": "4px"})],
            style={"fontSize": "12px", "color": S.TEXT2, "marginBottom": "4px"},
        ),
        _config_card(df_agg),
        S.charts_toc(_CHARTS),

        # ── Dynamic content ───────────────────────────────────────────────────
        html.Div(id="rq5-kpis"),
        html.Div(id="rq5-charts"),
    ])


def _schema_hint() -> html.Div:
    """Hint shown when data file is missing."""
    rows = [
        ("dataset",      "name of the NN/tree benchmark dataset"),
        ("model",        "model type: mlp, cnn_1d, transformer, xgboost, etc."),
        ("library",      "explanation library: shap, captum, shapiq, woodelf, etc."),
        ("approximator", "method used: kernel, permutation, gradient, path_dependent, etc."),
        ("device",       "hardware device: cpu or gpu"),
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
        "Create rq5_gpu_cpu_comparison_aggregated.csv inside results/converted/ to populate this page.",
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
    df_agg = _load(_AGGREGATED)
    df_seed = _load(_BY_SEED)
    if df_agg.empty:
        return html.Div(), html.Div()

    # Apply dropdown filters
    if ds != "__all__":
        df_agg = df_agg[df_agg["dataset"] == ds]
        df_seed = df_seed[df_seed["dataset"] == ds]
    if mdl != "__all__":
        df_agg = df_agg[df_agg["model"] == mdl]
        df_seed = df_seed[df_seed["model"] == mdl]
    if dev_val != "__all__":
        df_agg = df_agg[df_agg["device"].astype(str).str.lower().eq(dev_val.lower())]
        df_seed = df_seed[df_seed["device"].astype(str).str.lower().eq(dev_val.lower())]

    # Calculate leaderboard / stats
    # For speedup KPI
    sub = df_agg[df_agg["runtime_median"].notna() & df_agg["device"].notna()].copy()
    
    speedup_text = "—"
    if not sub.empty:
        grp = sub.groupby(["method", "device"])["runtime_median"].median().reset_index()
        pivot = grp.pivot(index="method", columns="device", values="runtime_median").dropna(subset=["cpu", "gpu"])
        if not pivot.empty:
            speedups = pivot["cpu"] / pivot["gpu"]
            med_speedup = speedups.median()
            speedup_text = f"{med_speedup:.1f}x speedup"

    n_models = df_agg["model"].dropna().nunique() if not df_agg.empty else 0
    n_libs = df_agg["library"].dropna().nunique() if not df_agg.empty else 0
    
    # Fastest GPU backend
    fastest_gpu = "—"
    gpu_runs = sub[sub["device"] == "gpu"]
    if not gpu_runs.empty:
        fastest_grp = gpu_runs.groupby("method")["runtime_median"].median()
        if not fastest_grp.empty:
            fastest_gpu = fastest_grp.idxmin()

    kpis = S.kpi_row(
        S.kpi_card(str(n_models), "Models Swept"),
        S.kpi_card(str(n_libs), "Libraries compared"),
        S.kpi_card(speedup_text, "Median GPU Speedup", S.GREEN),
        S.kpi_card(fastest_gpu, "Fastest GPU Method", S.ACCENT),
    )

    warns = []
    if df_agg.empty:
        warns.append(S.warning_note("No data matches the current filter selection."))

    charts = html.Div([
        *warns,
        S.section(
            _CHARTS[0]["title"], _CHARTS[0]["subtitle"],
            dcc.Graph(figure=S.fig_hardware_comparison(df_agg),
                      config={"displayModeBar": False}, style={"padding": "8px"}),
            section_id=_CHARTS[0]["section_id"],
        ),
        S.section(
            _CHARTS[1]["title"], _CHARTS[1]["subtitle"],
            dcc.Graph(figure=S.fig_hardware_speedup(df_agg),
                      config={"displayModeBar": False}, style={"padding": "8px"}),
            section_id=_CHARTS[1]["section_id"],
        ),
        S.section(
            _CHARTS[2]["title"], _CHARTS[2]["subtitle"],
            dcc.Graph(figure=S.fig_rho_vs_runtime_by_hardware(df_seed),
                      config={"displayModeBar": False}, style={"padding": "8px"}),
            section_id=_CHARTS[2]["section_id"],
        ),
        S.interpretation_note(_INTERP),
    ])

    return kpis, charts
