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
_WOODELF_DEPTH = os.path.join(_HERE, "results", "converted", "rq5_woodelf_depth_scaling.csv")


def _load(path) -> pd.DataFrame:
    df, src = S.try_load_data(path)
    return df if src is not None else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_RQ_HEADER = (
    "RQ5", "GPU vs CPU Comparison",
    "As a user running Shapley explanations on neural networks and tree ensembles, "
    "how much speedup does executing on GPU (CUDA) provide compared to CPU?"
)

_REMARKS = [
    "Benchmark compares neural network (captum, shapiq) and tree (woodelf) explanation methods across CPU and GPU devices.",
    "Configurations are identical between both devices to ensure a direct hardware comparison.",
    "woodelf runs exact tree Shapley values on XGBoost / LightGBM / Random Forest; its runtime is the median across the max_depth sweep per seed. GPU acceleration only pays off once the per-call overhead is amortised — for these tree sizes the CPU backend is often faster.",
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
    dict(
        section_id = "rq5-captum-dividends-section",
        title      = "Captum Hardware Acceleration Dividends",
        subtitle   = "Grouped paired bar chart comparing Captum estimators (Gradient SHAP, DeepLIFT SHAP) on CPU vs GPU across architectures.",
        fn         = S.fig_captum_hardware_dividends,
    ),
    dict(
        section_id = "rq5-woodelf-scaling-section",
        title      = "woodelf Tree-Explainer Scaling — CPU vs GPU",
        subtitle   = "Log-log runtime of the woodelf tree explainer against maximum tree depth (the swept variable), split by imputation (path-dependent / interventional) and device. CPU stays roughly flat while GPU runtime climbs steeply with depth.",
        fn         = S.fig_woodelf_depth_scaling,
    ),
]

_INTERP = (
    "How to read this page: check the runtime comparison chart to see the raw seconds gained or lost by moving to a GPU. "
    "The Speedup Factor chart quantifies the relative hardware efficiency — a factor above 1× means the GPU wins, while "
    "a factor below 1× (as seen for the woodelf tree backend at these sizes) means the GPU overhead (kernel launch, data "
    "transfer) is not yet justified. Use it to decide whether the accelerator pays off for your dataset and budget."
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
        _config_card(df_agg),

        # ── Dynamic content ───────────────────────────────────────────────────
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
    Output("rq5-charts", "children"),
    Input("rq5-ds",     "value"),
    Input("rq5-mdl",    "value"),
    Input("rq5-device", "value"),
)
def update_rq5(ds, mdl, dev_val):
    df_agg = _load(_AGGREGATED)
    df_seed = _load(_BY_SEED)
    df_depth = _load(_WOODELF_DEPTH)
    if df_agg.empty:
        return html.Div()

    # Apply dropdown filters
    if ds != "__all__":
        df_agg = df_agg[df_agg["dataset"] == ds]
        df_seed = df_seed[df_seed["dataset"] == ds]
        if not df_depth.empty:
            df_depth = df_depth[df_depth["dataset"] == ds]
    if mdl != "__all__":
        df_agg = df_agg[df_agg["model"] == mdl]
        df_seed = df_seed[df_seed["model"] == mdl]
        if not df_depth.empty:
            df_depth = df_depth[df_depth["model"] == mdl]
    if dev_val != "__all__":
        df_agg = df_agg[df_agg["device"].astype(str).str.lower().eq(dev_val.lower())]
        df_seed = df_seed[df_seed["device"].astype(str).str.lower().eq(dev_val.lower())]
        if not df_depth.empty:
            df_depth = df_depth[df_depth["device"].astype(str).str.lower().eq(dev_val.lower())]

    warns = []
    if df_agg.empty:
        warns.append(S.warning_note("No data matches the current filter selection."))

    charts = html.Div([
        *warns,
        S.section(
            _CHARTS[0]["title"], _CHARTS[0]["subtitle"],
            dcc.Graph(figure=S.fig_hardware_comparison(df_agg),
                      config=S.graph_config("rq5_hardware_comparison"),
                      style={"padding": "8px"}),
            section_id=_CHARTS[0]["section_id"],
        ),
        S.section(
            _CHARTS[1]["title"], _CHARTS[1]["subtitle"],
            dcc.Graph(figure=S.fig_hardware_speedup(df_seed),
                      config=S.graph_config("rq5_hardware_speedup"),
                      style={"padding": "8px"}),
            section_id=_CHARTS[1]["section_id"],
        ),
        S.section(
            _CHARTS[2]["title"], _CHARTS[2]["subtitle"],
            dcc.Graph(figure=S.fig_rho_vs_runtime_by_hardware(df_seed),
                      config=S.graph_config("rq5_rho_vs_runtime"),
                      style={"padding": "8px"}),
            section_id=_CHARTS[2]["section_id"],
        ),
        S.section(
            _CHARTS[3]["title"], _CHARTS[3]["subtitle"],
            dcc.Graph(figure=S.fig_captum_hardware_dividends(df_seed),
                      config=S.graph_config("rq5_captum_hardware_dividends"),
                      style={"padding": "8px"}),
            section_id=_CHARTS[3]["section_id"],
        ),
        S.section(
            _CHARTS[4]["title"], _CHARTS[4]["subtitle"],
            dcc.Graph(figure=S.fig_woodelf_depth_scaling(df_depth),
                      config=S.graph_config("rq5_woodelf_scaling"),
                      style={"padding": "8px"}),
            section_id=_CHARTS[4]["section_id"],
        ),
        S.interpretation_note(_INTERP),
    ])

    return charts
