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
    dict(
        section_id = "rq5-captum-dividends-section",
        title      = "Captum Hardware Acceleration Dividends",
        subtitle   = "Grouped paired bar chart comparing Captum estimators (Gradient SHAP, DeepLIFT SHAP) on CPU vs GPU across architectures.",
        fn         = S.fig_captum_hardware_dividends,
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

def _pill(text: str, bg: str, color: str, tooltip: str | None = None) -> html.Span:
    style = {
        "fontSize": "11px", "fontWeight": "500", "color": color,
        "background": bg, "borderRadius": "4px",
        "padding": "2px 8px", "marginRight": "4px", "marginBottom": "4px",
        "display": "inline-block",
    }
    if tooltip:
        style["cursor"] = "help"
        return html.Span(text, title=tooltip, style=style)
    return html.Span(text, style=style)


def _col(heading, items, bg, color, tooltips: dict | None = None) -> html.Div:
    tooltips = tooltips or {}
    return html.Div([
        html.Div(heading, style={
            "fontSize": "10px", "fontWeight": "700", "color": S.TEXT2,
            "textTransform": "uppercase", "letterSpacing": "0.07em",
            "marginBottom": "8px",
        }),
        html.Div([_pill(i, bg, color, tooltip=tooltips.get(i)) for i in items]),
    ], style={"flex": "1", "minWidth": "160px"})


def _config_card(df) -> html.Div:
    """Compact benchmark overview — config card."""
    n_ds = df["dataset"].dropna().nunique() if not df.empty else 0
    n_mdl = df["model"].dropna().nunique() if not df.empty else 0
    n_libs = df["library"].dropna().nunique() if not df.empty else 0
    n_approx = df["approximator"].dropna().nunique() if not df.empty else 0
    n_seeds = int(df["seed_count"].iloc[0]) if not df.empty and "seed_count" in df.columns else 10

    swept_items = [f"{n_ds} datasets", f"{n_mdl} models", f"{n_libs} libraries",
                   f"{n_approx} approximators", f"{n_seeds} seeds", "2 devices (CPU, GPU)"]
    swept_tooltips = {}
    if not df.empty:
        swept_tooltips = {
            swept_items[0]: ", ".join(sorted(df["dataset"].dropna().unique())),
            swept_items[1]: ", ".join(sorted(df["model"].dropna().unique())),
            swept_items[2]: ", ".join(sorted(df["library"].dropna().unique())),
            swept_items[3]: ", ".join(sorted(df["approximator"].dropna().unique())),
        }

    left  = _col("Swept", swept_items, "#EEF2FF", S.ACCENT, tooltips=swept_tooltips)
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

        html.Div(id="rq5-warnings"),

        # ── Dynamic content — one static section wrapper per chart, only the
        # nested graph placeholder is filled in by the callback. The scatter
        # section's seed checklist has to live here (not inside the dynamic
        # output) or it would reset/race every time the callback re-renders.
        S.section(
            _CHARTS[0]["title"], _CHARTS[0]["subtitle"],
            html.Div(id="rq5-runtime-comparison-chart"),
            section_id=_CHARTS[0]["section_id"],
        ),
        S.section(
            _CHARTS[1]["title"], _CHARTS[1]["subtitle"],
            html.Div(id="rq5-speedup-chart"),
            section_id=_CHARTS[1]["section_id"],
        ),
        S.section(
            _CHARTS[2]["title"], _CHARTS[2]["subtitle"],
            html.Div([
                _seed_checklist(),
                html.Div(id="rq5-scatter-chart"),
            ]),
            section_id=_CHARTS[2]["section_id"],
        ),
        S.section(
            _CHARTS[3]["title"], _CHARTS[3]["subtitle"],
            html.Div(id="rq5-captum-dividends-chart"),
            section_id=_CHARTS[3]["section_id"],
        ),
        S.interpretation_note(_INTERP),
    ])


def _seed_checklist() -> html.Div:
    """Per-seed inclusion toggles, scoped to the Speed vs Accuracy scatter only."""
    return html.Div([
        html.Span("Seeds:", style={
            "fontSize": "11px", "fontWeight": "600", "color": S.TEXT2,
            "marginRight": "10px", "flexShrink": "0",
        }),
        dcc.Checklist(
            id="rq5-scatter-seeds",
            options=[{"label": str(s), "value": s} for s in range(10)],
            value=list(range(10)),
            inline=True,
            inputStyle={"marginRight": "4px"},
            labelStyle={"marginRight": "12px", "fontSize": "12px",
                        "cursor": "pointer", "color": S.TEXT},
        ),
    ], style={
        "display": "flex", "alignItems": "center", "flexWrap": "wrap",
        "padding": "8px 12px",
        "borderBottom": f"1px solid {S.BORDER}",
        "background": S.BG,
    })


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
    Output("rq5-warnings",                   "children"),
    Output("rq5-runtime-comparison-chart",   "children"),
    Output("rq5-speedup-chart",              "children"),
    Output("rq5-scatter-chart",              "children"),
    Output("rq5-captum-dividends-chart",     "children"),
    Input("rq5-ds",             "value"),
    Input("rq5-lib",            "value"),
    Input("rq5-mdl",            "value"),
    Input("rq5-device",         "value"),
    Input("rq5-scatter-seeds",  "value"),
)
def update_rq5(ds, libs, mdl, dev_val, scatter_seeds):
    df_agg = _load(_AGGREGATED)
    df_seed = _load(_BY_SEED)
    if df_agg.empty:
        empty = html.Div()
        return empty, empty, empty, empty, empty

    # Apply dropdown filters
    df_agg = S.filter_by_column(df_agg, "dataset", ds)
    df_seed = S.filter_by_column(df_seed, "dataset", ds)
    df_agg = S.filter_by_column(df_agg, "library", libs)
    df_seed = S.filter_by_column(df_seed, "library", libs)
    if mdl != "__all__":
        df_agg = df_agg[df_agg["model"] == mdl]
        df_seed = df_seed[df_seed["model"] == mdl]
    if dev_val != "__all__":
        df_agg = df_agg[df_agg["device"].astype(str).str.lower().eq(dev_val.lower())]
        df_seed = df_seed[df_seed["device"].astype(str).str.lower().eq(dev_val.lower())]

    warn = html.Div()
    if df_agg.empty:
        warn = S.warning_note("No data matches the current filter selection.")

    # Speed vs Accuracy scatter — seed selection is scoped to this chart only.
    if scatter_seeds is None:
        scatter_seeds = list(range(10))
    scatter_df = df_seed[df_seed["seed"].isin(scatter_seeds)]

    return (
        warn,
        dcc.Graph(figure=S.fig_hardware_comparison(df_agg),
                  config=S.graph_config("rq5_hardware_comparison"),
                  style={"padding": "8px"}),
        dcc.Graph(figure=S.fig_hardware_speedup(df_seed),
                  config=S.graph_config("rq5_hardware_speedup"),
                  style={"padding": "8px"}),
        dcc.Graph(figure=S.fig_rho_vs_runtime_by_hardware(scatter_df),
                  config=S.graph_config("rq5_rho_vs_runtime"),
                  style={"padding": "8px"}),
        dcc.Graph(figure=S.fig_captum_hardware_dividends(df_seed),
                  config=S.graph_config("rq5_captum_hardware_dividends"),
                  style={"padding": "8px"}),
    )
