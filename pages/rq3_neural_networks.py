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
from dash import Input, Output, callback, dcc, html

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared as S

dash.register_page(
    __name__,
    path="/rq3",
    name="RQ3 — Neural Networks",
    title="RQ3 — Neural Networks",
)

_HERE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV    = os.path.join(_HERE, "results_config-neural-networks-RQ3.csv")
_CSV_FB = os.path.join(_HERE, "results_nn.csv")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION — edit here to change what the page shows
# ═════════════════════════════════════════════════════════════════════════════

_RQ_HEADER = (
    "RQ3", "Neural Networks",
    "As a user with a neural network, I want to know which library is the fastest?"
)

# Notes shown on the page directly below the research question.
_REMARKS = [
    "Remark: Use of variance in architecture rather than varying the size.",
    "Open question: ShapIQ supports ProxySPEX \u2014 would this also be interesting?",
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
]

_INTERP = (
    "How to read this page: if runtime is your only concern, pick the method with the shortest "
    "bar in the ranking chart. Then check the Speed vs Accuracy scatter to confirm it doesn't "
    "sacrifice too much quality for that speed gain."
)


def layout(**kwargs):
    df, src = S.try_load_data(_CSV, _CSV_FB)

    if src is None:
        return html.Div([
            S.rq_header(*_RQ_HEADER),
            *[S.info_note(r) for r in _REMARKS],
            S.missing_data_banner(_CSV),
            _schema_hint(),
        ])

    datasets = [{"label": "All datasets", "value": "__all__"}] + \
               [{"label": d, "value": d} for d in sorted(df["dataset"].dropna().unique())]
    libs     = sorted(df["library"].dropna().unique())
    budgets  = sorted(df["budget"].dropna().unique()) if not df.empty else []

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
        S.charts_toc(_CHARTS),
        S.filter_bar(
            S.filter_dropdown("Dataset", "rq3-ds", datasets, "__all__", "220px"),
            S.filter_checklist(
                "Library",
                "rq3-lib",
                [{"label": f"  {lib}", "value": lib} for lib in libs],
                libs,
            ),
            S.filter_checklist(
                "Budget",
                "rq3-budget",
                [{"label": f"  {int(b)}", "value": float(b)} for b in budgets],
                [float(b) for b in budgets],
            ),
        ),

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
    Input("rq3-lib",    "value"),
    Input("rq3-budget", "value"),
)
def update_rq3(ds, libs, budgets):
    df, src = S.try_load_data(_CSV, _CSV_FB)
    if src is None or df.empty:
        return html.Div(), html.Div()

    if ds != "__all__": df = df[df["dataset"] == ds]
    if libs:            df = df[df["library"].isin(libs)]
    if budgets:         df = df[df["budget"].isin(budgets)]

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
        S.kpi_card(fastest_m, "Fastest library / method", S.ACCENT),
        S.kpi_card(f"{fastest_rt:.3f} s" if not np.isnan(fastest_rt) else "—",
                   "Median runtime (fastest)", S.GREEN),
        S.kpi_card(str(n_libs), "Libraries compared"),
        S.kpi_card(f"{fail_of_fast:.0f} %" if not np.isnan(fail_of_fast) else "—",
                   "Failure rate of fastest method",
                   S.RED if (not np.isnan(fail_of_fast) and fail_of_fast > 10) else S.GREEN),
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
