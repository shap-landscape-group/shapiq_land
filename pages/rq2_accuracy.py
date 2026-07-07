"""
pages/rq2_accuracy.py  —  RQ2: Approximation Accuracy

Edit the PAGE CONFIGURATION block below to change what is shown.
The layout / filter / callback logic beneath it should rarely need touching.
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
    path="/rq2",
    name="RQ2 — Accuracy",
    title="RQ2 — Approximation Accuracy",
)

_HERE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV     = os.path.join(_HERE, "results_config-accuracy.csv")
_CSV_FB  = os.path.join(_HERE, "results_accuracy.csv")
_CSV_FB2 = os.path.join(_HERE, "results.csv")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION — edit here to change what the page shows
# ═════════════════════════════════════════════════════════════════════════════

_RQ_HEADER = (
    "RQ2", "Approximation Accuracy",
    "As a user using shapley approximations, I want to know how good the values "
    "actually are so that I can trust the explanations without wasting too much "
    "computing time needed for exact values."
)

# Notes shown on the page directly below the research question.
_REMARKS = [
    "Remark: Use of Spearman Rank Correlation.",
]


def _agg(df):
    """Aggregate raw runs to one row per method — used by Pareto and Ranking charts."""
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
        section_id = "rq2-pareto-section",
        title      = "Speed–Accuracy Pareto Frontier",
        subtitle   = "Colored markers are Pareto-optimal: no other method is both faster AND "
                     "more accurate. Gray = dominated. Hover for details.",
        fn         = lambda df: S.fig_pareto(_agg(df)),
    ),
    dict(
        section_id = "rq2-budget-section",
        title      = "Spearman ρ vs Budget",
        subtitle   = "How much budget before rank-order agreement plateaus? "
                     "Failed runs excluded. Flat line = already converged. Reference: ρ = 0.9.",
        fn         = S.fig_budget_rho,
    ),
    dict(
        section_id = "rq2-runtime-budget-section",
        title      = "Runtime vs Budget",
        subtitle   = "What does each additional budget unit cost in wall-clock time?",
        fn         = S.fig_runtime_vs_budget,
    ),
    dict(
        section_id = "rq2-conv-section",
        title      = "ρ Convergence Curve",
        subtitle   = "Rank-order agreement with exact Shapley values as budget grows. "
                     "Values near 1.0 = the approximation already ranks features correctly.",
        fn         = lambda df: S.fig_metric_vs_budget(df, "mean_sample_rho"),
    ),
    dict(
        section_id = "rq2-distribution-section",
        title      = "Spearman ρ Distribution",
        subtitle   = "Box + strip plot across all runs. Reference: ρ ≥ 0.9. "
                     "Wide spread means inconsistent results.",
        fn         = S.fig_distribution,
    ),
    dict(
        section_id = "rq2-ranking-section",
        title      = "Method Ranking by Spearman ρ",
        subtitle   = "Bars = median Spearman ρ per method. "
                     "Failure labels on the right (red if > 10 %).",
        fn         = lambda df: S.fig_leaderboard_bars(S.compute_leaderboard(df)),
    ),
]

_INTERP = (
    "How to read this page: use the Pareto chart to find methods that are both fast and accurate. "
    "Use the Budget charts to find the minimum budget where quality plateaus — spending more "
    "beyond that point wastes compute without improving accuracy."
)


# ═════════════════════════════════════════════════════════════════════════════
#  Layout
# ═════════════════════════════════════════════════════════════════════════════

def layout(**kwargs):
    df, src = S.try_load_data(_CSV, _CSV_FB, _CSV_FB2)

    datasets = [{"label": "All datasets", "value": "__all__"}] + \
               [{"label": d, "value": d} for d in sorted(df["dataset"].dropna().unique())]
    models   = [{"label": "All models",   "value": "__all__"}] + \
               [{"label": m, "value": m} for m in sorted(df["model"].dropna().unique())]
    budgets  = sorted(df["budget"].dropna().unique()) if not df.empty else []

    source_tag = html.Div(
        [html.Span("Data source: ", style={"fontWeight": "600"}),
         html.Code(os.path.basename(src) if src else "—",
                   style={"fontFamily": "monospace", "fontSize": "11px",
                          "background": S.BG, "padding": "2px 6px",
                          "borderRadius": "4px"})],
        style={"fontSize": "12px", "color": S.TEXT2, "marginBottom": "4px"},
    ) if src else S.missing_data_banner(_CSV)

    return html.Div([
        S.rq_header(*_RQ_HEADER),
        *[S.info_note(r) for r in _REMARKS],
        source_tag,
        S.data_summary_card(df),
        S.charts_toc(_CHARTS),
        S.filter_bar(
            S.filter_dropdown("Dataset", "rq2-ds",  datasets, "__all__", "220px"),
            S.filter_dropdown("Model",   "rq2-mdl", models,   "__all__", "200px"),
            S.filter_checklist(
                "Budget",
                "rq2-budget",
                [{"label": f"  {int(b)}", "value": float(b)} for b in budgets],
                [float(b) for b in budgets],
            ),
        ),
        html.Div(id="rq2-kpis"),
        html.Div(id="rq2-charts"),
    ])


# ═════════════════════════════════════════════════════════════════════════════
#  Callback
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("rq2-kpis",   "children"),
    Output("rq2-charts", "children"),
    Input("rq2-ds",     "value"),
    Input("rq2-mdl",    "value"),
    Input("rq2-budget", "value"),
)
def update_rq2(ds, mdl, budgets):
    df, _ = S.try_load_data(_CSV, _CSV_FB, _CSV_FB2)

    if ds  != "__all__": df = df[df["dataset"] == ds]
    if mdl != "__all__": df = df[df["model"]   == mdl]
    if budgets:          df = df[df["budget"].isin(budgets)]

    lb = S.compute_leaderboard(df)

    # ── KPIs ──────────────────────────────────────────────────────────────
    best_rho  = lb["rho_median"].max()              if not lb.empty else float("nan")
    best_m    = lb.iloc[0]["method"]                if not lb.empty else "—"
    zero_fail = lb[lb["failure_rate"] == 0].shape[0] if not lb.empty else 0
    n_budgets = df["budget"].dropna().nunique()      if not df.empty else 0

    min_budget_rho90 = "—"
    if not df.empty and df["budget"].notna().any():
        bq = (
            df[~df["is_failure"]]
            .groupby(["method", "budget"])["mean_sample_rho"]
            .median().reset_index()
        )
        threshold_rows = bq[bq["mean_sample_rho"] >= 0.9]
        if not threshold_rows.empty:
            min_budget_rho90 = str(int(threshold_rows.groupby("method")["budget"].min().min()))

    kpis = S.kpi_row(
        S.kpi_card(f"ρ = {best_rho:.3f}" if not np.isnan(best_rho) else "—",
                   f"Best median ρ ({best_m})", S.GREEN),
        S.kpi_card(str(zero_fail), "Methods with 0 % failure rate",
                   S.GREEN if zero_fail else S.RED),
        S.kpi_card(str(n_budgets), "Budget settings compared"),
        S.kpi_card(f"Budget ≥ {min_budget_rho90}", "Min budget for ρ ≥ 0.9", S.ACCENT),
    )

    # ── Warnings ──────────────────────────────────────────────────────────
    warns = []
    if df.empty:
        warns.append(S.warning_note("No data matches the current filter selection."))
    elif df["budget"].dropna().nunique() < 2:
        warns.append(S.warning_note(
            "Select at least two budget values to see convergence curves. "
            "Enable all budget checkboxes above."
        ))

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
        S.info_note(
            "Attribution vectors are not stored in the benchmark — only scalar metrics per run. "
            "Convergence is measured via proxy metrics: relative MAE, sign agreement, and "
            "Spearman ρ. These are strongly correlated with direct vector similarity."
        ),
        S.interpretation_note(_INTERP),
    ])

    return kpis, charts
