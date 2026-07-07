"""
pages/rq4_trees.py  —  RQ4: Tree Models

Edit the PAGE CONFIGURATION block below to change what is shown.
The layout / filter / callback logic beneath it should rarely need touching.

Data file: results_config-tree.csv (falls back to results_trees.csv).
The complexity axis auto-detects: tree_depth > max_depth > n_estimators > model.
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
    path="/rq4",
    name="RQ4 — Tree Models",
    title="RQ4 — Tree Models",
)

_HERE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV    = os.path.join(_HERE, "results_config-tree.csv")
_CSV_FB = os.path.join(_HERE, "results_trees.csv")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION — edit here to change what the page shows
# ═════════════════════════════════════════════════════════════════════════════

_RQ_HEADER = (
    "RQ4", "Tree Models",
    "As a user with deep/complex tree models, I want to know which library handles "
    "extreme tree depths efficiently without hitting computational bottlenecks or "
    "breaking points?"
)

# Notes shown on the page directly below the research question.
_REMARKS = [
    "Sweep background: higher depth would be better, but 100 is already good "
    "(only more if time left).",
]


def _build_charts(comp_col: str) -> list:
    """Returns the chart manifest for the current data's complexity column.

    Called from both layout() and update_rq4() so that the TOC and chart sections
    always reflect the actual complexity axis in the loaded CSV.

    To add or remove charts, edit the list returned here.
    section_id values must be unique and stable (used as DOM anchors).
    """
    label = comp_col.replace("_", " ").title()
    return [
        dict(
            section_id = "rq4-failure-section",
            title      = f"Failure Rate Heatmap — Method × {label}",
            subtitle   = "Red cells = breaking points. "
                         "The first red cell in each row shows where a method starts to fail.",
            fn         = S.fig_failure_vs_complexity,
        ),
        dict(
            section_id = "rq4-runtime-section",
            title      = f"Runtime vs {label}",
            subtitle   = "Steep slope = computational bottleneck as complexity grows. "
                         "Look for flat lines — those libraries scale gracefully.",
            fn         = S.fig_runtime_vs_complexity,
        ),
        dict(
            section_id = "rq4-quality-section",
            title      = f"Spearman ρ vs {label}",
            subtitle   = "Does approximation quality hold up at extreme depths? "
                         "Declining lines indicate the method loses accuracy for complex trees.",
            fn         = S.fig_rho_vs_complexity,
        ),
        dict(
            section_id = "rq4-ranking-section",
            title      = "Method Quality Ranking",
            subtitle   = "Median Spearman ρ aggregated across all complexity settings "
                         "in the current selection.",
            fn         = lambda df: S.fig_leaderboard_bars(S.compute_leaderboard(df)),
        ),
    ]


_INTERP = (
    "How to read this page: the failure heatmap is the key chart — find the first red cell "
    "in each row to locate a library's breaking point. Then check the runtime chart to see "
    "if a method that doesn't break is also fast."
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

    datasets  = [{"label": "All datasets", "value": "__all__"}] + \
               [{"label": d, "value": d} for d in sorted(df["dataset"].dropna().unique())]
    libs      = sorted(df["library"].dropna().unique())
    comp_col  = S._complexity_col(df)
    charts_man = _build_charts(comp_col)

    comp_vals = sorted(df[comp_col].dropna().unique()) if not df.empty else []
    try:
        comp_vals = sorted(int(v) for v in comp_vals)
        comp_opts = [{"label": f"  {v}", "value": v} for v in comp_vals]
    except (ValueError, TypeError):
        comp_opts = [{"label": f"  {v}", "value": v} for v in comp_vals]

    return html.Div([
        S.rq_header(*_RQ_HEADER),
        *[S.info_note(r) for r in _REMARKS],
        html.Div(
            [html.Span("Data source: ", style={"fontWeight": "600"}),
             html.Code(os.path.basename(src),
                       style={"fontFamily": "monospace", "fontSize": "11px",
                              "background": S.BG, "padding": "2px 6px",
                              "borderRadius": "4px"}),
             html.Span("  •  complexity axis: ", style={"marginLeft": "12px"}),
             html.Code(comp_col,
                       style={"fontFamily": "monospace", "fontSize": "11px",
                              "background": S.BG, "padding": "2px 6px",
                              "borderRadius": "4px"})],
            style={"fontSize": "12px", "color": S.TEXT2, "marginBottom": "4px"},
        ),
        S.data_summary_card(df),
        S.charts_toc(charts_man),
        S.filter_bar(
            S.filter_dropdown("Dataset", "rq4-ds", datasets, "__all__", "220px"),
            S.filter_checklist(
                "Library",
                "rq4-lib",
                [{"label": f"  {lib}", "value": lib} for lib in libs],
                libs,
            ),
            S.filter_checklist(
                comp_col.replace("_", " ").title(),
                "rq4-complexity",
                comp_opts,
                [o["value"] for o in comp_opts],
            ),
        ),

        # ── Dynamic content ───────────────────────────────────────────────────
        html.Div(id="rq4-kpis"),
        html.Div(id="rq4-charts"),
    ])


def _schema_hint() -> html.Div:
    rows = [
        ("dataset",          "benchmark dataset name"),
        ("model",            "model type: random_forest, gradient_boosting, xgboost, etc."),
        ("tree_depth",       "integer tree depth — this drives the main complexity axis"),
        ("n_estimators",     "number of trees (optional; used if tree_depth absent)"),
        ("library",          "explanation library"),
        ("approximator",     "approximation method"),
        ("runtime_s",        "wall-clock seconds"),
        ("relative_mae",     "approximation error vs exact values (optional)"),
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
        "Create results_trees.csv with at least the columns below. "
        "The tree_depth column is the key — without it the page uses 'model' as the x-axis.",
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
    Output("rq4-kpis",   "children"),
    Output("rq4-charts", "children"),
    Input("rq4-ds",         "value"),
    Input("rq4-lib",        "value"),
    Input("rq4-complexity", "value"),
)
def update_rq4(ds, libs, comp_vals):
    df, src = S.try_load_data(_CSV, _CSV_FB)
    if src is None or df.empty:
        return html.Div(), html.Div()

    comp_col      = S._complexity_col(df)
    charts_man    = _build_charts(comp_col)

    if ds != "__all__": df = df[df["dataset"] == ds]
    if libs:            df = df[df["library"].isin(libs)]
    if comp_vals:       df = df[df[comp_col].isin(comp_vals)]

    lb = S.compute_leaderboard(df)

    # ── KPIs ──────────────────────────────────────────────────────────────
    n_complexity = df[comp_col].dropna().nunique() if not df.empty else 0
    max_comp     = df[comp_col].max() if not df.empty and df[comp_col].notna().any() else "—"
    n_libs       = df["library"].dropna().nunique() if not df.empty else 0

    most_robust = "—"
    if not df.empty and df[comp_col].notna().any():
        try:
            max_comp_val = pd.to_numeric(df[comp_col], errors="coerce").max()
            hi_sub = df[pd.to_numeric(df[comp_col], errors="coerce") == max_comp_val]
        except Exception:
            hi_sub = df[df[comp_col] == df[comp_col].max()]
        if not hi_sub.empty:
            fr_by_method = hi_sub.groupby("method")["is_failure"].mean()
            robust = fr_by_method[fr_by_method < 0.5]
            if not robust.empty:
                most_robust = robust.idxmin()

    kpis = S.kpi_row(
        S.kpi_card(str(n_complexity), f"{comp_col} settings"),
        S.kpi_card(str(max_comp),     f"Highest {comp_col} benchmarked"),
        S.kpi_card(str(n_libs),       "Libraries compared"),
        S.kpi_card(most_robust,       f"Most robust at max {comp_col}", S.GREEN),
    )

    # ── Warnings ──────────────────────────────────────────────────────────
    warns = []
    if df.empty:
        warns.append(S.warning_note("No data matches the current filter selection."))
    elif n_complexity < 2:
        warns.append(S.warning_note(
            f"Only one '{comp_col}' value in the current selection. "
            "Select more values in the filter above to see scaling trends."
        ))

    # ── Charts — driven by _build_charts() manifest above ───────────────────
    charts = html.Div([
        *warns,
        *[
            S.section(
                c["title"], c["subtitle"],
                dcc.Graph(figure=c["fn"](df),
                          config={"displayModeBar": False}, style={"padding": "8px"}),
                section_id=c["section_id"],
            )
            for c in charts_man
        ],
        S.interpretation_note(_INTERP),
    ])

    return kpis, charts
