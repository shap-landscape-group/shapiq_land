"""
pages/home.py — Landing page / overview.

Shows a global leaderboard and links to each RQ page.
Data comes from results.csv (shared with RQ1 & RQ2).
"""
import os
import sys

import dash
import numpy as np
from dash import Input, Output, callback, dcc, html

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared as S

dash.register_page(__name__, path="/", name="Home", title="Benchmark Explorer")

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV     = os.path.join(_HERE, "results_config-dimensionality.csv")
_CSV_FB  = os.path.join(_HERE, "results_config-accuracy.csv")
_CSV_FB2 = os.path.join(_HERE, "results.csv")   # final fallback

# ── Metric reference rows ─────────────────────────────────────────────────────
_METRIC_ROWS = [
    ("mean_sample_rho", "mean Spearman ρ(approx, exact) per sample",
     "0–1, higher is better. Measures whether feature attribution rankings agree with ground truth. Primary quality metric."),
    ("relative_mae",    "mean |approx − exact| / mean |exact|",
     "Lower is better. > 1 means worse than predicting the mean — flagged as failure."),
    ("sign_agreement",  "fraction of features with correct sign",
     "0–1, higher is better. Captures whether direction of influence is correctly identified."),
    ("is_failure",      "relative_mae > 1  OR  relative_mae is NaN",
     "Boolean flag. Failed runs are excluded from quality analysis and counted separately."),
]




def layout(**kwargs):  # called on every page visit → always fresh data
    df, src = S.try_load_data(_CSV, _CSV_FB, _CSV_FB2)

    # Build filter options from actual data
    all_opt   = [{"label": "All", "value": "__all__"}]
    datasets  = all_opt + [{"label": d, "value": d}
                           for d in sorted(df["dataset"].dropna().unique())]
    models    = all_opt + [{"label": m, "value": m}
                           for m in sorted(df["model"].dropna().unique())]

    th_s = {"fontSize": "10px", "fontWeight": "600", "color": S.TEXT2,
            "textTransform": "uppercase", "letterSpacing": "0.05em",
            "padding": "8px 12px", "borderBottom": f"2px solid {S.BORDER}",
            "textAlign": "left", "background": S.BG}
    td_s = {"fontSize": "12px", "padding": "8px 12px",
            "borderBottom": f"1px solid {S.BORDER}", "verticalAlign": "top"}

    return html.Div([
        # ── Hero ─────────────────────────────────────────────────────────────
        html.Div([
            html.Div("SHAP-IQ",
                     style={"fontSize": "10px", "fontWeight": "700",
                            "color": "rgba(255,255,255,0.65)",
                            "letterSpacing": "0.16em", "textTransform": "uppercase",
                            "marginBottom": "12px"}),
            html.H1("Shapley Approximation Benchmark",
                    style={"fontSize": "26px", "fontWeight": "700",
                           "color": "white", "margin": "0 0 10px",
                           "letterSpacing": "-0.02em", "lineHeight": "1.25"}),
            html.P(
                "Four research questions evaluate Shapley approximation libraries "
                "across dimensionality, accuracy, neural networks, and tree model complexity.",
                style={"color": "rgba(255,255,255,0.75)", "fontSize": "13px",
                       "lineHeight": "1.75", "margin": "0 0 22px", "maxWidth": "600px"},
            ),
            # Quick-link buttons
            html.Div([
                html.A(f"→ {label}", href=href, style={
                    "display": "inline-block", "padding": "7px 16px",
                    "borderRadius": "6px", "fontSize": "12px", "fontWeight": "600",
                    "color": "white", "textDecoration": "none",
                    "background": "rgba(255,255,255,0.14)",
                    "border": "1px solid rgba(255,255,255,0.25)",
                    "marginRight": "8px", "marginBottom": "4px",
                    "transition": "background 0.15s",
                })
                for href, label in [
                    ("/rq1", "RQ1 — Dimensionality"),
                    ("/rq2", "RQ2 — Accuracy"),
                    ("/rq3", "RQ3 — Neural Networks"),
                    ("/rq4", "RQ4 — Tree Models"),
                ]
            ]),
        ], style={
            "background": f"linear-gradient(135deg, {S.ACCENT} 0%, #7C3AED 55%, {S.PINK} 100%)",
            "borderRadius": "14px", "padding": "40px 32px",
            "marginBottom": "28px",
        }),

        # ── Global filter bar ────────────────────────────────────────────────
        S.filter_bar(
            S.filter_dropdown("Dataset", "home-ds-select", datasets, "__all__", "220px"),
            S.filter_dropdown("Model",   "home-mdl-select", models,  "__all__", "200px"),
        ),

        # ── KPI strip + leaderboard ──────────────────────────────────────────
        html.Div(id="home-kpi-strip"),
        S.section(
            "Global Method Summary",
            "All methods ranked by median Spearman ρ (rank correlation with exact Shapley values). "
            "Ties broken by median runtime (faster = higher). Reflects the current filter selection.",
            html.Div(id="home-leaderboard-table"),
        ),

        # ── Metric reference ─────────────────────────────────────────────────
        S.section(
            "Metric Reference",
            "Definitions of all quality metrics used across every page.",
            html.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Metric",         style=th_s),
                        html.Th("Formula",        style=th_s),
                        html.Th("Interpretation", style=th_s),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(name,   style={**td_s, "fontFamily": "monospace",
                                                   "color": S.ACCENT, "whiteSpace": "nowrap"}),
                            html.Td(formula, style={**td_s, "fontFamily": "monospace",
                                                    "color": S.TEXT2, "fontSize": "11px",
                                                    "whiteSpace": "nowrap"}),
                            html.Td(interp, style=td_s),
                        ])
                        for name, formula, interp in _METRIC_ROWS
                    ]),
                ],
                style={"width": "100%", "borderCollapse": "collapse"},
            ),
        ),

        # ── Library capability matrix ─────────────────────────────────────────
        S.section(
            "Library Capability Overview",
            "Which libraries support which explanation types. "
            "Grey = planned but not yet benchmarked.",
            html.Div(id="home-capability-table"),
        ),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("home-kpi-strip",        "children"),
    Output("home-leaderboard-table","children"),
    Output("home-capability-table", "children"),
    Input("home-ds-select",  "value"),
    Input("home-mdl-select", "value"),
)
def update_home(ds, mdl):
    df, _ = S.try_load_data(_CSV, _CSV_FB, _CSV_FB2)

    if ds  != "__all__": df = df[df["dataset"] == ds]
    if mdl != "__all__": df = df[df["model"]   == mdl]

    lb = S.compute_leaderboard(df)

    # KPI strip
    best     = lb.iloc[0] if not lb.empty else None
    n_meth   = lb.shape[0]
    worst_fr = lb["failure_rate"].max() * 100 if not lb.empty else 0
    worst_m  = lb.loc[lb["failure_rate"].idxmax(), "method"] if not lb.empty else "—"

    kpis = S.kpi_row(
        S.kpi_card(str(n_meth), "Methods compared"),
        S.kpi_card(best["method"] if best is not None else "—",
                   "Top ranked", S.ACCENT),
        S.kpi_card(f"ρ = {best['rho_median']:.3f}" if best is not None else "—",
                   "Best median Spearman ρ", S.GREEN),
        S.kpi_card(f"{worst_fr:.0f} %",
                   f"Worst failure rate ({worst_m})",
                   S.RED if worst_fr > 10 else S.GREEN),
    )

    table      = S.build_leaderboard_datatable(lb, table_id="home-lb-table")
    cap_table  = S.capability_matrix_table(set(df["library"].dropna().unique()) if not df.empty else set())

    return kpis, table, cap_table
