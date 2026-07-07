"""
pages/rq1_dimensionality.py  —  RQ1: Dimensionality

Edit the PAGE CONFIGURATION block below to change what is shown.
The layout / filter / callback logic beneath it should rarely need touching.
"""
import os
import sys

import dash
from dash import Input, Output, callback, dcc, html

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared as S

dash.register_page(
    __name__,
    path="/rq1",
    name="RQ1 — Dimensionality",
    title="RQ1 — Dimensionality",
)

_HERE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV     = os.path.join(_HERE, "results_config-dimensionality.csv")
_CSV_FB  = os.path.join(_HERE, "results_dimensionality.csv")
_CSV_FB2 = os.path.join(_HERE, "results.csv")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION — edit here to change what the page shows
# ═════════════════════════════════════════════════════════════════════════════

_RQ_HEADER = (
    "RQ1", "Dimensionality",
    "As a user with a dataset that has many features, I want to find the fastest "
    "model-agnostic library for high-dimensional datasets?"
)

# Notes shown on the page directly below the research question.
_REMARKS = [
    "Also test the inverse: datasets with a small number of features.",
    "Remark: Use of multiple seeds.",
    "Scope: captum is excluded from model-agnostic questions; "
    "no Shapley value interactions in the model-agnostic part.",
]

# Charts rendered in order.
#   title      — section heading
#   subtitle   — one-line description shown below the heading
#   fn         — S.fig_* function that accepts the filtered DataFrame
#   section_id — stable HTML id for advisor deep-links (optional)
_CHARTS = [
    dict(
        title      = "Runtime vs Number of Features",
        subtitle   = "Median runtime on a log–log scale. "
                     "A steeper slope means worse scalability.",
        fn         = S.fig_runtime_vs_features,
        section_id = "rq1-runtime-section",
    ),
    dict(
        title      = "Spearman ρ vs Number of Features",
        subtitle   = "Does rank-order agreement with exact Shapley values hold as "
                     "dimensionality grows? Declining lines signal accuracy concerns. "
                     "Reference line: ρ = 0.9.",
        fn         = S.fig_rho_vs_features,
        section_id = "rq1-quality-section",
    ),
    dict(
        title      = "Failure Rate Heatmap",
        subtitle   = "Fraction of failed runs per method × feature count. "
                     "Red cells mark where a method becomes unreliable.",
        fn         = S.fig_failure_heatmap_by_features,
        section_id = "rq1-failure-section",
    ),
]

_INTERP = (
    "How to read this page: look for methods whose runtime line stays flat as "
    "n_features grows (good scalability) while ρ stays near 1.0 (accuracy maintained). "
    "The failure heatmap reveals where methods break down entirely."
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
    approxs  = sorted(df["approximator"].dropna().unique()) if not df.empty else []
    n_feats  = sorted(df["n_features"].dropna().unique())   if not df.empty else []

    source_tag = html.Div(
        [html.Span("Data source: ", style={"fontWeight": "600"}),
         html.Code(os.path.basename(src) if src else "—",
                   style={"fontFamily": "monospace", "fontSize": "11px",
                          "background": S.BG, "padding": "2px 6px", "borderRadius": "4px"})],
        style={"fontSize": "12px", "color": S.TEXT2, "marginBottom": "4px"},
    ) if src else S.missing_data_banner(_CSV)

    return html.Div([
        S.rq_header(*_RQ_HEADER),
        *[S.info_note(r) for r in _REMARKS],
        source_tag,
        S.data_summary_card(df),
        S.charts_toc(_CHARTS),
        S.filter_bar(
            S.filter_dropdown("Dataset", "rq1-ds",  datasets, "__all__", "220px"),
            S.filter_dropdown("Model",   "rq1-mdl", models,   "__all__", "200px"),
            S.filter_checklist(
                "n_features", "rq1-nf",
                [{"label": f"  {int(n)}", "value": n} for n in n_feats],
                n_feats,
            ),
            S.filter_checklist(
                "Approximator", "rq1-approx",
                [{"label": f"  {a}", "value": a} for a in approxs],
                approxs,
            ),
        ),
        html.Div(id="rq1-kpis"),
        html.Div(id="rq1-charts"),
    ])


# ═════════════════════════════════════════════════════════════════════════════
#  Filter helper
# ═════════════════════════════════════════════════════════════════════════════

def _apply_filters(df, ds, mdl, nf_vals, approxs):
    if ds  != "__all__": df = df[df["dataset"]      == ds]
    if mdl != "__all__": df = df[df["model"]        == mdl]
    if approxs:          df = df[df["approximator"].isin(approxs)]
    if nf_vals:          df = df[df["n_features"].isin([float(v) for v in nf_vals])]
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  Callback
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("rq1-kpis",   "children"),
    Output("rq1-charts", "children"),
    Input("rq1-ds",      "value"),
    Input("rq1-mdl",     "value"),
    Input("rq1-nf",      "value"),
    Input("rq1-approx",  "value"),
)
def update_rq1(ds, mdl, nf_vals, approxs):
    df, _ = S.try_load_data(_CSV, _CSV_FB, _CSV_FB2)
    fdf   = _apply_filters(df, ds, mdl, nf_vals or [], approxs or [])

    # ── KPIs ──────────────────────────────────────────────────────────────
    unique_nf = fdf["n_features"].dropna().nunique() if not fdf.empty else 0
    max_nf    = int(fdf["n_features"].max()) if not fdf.empty and fdf["n_features"].notna().any() else 0
    success   = (1 - fdf["is_failure"].mean()) if not fdf.empty else 0
    if not fdf.empty and fdf["n_features"].notna().any():
        hi = fdf[fdf["n_features"] == fdf["n_features"].max()]
        fastest_m = hi.groupby("method")["runtime_s"].median().idxmin() \
                    if not hi.empty and hi["runtime_s"].notna().any() else "—"
    else:
        fastest_m = "—"

    kpis = S.kpi_row(
        S.kpi_card(str(unique_nf), "Feature-count settings"),
        S.kpi_card(str(max_nf),    "Highest n_features benchmarked"),
        S.kpi_card(fastest_m,      f"Fastest method at n_features={max_nf}", S.ACCENT),
        S.kpi_card(f"{success * 100:.0f} %", "Overall success rate",
                   S.GREEN if success >= 0.8 else S.AMBER),
    )

    # ── Warnings ──────────────────────────────────────────────────────────
    warns = []
    if unique_nf < 2:
        warns.append(S.warning_note(
            "Select at least two n_features values to see scaling behaviour."
        ))

    # ── Special: side-by-side speed ranking at smallest vs largest selected ──
    side_by_side = []
    sel_nf = sorted(fdf["n_features"].dropna().unique()) if not fdf.empty else []
    if len(sel_nf) >= 2:
        lo_n = int(sel_nf[0])
        hi_n = int(sel_nf[-1])
        lo_df = fdf[fdf["n_features"] == sel_nf[0]]
        hi_df = fdf[fdf["n_features"] == sel_nf[-1]]
        side_by_side = [html.Div([
            html.Div([S.section(
                f"Speed Ranking — n_features = {lo_n}",
                "Fastest to slowest at the smallest selected feature count.",
                dcc.Graph(figure=S.fig_speed_ranking_at_nfeatures(lo_df, lo_n),
                          config={"displayModeBar": False}, style={"padding": "8px"}),
                section_id="rq1-speed-section",
            )], style={"flex": "1", "minWidth": "300px"}),
            html.Div([S.section(
                f"Speed Ranking — n_features = {hi_n}",
                "Fastest to slowest at the largest selected feature count.",
                dcc.Graph(figure=S.fig_speed_ranking_at_nfeatures(hi_df, hi_n),
                          config={"displayModeBar": False}, style={"padding": "8px"}),
            )], style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"})]

    # ── Main charts — driven by _CHARTS manifest above ────────────────────
    charts = html.Div([
        *warns,
        *side_by_side,
        *[
            S.section(
                c["title"], c["subtitle"],
                dcc.Graph(figure=c["fn"](fdf),
                          config={"displayModeBar": False}, style={"padding": "8px"}),
                section_id=c.get("section_id"),
            )
            for c in _CHARTS
        ],
        S.interpretation_note(_INTERP),
    ])

    return kpis, charts



