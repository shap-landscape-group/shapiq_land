"""
pages/rq4_trees.py  —  RQ4: Tree Models

Edit the PAGE CONFIGURATION block below to change what is shown.
The layout / filter / callback logic beneath it should rarely need touching.

Data file: results_config-tree-merged.csv (falls back to rq4_trees.csv).
The complexity axis auto-detects: tree_depth > max_depth > n_estimators > model.
This merged CSV carries a real ``max_depth`` column (4 → 80), so the page can
finally answer RQ4 directly: how does each library scale with tree depth?
"""

import shared as S
import os
import sys

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


dash.register_page(
    __name__,
    path="/rq4",
    name="RQ4 — Tree Models",
    title="RQ4 — Tree Models",
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Prefer the merged CSV (has a real max_depth column → true RQ4 answer);
# fall back to the older depth-less file so the page still renders.
_CSV = os.path.join(_HERE, "results", "results_config-tree-merged.csv")
_CSV_FALLBACK = os.path.join(_HERE, "results", "rq4_trees.csv")


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
    """Returns the chart manifest for the RQ4 tree-explanation benchmark.

    Called from both layout() and update_rq4() so the TOC and chart sections
    stay in sync. These charts key off the ``backend`` column (the actual
    algorithm variant, e.g. ``shapiq_tree_path_dependent``) rather than the
    generic ``library / approximator`` method label, because the tree CSVs
    leave ``approximator`` empty — which would otherwise merge genuinely
    different algorithms onto a single misleading line.

    To add or remove charts, edit the list returned here.
    section_id values must be unique and stable (used as DOM anchors).

    The first three charts use the ``max_depth`` axis (4 → 80) that the merged
    CSV finally provides — these are the ones that directly answer RQ4 ("which
    library handles extreme tree depths efficiently?"). They only render when a
    depth column is present; on the older depth-less CSV they show an empty-state
    message and the backend-level charts below carry the page.
    """
    has_depth = bool(comp_col) and comp_col in (
        "max_depth", "tree_depth", "depth")

    depth_charts = [
        dict(
            section_id="rq4-depth-runtime-section",
            title="Runtime vs Tree Depth",
            subtitle="The direct answer to RQ4. Each line is one algorithm backend; "
            "the x-axis is the maximum tree depth (4 → 80) and the y-axis is median "
            "runtime on a log scale. A rising line means the method gets more "
            "expensive as trees deepen — a potential bottleneck for deep models — "
            "while a flat line means the backend is depth-robust.",
            fn=S.fig_tree_runtime_vs_depth,
        ),
        dict(
            section_id="rq4-depth-scaling-section",
            title="Depth Scaling Factor (shallow → deep)",
            subtitle="How much each backend's runtime blows up going from the "
            "shallowest to the deepest tree in the sweep. A bar near 1× means the "
            "method barely notices depth (ideal for extreme trees); a large factor "
            "flags a backend whose cost explodes as depth grows. This is the "
            "single clearest ranking for the efficiency half of RQ4.",
            fn=S.fig_tree_depth_scaling_factor,
        ),
        dict(
            section_id="rq4-depth-quality-section",
            title="Approximation Quality vs Tree Depth",
            subtitle="Efficiency is only half the story: a backend that stays fast "
            "but loses rank fidelity on deep trees is still a poor choice. Lines "
            "that stay near the top (and above the ρ = 0.9 threshold) keep their "
            "explanations trustworthy even at extreme depths.",
            fn=S.fig_tree_quality_vs_depth,
        ),
    ]

    backend_charts = [
        dict(
            section_id="rq4-scaling-section",
            title="Runtime vs Number of Features",
            subtitle="The real complexity axis. Each line is one backend; a steep slope "

            "means the method becomes a computational bottleneck as the "
            "explanation problem grows in dimensionality. Flat lines scale "
            "gracefully.",
            fn=S.fig_tree_runtime_vs_features,
        ),
        dict(
            section_id="rq4-order-section",
            title="Cost of Interaction Order (1 → 2)",
            subtitle="Moving from main effects (order 1) to pairwise interactions "
            "(order 2) is the single biggest cost jump in the benchmark. "
            "Bars compare median runtime per library at each order (log scale).",
            fn=S.fig_tree_order_cost,
        ),
        dict(
            section_id="rq4-tradeoff-section",
            title="Speed vs Accuracy Trade-off",
            subtitle="Runtime (log x) against approximation quality (Spearman ρ) per "
            "backend. The sweet spot is the upper-left — fast and faithful. "
            "Points below ρ = 0.9 lose rank fidelity.",
            fn=S.fig_tree_accuracy_vs_runtime,
        ),
        dict(
            section_id="rq4-ranking-section",
            title="Approximation Quality by Backend",
            subtitle="Median Spearman ρ per algorithm variant across all settings in the "
            "current selection. Faded bars flag backends that produced no valid "
            "output; the dotted line marks the ρ = 0.9 fidelity threshold.",
            fn=S.fig_tree_quality_ranking,
        ),
    ]

    # Depth charts lead when the data supports them; otherwise the backend-level
    # charts (which still render on the old CSV) carry the page.
    return (depth_charts + backend_charts) if has_depth else backend_charts


_INTERP = (
    "How to read this page: the first three charts answer RQ4 head-on — runtime-vs-depth "
    "(with worst-case whiskers) shows how each backend scales as trees deepen, the "
    "depth-scaling-factor bars rank how much cost blows up from shallow to deep trees, and "
    "quality-vs-depth confirms fidelity survives. The remaining charts add context: how cost "
    "scales with feature count, the jump from main effects to pairwise interactions, and the "
    "overall speed-vs-accuracy trade-off (upper-left = fast and faithful)."
)


def layout(**kwargs):
    df, src = S.try_load_data(_CSV, _CSV_FALLBACK)

    if src is None:
        return html.Div([
            S.rq_header(*_RQ_HEADER),
            *[S.info_note(r) for r in _REMARKS],
            S.missing_data_banner(_CSV),
            _schema_hint(),
        ])

    datasets = [{"label": "All datasets", "value": "__all__"}] + \
        [{"label": d, "value": d}
            for d in sorted(df["dataset"].dropna().unique())]
    libs = sorted(df["library"].dropna().unique())
    comp_col = S._complexity_col(df)
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
            S.filter_dropdown("Dataset", "rq4-ds",
                              datasets, "__all__", "220px"),
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
    df, src = S.try_load_data(_CSV, _CSV_FALLBACK)
    if src is None or df.empty:
        return html.Div(), html.Div()

    comp_col = S._complexity_col(df)
    charts_man = _build_charts(comp_col)

    if ds != "__all__":
        df = df[df["dataset"] == ds]
    if libs:
        df = df[df["library"].isin(libs)]
    if comp_vals:
        df = df[df[comp_col].isin(comp_vals)]

    # ── KPIs — describe the axes that actually vary in the tree benchmark ──
    def _label(backend) -> str:
        return backend.replace("_", " ") if isinstance(backend, str) else "—"

    n_backends = df["backend"].dropna().nunique(
    ) if "backend" in df.columns and not df.empty else 0

    feat_range = "—"
    if "n_features" in df.columns:
        feats = pd.to_numeric(df["n_features"], errors="coerce").dropna()
        if not feats.empty:
            lo, hi = int(feats.min()), int(feats.max())
            feat_range = f"{lo}" if lo == hi else f"{lo}–{hi}"

    # Fastest backend that actually returns valid values.
    fastest = "—"
    valid = df[~df["is_failure"] &
               df["runtime_s"].notna()] if not df.empty else df
    if not valid.empty and "backend" in valid.columns:
        rt_by_backend = valid.groupby("backend")["runtime_s"].median()
        if not rt_by_backend.empty:
            fastest = _label(rt_by_backend.idxmin())

    # Count of backends that never produce a valid explanation (breaking points).
    n_broken = 0
    if not df.empty and "backend" in df.columns:
        fr_by_backend = df.groupby("backend")["is_failure"].mean()
        n_broken = int((fr_by_backend >= 0.5).sum())

    kpis = S.kpi_row(
        S.kpi_card(str(n_backends),  "Algorithm variants (backends)"),
        S.kpi_card(feat_range,       "Feature counts benchmarked"),
        S.kpi_card(fastest,          "Fastest valid backend", S.GREEN),
        S.kpi_card(str(n_broken),    "Backends with no valid output",
                   S.RED if n_broken else S.TEXT2),
    )

    # ── Warnings ──────────────────────────────────────────────────────────
    warns = []
    if df.empty:
        warns.append(S.warning_note(
            "No data matches the current filter selection."))
    elif n_backends < 2:
        warns.append(S.warning_note(
            "Only one algorithm variant in the current selection. "
            "Widen the Library filter above to compare backends."
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
