"""
pages/rq4_trees.py  —  RQ4: Tree Models

Edit the PAGE CONFIGURATION block below to change what is shown.
The layout / filter / callback logic beneath it should rarely need touching.

Data source: results/rq4_trees.csv (raw)
    → results/converted/rq4_trees_by_seed.csv    (one row per run, loaded here)
    → results/converted/rq4_trees_pairwise.csv   (long-format agreement pairs)
    via results_converters/rq4_results_converter.py

The tree depth used throughout is always the *realized* depth the fitted tree
actually reached (never the requested sweep target) — trees asked for depth
50/80 routinely saturate well below that.

Tree explainers here are exact methods (path-dependent / interventional /
interaction TreeSHAP variants), not approximations trading accuracy for speed —
so "agreement" charts on this page measure cross-library consistency, and a dip
is a correctness red flag (bug/numerical instability), not an expected
cost/quality trade-off.

Chart set is based on TREES.md's 3/2/3 layout, plus depth-sweep diagnostics
(failure-rate-vs-depth, depth-scaling-factor, a quick-look overview line chart,
and a per-backend agreement ranking) added back into every tab they're
meaningful for. Every chart section carries a `mode` key (path_dependent /
interventional / interaction) so update_rq4() always filters to one
computation mode (a column precomputed by the converter, not re-derived from
the backend string at callback time) before calling a chart's `fn` — backends
from different modes are never mixed into the same panel/line color. Chart
entries additionally carry a `source` key: "by_seed" (default) reads the
per-run dataframe, "pairwise" reads the long-format agreement-pairs dataframe
(needed for the backend x backend heatmap, which can't be reconstructed from
the by-seed dataframe's own already-averaged agreement column).
"""

import os
import sys

import dash
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

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONV = os.path.join(_HERE, "results", "converted")
_CSV = os.path.join(_CONV, "rq4_trees_by_seed.csv")
_PAIRWISE = os.path.join(_CONV, "rq4_trees_pairwise.csv")


def _load(path: str) -> pd.DataFrame:
    """Read a converted CSV directly — unlike the raw-CSV pages (home, and
    formerly this one), everything here (mode, is_failure, same-mode-peer
    quality metrics) is already computed by
    results_converters/rq4_results_converter.py, so this deliberately skips
    shared.data.load_data()'s generic reprocessing rather than risk it
    recomputing is_failure from relative_mae and clobbering the converter's
    additivity_gap-based signal (see that module's docstring for why trees
    need the different signal)."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


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
_REMARKS = []

# (column, label) pairs to surface in the "Fixed parameters" box, in priority
# order — only ones that are actually present AND constant across every row
# of the current CSV get shown.
_FIXED_PARAM_COLS = [
    ("n_background",  "background samples"),
    ("n_eval",        "eval samples"),
    ("n_estimators",  "n_estimators"),
]

_CASE_OPTIONS = [
    {"label": "Path-dependent", "value": "path-dependent"},
    {"label": "Interventional", "value": "interventional"},
    {"label": "Interaction", "value": "interaction"},
]
_DEFAULT_CASE = "path-dependent"

# case-tab value -> mode key used by fig_tree_* functions (precomputed by the
# converter into the `mode` column on both the by-seed and pairwise CSVs)
_CASE_TO_MODE = {
    "path-dependent": "path_dependent",
    "interventional": "interventional",
    "interaction": "interaction",
}


_AGREEMENT_NOTE = (
    "mean_sample_rho for a backend is its average Spearman ρ against every "
    "*other* same-mode backend, not a distance to one fixed reference (e.g. "
    "shap isn't treated as ground truth) — it's a symmetric peer-consensus "
    "measure, so a low value means a backend disagrees with the rest of the "
    "pack, not that it's necessarily the one that's wrong."
)


def _pass_fail_entry(section_id: str, mode: str) -> dict:
    return dict(
        section_id=section_id, mode=mode,
        title="Library × Model: Valid Output (by dataset)",
        subtitle="One panel per dataset, library × model within each — green = "
        "every run produced valid output, red = at least one failed outright "
        "(not a quality issue, the run itself never produced usable values). "
        "A library that's red in every panel has a model-only problem (e.g. "
        "fails on xgboost regardless of dataset); one that's red in a single "
        "panel has a problem tied to that specific dataset too. Hover a red "
        "cell for the exact failure rate.",
        fn=S.fig_tree_pass_fail_matrix,
    )


def _pathdep_charts() -> list:
    return [
        _pass_fail_entry("rq4-pd-pass-fail", "path_dependent"),
        dict(
            section_id="rq4-pd-agreement", mode="path_dependent",
            title="Cross-library Agreement (per model)",
            subtitle="Cell (A, B) = mean sign_agreement: the fraction of "
            "feature-level Shapley values where A and B agree on the *direction* "
            "of a feature's contribution (positive vs. negative), averaged over "
            "every sample/config both computed a valid result for. 1.0 = every "
            "feature's sign matched every time — it's not a distance/error score "
            "and not runtime. One panel per model family; dataset/depth/seed are "
            "pooled within each panel since this is a correctness check, not a "
            "scaling question — cells well below 1.0 flag a numerical/"
            "implementation bug, not an approximation trade-off.",
            fn=S.fig_tree_agreement_heatmap_by_model,
            source="pairwise",
        ),
        dict(
            section_id="rq4-pd-ranking", mode="path_dependent",
            title="Cross-Library Agreement by Backend",
            subtitle="Single-number companion to the heatmap above, one bar per "
            f"backend. {_AGREEMENT_NOTE} Faded bars flag backends that produced "
            "no valid output at all.",
            fn=S.fig_tree_quality_ranking,
        ),
        dict(
            section_id="rq4-pd-runtime-overview", mode="path_dependent",
            title="Runtime vs Tree Depth (quick overview)",
            subtitle="A fast global glance, pooling whichever datasets/models are "
            "currently selected — see the faceted breakdown right below for the "
            "dataset × model detail this necessarily hides.",
            fn=S.fig_tree_runtime_vs_depth,
        ),
        dict(
            section_id="rq4-pd-depth", mode="path_dependent",
            title="Runtime vs Max Depth",
            subtitle="The core depth-cost sweep. One panel per dataset × model "
            "family; each line is a library, with a median + range band across "
            "the (up to 10) seeds in that panel. A rising line means the method "
            "gets more expensive as trees deepen; a flat line is depth-robust.",
            fn=lambda d: S.fig_tree_runtime_vs_depth_faceted(d, facet_model_cols=True),
        ),
        dict(
            section_id="rq4-pd-scaling", mode="path_dependent",
            title="Depth Scaling Factor (shallow → deep)",
            subtitle="How much each backend's runtime blows up from the shallowest "
            "to the deepest tree, computed within each dataset/model combo first "
            "and then summarized (median) so no single dataset dominates the "
            "baseline. A bar near 1× means the method barely notices depth.",
            fn=S.fig_tree_depth_scaling_factor,
        ),
        dict(
            section_id="rq4-pd-agreement-overview", mode="path_dependent",
            title="Cross-Library Agreement vs Tree Depth (quick overview)",
            subtitle="Does agreement survive deep trees? Same quick-overview "
            f"caveat as the runtime chart above (pools all selected datasets/"
            f"models). {_AGREEMENT_NOTE}",
            fn=S.fig_tree_quality_vs_depth,
        ),
    ]


def _interventional_charts() -> list:
    return [
        _pass_fail_entry("rq4-iv-pass-fail", "interventional"),
        dict(
            section_id="rq4-iv-ranking", mode="interventional",
            title="Cross-Library Agreement by Backend",
            subtitle=f"{_AGREEMENT_NOTE} Faded bars flag backends that produced no "
            "valid output at all.",
            fn=S.fig_tree_quality_ranking,
        ),
        dict(
            section_id="rq4-iv-runtime-overview", mode="interventional",
            title="Runtime vs Tree Depth (quick overview)",
            subtitle="A fast global glance, pooling whichever datasets/models are "
            "currently selected — see the faceted breakdown right below.",
            fn=S.fig_tree_runtime_vs_depth,
        ),
        dict(
            section_id="rq4-iv-depth", mode="interventional",
            title="Runtime vs Max Depth",
            subtitle="Same layout as the path-dependent version, interventional "
            "backends only — is interventional computation more expensive than "
            "path-dependent at the same depth? (shap's true-value reference "
            "backend is excluded — it isn't a competing method.)",
            fn=lambda d: S.fig_tree_runtime_vs_depth_faceted(d, facet_model_cols=True),
        ),
        dict(
            section_id="rq4-iv-scaling", mode="interventional",
            title="Depth Scaling Factor (shallow → deep)",
            subtitle="Same construction as the path-dependent version, "
            "interventional backends only.",
            fn=S.fig_tree_depth_scaling_factor,
        ),
        dict(
            section_id="rq4-iv-agreement-overview", mode="interventional",
            title="Cross-Library Agreement vs Tree Depth (quick overview)",
            subtitle=f"Quick-overview version (pools all selected datasets/models). "
            f"{_AGREEMENT_NOTE}",
            fn=S.fig_tree_quality_vs_depth,
        ),
    ]


def _interaction_charts() -> list:
    return [
        _pass_fail_entry("rq4-in-pass-fail", "interaction"),
        dict(
            section_id="rq4-in-failure-depth", mode="interaction",
            title="Failure Rate vs Tree Depth",
            subtitle="Interaction (order-2) computation is expensive enough that "
            "shapiq/woodelf time out on deeper trees rather than hitting a fixed "
            "per-model incompatibility — this shows exactly where that failure "
            "rate creeps up as trees get deeper, which the library × model "
            "matrix above can't show.",
            fn=S.fig_tree_failure_vs_depth,
        ),
        dict(
            section_id="rq4-in-agreement", mode="interaction",
            title="Cross-library Agreement (per model)",
            subtitle="Same metric as the path-dependent heatmap (mean sign_agreement "
            "on the direction of each feature's contribution, 1.0 = always matches), "
            "computed on order-2 pairwise interaction values instead — do the exact "
            "interaction methods agree with each other?",
            fn=S.fig_tree_agreement_heatmap_by_model,
            source="pairwise",
        ),
        dict(
            section_id="rq4-in-ranking", mode="interaction",
            title="Cross-Library Agreement by Backend",
            subtitle=f"{_AGREEMENT_NOTE} Faded bars flag backends that produced no "
            "valid output at all.",
            fn=S.fig_tree_quality_ranking,
        ),
        dict(
            section_id="rq4-in-runtime-overview", mode="interaction",
            title="Runtime vs Tree Depth (quick overview)",
            subtitle="A fast global glance across the currently selected models "
            "(interaction data currently only covers the `bike` dataset).",
            fn=S.fig_tree_runtime_vs_depth,
        ),
        dict(
            section_id="rq4-in-order-cost", mode=None,
            title="Cost of Interaction Order (1 → 2)",
            subtitle="Bars compare median runtime per library at order 1 (main "
            "effects, path-dependent backends) vs order 2 (pairwise interactions) "
            "— the only chart on this page that spans two computation modes on "
            "purpose. For shap and shapiq, order 2 is a genuine, large cost jump. "
            "fasttreeshap and woodelf instead time out on a large share of their "
            "order-2 runs, so their median there is measured only on the easy "
            "runs that finished — the hatched, †-marked bars flag this "
            "survivorship bias rather than a real speed-up; see the failure-rate "
            "chart above for how often that happens.",
            fn=S.fig_tree_order_cost,
            custom_filter=lambda df: df[df["mode"].isin(["path_dependent", "interaction"])],
        ),
        dict(
            section_id="rq4-in-depth", mode="interaction",
            title="Runtime vs Max Depth",
            subtitle="Rows = dataset only; model family isn't faceted here (there's "
            "currently only one dataset with interaction data, so this is a single "
            "panel) — narrow the Model filter above to isolate one model family, "
            "otherwise each line pools across whichever models are selected.",
            fn=lambda d: S.fig_tree_runtime_vs_depth_faceted(d, facet_model_cols=False),
        ),
        dict(
            section_id="rq4-in-scaling", mode="interaction",
            title="Depth Scaling Factor (shallow → deep)",
            subtitle="Same construction as the path-dependent version, order-2 "
            "interaction backends only.",
            fn=S.fig_tree_depth_scaling_factor,
        ),
        dict(
            section_id="rq4-in-agreement-overview", mode="interaction",
            title="Cross-Library Agreement vs Tree Depth (quick overview)",
            subtitle=f"{_AGREEMENT_NOTE}",
            fn=S.fig_tree_quality_vs_depth,
        ),
    ]


_CASE_BUILDERS = {
    "path-dependent": _pathdep_charts,
    "interventional": _interventional_charts,
    "interaction": _interaction_charts,
}


def _build_charts(case: str) -> list:
    """Returns the chart manifest for one RQ4 case tab.

    Each entry carries a `mode` key — update_rq4() filters the dataframe to that
    computation mode before calling the chart's `fn`, so backends from different
    modes are never mixed into the same panel/line color. The one exception is
    "Cost of Interaction Order", which is *about* comparing two modes (order-1
    path-dependent vs. order-2 interaction runtime) — it sets `mode=None` and
    supplies its own `custom_filter` instead. It's fine (and expected) for
    different tabs to show a different number/kind of charts — order-2
    interaction data only exists for one dataset, for example.

    Entries also carry a `source` key ("by_seed" if absent, or "pairwise").
    The two agreement-heatmap entries set `source="pairwise"` — they need the
    long-format backend x backend comparison table (results/converted/
    rq4_trees_pairwise.csv), which can't be reconstructed from the by-seed
    dataframe's own already-peer-averaged mean_sample_rho/sign_agreement
    columns.
    """
    return _CASE_BUILDERS.get(case, _pathdep_charts)()


_INTERP = (
    "How to read this page: tree explainers here are exact methods, not "
    "approximations trading accuracy for speed, so the agreement heatmaps, "
    "rankings and ρ-vs-depth charts all measure cross-library correctness/"
    "consistency — a dip signals a numerical bug, not an expected cost/quality "
    "trade-off. Path-dependent and Interventional share the same layout: a "
    "quick-overview line chart first, then a small-multiple panel per "
    "dataset × model (median + range band across seeds) so nothing gets "
    "averaged across incomparable feature counts or model families, then the "
    "depth-scaling-factor ranking. Interactions add the order-1-vs-order-2 cost "
    "chart and the quadratic-blowup-by-feature-count chart, since that's the "
    "reason interaction_max_features caps the feature count at all — its depth "
    "and feature charts are currently single-panel since the data only covers "
    "one dataset (`bike`) so far."
)


def _pill(text: str, bg: str, color: str) -> html.Span:
    return html.Span(text, style={
        "fontSize": "11px", "fontWeight": "500", "color": color,
        "background": bg, "borderRadius": "4px",
        "padding": "2px 8px", "marginRight": "4px", "marginBottom": "4px",
        "display": "inline-block",
    })


def _col(heading: str, items: list, bg: str, color: str) -> html.Div:
    return html.Div([
        html.Div(heading, style={
            "fontSize": "10px", "fontWeight": "700", "color": S.TEXT2,
            "textTransform": "uppercase", "letterSpacing": "0.07em",
            "marginBottom": "8px",
        }),
        html.Div([_pill(i, bg, color) for i in items]),
    ], style={"flex": "1", "minWidth": "160px"})


def _config_card(df: pd.DataFrame) -> html.Div:
    """"Benchmark at a glance" card, matching the RQ1/RQ2/RQ3/RQ5 layout —
    a handful of curated facts read directly from the data (not hardcoded, so
    it stays correct if the sweep config changes), instead of the old
    data-source line + fixed-parameter chips + raw per-column value dump,
    which repeated the dataset/library/model lists already visible in the
    filters below and buried the few facts that actually matter.
    """
    n_ds = df["dataset"].nunique() if "dataset" in df.columns else 0
    n_models = df["model"].nunique() if "model" in df.columns else 0
    n_libs = df["library"].nunique() if "library" in df.columns else 0
    n_seeds = df["seed"].nunique() if "seed" in df.columns else 0

    swept_items = [f"{n_ds} datasets", f"{n_models} model families",
                   f"{n_libs} libraries", "3 computation modes"]
    if "max_depth" in df.columns:
        depths = pd.to_numeric(df["max_depth"], errors="coerce").dropna()
        if not depths.empty:
            swept_items.append(
                f"realized max_depth: {int(depths.min())}–{int(depths.max())}")

    fixed_items = []
    for col, label in _FIXED_PARAM_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna().unique()
        if len(vals) != 1:
            continue
        v = vals[0]
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        fixed_items.append(f"{label} = {v}")
    if n_seeds:
        fixed_items.append(f"{n_seeds} seeds")
    # There's no literal "timeout" column — infer it from the data instead: a
    # cluster of runs sitting right at the same runtime ceiling is a strong
    # signal of a per-run wall-clock budget (see the interaction-tab failures,
    # which are timeouts, not bugs).
    if "runtime_s" in df.columns:
        max_rt = df["runtime_s"].max()
        if pd.notna(max_rt) and (df["runtime_s"] > max_rt - 1).sum() >= 5:
            fixed_items.append(f"backend timeout (inferred) ≈ {max_rt:.0f}s")

    left  = _col("Swept", swept_items, "#EEF2FF", S.ACCENT)
    mid   = _col("Fixed", fixed_items or ["—"], "#F1F5F9", S.TEXT2)
    right = _col("Measured",
                 ["runtime_s", "sign_agreement / mean_sample_rho (cross-library "
                  "agreement)", "additivity_gap"],
                 "#F0FDF4", S.GREEN)

    return html.Div([
        html.Div("Benchmark at a glance", style={
            "fontSize": "13px", "fontWeight": "700", "color": S.TEXT,
            "marginBottom": "14px",
        }),
        html.Div([left, mid, right],
                 style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
    ], style={
        "background": S.CARD, "borderRadius": "12px",
        "border": f"1px solid {S.BORDER}", "padding": "20px 24px",
        "marginBottom": "20px",
    })


def layout(**kwargs):
    df = _load(_CSV)

    if df.empty:
        return html.Div([
            S.rq_header(*_RQ_HEADER),
            *[S.info_note(r) for r in _REMARKS],
            S.missing_data_banner(_CSV),
            _schema_hint(),
        ])

    datasets = sorted(df["dataset"].dropna().unique())
    libs = sorted(df["library"].dropna().unique())
    models = sorted(df["model"].dropna().unique()) if "model" in df.columns else []

    return html.Div([
        S.rq_header(*_RQ_HEADER),
        *[S.info_note(r) for r in _REMARKS],
        _config_card(df),
        S.filter_bar(
            S.filter_checklist(
                "Dataset", "rq4-ds",
                [{"label": f"  {d}", "value": d} for d in datasets], datasets,
            ),
            S.filter_checklist(
                "Library", "rq4-lib",
                [{"label": f"  {lib}", "value": lib} for lib in libs], libs,
            ),
            S.filter_checklist(
                "Model", "rq4-model",
                [{"label": f"  {m}", "value": m} for m in models], models,
            ),
        ),
        html.Div([
            html.Label("Case focus", style={
                       "fontWeight": "600", "fontSize": "13px"}),
            dcc.Tabs(
                id="rq4-case",
                value=_DEFAULT_CASE,
                children=[
                    dcc.Tab(
                        label=opt["label"],
                        value=opt["value"],
                        style={
                            "padding": "8px 12px",
                            "border": f"1px solid {S.BORDER}",
                            "borderRadius": "999px",
                            "marginRight": "8px",
                            "marginTop": "6px",
                            "background": "white",
                            "color": S.TEXT2,
                            "fontSize": "12px",
                            "fontWeight": "500",
                        },
                        selected_style={
                            "padding": "8px 12px",
                            "border": f"1px solid {S.ACCENT}",
                            "borderRadius": "999px",
                            "marginRight": "8px",
                            "marginTop": "6px",
                            "background": S.ACCENT,
                            "color": "white",
                            "fontSize": "12px",
                            "fontWeight": "600",
                        },
                    )
                    for opt in _CASE_OPTIONS
                ],
                parent_style={"marginTop": "6px"},
                style={"display": "flex", "flexWrap": "wrap", "gap": "4px"},
            ),
        ], style={"margin": "8px 0 12px", "padding": "8px 12px",
                  "border": f"1px solid {S.BORDER}", "borderRadius": "8px",
                  "background": S.BG}),

        # ── Dynamic content ───────────────────────────────────────────────────
        html.Div(id="rq4-kpis"),
        html.Div(id="rq4-charts"),
    ])


def _schema_hint() -> html.Div:
    rows = [
        ("dataset",          "benchmark dataset name"),
        ("model",            "model type: random_forest, lightgbm, xgboost"),
        ("max_depth",        "realized tree depth — this drives the depth axis"),
        ("max_depth_config", "requested/target depth cap (context only, not plotted)"),
        ("library",          "explanation library"),
        ("approximator",     "unused for tree backends — always empty"),
        ("runtime_s",        "wall-clock seconds"),
        ("relative_mae",     "cross-implementation MAE vs. same-mode peers (optional) — "
                              "these are exact methods, so this flags bugs/numerical "
                              "instability, not approximation error"),
        ("sign_agreement / mean_sample_rho", "cross-library agreement proxies (optional)"),
    ]
    th_s = {"fontSize": "10px", "fontWeight": "600", "color": S.TEXT2,
            "textTransform": "uppercase", "letterSpacing": "0.05em",
            "padding": "8px 12px", "borderBottom": f"2px solid {S.BORDER}",
            "textAlign": "left", "background": S.BG}
    td_s = {"fontSize": "12px", "padding": "8px 12px",
            "borderBottom": f"1px solid {S.BORDER}"}
    return S.section(
        "Expected CSV Schema",
        "Create rq4_trees.csv with at least the columns below. "
        "The max_depth column is the key — without it the depth charts show an "
        "empty-state message.",
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
    Input("rq4-ds",    "value"),
    Input("rq4-lib",   "value"),
    Input("rq4-model", "value"),
    Input("rq4-case",  "value"),
)
def update_rq4(datasets, libs, models, case):
    df = _load(_CSV)
    if df.empty:
        return html.Div(), html.Div()
    pairwise_df = _load(_PAIRWISE)

    charts_man = _build_charts(case)

    if datasets:
        df = df[df["dataset"].isin(datasets)]
        pairwise_df = pairwise_df[pairwise_df["dataset"].isin(datasets)]
    if libs:
        df = df[df["library"].isin(libs)]
        # Only the "anchor" side of each pair is restricted to the selected
        # libraries, matching what the row's own backend/library filter did
        # before conversion — the peer it's compared against is never itself
        # filtered out, since pairwise_metrics compares every row against
        # every same-mode peer regardless of the Library selection.
        pairwise_df = pairwise_df[pairwise_df["library_a"].isin(libs)]
    if models and "model" in df.columns:
        df = df[df["model"].isin(models)]
        pairwise_df = pairwise_df[pairwise_df["model"].isin(models)]

    # KPIs/warnings describe whichever mode the current case tab covers.
    kpi_mode = _CASE_TO_MODE.get(case, "path_dependent")
    kpi_df = df[df["mode"] == kpi_mode]

    # ── KPIs — describe the axes that actually vary in the tree benchmark ──
    def _label(backend) -> str:
        return backend.replace("_", " ") if isinstance(backend, str) else "—"

    n_backends = kpi_df["backend"].dropna().nunique() \
        if "backend" in kpi_df.columns and not kpi_df.empty else 0

    feat_range = "—"
    if "n_features" in kpi_df.columns:
        feats = pd.to_numeric(kpi_df["n_features"], errors="coerce").dropna()
        if not feats.empty:
            lo, hi = int(feats.min()), int(feats.max())
            feat_range = f"{lo}" if lo == hi else f"{lo}–{hi}"

    # Fastest backend that actually returns valid values.
    fastest = "—"
    valid = kpi_df[~kpi_df["is_failure"] &
                   kpi_df["runtime_s"].notna()] if not kpi_df.empty else kpi_df
    if not valid.empty and "backend" in valid.columns:
        rt_by_backend = valid.groupby("backend")["runtime_s"].median()
        if not rt_by_backend.empty:
            fastest = _label(rt_by_backend.idxmin())

    # Count of backends that never produce a valid explanation (breaking points).
    n_broken = 0
    if not kpi_df.empty and "backend" in kpi_df.columns:
        fr_by_backend = kpi_df.groupby("backend")["is_failure"].mean()
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
    elif kpi_df.empty:
        warns.append(S.warning_note(
            f"No rows remain for the selected case focus: {case}."))
    elif n_backends < 2:
        warns.append(S.warning_note(
            "Only one algorithm variant in the current selection. "
            "Widen the Library filter above to compare backends."
        ))

    # ── Charts — driven by _build_charts() manifest above ───────────────────
    def _chart_df(c: dict) -> pd.DataFrame:
        if c.get("custom_filter") is not None:
            return c["custom_filter"](df)
        if c.get("source") == "pairwise":
            return pairwise_df[pairwise_df["mode"] == c["mode"]] \
                if c["mode"] is not None else pairwise_df
        return df[df["mode"] == c["mode"]] if c["mode"] is not None else df

    charts = html.Div([
        *warns,
        *[
            S.section(
                c["title"], c["subtitle"],
                dcc.Graph(figure=c["fn"](_chart_df(c)),
                          config={"displayModeBar": False}, style={"padding": "8px"}),
                section_id=c["section_id"],
            )
            for c in charts_man
        ],
        S.interpretation_note(_INTERP),
    ])

    return kpis, charts
