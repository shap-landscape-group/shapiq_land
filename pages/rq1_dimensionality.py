"""
pages/rq1_dimensionality.py — RQ1: Dimensionality

Research question:
    "As a user with a dataset that has many features, I want to find the
    fastest model-agnostic library for high-dimensional datasets."

Data flow:
    RQ1+RQ2/results_config-dimensionality*.csv
        → results_converters/rq1_results_converter.py   (validation,
          flags, seed aggregation — run it after refreshing raw data)
        → results/converted/rq1_*.csv                    (loaded here)
        → figures on this page

This page contains NO aggregation across seeds — that happened in the
converter. Callbacks only filter; any residual display-level pooling
(e.g. "All models" = median across the 4 models) is done inside the
figure builders and stated in the figure comment plus the on-chart note.

Data limitations respected here:
    * Every method in the RQ1 files is an approximation. Pairwise metrics
      are cross-method AGREEMENT, never called accuracy.
    * The 1,000-feature stress test (1 dataset, 1 seed, 1 budget, 7 methods)
      is shown in its own clearly-labelled section, never on the standard
      scaling curves.
    * lightshap runs that hit the ~600 s execution cap are excluded from
      runtime medians (their true runtime is unknown) and surfaced in the
      feasibility heatmap instead.
"""
import os
import sys

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared as S

dash.register_page(
    __name__,
    path="/rq1",
    name="RQ1 — Dimensionality",
    title="RQ1 — Dimensionality",
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONV = os.path.join(_HERE, "results", "converted")

_AGGREGATED = os.path.join(_CONV, "rq1_scaling_aggregated.csv")
_FEASIBILITY = os.path.join(_CONV, "rq1_feasibility.csv")
_EXTREME = os.path.join(_CONV, "rq1_extreme_stress_test.csv")

# ─────────────────────────────────────────────────────────────────────────────
_RQ_HEADER = (
    "RQ1", "Dimensionality",
    "As a user with a dataset that has many features, I want to find the "
    "fastest model-agnostic library for high-dimensional datasets?",
)

# Interpretation revised for the new data: runtime/feasibility claims only.
# The old text implied a quality axis ("while quality stays high") — RQ1
# data contains no exact reference, so no quality-vs-truth claim is made.
_INTERP = (
    "Compare how steeply each method's runtime grows as n_features increases "
    "— flatter is better. The feasibility heatmap shows where methods hit the "
    "10-minute execution cap (all such runs are lightshap). Cross-method "
    "agreement is consistency between approximations, not accuracy against "
    "an exact reference — the RQ1 benchmark contains approximations only. "
    "The 1,000-feature stress test is a separate one-shot experiment and is "
    "not comparable to the standard scaling grid."
)

_DATASET_ORDER = ["ames_housing", "covertype", "diabetes_130", "gisette"]

# Line style: color = library, dash = approximator. 7 methods resolve into
# 4 colors × 2 dashes without needing a 7-color palette.
_APPROX_DASH = {"kernel": "solid", "permutation": "dot"}


def _lib_color(lib: str) -> str:
    return S.LIB_COLOR.get(lib, S.ACCENT)


def _load(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  Local layout helpers (unchanged style from the previous page version)
# ─────────────────────────────────────────────────────────────────────────────

def _pill(text, bg="#EEF2FF", color=S.ACCENT) -> html.Span:
    return html.Span(text, style={
        "display": "inline-block", "background": bg, "color": color,
        "fontSize": "11px", "fontWeight": "600",
        "padding": "3px 9px", "borderRadius": "4px",
        "marginRight": "5px", "marginBottom": "5px",
        "border": f"1px solid {color}40",
        "whiteSpace": "nowrap",
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


def _config_card() -> html.Div:
    """Benchmark-at-a-glance card, updated for the new experimental grid."""
    left = _col("Swept",
                ["4 datasets", "4 models", "n_features: 14–256",
                 "4 libraries", "2 approximators",
                 "budget: 512, 1024", "10 seeds"],
                "#EEF2FF", S.ACCENT)
    mid = _col("Fixed",
               ["n_background = 100", "n_eval = 10", "imputer = marginal",
                "10-min execution cap"],
               "#F1F5F9", S.TEXT2)
    right = _col("Measured",
                 ["runtime_s", "n_model_evals",
                  "cross-method agreement (no exact reference)"],
                 "#F0FDF4", S.GREEN)
    stress = _col("Separate stress test",
                  ["gisette @ 1,000 features", "seed 0 only",
                   "budget 2048", "7 methods"],
                  "#FEF3C7", "#92400E")

    return html.Div([
        html.Div("Benchmark at a glance", style={
            "fontSize": "13px", "fontWeight": "700", "color": S.TEXT,
            "marginBottom": "14px",
        }),
        html.Div([left, mid, right, stress],
                 style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
    ], style={
        "background": S.CARD, "borderRadius": "12px",
        "border": f"1px solid {S.BORDER}", "padding": "20px 24px",
        "marginBottom": "20px",
    })


def _col_note(*parts: str) -> html.Div:
    """Compact data-provenance line inside a chart card."""
    children = []
    for i, part in enumerate(parts):
        children.append(html.Span(part, style={"whiteSpace": "pre"}))
        if i < len(parts) - 1:
            children.append(html.Span("  ·  ", style={"color": S.BORDER}))
    return html.Div(children, style={
        "fontSize": "10px", "color": S.TEXT2, "fontFamily": "monospace",
        "padding": "4px 12px 6px",
        "borderBottom": f"1px solid {S.BORDER}",
        "background": S.BG,
        "letterSpacing": "0.01em",
    })


def _axis_toggle(cid: str, options: dict, default, label="Axis") -> html.Div:
    return html.Div([
        html.Span(f"{label}:", style={
            "fontSize": "11px", "fontWeight": "600", "color": S.TEXT2,
            "marginRight": "10px", "flexShrink": "0",
        }),
        dcc.RadioItems(
            id=cid,
            options=[{"label": v, "value": k} for k, v in options.items()],
            value=default,
            inline=True,
            inputStyle={"marginRight": "4px"},
            labelStyle={"marginRight": "16px", "fontSize": "12px",
                        "cursor": "pointer", "color": S.TEXT},
        ),
    ], style={
        "display": "flex", "alignItems": "center",
        "padding": "8px 12px",
        "borderBottom": f"1px solid {S.BORDER}",
        "background": S.BG,
        "borderRadius": "10px 10px 0 0",
    })


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE RQ1-F1 — Runtime / model-eval scaling by feature count
#
# Question answered:
#     How does computational cost change as dimensionality increases, and
#     which methods scale most effectively?
#
# Source:
#     rq1_scaling_aggregated.csv
#
# Raw CSV inputs:
#     dataset, model, n_features, seed, library, approximator, budget,
#     runtime_s, n_model_evals
#
# Row selection:
#     Standard dimensionality experiment only — the 1,000-feature stress
#     test is a separate figure (RQ1-F5). One budget at a time (radio),
#     because overlaying both budgets doubles the line count without
#     changing the scaling story; the budget effect has its own figure
#     (RQ1-F2). Time-capped runs were excluded from medians in the
#     converter.
#
# Grouping:
#     dataset × model × n_features × method × budget (from converter).
#     Display-level: "All models" pools the 4 models by median and says so
#     in the on-chart note; "All datasets" draws a 2×2 small-multiple grid
#     (one panel per dataset) instead of mixing incomparable feature grids
#     into one line.
#
# Seed aggregation:
#     Median across 10 seeds; band = q25–q75 (from converter).
#
# Visual encoding:
#     x = n_features (log)      y = median cost (log)
#     color = library           dash = approximator
#     band = seed IQR           panel = dataset (when All datasets)
#
# Why:
#     Log-log slopes make scaling behaviour directly comparable; small
#     multiples preserve each dataset's own feature grid (14→79 for
#     ames_housing vs 14→256 for gisette) instead of averaging them.
#
# Interpretation supported:
#     Methods whose line rises more slowly handle added features better
#     within the tested range.
#
# Limitation:
#     Measures computational scaling only — no statement about attribution
#     accuracy is possible from RQ1 data.
# ─────────────────────────────────────────────────────────────────────────────

def _scaling_traces(sub: pd.DataFrame, metric: str, showlegend: bool):
    """Median line + q25–q75 band per method for one dataset panel."""
    traces = []
    for method, mdf in sub.groupby("method"):
        lib = mdf["library"].iloc[0]
        approx = mdf["approximator"].iloc[0]
        color = _lib_color(lib)
        mdf = mdf.sort_values("n_features")

        if metric == "runtime_s":
            y, ylo, yhi = mdf["runtime_median"], mdf["runtime_q25"], mdf["runtime_q75"]
        else:
            y = mdf["evals_median"]
            ylo = yhi = None

        # Seed IQR band (runtime only — eval counts are deterministic per
        # config, their seed spread is negligible).
        if ylo is not None:
            traces.append(go.Scatter(
                x=pd.concat([mdf["n_features"], mdf["n_features"][::-1]]),
                y=pd.concat([yhi, ylo[::-1]]),
                fill="toself", fillcolor=color + "22",
                line=dict(width=0), hoverinfo="skip",
                legendgroup=method, showlegend=False,
            ))
        traces.append(go.Scatter(
            x=mdf["n_features"], y=y,
            mode="lines+markers", name=method,
            legendgroup=method, showlegend=showlegend,
            line=dict(color=color, width=2, dash=_APPROX_DASH.get(approx, "solid")),
            marker=dict(size=7, color=color, line=dict(color="white", width=1.2)),
            hovertemplate=(
                f"<b>{method}</b><br>n_features: %{{x}}<br>"
                f"{'median runtime: %{y:.3g} s' if metric == 'runtime_s' else 'median model evals: %{y:.3g}'}"
                "<extra></extra>"
            ),
        ))
    return traces


def _build_runtime_scaling_figure(agg: pd.DataFrame, dataset: str,
                                  metric: str) -> go.Figure:
    if agg.empty:
        return S.fig_empty("No converted data — run results_converters/rq1_results_converter.py")

    y_title = ("Median runtime (s) — log" if metric == "runtime_s"
               else "Median model evaluations — log")

    if dataset != "__all__":
        sub = agg[agg["dataset"] == dataset]
        if sub.empty:
            return S.fig_empty()
        fig = go.Figure()
        for tr in _scaling_traces(sub, metric, showlegend=True):
            fig.add_trace(tr)
        fig.update_layout(
            **S._CHART_LAYOUT, height=440, margin=S._MARGIN, legend=S._LEGEND_H,
            xaxis=dict(title="n_features (log)", type="log",
                       gridcolor=S.BORDER, zeroline=False),
            yaxis=dict(title=y_title, type="log",
                       gridcolor=S.BORDER, zeroline=False),
        )
        return fig

    # "All datasets": small multiples — one panel per dataset, because the
    # datasets use different feature grids and pooling them into one line
    # would average incomparable x-positions.
    datasets = [d for d in _DATASET_ORDER if d in agg["dataset"].unique()]
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=datasets,
        horizontal_spacing=0.09, vertical_spacing=0.14,
    )
    for i, ds in enumerate(datasets):
        row, col = divmod(i, 2)
        sub = agg[agg["dataset"] == ds]
        for tr in _scaling_traces(sub, metric, showlegend=(i == 0)):
            fig.add_trace(tr, row=row + 1, col=col + 1)
    fig.update_xaxes(type="log", gridcolor=S.BORDER, zeroline=False,
                     title_text="n_features")
    fig.update_yaxes(type="log", gridcolor=S.BORDER, zeroline=False)
    fig.update_layout(
        **S._CHART_LAYOUT, height=640, legend=S._LEGEND_H,
        margin=dict(l=55, r=16, t=60, b=48),
    )
    fig.update_annotations(font_size=12)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE RQ1-F2 — Budget effect on runtime (512 → 1024)
#
# Question answered:
#     How does doubling the sampling budget affect runtime per method?
#
# Source:
#     rq1_scaling_aggregated.csv
#
# Raw CSV inputs:
#     library, approximator, budget, runtime_s (dataset, model, n_features
#     as grouping context)
#
# Row selection:
#     Standard experiment; both budgets required per configuration cell —
#     cells where either budget's runtime median is missing (all runs
#     capped) are dropped.
#
# Grouping:
#     ratio = runtime_median(budget 1024) / runtime_median(budget 512)
#     computed per dataset × model × n_features × method, then the
#     distribution of ratios is shown per method as a box. This keeps
#     every experiment cell visible instead of collapsing to one bar.
#
# Seed aggregation:
#     Already median-per-cell from the converter; the box shows spread
#     ACROSS experiment cells, not across seeds.
#
# Visual encoding:
#     x = method   y = runtime ratio (linear)   reference line at 2.0
#
# Why:
#     For pure sampling methods runtime should scale ~linearly with
#     budget (ratio ≈ 2). Deviations reveal fixed overheads (ratio < 2)
#     or superlinear cost growth (ratio > 2).
#
# Interpretation supported:
#     Methods with ratios well below 2 are overhead-dominated at these
#     budgets; increasing their budget is comparatively cheap.
#
# Limitation:
#     Only two budgets exist in RQ1 — no full budget curve (RQ2 has three
#     budgets at fixed dimensionality).
# ─────────────────────────────────────────────────────────────────────────────

def _build_budget_effect_figure(agg: pd.DataFrame, dataset: str) -> go.Figure:
    if agg.empty:
        return S.fig_empty()
    sub = agg if dataset == "__all__" else agg[agg["dataset"] == dataset]

    cell_keys = ["dataset", "model", "n_features", "method", "library"]
    wide = sub.pivot_table(index=cell_keys, columns="budget",
                           values="runtime_median").reset_index()
    if 512 not in wide.columns or 1024 not in wide.columns:
        return S.fig_empty("Need both budgets in the current filter")
    wide = wide.dropna(subset=[512, 1024])
    wide["ratio"] = wide[1024] / wide[512]

    fig = go.Figure()
    order = (wide.groupby("method")["ratio"].median()
             .sort_values().index.tolist())
    for method in order:
        mdf = wide[wide["method"] == method]
        color = _lib_color(mdf["library"].iloc[0])
        fig.add_trace(go.Box(
            y=mdf["ratio"], name=method,
            marker_color=color, line_color=color,
            boxpoints="all", jitter=0.35, pointpos=0,
            marker=dict(size=4, opacity=0.5),
            fillcolor="rgba(0,0,0,0)",
            hovertemplate=f"<b>{method}</b><br>runtime ×%{{y:.2f}} at 2× budget<extra></extra>",
            showlegend=False,
        ))
    fig.add_hline(y=2.0, line=dict(color=S.TEXT2, width=1.2, dash="dot"),
                  annotation_text="2× — linear budget scaling",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color=S.TEXT2))
    fig.update_layout(
        **S._CHART_LAYOUT, height=400,
        margin=dict(l=55, r=16, t=36, b=90),
        xaxis=dict(tickangle=-25, gridcolor="rgba(0,0,0,0)", automargin=True),
        yaxis=dict(title="runtime(1024) / runtime(512)",
                   gridcolor=S.BORDER, zeroline=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE RQ1-F3 — Execution-cap feasibility heatmap
#
# Question answered:
#     Which method × dataset × dimensionality combinations fail to finish
#     inside the 10-minute execution cap?
#
# Source:
#     rq1_feasibility.csv
#
# Raw CSV inputs:
#     library, approximator, dataset, n_features, seed, runtime_s
#
# Row selection:
#     Standard experiment. A run counts as capped when runtime_s ≥ 595 s
#     (the raw data clusters at 600 ± 0.02 s — the configured cap).
#
# Grouping:
#     method × dataset × n_features. Models and budgets are pooled ON
#     PURPOSE (stated here and in the converter): a cap in any
#     model/budget combination is a deployment risk for that method at
#     that dimensionality. Denominator = 80 runs per cell
#     (4 models × 2 budgets × 10 seeds).
#
# Seed aggregation:
#     Fraction of capped runs over all runs in the cell.
#
# Visual encoding:
#     x = dataset @ n_features    y = method    cell = % capped runs
#     green (0 %) → red (100 %)
#
# Why:
#     "Fastest library" is meaningless for configurations that cannot
#     finish; this chart maps the feasibility boundary directly.
#
# Interpretation supported:
#     Non-zero cells mark configurations that cannot be safely deployed
#     under a 10-minute budget.
#
# Limitation:
#     Capped runs still produced output — the cap indicates runtime
#     infeasibility, not a crash.
# ─────────────────────────────────────────────────────────────────────────────

def _build_feasibility_heatmap(feas: pd.DataFrame, dataset: str) -> go.Figure:
    if feas.empty:
        return S.fig_empty()
    sub = feas if dataset == "__all__" else feas[feas["dataset"] == dataset]
    if sub.empty:
        return S.fig_empty()

    sub = sub.copy()
    sub["cell_label"] = (sub["dataset"] + " @ " +
                         sub["n_features"].astype(int).astype(str))
    col_order = (sub[["cell_label", "dataset", "n_features"]]
                 .drop_duplicates()
                 .sort_values(["dataset", "n_features"])["cell_label"].tolist())

    pivot = (sub.pivot_table(index="method", columns="cell_label",
                             values="time_cap_fraction")
             .reindex(columns=col_order))
    z = pivot.values * 100
    text = [[f"{v:.0f}%" if not np.isnan(v) else "—" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=list(pivot.columns), y=list(pivot.index),
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#D1FAE5"], [0.5, "#FEF3C7"], [1, "#FEE2E2"]],
        zmin=0, zmax=100,
        colorbar=dict(title="% capped", thickness=14, len=0.8),
        hovertemplate=("Method: <b>%{y}</b><br>Cell: <b>%{x}</b><br>"
                       "Capped runs: %{z:.1f}%<extra></extra>"),
    ))
    fig.update_layout(
        **S._CHART_LAYOUT, height=max(280, len(pivot) * 40 + 110),
        xaxis=dict(title="dataset @ n_features", tickangle=-30,
                   gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="", gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=16, t=20, b=90),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE RQ1-F4 — Cross-method agreement vs feature count
#
# Question answered:
#     Do the approximate methods keep agreeing with each other as
#     dimensionality grows, or do their outputs drift apart?
#
# Source:
#     rq1_scaling_aggregated.csv (column cross_method_rho_median)
#
# Raw CSV inputs:
#     pairwise_metrics JSON (mean_sample_rho against the other 13 methods
#     in the same dataset × model × n_features × seed cell)
#
# Row selection:
#     Standard experiment, one budget at a time (same radio as RQ1-F1).
#
# Grouping:
#     Per run, the converter averaged rho over the 13 other methods; here
#     the median of that consensus score is drawn per method ×
#     n_features. "All models"/"All datasets" pool by median with an
#     explicit note.
#
# Seed aggregation:
#     Median across 10 seeds (converter).
#
# Visual encoding:
#     x = n_features (log)   y = median consensus ρ (linear)
#     color = library, dash = approximator
#
# Why:
#     Without ground truth, mutual agreement is the only quality-adjacent
#     signal RQ1 data supports. Falling agreement at high dimensions means
#     at least some methods drift — it cannot say which one is right.
#
# Interpretation supported:
#     Configurations where consensus stays high behave consistently as
#     dimensions grow; a drop marks budget starvation relative to M.
#
# Limitation:
#     Agreement is NOT accuracy: all methods could agree and still be
#     wrong. Accuracy claims belong to RQ2, which has an exact reference.
# ─────────────────────────────────────────────────────────────────────────────

def _build_agreement_figure(agg: pd.DataFrame, dataset: str) -> go.Figure:
    if agg.empty or "cross_method_rho_median" not in agg.columns:
        return S.fig_empty()
    sub = agg if dataset == "__all__" else agg[agg["dataset"] == dataset]
    sub = sub.dropna(subset=["cross_method_rho_median"])
    if sub.empty:
        return S.fig_empty()

    # Display-level pooling across datasets/models (median), stated in the
    # on-chart note. Each dataset has its own feature grid, so when all
    # datasets are shown the x-axis unions the grids; lines are per method.
    grp = (sub.groupby(["method", "library", "approximator", "n_features"])
           ["cross_method_rho_median"].median().reset_index())

    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        color = _lib_color(mdf["library"].iloc[0])
        dash = _APPROX_DASH.get(mdf["approximator"].iloc[0], "solid")
        mdf = mdf.sort_values("n_features")
        fig.add_trace(go.Scatter(
            x=mdf["n_features"], y=mdf["cross_method_rho_median"],
            mode="lines+markers", name=method,
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=7, color=color, line=dict(color="white", width=1.2)),
            hovertemplate=(f"<b>{method}</b><br>n_features: %{{x}}<br>"
                           "consensus ρ: %{y:.3f}<extra></extra>"),
        ))
    fig.update_layout(
        **S._CHART_LAYOUT, height=420, margin=S._MARGIN, legend=S._LEGEND_H,
        xaxis=dict(title="n_features (log)", type="log",
                   gridcolor=S.BORDER, zeroline=False),
        yaxis=dict(title="Median cross-method agreement ρ  (NOT accuracy)",
                   gridcolor=S.BORDER, zeroline=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE RQ1-F5 — 1,000-feature stress test (separate experiment)
#
# Question answered:
#     Which methods remain computationally feasible when the feature count
#     is pushed to 1,000?
#
# Source:
#     rq1_extreme_stress_test.csv
#
# Raw CSV inputs:
#     model, library, approximator, runtime_s, n_model_evals
#     (gisette, n_features = 1000, seed 0, budget 2048 — fixed)
#
# Row selection:
#     The entire extreme file (28 rows). Page filters are NOT applied:
#     this experiment covers one dataset and one seed, so topbar filters
#     would silently empty or bias it.
#
# Grouping:
#     None — one bar per run (method × model). A single seed means any
#     aggregation would fake statistical support.
#
# Seed aggregation:
#     None possible (seed 0 only) — stated in the section subtitle.
#
# Visual encoding:
#     y = method grouped by model    x = runtime_s (log)
#     hatched red = run exceeded the 10-minute cap (runtime is the
#     actual wall clock here, up to ~3.5 h — the cap did not stop it)
#
# Why:
#     Horizontal bars on a log axis keep 12.9 s and 12,724 s readable in
#     one view; per-model grouping shows the model-overhead effect at
#     extreme dimensionality.
#
# Interpretation supported:
#     Feasibility ranking at M = 1,000 — which methods stay in the
#     minutes range vs escalate to hours.
#
# Limitation:
#     One dataset, one seed, one budget: indicative, not statistical.
#     Never merge these values into the standard scaling curves.
# ─────────────────────────────────────────────────────────────────────────────

def _build_extreme_stress_figure(extreme: pd.DataFrame) -> go.Figure:
    if extreme.empty:
        return S.fig_empty("No stress-test data converted")

    sub = extreme.sort_values(["model", "runtime_s"]).copy()
    sub["label"] = sub["method"] + "  ·  " + sub["model"]

    colors = [_lib_color(l) for l in sub["library"]]
    patterns = ["/" if c else "" for c in sub["hit_time_cap"]]

    fig = go.Figure(go.Bar(
        y=sub["label"], x=sub["runtime_s"], orientation="h",
        marker=dict(color=colors, opacity=0.88,
                    line=dict(color="white", width=0.5),
                    pattern=dict(shape=patterns, fgcolor="#B91C1C", size=5)),
        text=[f"{v:,.0f} s" + ("  ⚠ over cap" if c else "")
              for v, c in zip(sub["runtime_s"], sub["hit_time_cap"])],
        textposition="outside", textfont=dict(size=10, color=S.TEXT),
        customdata=np.stack([sub["model"], sub["n_model_evals"]], axis=1),
        hovertemplate=("<b>%{y}</b><br>runtime: %{x:,.1f} s<br>"
                       "model evals: %{customdata[1]:,.0f}<extra></extra>"),
    ))
    fig.update_layout(
        **S._CHART_LAYOUT, height=max(420, len(sub) * 24 + 80),
        xaxis=dict(title="Runtime (s) — log scale  ·  hatched = exceeded 10-min cap",
                   type="log", gridcolor=S.BORDER, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True,
                   tickfont=dict(size=10)),
        margin=dict(l=10, r=120, t=20, b=48),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────────────────────────────

def layout(**kwargs):
    return html.Div([
        S.rq_header(*_RQ_HEADER),
        _config_card(),

        # RQ1-F1
        S.section(
            "RQ1-F1 · Cost scaling by feature count",
            "Median cost per method (10 seeds, band = q25–q75). "
            "'All datasets' shows one panel per dataset because each dataset "
            "has its own feature grid. 'All models' pools the 4 models by "
            "median — select a model to isolate it. Time-capped runs excluded.",
            html.Div([
                _axis_toggle("rq1-cost-metric",
                             {"runtime_s": "runtime (s)",
                              "n_model_evals": "model evaluations"},
                             "runtime_s", label="Metric"),
                _axis_toggle("rq1-budget", {"512": "512", "1024": "1024"},
                             "512", label="Budget"),
                _col_note(
                    "source: converted/rq1_scaling_aggregated.csv",
                    "agg: median(10 seeds) in converter · display: median over models when 'All models'",
                    "capped runs excluded from medians",
                ),
                html.Div(id="rq1-f1-chart", style={"padding": "8px"}),
            ]),
            section_id="rq1-f1-section",
        ),

        # RQ1-F2
        S.section(
            "RQ1-F2 · Budget effect on runtime (512 → 1024)",
            "Each point = one dataset × model × n_features cell. Ratio ≈ 2 "
            "means runtime scales linearly with budget; below 2 means fixed "
            "overhead dominates.",
            html.Div([
                _col_note(
                    "source: converted/rq1_scaling_aggregated.csv",
                    "ratio = runtime_median(1024) / runtime_median(512) per experiment cell",
                ),
                html.Div(id="rq1-f2-chart", style={"padding": "8px"}),
            ]),
            section_id="rq1-f2-section",
        ),

        # RQ1-F3
        S.section(
            "RQ1-F3 · Execution-cap feasibility",
            "Share of runs hitting the 10-minute cap per method × dataset × "
            "n_features. Models, budgets and seeds pooled (80 runs per cell): "
            "a cap anywhere is a deployment risk.",
            html.Div([
                _col_note(
                    "source: converted/rq1_feasibility.csv",
                    "capped = runtime_s ≥ 595 s (raw data clusters at 600 ± 0.02 s)",
                    "cell = capped / all runs over 4 models × 2 budgets × 10 seeds",
                ),
                html.Div(id="rq1-f3-chart", style={"padding": "8px"}),
            ]),
            section_id="rq1-f3-section",
        ),

        # RQ1-F4
        S.section(
            "RQ1-F4 · Cross-method agreement vs dimensionality",
            "Consensus between the 14 approximate configurations. This is "
            "mutual agreement, NOT accuracy — RQ1 has no exact reference. "
            "Falling agreement at high M signals budget starvation.",
            html.Div([
                _col_note(
                    "source: converted/rq1_scaling_aggregated.csv · cross_method_rho_median",
                    "per run: mean ρ vs the other 13 methods in the same cell → median over seeds",
                ),
                html.Div(id="rq1-f4-chart", style={"padding": "8px"}),
            ]),
            section_id="rq1-f4-section",
        ),

        # RQ1-F5
        S.section(
            "RQ1-F5 · Extreme stress test — 1,000 features (separate experiment)",
            "gisette @ 1,000 features, budget 2048, seed 0 only, 7 methods × "
            "4 models. One-shot feasibility check — page filters do not apply "
            "and these bars are not comparable with the scaling curves above.",
            html.Div([
                _col_note(
                    "source: converted/rq1_extreme_stress_test.csv",
                    "no aggregation (single seed) · hatched bars exceeded the 10-min cap",
                ),
                html.Div(id="rq1-f5-chart", style={"padding": "8px"}),
            ]),
            section_id="rq1-f5-section",
        ),

        S.interpretation_note(_INTERP),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Callback — filtering only; all statistics were computed in the converter
# ─────────────────────────────────────────────────────────────────────────────

def _apply_filters(df, ds, mdl, approxs):
    """Topbar filtering. Model pooling note: when mdl == '__all__' the
    figure builders pool models by median and say so on the chart."""
    if ds != "__all__" and "dataset" in df.columns:
        df = df[df["dataset"] == ds]
    if mdl != "__all__" and "model" in df.columns:
        df = df[df["model"] == mdl]
    if approxs and "approximator" in df.columns:
        df = df[df["approximator"].isin(approxs)]
    return df


@callback(
    Output("rq1-f1-chart", "children"),
    Output("rq1-f2-chart", "children"),
    Output("rq1-f3-chart", "children"),
    Output("rq1-f4-chart", "children"),
    Output("rq1-f5-chart", "children"),
    Input("rq1-ds", "value"),
    Input("rq1-mdl", "value"),
    Input("rq1-approx", "value"),
    Input("rq1-cost-metric", "value"),
    Input("rq1-budget", "value"),
)
def update_rq1(ds, mdl, approxs, cost_metric, budget):
    ds = ds or "__all__"
    mdl = mdl or "__all__"
    cost_metric = cost_metric or "runtime_s"
    budget = int(budget or 512)

    agg = _load(_AGGREGATED)
    feas = _load(_FEASIBILITY)
    extreme = _load(_EXTREME)

    agg_f = _apply_filters(agg, ds, mdl, approxs or [])

    # F1 + F4 show one budget at a time (radio); "All models" pooling by
    # median happens here, matching the note on the charts.
    f1_input = agg_f[agg_f["budget"] == budget]
    if mdl == "__all__" and not f1_input.empty:
        f1_input = (
            f1_input.groupby(["dataset", "n_features", "library",
                              "approximator", "budget", "method"],
                             as_index=False)
            .agg(runtime_median=("runtime_median", "median"),
                 runtime_q25=("runtime_q25", "median"),
                 runtime_q75=("runtime_q75", "median"),
                 evals_median=("evals_median", "median"),
                 cross_method_rho_median=("cross_method_rho_median", "median"))
        )

    feas_f = feas if not feas.empty else feas
    if approxs:
        feas_f = feas_f[feas_f["approximator"].isin(approxs)]

    def _g(fig):
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    return (
        _g(_build_runtime_scaling_figure(f1_input, ds, cost_metric)),
        _g(_build_budget_effect_figure(agg_f, ds)),
        _g(_build_feasibility_heatmap(feas_f, ds)),
        _g(_build_agreement_figure(f1_input, ds)),
        # F5 deliberately unfiltered — single-seed stress test (see comment).
        _g(_build_extreme_stress_figure(extreme)),
    )
