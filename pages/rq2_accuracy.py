"""
pages/rq2_accuracy.py — RQ2: Approximation Accuracy

Research question:
    "As a user using Shapley approximations, I want to know how good the
    values actually are so that I can trust the explanations without
    wasting too much computing time needed for exact values."

Data flow:
    RQ1+RQ2/results_config-accuracy.csv
        → results_converters/rq2_results_converter.py   (validation,
          reference matching, seed aggregation)
        → results/converted/rq2_*.csv                    (loaded here)
        → figures on this page

Reference definition (fixed in the converter, restated on every chart):
    All error metrics are computed against `shap_true_value`, matched
    within the same dataset × model × seed cell. lightshap_exact and
    shapiq_true_value agree with it to numerical noise (1e-15 / 1e-7
    median relative MAE) so the choice among those three is immaterial;
    dalex_true_value deviates by a median 3.2% and is therefore NOT used
    as a reference — that deviation is itself shown in RQ2-F5.

Design guarantees from the data (checked by the converter):
    * n_features fixed at 12 → accuracy is isolated from scaling effects.
    * 5 datasets × 4 models × 10 seeds × 25 methods, fully crossed.
    * Budgets 128 / 512 / 2048 for all approximators.
"""
import os
import sys

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared as S

dash.register_page(
    __name__,
    path="/rq2",
    name="RQ2 — Accuracy",
    title="RQ2 — Approximation Accuracy",
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONV = os.path.join(_HERE, "results", "converted")

_BY_SEED = os.path.join(_CONV, "rq2_accuracy_by_seed.csv")
_CONVERGENCE = os.path.join(_CONV, "rq2_convergence_aggregated.csv")
_RUNTIME_ACC = os.path.join(_CONV, "rq2_runtime_accuracy.csv")
_REF_AGREEMENT = os.path.join(_CONV, "rq2_reference_agreement.csv")

# ─────────────────────────────────────────────────────────────────────────────
_RQ_HEADER = (
    "RQ2", "Approximation Accuracy",
    "As a user using Shapley approximations, I want to know how good the values "
    "actually are so that I can trust the explanations without wasting too much "
    "compute time on exact values.",
)

# Interpretation rewritten for the new data: three budgets, one fixed
# dimensionality (12 features), explicit single oracle, and the measured
# dalex reference deviation.
_INTERP = (
    "All errors are measured against the exact shap_true_value oracle at a "
    "fixed 12 features. Error falls monotonically with budget for every "
    "method — kernel and permutation differ mainly in how fast they start "
    "converging (lightshap arrives near its floor already at budget 128). "
    "The Pareto chart shows what exactness costs: exact backends are often "
    "cheaper than high-budget approximations at this dimensionality. Note "
    "that dalex_true_value disagrees with the other three exact backends by "
    "a median 3.2% relative MAE (RQ2-F5), so 'exact' is not a single "
    "well-defined point across libraries."
)

_APPROX_DASH = {"kernel": "solid", "permutation": "dot"}
_BUDGETS = [128, 512, 2048]


def _lib_color(lib: str) -> str:
    return S.LIB_COLOR.get(lib, S.ACCENT)


def _load(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  Local layout helpers
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
    left = _col("Swept",
                ["5 datasets", "4 models", "budget: 128, 512, 2048",
                 "4 libraries", "2 approximators", "10 seeds"],
                "#EEF2FF", S.ACCENT)
    mid = _col("Fixed",
               ["n_features = 12", "n_background = 100",
                "n_eval = 10", "imputer = marginal"],
               "#F1F5F9", S.TEXT2)
    right = _col("Reference",
                 ["oracle = shap_true_value",
                  "lightshap_exact ≈ oracle (1e-15)",
                  "shapiq_true_value ≈ oracle (1e-7)",
                  "dalex_true_value deviates 3.2%"],
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


def _col_note(*parts: str) -> html.Div:
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


def _pool_models_datasets(conv: pd.DataFrame, ds: str, mdl: str) -> pd.DataFrame:
    """Display-level pooling for the convergence figures.

    When 'All datasets' / 'All models' is selected, the remaining
    dimension(s) are pooled by MEDIAN over the aggregated cells. This is
    stated on each affected chart's provenance note. Selecting a specific
    dataset and model shows unpooled converter output.
    """
    group_cols = ["library", "approximator", "budget", "method"]
    if ds != "__all__":
        conv = conv[conv["dataset"] == ds]
    else:
        group_cols = ["dataset"] + group_cols if False else group_cols
    if mdl != "__all__":
        conv = conv[conv["model"] == mdl]

    value_cols = [c for c in conv.columns
                  if c.endswith(("_median", "_q25", "_q75"))]
    return (conv.groupby(group_cols, as_index=False)[value_cols].median())


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE RQ2-F1 — Error convergence over budget (relative MAE)
#
# Question answered:
#     How does approximation error change with budget, and do methods
#     converge toward the exact values?
#
# Source:
#     rq2_convergence_aggregated.csv
#
# Raw CSV inputs:
#     budget, library, approximator, pairwise_metrics["shap_true_value"]
#     .relative_mae (dataset, model, seed as structure)
#
# Row selection:
#     Approximation runs only (references have no budget). Reference =
#     shap_true_value, matched within the same dataset × model × seed cell.
#
# Grouping:
#     method × budget after optional dataset/model filter. 'All' selections
#     pool the aggregated cells by median (display-level, stated on chart).
#
# Seed aggregation:
#     Median across 10 seeds, band = q25–q75 (converter).
#
# Visual encoding:
#     x = budget (log2-spaced categories 128/512/2048)
#     y = median relative MAE (log)   color = library, dash = approximator
#     band = seed IQR
#
# Why:
#     Log-y makes multiplicative error reduction visible as straight-ish
#     decay; the IQR band shows whether a method's advantage is stable
#     across seeds or an artifact of one lucky draw.
#
# Interpretation supported:
#     Whether spending 4× more budget buys a proportional error reduction,
#     and which methods start near their floor already at budget 128.
#
# Limitation:
#     Fixed 12 features — convergence speed at higher dimensionality is
#     not covered by this data (see RQ1 for scaling).
# ─────────────────────────────────────────────────────────────────────────────

def _build_convergence_figure(pooled: pd.DataFrame, metric: str) -> go.Figure:
    if pooled.empty:
        return S.fig_empty("No converted data — run results_converters/rq2_results_converter.py")

    is_mae = metric == "relative_mae"
    med, q25, q75 = (("relative_mae_median", "relative_mae_q25", "relative_mae_q75")
                     if is_mae else ("rho_median", "rho_q25", "rho_q75"))

    fig = go.Figure()
    for method, mdf in pooled.groupby("method"):
        lib = mdf["library"].iloc[0]
        color = _lib_color(lib)
        dash = _APPROX_DASH.get(mdf["approximator"].iloc[0], "solid")
        mdf = mdf.sort_values("budget")

        fig.add_trace(go.Scatter(
            x=pd.concat([mdf["budget"], mdf["budget"][::-1]]),
            y=pd.concat([mdf[q75], mdf[q25][::-1]]),
            fill="toself", fillcolor=color + "22",
            line=dict(width=0), hoverinfo="skip",
            legendgroup=method, showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=mdf["budget"], y=mdf[med],
            mode="lines+markers", name=method,
            legendgroup=method,
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=8, color=color, line=dict(color="white", width=1.2)),
            hovertemplate=(f"<b>{method}</b><br>budget: %{{x}}<br>"
                           f"{metric}: %{{y:.4g}}<extra></extra>"),
        ))

    if not is_mae:
        fig.add_hline(y=S.RHO_GOOD,
                      line=dict(color=S.GREEN, width=1.2, dash="dot"),
                      annotation_text=f"ρ = {S.RHO_GOOD}",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=S.GREEN))

    fig.update_layout(
        **S._CHART_LAYOUT, height=440, margin=S._MARGIN, legend=S._LEGEND_H,
        xaxis=dict(title="Budget (coalition evaluations) — log",
                   type="log", tickvals=_BUDGETS,
                   gridcolor=S.BORDER, zeroline=False),
        yaxis=dict(
            title=("Median relative MAE vs shap_true_value — log"
                   if is_mae else "Median Spearman ρ vs shap_true_value"),
            type="log" if is_mae else "linear",
            gridcolor=S.BORDER, zeroline=False,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE RQ2-F2 — Runtime vs error trade-off (Pareto view)
#
# Question answered:
#     Which method × budget combinations give the best accuracy per second,
#     and when is paying for the exact computation the better deal?
#
# Source:
#     rq2_runtime_accuracy.csv
#
# Raw CSV inputs:
#     runtime_s, pairwise relative_mae vs shap_true_value, budget,
#     library, approximator, backend
#
# Row selection:
#     Approximation configurations (21) as circles/diamonds; exact
#     reference backends as X markers. dalex_true_value carries its
#     measured 3.2% deviation instead of 0 so its position is honest.
#
# Grouping:
#     method × budget, medians over seeds (and over the dataset/model
#     cells remaining after the filter — display pooling stated on chart).
#
# Seed aggregation:
#     Median (converter).
#
# Visual encoding:
#     x = median runtime (log)     y = median relative MAE (log, floored
#     at 1e-16 for the exact backends so they render on a log axis)
#     color = library   symbol = budget (circle 128 / diamond 512 /
#     square 2048, X = exact)
#
# Why:
#     The practitioner's actual decision is a joint runtime/error choice;
#     medians per configuration keep the chart readable (a full scatter of
#     4,200 runs hides the frontier).
#
# Interpretation supported:
#     Points toward the lower-left dominate. Comparing X markers with
#     high-budget circles answers "is the approximation even worth it at
#     12 features?"
#
# Limitation:
#     At 12 features exact computation is cheap; at higher dimensionality
#     the exact option disappears entirely, so this frontier does NOT
#     generalise to RQ1's regime.
# ─────────────────────────────────────────────────────────────────────────────

_BUDGET_SYMBOL = {128: "circle", 512: "diamond", 2048: "square"}
_EXACT_MAE_FLOOR = 1e-16   # display floor so exact backends render on log-y


def _build_pareto_figure(ra: pd.DataFrame, ds: str, mdl: str) -> go.Figure:
    if ra.empty:
        return S.fig_empty()

    sub = ra.copy()
    if ds != "__all__":
        sub = sub[sub["dataset"] == ds]
    if mdl != "__all__":
        sub = sub[sub["model"] == mdl]
    if sub.empty:
        return S.fig_empty()

    # Display pooling over remaining dataset/model cells (median), stated
    # in the chart provenance note.
    grp = (sub.groupby(["library", "approximator", "budget", "method",
                        "is_reference"], dropna=False, as_index=False)
           .agg(runtime=("runtime_median", "median"),
                mae=("relative_mae_median", "median")))

    fig = go.Figure()

    apx = grp[~grp["is_reference"]]
    for (lib, budget), gdf in apx.groupby(["library", "budget"]):
        color = _lib_color(lib)
        fig.add_trace(go.Scatter(
            x=gdf["runtime"], y=gdf["mae"].clip(lower=_EXACT_MAE_FLOOR),
            mode="markers",
            name=f"{lib} · budget {int(budget)}",
            marker=dict(color=color, size=11,
                        symbol=_BUDGET_SYMBOL.get(int(budget), "circle"),
                        line=dict(color="white", width=1.2)),
            customdata=gdf[["method"]].values,
            hovertemplate=("<b>%{customdata[0]}</b> · budget " + str(int(budget)) +
                           "<br>runtime: %{x:.3g} s<br>"
                           "relative MAE: %{y:.3g}<extra></extra>"),
        ))

    refs = grp[grp["is_reference"]]
    for _, r in refs.iterrows():
        color = _lib_color(r["library"])
        fig.add_trace(go.Scatter(
            x=[r["runtime"]], y=[max(r["mae"], _EXACT_MAE_FLOOR)],
            mode="markers+text",
            name=f"{r['method']} (exact)",
            marker=dict(color=color, size=14, symbol="x",
                        line=dict(color="white", width=1)),
            text=[r["method"].replace("_true_value", "").replace("_exact", "")],
            textposition="top center", textfont=dict(size=9, color=S.TEXT2),
            hovertemplate=(f"<b>{r['method']}</b> (exact)<br>"
                           "runtime: %{x:.3g} s<br>"
                           "relative MAE: %{y:.3g}<extra></extra>"),
            showlegend=False,
        ))

    fig.update_layout(
        **S._CHART_LAYOUT, height=480, margin=S._MARGIN, legend=S._LEGEND_H,
        xaxis=dict(title="Median runtime (s) — log", type="log",
                   gridcolor=S.BORDER, zeroline=False),
        yaxis=dict(title="Median relative MAE vs shap_true_value — log "
                         "(exact backends floored at 1e-16)",
                   type="log", gridcolor=S.BORDER, zeroline=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE RQ2-F3 — Seed stability of the error
#
# Question answered:
#     How much does approximation error vary across random seeds — can a
#     single run be trusted?
#
# Source:
#     rq2_accuracy_by_seed.csv (seed-level rows kept exactly for this)
#
# Raw CSV inputs:
#     seed, budget, library, approximator, relative_mae vs shap_true_value
#
# Row selection:
#     Approximation runs at ONE budget (radio, default 512) so the box
#     width reflects seed noise, not budget differences.
#
# Grouping:
#     One box per method over all runs in the filter (10 seeds × remaining
#     dataset/model cells). Pooling across datasets/models is intentional
#     here and stated: the question is run-to-run variation a user would
#     face, whatever their data looks like.
#
# Seed aggregation:
#     None — the distribution IS the message.
#
# Visual encoding:
#     x = method     y = relative MAE (log)     box + all points
#
# Why:
#     Box plots over raw seed-level values expose spread and outliers that
#     medians hide; a method with lower median but huge spread is a worse
#     practical choice than the medians alone suggest.
#
# Interpretation supported:
#     Tight boxes = reproducible error; long upper whiskers = risk that a
#     single unlucky run is much worse than the median.
#
# Limitation:
#     Fixed 12 features; spread at higher dimensionality is untested here.
# ─────────────────────────────────────────────────────────────────────────────

def _build_seed_stability_figure(by_seed: pd.DataFrame, ds: str, mdl: str,
                                 budget: int) -> go.Figure:
    if by_seed.empty:
        return S.fig_empty()

    sub = by_seed[by_seed["budget"] == budget]
    if ds != "__all__":
        sub = sub[sub["dataset"] == ds]
    if mdl != "__all__":
        sub = sub[sub["model"] == mdl]
    sub = sub.dropna(subset=["relative_mae"])
    if sub.empty:
        return S.fig_empty()

    order = (sub.groupby("method")["relative_mae"].median()
             .sort_values().index.tolist())
    fig = go.Figure()
    for method in order:
        mdf = sub[sub["method"] == method]
        color = _lib_color(mdf["library"].iloc[0])
        fig.add_trace(go.Box(
            y=mdf["relative_mae"], name=method,
            boxpoints="all", jitter=0.4, pointpos=0,
            marker=dict(color=color, size=3.5, opacity=0.4,
                        line=dict(color="white", width=0.4)),
            line=dict(color=color, width=2), fillcolor="rgba(0,0,0,0)",
            hovertemplate=f"<b>{method}</b><br>relative MAE: %{{y:.4g}}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        **S._CHART_LAYOUT, height=440,
        margin=dict(l=55, r=16, t=36, b=100),
        xaxis=dict(tickangle=-25, gridcolor="rgba(0,0,0,0)", automargin=True),
        yaxis=dict(title=f"Relative MAE vs shap_true_value @ budget {budget} — log",
                   type="log", gridcolor=S.BORDER, zeroline=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE RQ2-F4 — Sign agreement vs budget
#
# Question answered:
#     How often does the approximation get the DIRECTION of a feature's
#     contribution right, and does more budget fix sign errors?
#
# Source:
#     rq2_convergence_aggregated.csv (sign_agreement_median)
#
# Raw CSV inputs:
#     pairwise sign_agreement vs shap_true_value, budget, library,
#     approximator
#
# Row selection / grouping / aggregation:
#     Same pipeline as RQ2-F1 (median over 10 seeds in the converter,
#     display pooling by median when 'All' is selected).
#
# Visual encoding:
#     x = budget (log)   y = median sign agreement (linear, zoomed to data)
#     color = library, dash = approximator
#
# Why:
#     For many users the sign (+/-) of an attribution matters more than
#     its magnitude; sign agreement is the metric closest to "does the
#     explanation point the right way".
#
# Interpretation supported:
#     Budgets where sign agreement plateaus near 1.0 — a cheaper
#     correctness criterion than full value convergence.
#
# Limitation:
#     Sign agreement saturates quickly; it cannot discriminate between
#     methods that are all above ~0.99 (use RQ2-F1 for that).
# ─────────────────────────────────────────────────────────────────────────────

def _build_sign_agreement_figure(pooled: pd.DataFrame) -> go.Figure:
    if pooled.empty or "sign_agreement_median" not in pooled.columns:
        return S.fig_empty()

    fig = go.Figure()
    for method, mdf in pooled.groupby("method"):
        color = _lib_color(mdf["library"].iloc[0])
        dash = _APPROX_DASH.get(mdf["approximator"].iloc[0], "solid")
        mdf = mdf.sort_values("budget")
        fig.add_trace(go.Scatter(
            x=mdf["budget"], y=mdf["sign_agreement_median"],
            mode="lines+markers", name=method,
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=8, color=color, line=dict(color="white", width=1.2)),
            hovertemplate=(f"<b>{method}</b><br>budget: %{{x}}<br>"
                           "sign agreement: %{y:.4f}<extra></extra>"),
        ))
    vals = pooled["sign_agreement_median"].dropna()
    y_lo = max(0.0, float(vals.min()) - 0.01) if len(vals) else 0.0
    fig.update_layout(
        **S._CHART_LAYOUT, height=400, margin=S._MARGIN, legend=S._LEGEND_H,
        xaxis=dict(title="Budget — log", type="log", tickvals=_BUDGETS,
                   gridcolor=S.BORDER, zeroline=False),
        yaxis=dict(title="Median sign agreement vs shap_true_value",
                   range=[y_lo, 1.005], gridcolor=S.BORDER, zeroline=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE RQ2-F5 — Do the exact reference implementations agree?
#
# Question answered:
#     Is "exact" a single well-defined target — do the four exact/reference
#     backends produce the same values?
#
# Source:
#     rq2_reference_agreement.csv
#
# Raw CSV inputs:
#     pairwise_metrics of the 4 reference rows (shap_true_value,
#     shapiq_true_value, lightshap_exact, dalex_true_value) against each
#     other, over all 200 dataset × model × seed cells.
#
# Row selection:
#     Reference backends only — no approximation rows.
#
# Grouping:
#     source backend × target backend; median relative MAE over the 200
#     cells (max shown in hover for the worst case).
#
# Seed aggregation:
#     Median over all cells (seeds included in the 200).
#
# Visual encoding:
#     4 × 4 heatmap, log-colored: green ≈ machine precision, red = percent-
#     level disagreement.
#
# Why:
#     This chart justifies the page's oracle choice and surfaces the key
#     data finding: dalex_true_value is ~3.2% away from the other three,
#     so cross-library "exact" values are not interchangeable.
#
# Interpretation supported:
#     shap/shapiq/lightshap exact solutions form one consistent oracle;
#     dalex's reference implementation measurably deviates.
#
# Limitation:
#     Cannot say WHICH side is mathematically correct — only that they
#     disagree; the three-vs-one pattern makes shap_true_value the safer
#     choice.
# ─────────────────────────────────────────────────────────────────────────────

def _build_reference_agreement_figure(ref: pd.DataFrame) -> go.Figure:
    if ref.empty:
        return S.fig_empty()

    order = ["shap_true_value", "shapiq_true_value",
             "lightshap_exact", "dalex_true_value"]
    pivot = (ref.pivot(index="source_backend", columns="target_backend",
                       values="relative_mae_median")
             .reindex(index=order, columns=order))

    # Log-transform for color: machine precision (1e-15) → percent level
    # (1e-2) spans 13 decades; linear coloring would render everything as
    # "identical" except the dalex row.
    z = pivot.values.astype(float)
    with np.errstate(divide="ignore"):
        z_log = np.log10(np.where(z > 0, z, 1e-16))
    text = [[("—" if np.isnan(v) else f"{v:.1e}") for v in row]
            for row in pivot.values]

    fig = go.Figure(go.Heatmap(
        z=z_log, x=list(pivot.columns), y=list(pivot.index),
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#D1FAE5"], [0.6, "#93C5FD"], [1, "#FEE2E2"]],
        zmin=-16, zmax=-1,
        colorbar=dict(title="log₁₀ rel. MAE", thickness=14, len=0.8),
        hovertemplate=("source: <b>%{y}</b><br>target: <b>%{x}</b><br>"
                       "median relative MAE: %{text}<extra></extra>"),
    ))
    fig.update_layout(
        **S._CHART_LAYOUT, height=340,
        xaxis=dict(title="compared against", gridcolor="rgba(0,0,0,0)",
                   tickangle=-20),
        yaxis=dict(title="reference backend", gridcolor="rgba(0,0,0,0)",
                   automargin=True),
        margin=dict(l=90, r=16, t=20, b=90),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────────────────────────────

def layout(**kwargs):
    return html.Div([
        S.rq_header(*_RQ_HEADER),
        _config_card(),

        # RQ2-F1
        S.section(
            "RQ2-F1 · Error convergence over budget",
            "Median error vs the exact shap_true_value oracle across 10 seeds "
            "(band = q25–q75). 'All datasets'/'All models' pool the aggregated "
            "cells by median — select specific values to unpool.",
            html.Div([
                _axis_toggle("rq2-conv-metric",
                             {"relative_mae": "relative MAE",
                              "mean_sample_rho": "Spearman ρ"},
                             "relative_mae", label="Metric"),
                _col_note(
                    "source: converted/rq2_convergence_aggregated.csv",
                    "reference: shap_true_value, matched within dataset × model × seed",
                    "agg: median(10 seeds) in converter · band: q25–q75",
                ),
                html.Div(id="rq2-f1-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-f1-section",
        ),

        # RQ2-F2
        S.section(
            "RQ2-F2 · Runtime vs error — is the approximation worth it?",
            "Each marker = one method × budget configuration (medians). "
            "X markers = exact backends; at 12 features they are often "
            "cheaper than high-budget approximations. dalex_true_value is "
            "plotted at its measured 3.2% deviation, not at zero.",
            html.Div([
                _col_note(
                    "source: converted/rq2_runtime_accuracy.csv",
                    "symbol: budget (○128 ◇512 □2048, ✕ exact) · color: library",
                    "display pooling: median over unfiltered dataset/model cells",
                ),
                html.Div(id="rq2-f2-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-f2-section",
        ),

        # RQ2-F3
        S.section(
            "RQ2-F3 · Seed stability of the error",
            "Raw seed-level error distributions at one budget. Tight boxes "
            "mean a single run is representative; long whiskers mean an "
            "unlucky seed can be far off the median.",
            html.Div([
                _axis_toggle("rq2-stability-budget",
                             {"128": "128", "512": "512", "2048": "2048"},
                             "512", label="Budget"),
                _col_note(
                    "source: converted/rq2_accuracy_by_seed.csv (no aggregation)",
                    "one point per run: 10 seeds × filtered dataset/model cells",
                ),
                html.Div(id="rq2-f3-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-f3-section",
        ),

        # RQ2-F4
        S.section(
            "RQ2-F4 · Sign agreement vs budget",
            "How often the approximation assigns the correct direction "
            "(+/−) to each attribution. Saturates near 1.0 quickly — use "
            "RQ2-F1 to separate methods beyond that point.",
            html.Div([
                _col_note(
                    "source: converted/rq2_convergence_aggregated.csv · sign_agreement_median",
                    "reference: shap_true_value · agg: median(10 seeds)",
                ),
                html.Div(id="rq2-f4-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-f4-section",
        ),

        # RQ2-F5
        S.section(
            "RQ2-F5 · Reference agreement check",
            "Median relative MAE between the four exact/reference backends "
            "over 200 dataset × model × seed cells. shap, shapiq and "
            "lightshap agree to numerical noise; dalex_true_value deviates "
            "by ~3.2% — 'exact' is not interchangeable across libraries.",
            html.Div([
                _col_note(
                    "source: converted/rq2_reference_agreement.csv",
                    "justifies the shap_true_value oracle choice (3-vs-1 pattern)",
                    "unaffected by page filters (backend-level check)",
                ),
                html.Div(id="rq2-f5-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-f5-section",
        ),

        S.interpretation_note(_INTERP),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Callback — filtering only; statistics computed in the converter
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("rq2-f1-chart", "children"),
    Output("rq2-f2-chart", "children"),
    Output("rq2-f3-chart", "children"),
    Output("rq2-f4-chart", "children"),
    Output("rq2-f5-chart", "children"),
    Input("rq2-ds", "value"),
    Input("rq2-mdl", "value"),
    Input("rq2-approx", "value"),
    Input("rq2-conv-metric", "value"),
    Input("rq2-stability-budget", "value"),
)
def update_rq2(ds, mdl, approxs, conv_metric, stability_budget):
    ds = ds or "__all__"
    mdl = mdl or "__all__"
    conv_metric = conv_metric or "relative_mae"
    stability_budget = int(stability_budget or 512)

    convergence = _load(_CONVERGENCE)
    runtime_acc = _load(_RUNTIME_ACC)
    by_seed = _load(_BY_SEED)
    ref_agreement = _load(_REF_AGREEMENT)

    if approxs:
        if not convergence.empty:
            convergence = convergence[convergence["approximator"].isin(approxs)]
        if not by_seed.empty:
            by_seed = by_seed[by_seed["approximator"].isin(approxs)]
        if not runtime_acc.empty:
            # keep exact reference rows visible regardless of the
            # approximator filter — they are the comparison baseline
            runtime_acc = runtime_acc[
                runtime_acc["is_reference"] |
                runtime_acc["approximator"].isin(approxs)]

    pooled = (_pool_models_datasets(convergence, ds, mdl)
              if not convergence.empty else convergence)

    def _g(fig):
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    return (
        _g(_build_convergence_figure(pooled, conv_metric)),
        _g(_build_pareto_figure(runtime_acc, ds, mdl)),
        _g(_build_seed_stability_figure(by_seed, ds, mdl, stability_budget)),
        _g(_build_sign_agreement_figure(pooled)),
        # F5 is a backend-level check — page filters deliberately not applied.
        _g(_build_reference_agreement_figure(ref_agreement)),
    )
