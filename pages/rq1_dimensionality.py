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
    path="/rq2",
    name="RQ2 — Dimensionality",
    title="RQ2 — Dimensionality",
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONV = os.path.join(_HERE, "results", "converted")

_AGGREGATED = os.path.join(_CONV, "rq1_scaling_aggregated.csv")
_FEASIBILITY = os.path.join(_CONV, "rq1_feasibility.csv")
_EXTREME = os.path.join(_CONV, "rq1_extreme_stress_test.csv")

# ─────────────────────────────────────────────────────────────────────────────
_RQ_HEADER = (
    "RQ2", "Dimensionality",
    "As a user with a dataset that has many features, I want to find the "
    "fastest model-agnostic library for high-dimensional datasets?",
)

# Interpretation revised for the new data: runtime/feasibility claims only.
# The old text implied a quality axis ("while quality stays high") — RQ1
# data contains no exact reference, so no quality-vs-truth claim is made.
_INTERP = (
    "Compare how steeply each method's runtime grows as n_features increases "
    "— flatter is better. The feasibility heatmap shows where methods hit the "
    "10-minute execution cap (all such runs are lightshap). F2 tracks whether "
    "the 7 methods still rank features alike as dimensionality grows (F2). "
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


def _band_fill(hex_color: str, alpha: float = 0.13) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha) — Plotly rejects 8-digit hex."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
#  Local layout helpers (unchanged style from the previous page version)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_NAMES = ["decision_tree", "gradient_boosting", "linear_regularized", "random_forest"]
_METHOD_NAMES = ["dalex / permutation", "lightshap / kernel", "lightshap / permutation",
                  "shap / kernel", "shap / permutation", "shapiq / kernel", "shapiq / permutation"]


def _config_card() -> html.Div:
    """Benchmark-at-a-glance card, updated for the new experimental grid."""
    left = S.stat_col("Swept",
                [("4 datasets", ", ".join(_DATASET_ORDER)),
                 ("4 models", ", ".join(_MODEL_NAMES)),
                 "n_features: 14–256",
                 ("7 methods", ", ".join(_METHOD_NAMES)),
                 "dalex: perm only",
                 "budget: 512, 1024", "10 seeds"],
                "#EEF2FF", S.ACCENT)
    mid = S.stat_col("Fixed",
               [("n_background = 100", "Number of background samples used to "
                 "estimate the reference/baseline distribution for each explanation."),
                ("n_eval = 10", "Number of evaluation points explained per cell."),
                ("imputer = marginal", "Missing/absent features are replaced by "
                 "sampling from their marginal distribution (feature independence "
                 "assumed), not conditioned on the other present features."),
                "10-min execution cap"],
               "#F1F5F9", S.TEXT2)
    right = S.stat_col("Measured",
                 ["runtime_s", "n_model_evals",
                  ("cross-method agreement ρ", "Spearman rank correlation between "
                   "each method's feature ranking and the other six methods' rankings.")],
                 "#F0FDF4", S.GREEN)
    stress = S.stat_col("Separate stress test",
                  [("gisette @ 1,000 features", "One-shot extreme test, not part "
                    "of the standard scaling grid above."),
                   "seed 0 only",
                   "budget 2048",
                   ("7 methods", ", ".join(_METHOD_NAMES))],
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


_DALEX_METHOD = "dalex / permutation"


def _budget_effect_note() -> html.Div:
    """F4 context — what the ratio means and why dalex is flagged."""
    return html.Div([
        html.Div(
            "Runtime ratio = runtime(1024) / runtime(512) per experiment cell. "
            "For shap, shapiq, and lightshap this tests whether wall-clock cost "
            "scales linearly with coalition samples — ratio ≈ 2 means yes, "
            "< 2 means fixed overhead dominates.",
            style={"fontSize": "12px", "color": S.TEXT, "lineHeight": "1.6",
                   "marginBottom": "4px"},
        ),
        html.Div(
            "★ dalex uses a fixed evaluation budget split across features — "
            "doubling the config budget does not add coalition samples the "
            "same way, so its ratio is not directly comparable. "
            "Median per cell · capped/failed cells excluded · "
            "source: converted/rq1_scaling_aggregated.csv",
            style={"fontSize": "11px", "color": S.TEXT2, "lineHeight": "1.55"},
        ),
    ], style={
        "background": "#F0F4FF",
        "border": f"1px solid {S.BORDER}",
        "borderRadius": "6px",
        "padding": "9px 13px",
        "margin": "0 0 2px",
    })


def _agreement_note() -> html.Div:
    """Compact agreement chart context — what is compared and how to read it."""
    return html.Div([
        html.Div(
            "Each line is one of the 7 approximation methods "
            "(shap, shapiq, and lightshap: kernel + permutation; "
            "dalex: permutation only). "
            "The y-axis is the mean Spearman ρ against the other six methods "
            "at the same budget — how closely their feature rankings agree "
            "as n_features grows.",
            style={"fontSize": "12px", "color": S.TEXT, "lineHeight": "1.6",
                   "marginBottom": "4px"},
        ),
        html.Div(
            "A drop at high M usually means methods diverge under budget pressure, "
            "not that one line is automatically correct. "
            "Same-budget pairwise comparisons only · median over 10 seeds · "
            "budget follows the F1 toggle · "
            "source: converted/rq1_scaling_aggregated.csv",
            style={"fontSize": "11px", "color": S.TEXT2, "lineHeight": "1.55"},
        ),
    ], style={
        "background": "#F0F4FF",
        "border": f"1px solid {S.BORDER}",
        "borderRadius": "6px",
        "padding": "9px 13px",
        "margin": "0 0 2px",
    })


def _filter_bar() -> html.Div:
    """Dataset / model / approximator filter row, rendered IN the page so
    IDs always exist in the DOM at load time (avoids topbar timing race)."""
    df = _load(_AGGREGATED)
    datasets  = [{"label": "All datasets", "value": "__all__"}] + \
                [{"label": d, "value": d} for d in sorted(df["dataset"].dropna().unique())] \
                if not df.empty else [{"label": "All datasets", "value": "__all__"}]
    models    = [{"label": "All models", "value": "__all__"}] + \
                [{"label": m, "value": m} for m in sorted(df["model"].dropna().unique())] \
                if not df.empty else [{"label": "All models", "value": "__all__"}]
    approxs   = sorted(df["approximator"].dropna().unique()) if not df.empty else []

    lbl_style = {"fontSize": "10px", "fontWeight": "700", "color": S.TEXT2,
                 "textTransform": "uppercase", "letterSpacing": "0.06em",
                 "marginBottom": "3px"}
    return html.Div([
        html.Div([
            html.Div("Dataset", style=lbl_style),
            dcc.Dropdown(id="rq2-ds", options=datasets, value="__all__",
                         clearable=False,
                         style={"width": "150px", "fontSize": "12px", "minHeight": "28px"}),
        ], style={"marginRight": "12px"}),
        html.Div([
            html.Div("Model", style=lbl_style),
            dcc.Dropdown(id="rq2-mdl", options=models, value="__all__",
                         clearable=False,
                         style={"width": "140px", "fontSize": "12px", "minHeight": "28px"}),
        ], style={"marginRight": "16px"}),
        html.Div([
            html.Div("Approximator", style=lbl_style),
            dcc.Checklist(
                id="rq2-approx",
                options=[{"label": f" {a}", "value": a} for a in approxs],
                value=list(approxs),
                inline=True,
                inputStyle={"marginRight": "3px"},
                labelStyle={"marginRight": "8px", "fontSize": "12px", "cursor": "pointer"},
            ),
        ]),
    ], style={
        "display": "flex", "alignItems": "flex-end", "flexWrap": "wrap",
        "gap": "4px", "padding": "10px 16px 10px",
        "background": S.BG, "borderBottom": f"1px solid {S.BORDER}",
        "marginBottom": "4px",
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

def _nf_ticks(sub_df: pd.DataFrame) -> dict:
    """Log-scale x-axis ticks only at actual n_features present in sub_df."""
    vals = sorted(sub_df["n_features"].unique().tolist())
    return dict(tickmode="array", tickvals=vals,
                ticktext=[str(int(v)) for v in vals])


def _runtime_ticks() -> dict:
    """Human-readable log-scale ticks for runtime (s) axes."""
    vals = [0.01, 0.1, 1, 10, 100, 600]
    return dict(tickmode="array", tickvals=vals,
                ticktext=["0.01 s", "0.1 s", "1 s", "10 s", "100 s", "600 s"])


def _stress_runtime_ticks() -> dict:
    """Human-readable log-scale ticks for the extreme-stress runtime axis."""
    vals = [10, 30, 100, 300, 1000, 3000, 10000]
    return dict(tickmode="array", tickvals=vals,
                ticktext=["10 s", "30 s", "100 s", "300 s",
                          "1000 s", "3000 s", "10000 s"])


_SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def _pow10_label(exp: int) -> str:
    """Format an integer exponent as a consistent 10ⁿ axis label."""
    if exp == 0:
        return "10⁰"
    digits = str(abs(exp))
    sup_digits = "".join(_SUP[int(d)] for d in digits)
    if exp < 0:
        return f"10⁻{sup_digits}"
    return f"10{sup_digits}"


def _norm_evals(sub_df: pd.DataFrame) -> np.ndarray:
    return (sub_df["evals_median"].to_numpy()
            / (2.0 ** sub_df["n_features"].to_numpy()))


def _log_decade_step(span: int, target: int = 7) -> int:
    if span <= target - 1:
        return 1
    for step in (1, 2, 5, 10, 20, 50, 100):
        if span // step <= target - 1:
            return step
    return max(10, (span + target - 2) // (target - 1))


def _evals_ticks(sub_df: pd.DataFrame) -> dict:
    """Log-scale y-axis ticks for model evals / 2ⁿ — always 10ⁿ labels."""
    vals = _norm_evals(sub_df)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return {}
    lo, hi = float(vals.min()), float(vals.max())
    exp_lo = int(np.floor(np.log10(lo)))
    exp_hi = int(np.ceil(np.log10(hi)))
    step = _log_decade_step(exp_hi - exp_lo)
    start = (exp_lo // step) * step
    if start > exp_lo:
        start -= step
    exps = list(range(start, exp_hi + 1, step))
    if exps[0] > exp_lo:
        exps.insert(0, exp_lo)
    if exps[-1] < exp_hi:
        exps.append(exp_hi)
    tickvals = [10.0 ** e for e in exps]
    ticktext = [_pow10_label(e) for e in exps]
    return dict(tickmode="array", tickvals=tickvals, ticktext=ticktext)


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
            hover_suffix = "median runtime: %{y:.3g} s"
            customdata = None
        else:
            # Normalise by 2^n_features: shows "model calls per slot in the
            # full coalition lattice".  Values > 1 at small n_features are
            # expected — each coalition is probed against n_background
            # background samples, so raw call counts exceed 2^n_features
            # until the feature space grows large enough.  The exponential
            # denominator makes the coverage gap visible as a clean
            # downward slope on a log scale.
            norm_factor = 2.0 ** mdf["n_features"].values
            y = mdf["evals_median"].values / norm_factor
            raw = mdf["evals_median"].values
            customdata = np.stack([raw], axis=1)
            hover_suffix = "evals / 2ⁿ: %{y:.3g}<br>raw model evals: %{customdata[0]:,.0f}"
            ylo = yhi = None

        # Seed IQR band (runtime only — eval counts are deterministic per
        # config, their seed spread is negligible).
        if ylo is not None:
            traces.append(go.Scatter(
                x=pd.concat([mdf["n_features"], mdf["n_features"][::-1]]),
                y=pd.concat([yhi, ylo[::-1]]),
                fill="toself", fillcolor=_band_fill(color),
                line=dict(width=0), hoverinfo="skip",
                legendgroup=method, showlegend=False,
            ))
        scatter_kwargs = dict(
            x=mdf["n_features"], y=y,
            mode="lines+markers", name=method,
            legendgroup=method, showlegend=showlegend,
            line=dict(color=color, width=2, dash=_APPROX_DASH.get(approx, "solid")),
            marker=dict(size=7, color=color, line=dict(color="white", width=1.2)),
            hovertemplate=f"<b>{method}</b><br>n_features: %{{x}}<br>{hover_suffix}<extra></extra>",
        )
        if customdata is not None:
            scatter_kwargs["customdata"] = customdata
        traces.append(go.Scatter(**scatter_kwargs))
    return traces


def _build_runtime_scaling_figure(agg: pd.DataFrame, dataset: str,
                                  metric: str) -> go.Figure:
    if agg.empty:
        return S.fig_empty("No converted data — run results_converters/rq1_results_converter.py")

    y_title = ("Median runtime (s)" if metric == "runtime_s"
               else "Model evals / 2ⁿ features")

    if dataset != "__all__":
        sub = agg[agg["dataset"] == dataset]
        if sub.empty:
            return S.fig_empty()
        fig = go.Figure()
        for tr in _scaling_traces(sub, metric, showlegend=True):
            fig.add_trace(tr)
        y_ticks = (_runtime_ticks() if metric == "runtime_s"
                   else _evals_ticks(sub))
        fig.update_layout(
            **S._CHART_LAYOUT, height=440, margin=S._MARGIN, legend=S._LEGEND_H,
            xaxis=dict(title="n_features", type="log",
                       gridcolor=S.BORDER, zeroline=False, **_nf_ticks(sub)),
            yaxis=dict(title=y_title, type="log",
                       gridcolor=S.BORDER, zeroline=False, **y_ticks),
        )
        return fig

    # "All datasets": small multiples — one panel per dataset, because the
    # datasets use different feature grids and pooling them into one line
    # would average incomparable x-positions.
    datasets = [d for d in _DATASET_ORDER if d in agg["dataset"].unique()]
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=datasets,
        horizontal_spacing=0.12, vertical_spacing=0.16,
    )
    # axis key suffix: subplot (row, col) → "" / "2" / "3" / "4"
    _ax_sfx = {(1, 1): "", (1, 2): "2", (2, 1): "3", (2, 2): "4"}
    for i, ds in enumerate(datasets):
        row, col = divmod(i, 2)
        sub = agg[agg["dataset"] == ds]
        for tr in _scaling_traces(sub, metric, showlegend=(i == 0)):
            fig.add_trace(tr, row=row + 1, col=col + 1)
        sfx = _ax_sfx[(row + 1, col + 1)]
        # Only label y-axis on left column panels to avoid crowding.
        y_label = y_title if col == 0 else ""
        y_ticks = (_runtime_ticks() if metric == "runtime_s"
                   else _evals_ticks(sub))
        fig.update_layout(**{
            f"xaxis{sfx}": dict(type="log", gridcolor=S.BORDER, zeroline=False,
                                title_text="n_features", **_nf_ticks(sub)),
            f"yaxis{sfx}": dict(type="log", gridcolor=S.BORDER, zeroline=False,
                                title_text=y_label, **y_ticks),
        })
    fig.update_layout(
        **S._CHART_LAYOUT, height=680, legend=S._LEGEND_H,
        margin=dict(l=60, r=16, t=60, b=48),
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
#     dalex defines budget as a fixed total divided across features — its
#     512→1024 ratio is flagged (★) and not comparable to the others.
#     Only two budgets exist in RQ1 — no full budget curve (RQ2 has three
#     budgets at fixed dimensionality).
# ─────────────────────────────────────────────────────────────────────────────

def _build_budget_effect_figure(agg: pd.DataFrame, dataset: str) -> go.Figure:
    """
    RQ2-F4 — Does doubling the budget (512 → 1024) double the runtime?

    A ratio of exactly 2 means runtime grows linearly with budget (all cost is
    model evaluations). Ratio < 2 means fixed overhead (coalition construction,
    Python startup, etc.) dominates — doubling the budget costs less than 2×.
    Ratio >> 2 signals an anomaly (near-zero denominator) and those cells are
    excluded.

    Box plot: distribution of ratios per method across all experiment cells.
    """
    if agg.empty:
        return S.fig_empty()
    sub = agg if dataset == "__all__" else agg[agg["dataset"] == dataset]

    cell_keys = ["dataset", "model", "n_features", "method", "library"]

    # Exclude cells where runtime_median is unreliable:
    #   time_cap_count > 0  → runtime truncated at ~600s cap
    #   evals_missing_count > 0 → structural failure (budget < 2M+2), leaving
    #     near-zero runtimes that make the ratio meaningless.
    clean = sub[
        (sub["time_cap_count"] == 0) &
        (sub["evals_missing_count"] == 0)
    ]

    wide = clean.pivot_table(index=cell_keys, columns="budget",
                             values="runtime_median").reset_index()
    if 512 not in wide.columns or 1024 not in wide.columns:
        return S.fig_empty("Need both budgets in the current filter")
    wide = wide.dropna(subset=[512, 1024])
    wide["ratio"] = wide[1024] / wide[512]

    order = (wide.groupby("method")["ratio"].median()
             .sort_values().index.tolist())

    fig = go.Figure()

    for method in order:
        mdf = wide[wide["method"] == method]
        lib = mdf["library"].iloc[0]
        color = _lib_color(lib)
        is_dalex = method == _DALEX_METHOD
        label = f"{method} ★" if is_dalex else method
        hover_extra = (
            "<br><i>★ fixed budget split across features — not comparable</i>"
            if is_dalex else ""
        )
        fig.add_trace(go.Box(
            y=mdf["ratio"], name=label,
            marker_color=color,
            line_color=color,
            line_width=2 if is_dalex else 1.5,
            boxpoints="all", jitter=0.35, pointpos=0,
            marker=dict(size=4, opacity=0.35 if is_dalex else 0.5),
            fillcolor="rgba(0,0,0,0)",
            hovertemplate=(
                f"<b>{method}</b><br>ratio=%{{y:.2f}}{hover_extra}<extra></extra>"
            ),
            showlegend=False,
        ))
    fig.update_layout(
        **S._CHART_LAYOUT, height=400,
        margin=dict(l=55, r=16, t=36, b=90),
        xaxis=dict(tickangle=-25, gridcolor="rgba(0,0,0,0)", automargin=True),
        yaxis=dict(title="runtime(1024) / runtime(512)",
                   gridcolor=S.BORDER, zeroline=False),
    )

    fig.add_hline(y=2.0, line=dict(color=S.TEXT2, width=1.2, dash="dot"),
                  annotation_text="ratio = 2 → linear scaling",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color=S.TEXT2))
    fig.add_hline(y=1.0, line=dict(color=S.BORDER, width=0.8, dash="dot"),
                  annotation_text="ratio = 1 → no extra cost",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color=S.TEXT2))
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
                 .sort_values(["n_features", "dataset"])["cell_label"].tolist())

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
#     pairwise_metrics JSON (mean_sample_rho against the other 6 methods
#     in the same dataset × model × n_features × seed cell)
#
# Row selection:
#     Standard experiment, one budget at a time (same radio as RQ1-F1).
#
# Grouping:
#     Per run, the converter averaged ρ over the other 6 methods at the
#     same budget; here the median of that score is drawn per method ×
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
        xaxis=dict(title="n_features", type="log",
                   gridcolor=S.BORDER, zeroline=False, **_nf_ticks(sub)),
        yaxis=dict(title="Median cross-method agreement (ρ)",
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
                   type="log", gridcolor=S.BORDER, zeroline=False,
                   **_stress_runtime_ticks()),
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
            "Cost scaling by feature count",
            "Median cost per method (10 seeds, band = q25–q75). "
            "'All datasets' shows one panel per dataset because each dataset "
            "has its own feature grid. 'All models' pools the 4 models by "
            "median — select a model to isolate it. Time-capped runs excluded. "
            "When 'model evaluations' is selected the y-axis shows raw calls "
            "normalised by 2ⁿ features (the size of the full coalition lattice) "
            "so the exponential coverage gap is directly visible as a downward "
            "slope — values > 1 at small n are expected because each coalition "
            "is probed against a background dataset.",
            html.Div([
                _axis_toggle("rq2-cost-metric",
                             {"runtime_s": "runtime (s)",
                              "n_model_evals": "model evaluations"},
                             "runtime_s", label="Metric"),
                _axis_toggle("rq2-budget", {"512": "512", "1024": "1024"},
                             "512", label="Budget"),
                _col_note(
                    "source: converted/rq1_scaling_aggregated.csv",
                    "agg: median(10 seeds) in converter · display: median over models when 'All models'",
                    "capped runs excluded from medians",
                ),
                html.Div(id="rq2-f1-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-f1-section",
        ),

        # RQ2-F2 — cross-method agreement
        S.section(
            "Cross-method agreement vs dimensionality",
            "",
            html.Div([
                _agreement_note(),
                html.Div(id="rq2-f2-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-f2-section",
        ),

        # RQ2-F3
        S.section(
            "Execution-cap feasibility",
            "Share of runs hitting the 10-minute cap per method × dataset × "
            "n_features. Models, budgets and seeds pooled (80 runs per cell): "
            "a cap anywhere is a deployment risk.",
            html.Div([
                _col_note(
                    "source: converted/rq1_feasibility.csv",
                    "capped = runtime_s ≥ 595 s (raw data clusters at 600 ± 0.02 s)",
                    "cell = capped / all runs over 4 models × 2 budgets × 10 seeds",
                ),
                html.Div(id="rq2-f3-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-f3-section",
        ),

        # RQ2-F4 — budget effect
        S.section(
            "Budget effect on runtime (512 → 1024)",
            "",
            html.Div([
                _budget_effect_note(),
                html.Div(id="rq2-f4-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-f4-section",
        ),

        # RQ1-F5
        S.section(
            "Extreme stress test — 1,000 features (separate experiment)",
            "gisette @ 1,000 features, budget 2048, seed 0 only, 7 methods × "
            "4 models. One-shot feasibility check — page filters do not apply "
            "and these bars are not comparable with the scaling curves above.",
            html.Div([
                _col_note(
                    "source: converted/rq1_extreme_stress_test.csv",
                    "no aggregation (single seed) · hatched bars exceeded the 10-min cap",
                ),
                html.Div(id="rq2-f5-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-f5-section",
        ),

        S.interpretation_note(_INTERP),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Callback — filtering only; all statistics were computed in the converter
# ─────────────────────────────────────────────────────────────────────────────

def _apply_filters(df, ds, mdl, approxs):
    """Filter by topbar store values. approxs may be None (= all) or a list."""
    ds = ds or "__all__"
    mdl = mdl or "__all__"
    if ds != "__all__" and "dataset" in df.columns:
        df = df[df["dataset"] == ds]
    if mdl != "__all__" and "model" in df.columns:
        df = df[df["model"] == mdl]
    if approxs and "approximator" in df.columns:
        df = df[df["approximator"].isin(approxs)]
    return df


@callback(
    Output("rq2-f1-chart", "children"),
    Output("rq2-f2-chart", "children"),
    Output("rq2-f3-chart", "children"),
    Output("rq2-f4-chart", "children"),
    Output("rq2-f5-chart", "children"),
    Input("rq2-ds", "data"),
    Input("rq2-mdl", "data"),
    Input("rq2-approx", "data"),
    Input("rq2-cost-metric", "value"),
    Input("rq2-budget", "value"),
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

    # F1 + F2 (agreement) show one budget at a time (radio); "All models"
    # pooling by median happens here, matching the note on the charts.
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

    def _g(fig, filename):
        return dcc.Graph(figure=fig, config=S.graph_config(filename))

    return (
        _g(_build_runtime_scaling_figure(f1_input, ds, cost_metric),
           "rq1_f1_cost_scaling"),
        _g(_build_agreement_figure(f1_input, ds), "rq1_f2_agreement"),
        _g(_build_feasibility_heatmap(feas_f, ds), "rq1_f3_feasibility"),
        _g(_build_budget_effect_figure(agg_f, ds), "rq1_f4_budget_effect"),
        # F5 deliberately unfiltered — single-seed stress test (see comment).
        _g(_build_extreme_stress_figure(extreme), "rq1_f5_stress_test"),
    )
