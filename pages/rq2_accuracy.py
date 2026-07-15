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
from dash import Input, Output, callback, ctx, dcc, html, no_update

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared as S

dash.register_page(
    __name__,
    path="/rq1",
    name="RQ1 — Accuracy",
    title="RQ1 — Approximation Accuracy",
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONV = os.path.join(_HERE, "results", "converted")

_BY_SEED = os.path.join(_CONV, "rq2_accuracy_by_seed.csv")
_CONVERGENCE = os.path.join(_CONV, "rq2_convergence_aggregated.csv")
_RUNTIME_ACC = os.path.join(_CONV, "rq2_runtime_accuracy.csv")
_REF_AGREEMENT = os.path.join(_CONV, "rq2_reference_agreement.csv")

# ─────────────────────────────────────────────────────────────────────────────
_RQ_HEADER = (
    "RQ1", "Approximation Accuracy",
    "As a user using Shapley approximations, I want to know how good the values "
    "actually are so that I can trust the explanations without wasting too much "
    "compute time on exact values.",
)

# Interpretation rewritten for the new data: three budgets, one fixed
# dimensionality (12 features), explicit single oracle, and the measured
# dalex reference deviation.
_INTERP = (
    "RQ1-F1 first establishes the measurement basis: shap, shapiq and "
    "lightshap agree to within numerical noise, so shap_true_value is a "
    "trustworthy oracle. dalex_true_value deviates by ~3.2% and is excluded "
    "from the reference role. All subsequent error metrics are computed "
    "against shap_true_value at a fixed 12 features. Error falls "
    "monotonically with budget for every method — kernel and permutation "
    "differ mainly in how fast they start converging (lightshap arrives "
    "near its floor already at budget 128). The Pareto chart (RQ1-F3) shows "
    "what exactness costs: exact backends are often cheaper than high-budget "
    "approximations at this dimensionality. The sign agreement toggle on "
    "RQ1-F2 shows that direction errors disappear before magnitude errors — "
    "magnitude convergence (relative MAE) is the more demanding test."
)

_APPROX_DASH = {"kernel": "solid", "permutation": "dot"}
_BUDGETS = [128, 512, 2048]


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
#  Local layout helpers
# ─────────────────────────────────────────────────────────────────────────────

_DATASET_NAMES = ["ames_housing", "bike", "covertype", "diabetes_130", "gisette"]
_MODEL_NAMES = ["decision_tree", "gradient_boosting", "linear_regularized", "random_forest"]
_LIBRARY_NAMES = ["dalex", "lightshap", "shap", "shapiq"]


def _config_card() -> html.Div:
    left = S.stat_col("Swept",
                [("5 datasets", ", ".join(_DATASET_NAMES)),
                 ("4 models", ", ".join(_MODEL_NAMES)),
                 "budget: 128, 512, 2048",
                 ("4 libraries", ", ".join(_LIBRARY_NAMES)),
                 ("2 approximators", "kernel, permutation"),
                 "10 seeds"],
                "#EEF2FF", S.ACCENT)
    mid = S.stat_col("Fixed",
               ["n_features = 12",
                ("n_background = 100", "Number of background samples used to "
                 "estimate the reference/baseline distribution for each explanation."),
                ("n_eval = 10", "Number of evaluation points explained per cell."),
                ("imputer = marginal", "Missing/absent features are replaced by "
                 "sampling from their marginal distribution (feature independence "
                 "assumed), not conditioned on the other present features.")],
               "#F1F5F9", S.TEXT2)
    right = S.stat_col("Reference",
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


_METRIC_META = {
    "relative_mae": {
        "label": "Relative MAE",
        "direction": "lower",
        "line1": (
            "Mean absolute attribution error as a fraction of the exact values. "
            "0 = perfect; 0.05 ≈ 5 % off on average. "
            "Log scale — each gridline is a 10× improvement."
        ),
        "line2": (
            "Solid = kernel · dashed = permutation. "
            "Error bars = q25–q75 across 10 seeds. "
            "A flat slope means the method already hit its floor at budget 128."
        ),
    },
    "mean_sample_rho": {
        "label": "Spearman ρ",
        "direction": "higher",
        "line1": (
            "Rank correlation between approximated and exact attributions. "
            "1 = features ranked in the right order; 0 = no correlation. "
            "Green line = ρ 0.9 ('good enough') threshold."
        ),
        "line2": "Solid = kernel · dashed = permutation. Error bars = q25–q75 across 10 seeds.",
    },
    "sign_agreement": {
        "label": "Sign agreement",
        "direction": "higher",
        "line1": (
            "Fraction of features with the correct positive / negative direction. "
            "1.0 = all signs right; 0.5 = random. "
            "Green line = perfect agreement."
        ),
        "line2": "Solid = kernel · dashed = permutation. Error bars = q25–q75 across 10 seeds.",
    },
    "relative_additivity_gap": {
        "label": "Additivity gap",
        "direction": "lower",
        "line1": (
            "How far Σ φ̂ᵢ deviates from f(x) − baseline, relative to that gap. "
            "0 = attributions sum exactly to the model output. "
            "Log scale."
        ),
        "line2": (
            "Most methods enforce this algebraically — they cluster at the green "
            "machine-precision line (~10⁻¹⁵). "
            "shapiq / kernel does not: sits ~10⁻⁹ to 10⁻⁸, five orders of magnitude higher. "
            "Solid = kernel · dashed = permutation."
        ),
    },
}

_DIR_BADGE = {
    "lower": ("#D1FAE5", "#065F46", "▼ lower is better"),
    "higher": ("#D1FAE5", "#065F46", "▲ higher is better"),
}


def _metric_explainer(metric: str) -> html.Div:
    """Concise two-line per-metric explanation below the F2 toggle."""
    meta = _METRIC_META.get(metric, {})
    if not meta:
        return html.Div()
    bg, fg, badge_text = _DIR_BADGE[meta["direction"]]
    return html.Div([
        html.Div([
            html.Span(meta["line1"],
                      style={"fontSize": "12px", "color": S.TEXT, "lineHeight": "1.6"}),
            html.Span(badge_text, style={
                "fontSize": "10px", "fontWeight": "700", "color": fg,
                "background": bg, "borderRadius": "4px",
                "padding": "1px 7px", "marginLeft": "10px",
                "verticalAlign": "middle", "whiteSpace": "nowrap",
            }),
        ], style={"marginBottom": "4px"}),
        html.Div(meta["line2"],
                 style={"fontSize": "11px", "color": S.TEXT2, "lineHeight": "1.55"}),
    ], style={
        "background": "#F0F4FF",
        "border": f"1px solid {S.BORDER}",
        "borderRadius": "6px",
        "padding": "9px 13px",
        "margin": "6px 0 2px",
    })


def _fmt_rel_mae_pct(v: float) -> str:
    """Format relative MAE as a human-readable percentage for cell labels."""
    if np.isnan(v):
        return "—"
    pct = v * 100
    if pct < 1e-8:
        return "~0 %"
    if pct < 0.01:
        s = f"{pct:.6f}".rstrip("0").rstrip(".")
        return f"{s} %"
    return f"{pct:.2g} %"


def _f1_note() -> html.Div:
    """Compact F1 context — only what the heatmap cannot show on its own."""
    return html.Div([
        html.Div(
            "Approximation error needs a fixed reference first. "
            "shap_true_value is that reference: lightshap_exact matches it at "
            "~0 % (machine precision), shapiq_true_value at ~0.00005 % — "
            "negligible but visibly higher — while dalex_true_value sits at "
            "~3.2 % from a different exact-computation path, not seed noise.",
            style={"fontSize": "12px", "color": S.TEXT, "lineHeight": "1.6",
                   "marginBottom": "4px"},
        ),
        html.Div(
            "Pairwise median relative MAE · median over all dataset × model × seed cells "
            "· page filters do not apply · source: converted/rq2_reference_agreement.csv",
            style={"fontSize": "11px", "color": S.TEXT2, "lineHeight": "1.55"},
        ),
    ], style={
        "background": "#F0F4FF",
        "border": f"1px solid {S.BORDER}",
        "borderRadius": "6px",
        "padding": "9px 13px",
        "margin": "0 0 2px",
    })


_CONVERGENCE_METRICS = {"relative_mae", "mean_sample_rho"}

def _f2_section_title(metric: str) -> str:
    label = _METRIC_META.get(metric, {}).get("label", metric)
    kind = "Error convergence" if metric in _CONVERGENCE_METRICS else "Quality check"
    return f"{kind} — {label}"


def _filter_bar() -> html.Div:
    """Dataset / model / approximator filter row rendered IN the page so
    IDs always exist in the DOM at load time (avoids topbar timing race)."""
    df = _load(_CONVERGENCE)
    datasets = [{"label": "All datasets", "value": "__all__"}] + \
               [{"label": d, "value": d} for d in sorted(df["dataset"].dropna().unique())] \
               if not df.empty else [{"label": "All datasets", "value": "__all__"}]
    models   = [{"label": "All models", "value": "__all__"}] + \
               [{"label": m, "value": m} for m in sorted(df["model"].dropna().unique())] \
               if not df.empty else [{"label": "All models", "value": "__all__"}]
    approxs  = sorted(df["approximator"].dropna().unique()) if not df.empty else []

    lbl_style = {"fontSize": "10px", "fontWeight": "700", "color": S.TEXT2,
                 "textTransform": "uppercase", "letterSpacing": "0.06em",
                 "marginBottom": "3px"}
    return html.Div([
        html.Div([
            html.Div("Dataset", style=lbl_style),
            dcc.Dropdown(id="rq1-ds", options=datasets, value="__all__",
                         clearable=False,
                         style={"width": "150px", "fontSize": "12px", "minHeight": "28px"}),
        ], style={"marginRight": "12px"}),
        html.Div([
            html.Div("Model", style=lbl_style),
            dcc.Dropdown(id="rq1-mdl", options=models, value="__all__",
                         clearable=False,
                         style={"width": "140px", "fontSize": "12px", "minHeight": "28px"}),
        ], style={"marginRight": "16px"}),
        html.Div([
            html.Div("Approximator", style=lbl_style),
            dcc.Checklist(
                id="rq1-approx",
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


def _metric_toggle() -> html.Div:
    """F2 metric toggle — two visually grouped RadioItems, kept mutually
    exclusive via sync_f2_metric callback."""
    _grp_lbl = {"fontSize": "10px", "fontWeight": "700", "color": S.TEXT2,
                "textTransform": "uppercase", "letterSpacing": "0.07em",
                "marginRight": "8px", "flexShrink": "0"}
    _radio_lbl = {"marginRight": "16px", "fontSize": "12px",
                  "cursor": "pointer", "color": S.TEXT}
    return html.Div([
        html.Span("Convergence:", style=_grp_lbl),
        dcc.RadioItems(
            id="rq1-conv-metric",
            options=[
                {"label": "relative MAE", "value": "relative_mae"},
                {"label": "Spearman ρ",   "value": "mean_sample_rho"},
            ],
            value="relative_mae",
            inline=True,
            inputStyle={"marginRight": "4px"},
            labelStyle=_radio_lbl,
        ),
        html.Span("│", style={"color": S.BORDER, "margin": "0 14px",
                              "fontSize": "16px", "lineHeight": "1"}),
        html.Span("Quality check:", style=_grp_lbl),
        dcc.RadioItems(
            id="rq1-conv-metric-quality",
            options=[
                {"label": "Sign agreement",  "value": "sign_agreement"},
                {"label": "Additivity gap",  "value": "relative_additivity_gap"},
            ],
            value=None,
            inline=True,
            inputStyle={"marginRight": "4px"},
            labelStyle=_radio_lbl,
        ),
    ], style={
        "display": "flex", "alignItems": "center", "flexWrap": "wrap",
        "padding": "8px 12px",
        "borderBottom": f"1px solid {S.BORDER}",
        "background": S.BG,
        "borderRadius": "10px 10px 0 0",
        "gap": "4px",
    })


def _seed_checklist() -> html.Div:
    """Per-seed inclusion toggles for the seed-stability box plot."""
    return html.Div([
        html.Span("Seeds:", style={
            "fontSize": "11px", "fontWeight": "600", "color": S.TEXT2,
            "marginRight": "10px", "flexShrink": "0",
        }),
        dcc.Checklist(
            id="rq1-stability-seeds",
            options=[{"label": str(s), "value": s} for s in range(10)],
            value=list(range(10)),
            inline=True,
            inputStyle={"marginRight": "4px"},
            labelStyle={"marginRight": "12px", "fontSize": "12px",
                        "cursor": "pointer", "color": S.TEXT},
        ),
    ], style={
        "display": "flex", "alignItems": "center", "flexWrap": "wrap",
        "padding": "8px 12px",
        "borderBottom": f"1px solid {S.BORDER}",
        "background": S.BG,
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
    is_sign = metric == "sign_agreement"
    is_gap = metric == "relative_additivity_gap"
    if is_mae:
        med, q25, q75 = "relative_mae_median", "relative_mae_q25", "relative_mae_q75"
    elif is_sign:
        med, q25, q75 = "sign_agreement_median", "sign_agreement_q25", "sign_agreement_q75"
    elif is_gap:
        med, q25, q75 = ("rel_additivity_gap_median",
                         "rel_additivity_gap_q25",
                         "rel_additivity_gap_q75")
    else:
        med, q25, q75 = "rho_median", "rho_q25", "rho_q75"

    fig = go.Figure()
    for method, mdf in pooled.groupby("method"):
        lib = mdf["library"].iloc[0]
        color = _lib_color(lib)
        dash = _APPROX_DASH.get(mdf["approximator"].iloc[0], "solid")
        mdf = mdf.sort_values("budget")

        fig.add_trace(go.Scatter(
            x=pd.concat([mdf["budget"], mdf["budget"][::-1]]),
            y=pd.concat([mdf[q75], mdf[q25][::-1]]),
            mode="none", fill="toself", fillcolor=_band_fill(color),
            hoverinfo="skip",
            legendgroup=method, showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=mdf["budget"], y=mdf[med],
            mode="lines", name=method,
            legendgroup=method,
            line=dict(color=color, width=2, dash=dash),
            hovertemplate=(f"<b>{method}</b><br>budget: %{{x}}<br>"
                           f"{metric}: %{{y:.4g}}<extra></extra>"),
        ))


    if is_sign:
        sign_vals = pooled["sign_agreement_median"].dropna()
        y_lo = max(0.0, float(sign_vals.min()) - 0.02) if len(sign_vals) else 0.0
        fig.add_hline(y=1.0,
                      line=dict(color=S.GREEN, width=1.2, dash="dot"),
                      annotation_text="perfect agreement",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=S.GREEN))
        y_cfg = dict(title="Sign agreement rate",
                     range=[y_lo, 1.015], gridcolor=S.BORDER, zeroline=False)
    elif is_gap:
        # Log scale over a fixed window: machine-precision cluster (~1e-15)
        # in the lower third, shapiq/kernel (~1e-9 to 1e-7) in the upper
        # half. -17.5 gives headroom below the ~1e-15 floor so those methods
        # don't disappear into the axis line.
        fig.add_hline(y=1e-15,
                      line=dict(color=S.GREEN, width=1.2, dash="dot"),
                      annotation_text="machine precision",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=S.GREEN))
        y_cfg = dict(
            title="Relative additivity gap",
            type="log", gridcolor=S.BORDER, zeroline=False,
            range=[-17.5, -7.5],  # log₁₀ units: 1e-17.5 … 1e-7.5
            exponentformat="power",
            tickmode="array",
            tickvals=[1e-16, 1e-14, 1e-12, 1e-10, 1e-8],
        )
    elif is_mae:
        y_cfg = dict(
            title="Attribution error vs exact values",
            type="log", gridcolor=S.BORDER, zeroline=False,
            tickmode="array",
            tickvals=[0.01, 0.05, 0.1, 0.5, 1.0],
            ticktext=["1 %", "5 %", "10 %", "50 %", "100 %"],
        )
    else:
        fig.add_hline(y=S.RHO_GOOD,
                      line=dict(color=S.GREEN, width=1.2, dash="dot"),
                      annotation_text=f"ρ = {S.RHO_GOOD}",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=S.GREEN))
        y_cfg = dict(title="Spearman rank correlation (ρ)",
                     gridcolor=S.BORDER, zeroline=False)

    fig.update_layout(
        **S._CHART_LAYOUT, height=440, margin=S._MARGIN, legend=S._LEGEND_H,
        xaxis=dict(title="Budget", type="log",
                   tickmode="array", tickvals=_BUDGETS,
                   ticktext=[str(b) for b in _BUDGETS],
                   gridcolor=S.BORDER, zeroline=False),
        yaxis=y_cfg,
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
#     color = library   ring fill = budget (thin edge 128 / wider 512 /
#     solid 2048) — same saturation, fill grows inward from the rim
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

_BUDGET_OUTER_SIZE = 12
# Center-mask diameter (px) — carves the hollow from inside, leaving a
# coloured ring at the outer edge. None = fully filled (2048).
_BUDGET_INNER_MASK = {128: 10, 512: 7}
# circle-open stroke widths for legend swatches (≈ same ring thickness).
_BUDGET_LEGEND_STROKE = {128: 2.2, 512: 5.8}
_PLOT_BG = "#F8FAFF"   # must match S._CHART_LAYOUT plot_bgcolor


def _budget_legend_marker(color: str, budget: int) -> dict:
    """Legend swatch — ring appearance without the plot-bg mask layer."""
    if budget == 2048:
        return dict(size=_BUDGET_OUTER_SIZE, symbol="circle", color=color,
                    line=dict(color=color, width=1.8))
    return dict(
        size=_BUDGET_OUTER_SIZE, symbol="circle-open", color="rgba(0,0,0,0)",
        line=dict(color=color, width=_BUDGET_LEGEND_STROKE[budget]),
    )


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

    # Approximation runs only — exact backends sit at MAE ≈ 0 which forces
    # the log y-axis to span 16 decades and squashes all approximation
    # differences into a tiny band at the top.
    apx = sub[~sub["is_reference"]]

    # Pool remaining dataset × model cells by median.
    grp = (apx.groupby(["library", "approximator", "budget", "method"],
                       as_index=False)
           .agg(runtime=("runtime_median", "median"),
                mae=("relative_mae_median", "median")))

    if grp.empty:
        return S.fig_empty()

    fig = go.Figure()
    hover = ("<b>%{customdata[0]}</b> · budget %{meta}<br>"
             "runtime: %{x:.3g} s<br>"
             "attribution error: %{y:.1%}<extra></extra>")
    for (lib, budget), gdf in grp.groupby(["library", "budget"]):
        color = _lib_color(lib)
        b = int(budget)
        group = f"pareto-{lib}-{b}"
        fig.add_trace(go.Scatter(
            x=gdf["runtime"], y=gdf["mae"],
            mode="markers",
            legendgroup=group,
            showlegend=False,
            meta=b,
            marker=dict(size=_BUDGET_OUTER_SIZE, symbol="circle", color=color,
                        line=dict(color=color, width=1.8)),
            customdata=gdf[["method"]].values,
            hovertemplate=hover,
        ))
        mask = _BUDGET_INNER_MASK.get(b)
        if mask is not None:
            # Hollow out the centre so only an edge ring stays coloured.
            fig.add_trace(go.Scatter(
                x=gdf["runtime"], y=gdf["mae"],
                mode="markers",
                legendgroup=group,
                showlegend=False,
                meta=b,
                marker=dict(size=mask, symbol="circle", color=_PLOT_BG,
                            line=dict(width=0, color=_PLOT_BG)),
                customdata=gdf[["method"]].values,
                hovertemplate=hover,
            ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name=f"{lib} · budget {b}",
            legendgroup=group,
            marker=_budget_legend_marker(color, b),
        ))

    fig.update_layout(
        **S._CHART_LAYOUT, height=480, margin=S._MARGIN, legend=S._LEGEND_H,
        xaxis=dict(title="Runtime (seconds)", type="log",
                   gridcolor=S.BORDER, zeroline=False,
                   tickmode="array",
                   tickvals=[0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300],
                   ticktext=["0.03 s", "0.1 s", "0.3 s", "1 s",
                             "3 s", "10 s", "30 s", "100 s", "300 s"]),
        yaxis=dict(
            title="Attribution error vs exact values",
            type="log", gridcolor=S.BORDER, zeroline=False,
            tickmode="array",
            tickvals=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
            ticktext=["1 %", "2 %", "5 %", "10 %", "20 %", "50 %", "100 %"],
        ),
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

_LINEAR_MODEL = "linear_regularized"
_LINEAR_Y_PAD = -0.004   # −0.4 % — lifts ~0 % points off the axis line
_NONLINEAR_MODELS = ("decision_tree", "gradient_boosting", "random_forest")


def _stability_method_order(sub: pd.DataFrame) -> list:
    return (sub.groupby("method")["relative_mae"].median()
            .sort_values().index.tolist())


def _add_stability_boxes(fig, sub: pd.DataFrame, order: list):
    """One box per method on a single full-height panel."""
    for method in order:
        mdf = sub[sub["method"] == method]
        if mdf.empty:
            continue
        color = _lib_color(mdf["library"].iloc[0])
        fig.add_trace(go.Box(
            y=mdf["relative_mae"], name=method, x=[method] * len(mdf),
            boxpoints="all", jitter=0.4, pointpos=0,
            marker=dict(color=color, size=3.5, opacity=0.4,
                        line=dict(color="white", width=0.4)),
            line=dict(color=color, width=2), fillcolor="rgba(0,0,0,0)",
            hovertemplate=(f"<b>{method}</b><br>"
                           "dataset: %{customdata[0]}<br>"
                           "model: %{customdata[1]}<br>"
                           "seed: %{customdata[2]}<br>"
                           "attribution error: %{y:.1%}<extra></extra>"),
            customdata=np.stack([
                mdf["dataset"].astype(str),
                mdf["model"].astype(str),
                mdf["seed"].astype(str),
            ], axis=1),
            showlegend=False,
        ))


def _stability_linear_yaxis(vals: pd.Series, *, ceil: float) -> dict:
    """Linear % axis with a small pad below 0 so ~0 % points sit above the line."""
    clean = vals.replace(0, np.nan).dropna()
    y_max = min(ceil, max(0.05, float(clean.max()) * 1.15)) if len(clean) else ceil
    step = 2 if y_max <= 0.20 else (5 if y_max <= 0.50 else 10)
    tick_vals = [i / 100 for i in range(0, int(y_max * 100) + step, step)]
    return dict(
        title="Attribution error vs exact values",
        type="linear", range=[_LINEAR_Y_PAD, y_max],
        tickmode="array", tickvals=tick_vals,
        ticktext=[f"{int(v * 100)} %" for v in tick_vals],
        gridcolor=S.BORDER, zeroline=True,
        zerolinecolor=S.BORDER, zerolinewidth=1,
    )


def _is_linear_stability_view(mdl: str, model_group: str) -> bool:
    return mdl == _LINEAR_MODEL or (mdl == "__all__" and model_group == "linear")


def _linear_stability_note() -> html.Div:
    """Responsive HTML note — wraps naturally on narrow viewports."""
    line = {"margin": "0", "fontSize": "10px", "lineHeight": "1.55",
            "color": S.TEXT2}
    return html.Div([
        html.P([
            html.Strong("6 of 7 methods"),
            " are at machine precision (~0%).",
        ], style={**line, "marginBottom": "6px"}),
        html.P([
            html.Strong("shap / kernel"),
            " is the outlier — KernelExplainer fits random coalitions by "
            "weighted regression and does not recover exact Shapley values "
            "on linear models at this budget, even though permutation methods do.",
        ], style=line),
    ], style={
        "marginBottom": "12px",
        "maxWidth": "100%",
        "overflowWrap": "break-word",
        "wordBreak": "break-word",
    })


def _graph(fig, *, responsive: bool = False) -> dcc.Graph:
    kwargs = {
        "figure": fig,
        "config": {"displayModeBar": False},
    }
    if responsive:
        kwargs["responsive"] = True
        kwargs["style"] = {"width": "100%", "minHeight": "440px"}
    return dcc.Graph(**kwargs)


def _stability_panel(fig: go.Figure, mdl: str, model_group: str) -> html.Div:
    children = []
    if _is_linear_stability_view(mdl, model_group):
        children.append(_linear_stability_note())
    children.append(_graph(fig, responsive=True))
    return html.Div(children, style={"width": "100%"})


def _build_seed_stability_figure(by_seed: pd.DataFrame, ds: str, mdl: str,
                                 budget: int,
                                 model_group: str = "nonlinear",
                                 seeds: list | None = None) -> go.Figure:
    if by_seed.empty:
        return S.fig_empty()

    sub = by_seed[by_seed["budget"] == budget]
    if ds != "__all__":
        sub = sub[sub["dataset"] == ds]
    if mdl != "__all__":
        sub = sub[sub["model"] == mdl]
    elif model_group == "linear":
        sub = sub[sub["model"] == _LINEAR_MODEL]
    else:
        sub = sub[sub["model"].isin(_NONLINEAR_MODELS)]

    if seeds is not None:
        seed_set = {int(s) for s in seeds}
        if not seed_set:
            return S.fig_empty("Select at least one seed.")
        sub = sub[sub["seed"].isin(seed_set)]

    sub = sub.dropna(subset=["relative_mae"])
    if sub.empty:
        return S.fig_empty()

    viewing_linear = _is_linear_stability_view(mdl, model_group)

    order = _stability_method_order(sub)
    fig = go.Figure()
    _add_stability_boxes(fig, sub, order)

    y_cfg = _stability_linear_yaxis(
        sub["relative_mae"],
        ceil=0.15 if viewing_linear else 1.5,
    )
    layout_margin = dict(l=60, r=16, t=36, b=110)

    fig.update_layout(
        **S._CHART_LAYOUT, height=440,
        margin=layout_margin,
        xaxis=dict(tickangle=-30, gridcolor="rgba(0,0,0,0)", automargin=True),
        yaxis=y_cfg,
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

    z_raw = pivot.values.astype(float)
    n = len(order)
    with np.errstate(divide="ignore"):
        z_log = np.full_like(z_raw, np.nan)
        for i in range(n):
            for j in range(n):
                if i != j and not np.isnan(z_raw[i, j]) and z_raw[i, j] > 0:
                    z_log[i, j] = np.log10(z_raw[i, j])

    text = [[_fmt_rel_mae_pct(z_raw[i, j]) if i != j else "—"
             for j in range(n)] for i in range(n)]
    hover_raw = [[("" if i == j else f"{z_raw[i, j]:.4e}")
                  for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=z_log, x=list(pivot.columns), y=list(pivot.index),
        text=text, texttemplate="%{text}",
        customdata=hover_raw,
        # One family: pale neutral at low end (0 % and 0.00005 % look
        # similar but distinguishable), ramping to dalex amber at 3.2 %.
        colorscale=[
            [0.0,  "#F5F8F4"],
            [0.60, "#F5F8F4"],
            [0.68, "#EFF3E8"],   # shapiq — barely warmer, same family
            [0.82, "#FEF3C7"],
            [0.92, "#FDE68A"],
            [1.0,  S.AMBER],     # dalex orange (#F59E0B)
        ],
        zmin=-16, zmax=-1.5,
        colorbar=dict(
            title="rel. MAE",
            tickmode="array",
            tickvals=[-15.5, -1.5],
            ticktext=["~0 % · 0.00005 %", "3.2 %"],
            thickness=14, len=0.8,
        ),
        hovertemplate=("source: <b>%{y}</b><br>target: <b>%{x}</b><br>"
                       "rel MAE: %{customdata}<br>"
                       "(%{text})<extra></extra>"),
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

        # RQ1-F1 — oracle validation (first: establishes the measurement basis)
        S.section(
            "Oracle validation — which 'exact' backend to trust",
            "",
            html.Div([
                _f1_note(),
                html.Div(id="rq1-f5-chart", style={"padding": "8px"}),
            ]),
            section_id="rq1-f1-section",
        ),

        # RQ1-F2 — error convergence (title and explainer are metric-driven)
        S.section(
            html.Div(id="rq1-f2-section-heading",
                     style={"fontSize": "15px", "fontWeight": "600",
                            "letterSpacing": "-0.01em", "color": S.TEXT}),
            "",
            html.Div([
                _metric_toggle(),
                html.Div(id="rq1-f2-metric-text"),
                _col_note(
                    "source: converted/rq2_convergence_aggregated.csv",
                    "reference: shap_true_value · agg: median(10 seeds) · error bars: q25–q75",
                ),
                html.Div(id="rq1-f1-chart", style={"padding": "8px"}),
            ]),
            section_id="rq1-f2-section",
        ),

        # RQ1-F3 — Pareto: runtime vs error
        S.section(
            "Runtime vs error trade-off",
            "Approximation methods only — exact backends are excluded because "
            "their MAE ≈ 0 would compress the entire approximation range into "
            "a sliver at the top of the axis. "
            "Each marker is one method × budget: x = how long it takes, "
            "y = how far off the attributions are on average (relative MAE — "
            "e.g. 0.05 means attributions are ~5% away from exact values). "
            "Lower-left is better. Points of the same colour use identical "
            "saturation — only the filled ring width differs: a thin edge "
            "rim for budget 128, a wider inward fill for 512, and solid "
            "for 2048.",
            html.Div([
                _col_note(
                    "source: converted/rq2_runtime_accuracy.csv",
                    "ring: budget (thin edge 128 · wider 512 · solid 2048) · color: library",
                    "pooling: median over all dataset × model cells",
                ),
                html.Div(id="rq1-f2-chart", style={"padding": "8px"}),
            ]),
            section_id="rq1-f3-section",
        ),

        # RQ1-F4 — seed stability
        S.section(
            "Seed stability of the error",
            "Raw seed-level error distributions — no aggregation. "
            "Tight boxes mean a single run is representative; wide spread "
            "means you could get unlucky with a random seed. "
            "Use the Model toggle to switch between "
            "decision_tree · gradient_boosting · random_forest and "
            "linear_regularized. "
            "On linear_regularized most methods hit ~0 % error (Shapley values "
            "are analytically exact for linear models) — except shap/kernel, "
            "which uses random coalition regression and stays approximate "
            "(typically 1–10 % at budget 512). Both views use a linear y-axis "
            "with a small pad below 0 % so near-zero points sit above the "
            "axis line. Uncheck individual seeds to exclude lucky or unlucky "
            "coalition draws — hover any point to see its dataset, model, "
            "and seed.",
            html.Div([
                _axis_toggle("rq1-stability-budget",
                             {"128": "128", "512": "512", "2048": "2048"},
                             "512", label="Budget"),
                _axis_toggle("rq1-stability-models",
                             {"nonlinear": ("decision_tree · gradient_boosting · "
                                            "random_forest"),
                              "linear": "linear_regularized"},
                             "nonlinear", label="Model"),
                _seed_checklist(),
                _col_note(
                    "source: converted/rq2_accuracy_by_seed.csv (no aggregation)",
                    "approximators only: kernel + permutation (7 methods)",
                    "one point per run: filtered seeds × dataset × model cells",
                ),
                html.Div(id="rq1-f3-chart", style={"padding": "8px"}),
            ]),
            section_id="rq1-f4-section",
        ),

        S.interpretation_note(_INTERP),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Callback — filtering only; statistics computed in the converter
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("rq1-conv-metric", "value"),
    Output("rq1-conv-metric-quality", "value"),
    Input("rq1-conv-metric", "value"),
    Input("rq1-conv-metric-quality", "value"),
    prevent_initial_call=True,
)
def sync_f2_metric(conv_metric, quality_metric):
    """Keep the two metric radio groups mutually exclusive."""
    if ctx.triggered_id == "rq1-conv-metric-quality" and quality_metric:
        return None, quality_metric
    if ctx.triggered_id == "rq1-conv-metric" and conv_metric:
        return conv_metric, None
    return no_update, no_update


@callback(
    Output("rq1-f5-chart", "children"),            # F1 slot — reference agreement
    Output("rq1-f2-section-heading", "children"),  # F2 dynamic title
    Output("rq1-f2-metric-text", "children"),      # F2 metric explainer
    Output("rq1-f1-chart", "children"),            # F2 slot — convergence
    Output("rq1-f2-chart", "children"),            # F3 slot — pareto
    Output("rq1-f3-chart", "children"),            # F4 slot — seed stability
    Input("rq1-ds", "data"),
    Input("rq1-mdl", "data"),
    Input("rq1-approx", "data"),
    Input("rq1-conv-metric", "value"),
    Input("rq1-conv-metric-quality", "value"),
    Input("rq1-stability-budget", "value"),
    Input("rq1-stability-models", "value"),
    Input("rq1-stability-seeds", "value"),
)
def update_rq2(ds, mdl, approxs, conv_metric, quality_metric, stability_budget,
               stability_models, stability_seeds):
    ds = ds or "__all__"
    mdl = mdl or "__all__"
    conv_metric = quality_metric or conv_metric or "relative_mae"
    stability_budget = int(stability_budget or 512)
    stability_models = stability_models or "nonlinear"
    if stability_seeds is None:
        stability_seeds = list(range(10))

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
        return _graph(fig)

    return (
        # F1 — oracle validation (no page filters applied)
        _g(_build_reference_agreement_figure(ref_agreement)),
        # F2 dynamic title
        _f2_section_title(conv_metric),
        # F2 metric explainer
        _metric_explainer(conv_metric),
        # F2 — convergence
        _g(_build_convergence_figure(pooled, conv_metric)),
        # F3 — Pareto runtime vs error
        _g(_build_pareto_figure(runtime_acc, ds, mdl)),
        # F4 — seed stability (responsive chart + HTML note on linear view)
        _stability_panel(
            _build_seed_stability_figure(
                by_seed, ds, mdl, stability_budget,
                model_group=stability_models, seeds=stability_seeds),
            mdl, stability_models),
    )
