"""
pages/rq2_accuracy.py  —  RQ2: Approximation Accuracy

Benchmark axes of variation:
  Swept  : 3 seeds · n_background [50, 200] · budget [64, 512]
           · 4 models · 3 datasets · 4 libraries · 2 approximators
  Fixed  : n_features 4–14 · n_samples 1000 · n_eval 10 · imputer marginal
  Measured: runtime_s · n_model_evals · relative_mae · mean_sample_rho · sign_agreement

⚠ Caution: n_background=200 × budget=64 may produce unreliable results.
"""
import os
import sys

import dash
import numpy as np
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
_CSV  = os.path.join(_HERE, "results", "rq2_accuracy.csv")

# ─────────────────────────────────────────────────────────────────────────────
_RQ_HEADER = (
    "RQ2", "Approximation Accuracy",
    "As a user using Shapley approximations, I want to know how good the values "
    "actually are so that I can trust the explanations without wasting too much "
    "compute time on exact values.",
)

_INTERP = (
    "Use the Quality–Cost scatter to spot methods that achieve high \u03c1 cheaply. "
    "Use the Budget chart to find where quality plateaus \u2014 spending more beyond that "
    "wastes compute. The pairwise heatmap shows cross-library agreement and how close "
    "each approximation is to the exact Shapley values."
)

# ─────────────────────────────────────────────────────────────────────────────
#  Local helpers
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


def _fig_coverage(df) -> go.Figure:
    """Library \u00d7 approximator run-count heatmap."""
    if df.empty or "library" not in df.columns or "approximator" not in df.columns:
        return S.fig_empty("No data")
    counts = df.groupby(["library", "approximator"]).size().reset_index(name="n")
    pivot  = counts.pivot(index="library", columns="approximator", values="n").fillna(0)
    z      = pivot.values
    text   = [[f"{int(v)}" for v in row] for row in z]
    fig    = go.Figure(go.Heatmap(
        z=z, x=list(pivot.columns), y=list(pivot.index),
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#EEF2FF"], [1, S.ACCENT]],
        showscale=False,
        hovertemplate=(
            "Library: <b>%{y}</b><br>"
            "Approximator: <b>%{x}</b><br>"
            "Runs: %{z}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **S._CHART_LAYOUT,
        height=150,
        margin=dict(l=10, r=10, t=4, b=36),
        xaxis=dict(title="Approximator", gridcolor="rgba(0,0,0,0)",
                   tickfont=dict(size=10)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True,
                   tickfont=dict(size=10)),
    )
    return fig


def _config_card(df) -> html.Div:
    """Benchmark at a glance card with library \u00d7 approximator coverage."""
    nbgs    = sorted(df["n_background"].dropna().unique().astype(int)) \
              if "n_background" in df.columns and not df.empty else [50, 200]
    budgets = sorted(df["budget"].dropna().unique().astype(int)) \
              if "budget" in df.columns and not df.empty else [64, 512]

    left  = _col("Swept", [
        "3 datasets",
        "4 models",
        f"n_background: {', '.join(str(n) for n in nbgs)}",
        f"budget: {', '.join(str(b) for b in budgets)}",
        "4 libraries",
        "2 approximators",
        "3 seeds",
    ], "#EEF2FF", S.ACCENT)

    mid   = _col("Fixed", [
        "n_features: 4\u201314",
        "n_samples: 1000",
        "n_eval: 10",
        "imputer: marginal",
    ], "#F1F5F9", S.TEXT2)

    right = _col("Measured", [
        "runtime_s",
        "n_model_evals",
        "relative_mae",
        "mean_sample_rho",
        "sign_agreement",
    ], "#F0FDF4", S.GREEN)

    return html.Div([
        html.Div([
            html.Div("Benchmark at a glance", style={
                "fontSize": "13px", "fontWeight": "700", "color": S.TEXT,
                "marginBottom": "14px",
            }),
            html.Div([left, mid, right],
                     style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
        ], style={"flex": "1.4", "minWidth": "280px"}),
        html.Div([
            html.Div("Library \u00d7 approximator coverage  (runs per cell)", style={
                "fontSize": "10px", "fontWeight": "700", "color": S.TEXT2,
                "textTransform": "uppercase", "letterSpacing": "0.07em",
                "marginBottom": "6px",
            }),
            dcc.Graph(
                figure=_fig_coverage(df),
                config={"displayModeBar": False},
                style={"height": "150px"},
            ),
        ], style={"flex": "1", "minWidth": "240px"}),
    ], style={
        "display": "flex", "gap": "32px", "flexWrap": "wrap",
        "background": S.CARD, "borderRadius": "12px",
        "border": f"1px solid {S.BORDER}", "padding": "20px 24px",
        "marginBottom": "20px",
    })


def _col_note(*parts: str) -> html.Div:
    """Compact data-provenance annotation inside a chart card."""
    children = []
    for i, part in enumerate(parts):
        children.append(html.Span(part, style={"whiteSpace": "pre"}))
        if i < len(parts) - 1:
            children.append(html.Span("  \u00b7  ", style={"color": S.BORDER}))
    return html.Div(children, style={
        "fontSize": "10px", "color": S.TEXT2, "fontFamily": "monospace",
        "padding": "4px 12px 6px",
        "borderBottom": f"1px solid {S.BORDER}",
        "background": S.BG,
        "letterSpacing": "0.01em",
    })


def _axis_toggle(cid: str, options: dict, default: str,
                 label: str = "Axis", top: bool = False) -> html.Div:
    """Inline axis-selector row."""
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
        **({"borderRadius": "10px 10px 0 0"} if top else {}),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────────────────────────────

def layout(**kwargs):
    df, _ = S.try_load_data(_CSV)

    return html.Div([
        S.rq_header(*_RQ_HEADER),
        html.Div(id="rq2-kpis", style={"marginBottom": "20px"}),
        _config_card(df),

        # ── Quality\u2013Cost scatter ──────────────────────────────────────────
        S.section(
            "Quality vs Cost",
            "Each point = one library \u00d7 approximator \u00d7 budget \u00d7 n_background "
            "combination, aggregated (median) over seeds, models and datasets. "
            "Circle = n_background\u202050, diamond = n_background\u2002200. "
            "Red border = n_bg=200 \u00d7 budget\u202664 (low-budget warning).",
            html.Div([
                _axis_toggle(
                    "rq2-scatter-x",
                    {"runtime_s": "runtime (s)", "n_model_evals": "model evaluations"},
                    "runtime_s", label="x", top=True,
                ),
                _axis_toggle(
                    "rq2-scatter-y",
                    {"mean_sample_rho": "Spearman \u03c1",
                     "relative_mae": "relative MAE",
                     "sign_agreement": "sign agreement"},
                    "mean_sample_rho", label="y",
                ),
                _col_note(
                    "x \u2192 runtime_s  or  n_model_evals  (select above)",
                    "y \u2192 mean_sample_rho  or  relative_mae  or  sign_agreement",
                    "agg: median(seed, model, dataset) \u2192 per library \u00d7 approx \u00d7 budget \u00d7 n_background",
                ),
                html.Div(id="rq2-scatter-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-scatter-section",
        ),

        # ── Budget convergence ────────────────────────────────────────────
        S.section(
            "Quality vs Budget",
            "How much budget is needed before quality plateaus? "
            "Solid line / circle = n_background\u202050.  "
            "Dashed line / diamond = n_background\u2002200.  "
            "Failed runs excluded.",
            html.Div([
                _axis_toggle(
                    "rq2-budget-metric",
                    {"mean_sample_rho": "Spearman \u03c1", "relative_mae": "relative MAE"},
                    "mean_sample_rho", top=True,
                ),
                _col_note(
                    "x \u2192 budget",
                    "y \u2192 mean_sample_rho  or  relative_mae  (select above)",
                    "agg: median(seed, model, dataset) \u2192 per method \u00d7 budget \u00d7 n_background",
                    "failed runs excluded  (is_failure = relative_mae > 1.0 or NaN)",
                ),
                html.Div(id="rq2-budget-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-budget-section",
        ),

        # ── Cross-library pairwise heatmap ────────────────────────────────
        S.section(
            "Cross-library agreement heatmap",
            "How closely does each library\u2019s output agree with exact Shapley values "
            "and with every other library? "
            "Values averaged over all matching runs.",
            html.Div([
                _axis_toggle(
                    "rq2-pair-metric",
                    {"mean_sample_rho": "Spearman \u03c1", "relative_mae": "relative MAE"},
                    "mean_sample_rho", top=True,
                ),
                _col_note(
                    "source: pairwise_metrics JSON field per run",
                    "row = source library  \u00b7  col = compared-against library / exact",
                    "value = mean(metric) over all seeds, models, datasets, budgets, n_backgrounds in filter",
                ),
                html.Div(id="rq2-pairwise-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-pairwise-section",
        ),

        # ── Spearman \u03c1 distribution ────────────────────────────────────────
        S.section(
            "Spearman \u03c1 distribution",
            "Box + strip plot across all runs in the current filter. "
            "Wide spread = inconsistent results across seeds / datasets / models. "
            "Reference: \u03c1 \u2265 0.9.",
            html.Div([
                _col_note(
                    "y \u2192 mean_sample_rho  (from pairwise_metrics vs true_value reference)",
                    "each point = one run (seed \u00d7 dataset \u00d7 model \u00d7 budget \u00d7 n_background)",
                ),
                html.Div(id="rq2-distribution-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-distribution-section",
        ),

        # ── Method leaderboard ─────────────────────────────────────────────
        S.section(
            "Method ranking by Spearman \u03c1",
            "Bars = median \u03c1 per library \u00d7 approximator. "
            "Failure labels on the right (red if > 10\u202f%).",
            html.Div([
                _col_note(
                    "agg: median(mean_sample_rho) per method over all runs in current filter",
                    "failure rate = fraction of runs with relative_mae > 1.0  or  NaN",
                ),
                html.Div(id="rq2-leaderboard-chart", style={"padding": "8px"}),
            ]),
            section_id="rq2-leaderboard-section",
        ),

        S.interpretation_note(_INTERP),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Filter helper
# ─────────────────────────────────────────────────────────────────────────────

def _apply_filters(df, ds, mdl, nbg_vals, budget_vals, approxs):
    if ds and ds != "__all__":
        df = df[df["dataset"] == ds]
    if mdl and mdl != "__all__":
        df = df[df["model"] == mdl]
    if nbg_vals and "n_background" in df.columns:
        df = df[df["n_background"].isin([float(v) for v in nbg_vals])]
    if budget_vals and "budget" in df.columns:
        df = df[df["budget"].isin([float(v) for v in budget_vals])]
    if approxs and "approximator" in df.columns:
        df = df[df["approximator"].isin(approxs)]
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Callback
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("rq2-kpis",               "children"),
    Output("rq2-scatter-chart",      "children"),
    Output("rq2-budget-chart",       "children"),
    Output("rq2-pairwise-chart",     "children"),
    Output("rq2-distribution-chart", "children"),
    Output("rq2-leaderboard-chart",  "children"),
    Input("rq2-ds",            "value"),
    Input("rq2-mdl",           "value"),
    Input("rq2-nbg",           "value"),
    Input("rq2-budget-filt",   "value"),
    Input("rq2-approx",        "value"),
    Input("rq2-scatter-x",     "value"),
    Input("rq2-scatter-y",     "value"),
    Input("rq2-budget-metric", "value"),
    Input("rq2-pair-metric",   "value"),
)
def update_rq2(ds, mdl, nbg_vals, budget_vals, approxs,
               scatter_x, scatter_y, budget_metric, pair_metric):
    df, _ = S.try_load_data(_CSV)
    df    = _apply_filters(df, ds or "__all__", mdl or "__all__",
                           nbg_vals or [], budget_vals or [], approxs or [])

    scatter_x     = scatter_x     or "runtime_s"
    scatter_y     = scatter_y     or "mean_sample_rho"
    budget_metric = budget_metric or "mean_sample_rho"
    pair_metric   = pair_metric   or "mean_sample_rho"

    lb = S.compute_leaderboard(df)

    # ── KPIs ──────────────────────────────────────────────────────────────
    best_rho  = lb["rho_median"].max()               if not lb.empty else float("nan")
    best_m    = lb.iloc[0]["method"]                 if not lb.empty else "\u2014"
    zero_fail = lb[lb["failure_rate"] == 0].shape[0] if not lb.empty else 0
    n_methods = df["method"].nunique()               if not df.empty else 0

    min_budget_rho90 = "\u2014"
    if not df.empty and "budget" in df.columns and df["budget"].notna().any():
        bq = (
            df[~df["is_failure"]]
            .groupby(["method", "budget"])["mean_sample_rho"]
            .median().reset_index()
        )
        thr = bq[bq["mean_sample_rho"] >= 0.9]
        if not thr.empty:
            min_budget_rho90 = str(int(thr.groupby("method")["budget"].min().min()))

    kpis = S.kpi_row(
        
    )

    # ── Early exit when no data ────────────────────────────────────────────
    if df.empty:
        warn  = S.warning_note("No data matches the current filter selection.")
        empty = dcc.Graph(figure=S.fig_empty(), config={"displayModeBar": False})
        return kpis, warn, empty, empty, empty, empty

    # ── Warn about the problematic n_bg=200 \u00d7 budget=64 combo ─────────────
    warns = []
    if (
        "n_background" in df.columns and "budget" in df.columns and
        not df[(df["n_background"] == 200) & (df["budget"] <= 64)].empty
    ):
        warns.append(S.warning_note(
            "\u26a0 n_background=200 \u00d7 budget=64 is present in the current selection. "
            "These runs may be unreliable (insufficient evaluations relative to background). "
            "They appear with a red border in the scatter chart. "
            "Consider filtering out budget=64 when n_background=200 is selected."
        ))

    def _g(fig):
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    scatter_chart  = _g(S.fig_quality_vs_cost_rq2(df, scatter_x, scatter_y))
    budget_chart   = html.Div(warns + [_g(S.fig_budget_quality_lines(df, budget_metric))])
    pairwise_chart = _g(S.fig_pairwise_heatmap_rq2(df, pair_metric))
    dist_chart     = _g(S.fig_distribution(df))
    leader_chart   = _g(S.fig_leaderboard_bars(lb))

    return kpis, scatter_chart, budget_chart, pairwise_chart, dist_chart, leader_chart
