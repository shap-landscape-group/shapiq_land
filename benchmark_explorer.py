"""
Benchmark Explorer — polished Dash app for SHAP approximation results.

Design principles:
  - Default view answers: "Which method is fast, accurate, and reliable?"
  - Only 2 primary controls (dataset + model) are visible by default.
  - Advanced filters (n_features, budget) are collapsed.
  - Raw relative_mae is never shown directly; instead quality_score = clip(-log10(mae), 0, 12).
  - Failures are detected and flagged, not silently mixed with good results.
  - Four views: Leaderboard, Pareto frontier, Budget convergence, Method matrix.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

# ── paths ────────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "results.csv")

# ── quality metric constants ─────────────────────────────────────────────────
EPSILON = 1e-10          # floor so log10 doesn't blow up
Q_MAX = 12.0             # cap: anything this good is displayed as 12
Q_MIN = 0.0              # floor: bad methods score 0
FAILURE_MAE = 1.0        # relative_mae > 1  →  failure (100 % error)

# ── design tokens ────────────────────────────────────────────────────────────
BG = "#F7F9FC"
CARD = "#FFFFFF"
BORDER = "#DDE3EF"
ACCENT = "#4F6EF7"
GREEN = "#10B981"
RED = "#EF4444"
AMBER = "#F59E0B"
MUTED = "#CBD5E1"
TEXT = "#0F172A"
TEXT2 = "#64748B"

# One colour per library — used consistently across all charts
LIB_COLOR = {
    "shap":      ACCENT,
    "shapiq":    "#7C3AED",
    "lightshap": GREEN,
    "dalex":     AMBER,
}

FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, sans-serif"


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading & metric computation
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in ["runtime_s", "relative_mae", "budget", "sign_agreement",
                "mean_sample_rho", "n_model_evals", "n_features", "n_samples"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # keep only approximation rows (true_value rows lack all quality metrics)
    df = df[df["computation_type"] == "approximation"].copy()

    # friendly method label
    df["method"] = df["library"] + " / " + df["approximator"].fillna("?")

    # failure flag — any of these conditions marks a run as failed
    df["is_failure"] = (
        df["relative_mae"].isna()
        | (df["relative_mae"] > FAILURE_MAE)
        | df["sign_agreement"].isna()
        | df["mean_sample_rho"].isna()
    )

    # quality score: higher = better, 0–12 range
    # good methods cluster near 8–12, bad near 0–4, failures forced to 0
    df["quality_score"] = np.clip(
        -np.log10(df["relative_mae"].clip(lower=EPSILON)),
        Q_MIN, Q_MAX,
    )
    df.loc[df["is_failure"], "quality_score"] = 0.0

    return df


def compute_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per method and rank by a weighted combined score."""
    grp = (
        df.groupby(["method", "library", "approximator"])
        .agg(
            q_median=("quality_score", "median"),
            q_p25=("quality_score", lambda x: x.quantile(0.25)),
            runtime_median=("runtime_s", "median"),
            sign_median=("sign_agreement", "median"),
            rho_median=("mean_sample_rho", "median"),
            failure_rate=("is_failure", "mean"),
            n_runs=("runtime_s", "count"),
        )
        .reset_index()
    )

    def norm01(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=s.index)

    grp["combined"] = (
        0.40 * norm01(grp["q_median"])
        + 0.25 * norm01(grp["sign_median"].fillna(0))
        + 0.20 * norm01(grp["rho_median"].fillna(0))
        + 0.10 * norm01(-grp["runtime_median"])
        + 0.05 * norm01(1 - grp["failure_rate"])
    )

    grp = grp.sort_values("combined", ascending=False).reset_index(drop=True)
    grp.insert(0, "rank", range(1, len(grp) + 1))
    return grp


def pareto_mark(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    """Return boolean Series: True if point is on the Pareto front (min-x, max-y)."""
    idx = df.index
    flags = pd.Series(False, index=idx)
    sorted_idx = df[x_col].argsort()
    best_y = -np.inf
    for i in sorted_idx:
        y = df.loc[idx[i], y_col]
        if y > best_y:
            best_y = y
            flags.iloc[i] = True
    return flags


# ═══════════════════════════════════════════════════════════════════════════════
# Chart builders
# ═══════════════════════════════════════════════════════════════════════════════

# Only truly-shared, never-overridden keys live here.
# margin, legend, xaxis, yaxis are set per-chart to avoid duplicate-kwarg errors.
_CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family=FONT, color=TEXT, size=12),
    plot_bgcolor=BG,
    paper_bgcolor=CARD,
)

_LEGEND_H = dict(orientation="h", yanchor="bottom", y=1.02,
                 xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11))
_MARGIN    = dict(l=55, r=16, t=36, b=48)


def fig_pareto(agg: pd.DataFrame) -> go.Figure:
    if agg.empty:
        return go.Figure(layout=dict(title="No data for current filters", **_CHART_LAYOUT))

    agg = agg.copy().reset_index(drop=True)
    agg["is_pareto"] = pareto_mark(agg, "runtime_median", "q_median")

    fig = go.Figure()

    # ── dominated cloud ──
    dom = agg[~agg["is_pareto"]]
    if not dom.empty:
        fig.add_trace(go.Scatter(
            x=dom["runtime_median"], y=dom["q_median"],
            mode="markers",
            name="Dominated",
            marker=dict(color=MUTED, size=9, opacity=0.55,
                        line=dict(color="white", width=1)),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Runtime: %{x:.3f} s<br>"
                "Quality: %{y:.2f}<br>"
                "Sign agr.: %{customdata[1]:.3f}<br>"
                "Failure: %{customdata[2]:.0f}%<extra></extra>"
            ),
            customdata=dom[["method", "sign_median", "failure_rate"]]
                .assign(failure_rate=dom["failure_rate"] * 100).values,
        ))

    # ── Pareto frontier step line ──
    par = agg[agg["is_pareto"]].sort_values("runtime_median")
    if not par.empty:
        fig.add_trace(go.Scatter(
            x=par["runtime_median"], y=par["q_median"],
            mode="lines",
            line=dict(color=GREEN, width=1.5, dash="dot"),
            name="Pareto frontier",
            hoverinfo="skip",
        ))

        colors = [LIB_COLOR.get(lib, ACCENT) for lib in par["library"]]
        fig.add_trace(go.Scatter(
            x=par["runtime_median"], y=par["q_median"],
            mode="markers+text",
            name="Pareto-optimal",
            marker=dict(color=colors, size=14,
                        line=dict(color="white", width=2)),
            text=par["method"],
            textposition="top center",
            textfont=dict(size=9, color=TEXT2),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Runtime: %{x:.3f} s<br>"
                "Quality: %{y:.2f}<br>"
                "Sign agr.: %{customdata[1]:.3f}<br>"
                "ρ: %{customdata[3]:.3f}<br>"
                "Failure: %{customdata[2]:.0f}%<extra></extra>"
            ),
            customdata=par[["method", "sign_median", "failure_rate", "rho_median"]]
                .assign(failure_rate=par["failure_rate"] * 100).values,
        ))

    fig.update_layout(
        **_CHART_LAYOUT,
        height=440,
        margin=_MARGIN,
        legend=_LEGEND_H,
        xaxis=dict(title="Median runtime (s) — log scale", type="log",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Quality score  (higher = better)", gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_budget(df: pd.DataFrame) -> go.Figure:
    sub = df[~df["is_failure"] & df["budget"].notna()].copy()
    if sub.empty:
        return go.Figure(layout=dict(title="No non-failed runs for current filters",
                                     **_CHART_LAYOUT))

    grp = (
        sub.groupby(["method", "library", "budget"])
        .agg(q=("quality_score", "median"))
        .reset_index()
    )

    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("budget")
        fig.add_trace(go.Scatter(
            x=mdf["budget"], y=mdf["q"],
            mode="lines+markers",
            name=method,
            line=dict(color=LIB_COLOR.get(lib, ACCENT), width=2),
            marker=dict(size=8, color=LIB_COLOR.get(lib, ACCENT),
                        line=dict(color="white", width=1.5)),
            hovertemplate=(
                f"<b>{method}</b><br>"
                "Budget: %{x}<br>"
                "Quality: %{y:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_CHART_LAYOUT,
        height=400,
        margin=_MARGIN,
        legend=_LEGEND_H,
        xaxis=dict(title="Budget (approximation evaluations)",
                   tickvals=[64, 256, 512], gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median quality score", gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_raw_scatter(df: pd.DataFrame) -> go.Figure:
    """Log-log scatter: relative MAE vs runtime, one trace per backend+approximator combo."""
    sub = df[df["computation_type"] == "approximation"].copy()
    if sub.empty:
        return go.Figure(layout=dict(title="No data", **_CHART_LAYOUT))

    sub["combo"] = sub["backend"] + ", " + sub["approximator"].fillna("?")

    # marker shape by approximator
    shape_map = {"kernel": "circle", "permutation": "diamond"}

    # colour by library (reuse LIB_COLOR, fall back to ACCENT)
    fig = go.Figure()
    for combo, grp in sub.groupby("combo"):
        lib  = grp["library"].iloc[0]
        approx = grp["approximator"].iloc[0] if grp["approximator"].notna().any() else "?"
        color  = LIB_COLOR.get(lib, ACCENT)
        symbol = shape_map.get(approx, "circle")

        # size encodes n_features (small=4, medium=14/16, large=64)
        sizes = grp["n_features"].fillna(4).clip(lower=4)
        sizes = 7 + (sizes / sizes.max()) * 14   # range ~7–21

        fig.add_trace(go.Scatter(
            x=grp["runtime_s"],
            y=grp["relative_mae"],
            mode="markers",
            name=combo,
            marker=dict(
                color=color,
                symbol=symbol,
                size=sizes,
                opacity=0.75,
                line=dict(color="white", width=0.8),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Runtime: %{x:.3f} s<br>"
                "Relative MAE: %{y:.2e}<br>"
                "Budget: %{customdata[1]}<br>"
                "n_features: %{customdata[2]}<br>"
                "Dataset: %{customdata[3]}<br>"
                "Model: %{customdata[4]}<extra></extra>"
            ),
            customdata=grp[["method", "budget", "n_features", "dataset", "model"]].values,
        ))

    # failure threshold line at relative_mae = 1.0
    fig.add_hline(
        y=1.0,
        line=dict(color=RED, width=1.2, dash="dot"),
        annotation_text="failure threshold (relative MAE = 1)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=RED),
    )

    fig.update_layout(
        **_CHART_LAYOUT,
        height=520,
        margin=dict(l=70, r=20, t=30, b=60),
        legend=dict(
            title=dict(text="Backend, Approximator", font=dict(size=11)),
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=BORDER,
            borderwidth=1,
        ),
        xaxis=dict(
            title="Runtime (seconds)",
            type="log",
            gridcolor=BORDER,
            zeroline=False,
        ),
        yaxis=dict(
            title="Relative MAE  (lower = better)",
            type="log",
            gridcolor=BORDER,
            zeroline=False,
        ),
    )
    return fig


def fig_leaderboard_bars(lb: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: quality score bars with failure-rate text annotations."""
    if lb.empty:
        return go.Figure(layout=dict(**_CHART_LAYOUT))

    # sort ascending so best method appears at top of horizontal chart
    lb = lb.sort_values("combined", ascending=True).reset_index(drop=True)

    colors = [LIB_COLOR.get(lib, ACCENT) for lib in lb["library"]]
    fail_pcts = lb["failure_rate"] * 100

    fig = go.Figure()

    # main quality bars
    fig.add_trace(go.Bar(
        y=lb["method"],
        x=lb["q_median"],
        orientation="h",
        name="Quality score",
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color="white", width=0.5)),
        text=lb["q_median"].round(1).astype(str),
        textposition="outside",
        textfont=dict(size=11, color=TEXT),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Quality: %{x:.2f}<br>"
            "Runtime: %{customdata[0]:.3f} s<br>"
            "Sign agr.: %{customdata[1]:.3f}<br>"
            "Failure: %{customdata[2]:.0f}%<extra></extra>"
        ),
        customdata=lb[["runtime_median", "sign_median", "failure_rate"]]
            .assign(failure_rate=fail_pcts).values,
    ))

    # failure-rate annotations: text label right of bar, red if >10 %
    annotations = []
    for _, row in lb.iterrows():
        fp = row["failure_rate"] * 100
        if fp > 0:
            color = RED if fp > 10 else TEXT2
            label = f"✕ {fp:.0f}%" if fp > 10 else f"{fp:.0f}%"
            annotations.append(dict(
                x=Q_MAX * 1.15,
                y=row["method"],
                text=f"<b>{label}</b>" if fp > 10 else label,
                showarrow=False,
                xanchor="left",
                font=dict(size=11, color=color, family=FONT),
                xref="x",
                yref="y",
            ))

    row_height = 32
    chart_height = max(260, len(lb) * row_height + 60)

    fig.update_layout(
        **_CHART_LAYOUT,
        height=chart_height,
        annotations=annotations,
        xaxis=dict(
            title="Median quality score (0 – 12)",
            range=[0, Q_MAX * 1.35],   # extra space for failure labels on the right
            gridcolor=BORDER, zeroline=False,
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=20, t=30, b=48),
        showlegend=False,
    )
    return fig


def fig_distribution(df: pd.DataFrame) -> go.Figure:
    """Box + strip plot of quality score distribution per method."""
    if df.empty:
        return go.Figure(layout=dict(title="No data", **_CHART_LAYOUT))

    # sort methods by median quality descending
    order = (
        df.groupby("method")["quality_score"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig = go.Figure()

    for method in order:
        sub = df[df["method"] == method]
        lib = sub["library"].iloc[0]
        color = LIB_COLOR.get(lib, ACCENT)

        # jittered strip (all raw points)
        fig.add_trace(go.Box(
            y=sub["quality_score"],
            name=method,
            boxpoints="all",
            jitter=0.35,
            pointpos=0,
            marker=dict(
                color=color,
                size=4,
                opacity=0.45,
                line=dict(color="white", width=0.5),
            ),
            line=dict(color=color, width=2),
            fillcolor="rgba(0,0,0,0)",
            whiskerwidth=0.6,
            hovertemplate=(
                f"<b>{method}</b><br>"
                "Quality: %{y:.2f}<extra></extra>"
            ),
            showlegend=False,
        ))

    # add a shaded band for the "good" zone (quality >= 8)
    fig.add_hrect(
        y0=8, y1=Q_MAX,
        fillcolor=GREEN, opacity=0.05,
        line_width=0,
        annotation_text="Good zone (≥ 8)",
        annotation_position="top left",
        annotation_font=dict(size=10, color=GREEN),
    )

    fig.update_layout(
        **_CHART_LAYOUT,
        height=440,
        yaxis=dict(
            title="Quality score (0 – 12)",
            range=[-0.5, Q_MAX + 0.5],
            gridcolor=BORDER, zeroline=False,
        ),
        xaxis=dict(
            tickangle=-30,
            gridcolor="rgba(0,0,0,0)",
            automargin=True,
        ),
        margin=dict(l=55, r=16, t=36, b=100),
    )
    return fig


def fig_matrix(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.groupby(["library", "approximator"])
        .agg(q=("quality_score", "median"))
        .reset_index()
        .pivot(index="library", columns="approximator", values="q")
    )

    if pivot.empty:
        return go.Figure(layout=dict(title="No data", **_CHART_LAYOUT))

    z = pivot.values
    text = [[f"{v:.1f}" if not np.isnan(v) else "—" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(pivot.columns),
        y=list(pivot.index),
        text=text,
        texttemplate="%{text}",
        colorscale=[[0, "#FEF3C7"], [0.4, "#60A5FA"], [1, "#1E3A8A"]],
        colorbar=dict(title="Quality", thickness=14, len=0.8),
        hovertemplate="Library: <b>%{y}</b><br>Approximator: <b>%{x}</b><br>Quality: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        **_CHART_LAYOUT,
        height=300,
        xaxis=dict(title="Approximator", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Library", gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=90, r=16, t=20, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Layout helpers
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_card(value: str, label: str, color: str | None = None) -> html.Div:
    return html.Div(
        [
            html.Div(value, style={
                "fontSize": "24px", "fontWeight": "700",
                "color": color or TEXT, "lineHeight": "1",
            }),
            html.Div(label, style={
                "fontSize": "11px", "color": TEXT2,
                "marginTop": "4px", "fontWeight": "500",
                "textTransform": "uppercase", "letterSpacing": "0.05em",
            }),
        ],
        style={
            "flex": "1", "minWidth": "140px",
            "background": CARD, "border": f"1px solid {BORDER}",
            "borderRadius": "12px", "padding": "16px 20px",
            "boxShadow": "0 1px 3px rgba(15,23,42,0.06)",
        },
    )


def kpi_strip(df: pd.DataFrame, lb: pd.DataFrame) -> html.Div:
    n_methods = lb.shape[0]
    best = lb.iloc[0] if not lb.empty else None
    # worst per-method failure rate (most meaningful single number)
    worst_fail_pct = lb["failure_rate"].max() * 100 if not lb.empty else 0
    worst_method   = lb.loc[lb["failure_rate"].idxmax(), "method"] if not lb.empty else "—"

    cards = [
        kpi_card(str(n_methods), "Methods compared"),
        kpi_card(
            best["method"] if best is not None else "—",
            "Top ranked",
            ACCENT,
        ),
        kpi_card(
            f"{best['q_median']:.1f} / 12" if best is not None else "—",
            "Best quality score",
            GREEN,
        ),
        kpi_card(
            f"{worst_fail_pct:.0f} %",
            f"Worst failure rate ({worst_method})",
            RED if worst_fail_pct > 10 else GREEN,
        ),
    ]

    return html.Div(cards, style={
        "display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "20px",
    })


def section(title: str, subtitle: str, children) -> html.Div:
    return html.Div(
        [
            html.H3(title, style={"margin": "0 0 2px", "fontSize": "16px", "fontWeight": "600"}),
            html.P(subtitle, style={"margin": "0 0 14px", "fontSize": "12px", "color": TEXT2}),
            html.Div(
                children,
                style={
                    "background": CARD, "borderRadius": "12px",
                    "border": f"1px solid {BORDER}", "overflow": "hidden",
                },
            ),
        ],
        style={"marginBottom": "24px"},
    )


def build_leaderboard_datatable(lb: pd.DataFrame) -> dash_table.DataTable:
    display = lb[[
        "rank", "method", "q_median", "runtime_median",
        "sign_median", "rho_median", "failure_rate", "n_runs",
    ]].copy()

    display["q_median"] = display["q_median"].round(2)
    display["runtime_median"] = display["runtime_median"].round(3)
    display["sign_median"] = display["sign_median"].round(3)
    display["rho_median"] = display["rho_median"].round(3)
    display["failure_rate"] = (display["failure_rate"] * 100).round(1)

    columns = [
        {"name": "#",             "id": "rank"},
        {"name": "Method",        "id": "method"},
        {"name": "Quality (med)", "id": "q_median"},
        {"name": "Runtime (s)",   "id": "runtime_median"},
        {"name": "Sign agr.",     "id": "sign_median"},
        {"name": "Mean ρ",        "id": "rho_median"},
        {"name": "Failure %",     "id": "failure_rate"},
        {"name": "Runs",          "id": "n_runs"},
    ]

    return dash_table.DataTable(
        id="lb-table",
        data=display.to_dict("records"),
        columns=columns,
        sort_action="native",
        page_size=25,
        style_table={"overflowX": "auto"},
        style_header={
            "background": BG,
            "color": TEXT2,
            "fontWeight": "600",
            "fontSize": "11px",
            "textTransform": "uppercase",
            "letterSpacing": "0.05em",
            "border": "none",
            "borderBottom": f"1px solid {BORDER}",
            "padding": "10px 14px",
            "fontFamily": FONT,
        },
        style_cell={
            "fontFamily": FONT,
            "fontSize": "13px",
            "padding": "10px 14px",
            "border": "none",
            "borderBottom": f"1px solid {BORDER}",
            "color": TEXT,
            "background": CARD,
        },
        style_data_conditional=[
            {"if": {"row_index": 0},
             "background": "#EEF2FF", "fontWeight": "600"},
            {"if": {"column_id": "failure_rate",
                    "filter_query": "{failure_rate} > 20"},
             "color": RED},
            {"if": {"column_id": "q_median",
                    "filter_query": "{q_median} >= 8"},
             "color": GREEN, "fontWeight": "600"},
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════════

def build_app() -> Dash:
    df = load_data(CSV_PATH)

    datasets = sorted(df["dataset"].dropna().unique())
    models   = sorted(df["model"].dropna().unique())
    n_feats  = sorted(df["n_features"].dropna().unique())

    app = Dash(__name__, suppress_callback_exceptions=True)
    app._df = df  # stash so callers can inspect if needed

    # inject Inter font and minimal global style
    app.index_string = """<!DOCTYPE html>
<html>
<head>
  {%metas%}<title>Benchmark Explorer</title>{%favicon%}{%css%}
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { margin: 0; background: """ + BG + """; font-family: """ + FONT + """; color: """ + TEXT + """; }
    .rc-slider-track { background-color: """ + ACCENT + """ !important; }
    .rc-slider-handle { border-color: """ + ACCENT + """ !important; }
    details > summary { list-style: none; cursor: pointer; }
    details > summary::-webkit-details-marker { display: none; }
    .tab { font-family: """ + FONT + """; font-size: 13px; font-weight: 500;
           color: """ + TEXT2 + """; border: none !important;
           background: transparent !important; padding: 10px 18px; }
    .tab--selected { color: """ + ACCENT + """ !important; font-weight: 600 !important;
                     border-bottom: 2px solid """ + ACCENT + """ !important; }
    .tabs--content { border: none !important; }
  </style>
</head>
<body>
  {%app_entry%}
  <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

    # ── primary filter bar ──────────────────────────────────────────────────
    header = html.Div(
        [
            html.Div(
                [
                    html.Div("Benchmark Explorer",
                             style={"fontSize": "18px", "fontWeight": "700", "flex": "1"}),
                    html.Div("SHAP approximation method benchmark",
                             style={"fontSize": "12px", "color": TEXT2, "marginTop": "2px"}),
                ],
                style={"flex": "1"},
            ),
            html.Div(
                [
                    html.Div("Dataset", style={
                        "fontSize": "10px", "fontWeight": "600", "color": TEXT2,
                        "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "4px",
                    }),
                    dcc.Dropdown(
                        id="ds-select",
                        options=[{"label": "All datasets", "value": "__all__"}]
                                + [{"label": d, "value": d} for d in datasets],
                        value="__all__",
                        clearable=False,
                        style={"width": "210px", "fontSize": "13px"},
                    ),
                ],
                style={"marginRight": "14px"},
            ),
            html.Div(
                [
                    html.Div("Model", style={
                        "fontSize": "10px", "fontWeight": "600", "color": TEXT2,
                        "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "4px",
                    }),
                    dcc.Dropdown(
                        id="mdl-select",
                        options=[{"label": "All models", "value": "__all__"}]
                                + [{"label": m, "value": m} for m in models],
                        value="__all__",
                        clearable=False,
                        style={"width": "210px", "fontSize": "13px"},
                    ),
                ],
                style={"marginRight": "14px"},
            ),
            html.Div(
                [
                    html.Div("Features", style={
                        "fontSize": "10px", "fontWeight": "600", "color": TEXT2,
                        "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "4px",
                    }),
                    dcc.Dropdown(
                        id="nf-select",
                        options=[{"label": "All", "value": "__all__"}]
                                + [{"label": str(int(n)), "value": n} for n in n_feats],
                        value="__all__",
                        clearable=False,
                        style={"width": "130px", "fontSize": "13px"},
                    ),
                ],
                style={"marginRight": "14px"},
            ),
            html.Div(
                [
                    html.Div("Budget", style={
                        "fontSize": "10px", "fontWeight": "600", "color": TEXT2,
                        "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "4px",
                    }),
                    dcc.Checklist(
                        id="budget-check",
                        options=[{"label": f"  {int(b)}", "value": float(b)}
                                 for b in sorted(df["budget"].dropna().unique())],
                        value=[float(b) for b in sorted(df["budget"].dropna().unique())],
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "12px", "fontSize": "13px", "cursor": "pointer"},
                    ),
                ],
            ),
        ],
        style={
            "display": "flex", "alignItems": "center",
            "padding": "14px 28px",
            "background": CARD, "borderBottom": f"1px solid {BORDER}",
            "flexWrap": "wrap", "gap": "8px",
        },
    )

    # ── tabs ────────────────────────────────────────────────────────────────
    tabs = dcc.Tabs(
        id="tabs",
        value="about",
        children=[
            dcc.Tab(label="About",              value="about"),
            dcc.Tab(label="Raw Data",           value="raw"),
            dcc.Tab(label="Leaderboard",        value="leaderboard"),
            dcc.Tab(label="Method Matrix",      value="matrix"),
            dcc.Tab(label="Distribution",       value="distribution"),
            dcc.Tab(label="Budget Convergence", value="budget"),
            dcc.Tab(label="Pareto Frontier",    value="pareto"),
        ],
        style={"marginBottom": "18px"},
    )

    app.layout = html.Div(
        [
            header,
            html.Div(
                [
                    tabs,
                    html.Div(id="tab-content"),
                ],
                style={"padding": "24px 28px", "maxWidth": "1360px", "margin": "0 auto"},
            ),
        ],
        style={"minHeight": "100vh", "background": BG},
    )

    # ── shared filter function ───────────────────────────────────────────────
    def apply_filters(ds, mdl, nf, budgets):
        fdf = df.copy()
        if ds  != "__all__": fdf = fdf[fdf["dataset"]    == ds]
        if mdl != "__all__": fdf = fdf[fdf["model"]      == mdl]
        if nf  != "__all__": fdf = fdf[fdf["n_features"] == nf]
        if budgets:          fdf = fdf[fdf["budget"].isin(budgets)]
        return fdf

    # ── single master callback ───────────────────────────────────────────────
    @app.callback(
        Output("tab-content","children"),
        Input("tabs",         "value"),
        Input("ds-select",    "value"),
        Input("mdl-select",   "value"),
        Input("nf-select",    "value"),
        Input("budget-check", "value"),
    )
    def update(tab, ds, mdl, nf, budgets):
        fdf = apply_filters(ds, mdl, nf, budgets)

        # leaderboard always uses the full (failure-inclusive) filtered set
        lb = compute_leaderboard(fdf)

        # ── tab content ─────────────────────────────────────────────────────
        if tab == "leaderboard":
            weight_note = (
                "Combined score = 40 % quality · 25 % sign agreement "
                "· 20 % rank correlation · 10 % speed · 5 % reliability"
            )
            content = html.Div(
                [
                    section(
                        "Method Ranking",
                        "Bars = median quality score (higher = better). "
                        "Diamonds = failure rate (red if > 10 %). "
                        "Color = library.",
                        dcc.Graph(
                            figure=fig_leaderboard_bars(lb),
                            config={"displayModeBar": False},
                            style={"padding": "8px"},
                        ),
                    ),
                    section(
                        "Detailed Table",
                        weight_note,
                        build_leaderboard_datatable(lb),
                    ),
                ]
            )

        elif tab == "pareto":
            agg = (
                fdf.groupby(["method", "library", "approximator"])
                .agg(
                    q_median=("quality_score", "median"),
                    runtime_median=("runtime_s", "median"),
                    sign_median=("sign_agreement", "median"),
                    rho_median=("mean_sample_rho", "median"),
                    failure_rate=("is_failure", "mean"),
                )
                .reset_index()
            )
            content = section(
                "Pareto Frontier",
                "Pareto-optimal methods are faster AND more accurate than all dominated methods. "
                "Dominated points are shown in gray.",
                dcc.Graph(figure=fig_pareto(agg),
                          config={"displayModeBar": False},
                          style={"padding": "8px"}),
            )

        elif tab == "budget":
            content = section(
                "Quality vs Budget",
                "Does spending more budget (more model evaluations) actually improve accuracy? "
                "Failed runs are excluded.",
                dcc.Graph(figure=fig_budget(fdf),
                          config={"displayModeBar": False},
                          style={"padding": "8px"}),
            )

        elif tab == "matrix":
            content = section(
                "Method Quality Matrix",
                "Median quality score per library × approximator. "
                "Darker blue = higher score = more accurate. "
                "Failed runs are included but scored as 0.",
                dcc.Graph(figure=fig_matrix(fdf),
                          config={"displayModeBar": False},
                          style={"padding": "8px"}),
            )

        elif tab == "distribution":
            content = section(
                "Quality Score Distribution",
                "Each box shows the spread of quality scores across all runs of that method. "
                "Points are individual runs. Green band = quality ≥ 8 (excellent). "
                "Methods sorted best-first.",
                dcc.Graph(
                    figure=fig_distribution(fdf),
                    config={"displayModeBar": False},
                    style={"padding": "8px"},
                ),
            )

        elif tab == "raw":
            content = section(
                "Approximation Quality vs Runtime",
                "Each point is one benchmark run. Y axis = relative MAE (lower = better, log scale). "
                "X axis = runtime in seconds (log scale). "
                "Marker size = number of features. "
                "Dotted red line = failure threshold (relative MAE \u2265 1).",
                dcc.Graph(
                    figure=fig_raw_scatter(fdf),
                    config={"displayModeBar": True},
                    style={"padding": "8px"},
                ),
            )

        elif tab == "about":
            def h2(text):
                return html.H2(text, style={"fontSize": "15px", "fontWeight": "700",
                                            "margin": "28px 0 6px", "color": TEXT,
                                            "borderBottom": f"1px solid {BORDER}",
                                            "paddingBottom": "6px"})
            def p(text):
                return html.P(text, style={"margin": "0 0 10px", "lineHeight": "1.7",
                                           "color": TEXT, "fontSize": "13px"})
            def col_row(name, dtype, desc):
                return html.Tr([
                    html.Td(name, style={"fontFamily": "monospace", "fontSize": "12px",
                                        "padding": "6px 12px", "color": ACCENT,
                                        "whiteSpace": "nowrap", "verticalAlign": "top"}),
                    html.Td(dtype, style={"fontSize": "11px", "padding": "6px 12px",
                                          "color": TEXT2, "whiteSpace": "nowrap",
                                          "verticalAlign": "top"}),
                    html.Td(desc, style={"fontSize": "13px", "padding": "6px 12px",
                                         "color": TEXT, "lineHeight": "1.5"}),
                ])
            table_style = {"width": "100%", "borderCollapse": "collapse",
                           "fontSize": "13px", "marginBottom": "12px"}
            th_style   = {"fontSize": "10px", "fontWeight": "600", "color": TEXT2,
                          "textTransform": "uppercase", "letterSpacing": "0.05em",
                          "padding": "8px 12px", "borderBottom": f"2px solid {BORDER}",
                          "textAlign": "left", "background": BG}
            content = html.Div(
                [
                    html.H1("Benchmark Data — Reference Guide",
                            style={"fontSize": "20px", "fontWeight": "700",
                                   "margin": "0 0 4px", "color": TEXT}),
                    html.P("A structured overview of the raw benchmark data and the derived metrics used in this dashboard.",
                           style={"color": TEXT2, "fontSize": "13px", "margin": "0 0 8px"}),

                    h2("Background"),
                    p("This benchmark evaluates SHAP approximation methods — algorithms that estimate Shapley value "
                      "feature attributions without computing them exactly. The benchmark runs each method across "
                      "multiple datasets, model types, feature counts, and approximation budgets and records how "
                      "closely the approximated SHAP values match a reference exact computation."),

                    h2("Experimental Dimensions"),
                    html.Table(
                        [
                            html.Thead(html.Tr([
                                html.Th("Dimension", style=th_style),
                                html.Th("Values", style=th_style),
                            ])),
                            html.Tbody([
                                html.Tr([html.Td("Datasets", style={"padding":"6px 12px","fontSize":"13px","borderBottom":f"1px solid {BORDER}"}),
                                         html.Td(", ".join(sorted(df["dataset"].dropna().unique())), style={"padding":"6px 12px","fontSize":"13px","borderBottom":f"1px solid {BORDER}"})]),
                                html.Tr([html.Td("Models", style={"padding":"6px 12px","fontSize":"13px","borderBottom":f"1px solid {BORDER}"}),
                                         html.Td(", ".join(sorted(df["model"].dropna().unique())), style={"padding":"6px 12px","fontSize":"13px","borderBottom":f"1px solid {BORDER}"})]),
                                html.Tr([html.Td("Libraries", style={"padding":"6px 12px","fontSize":"13px","borderBottom":f"1px solid {BORDER}"}),
                                         html.Td(", ".join(sorted(df["library"].dropna().unique())), style={"padding":"6px 12px","fontSize":"13px","borderBottom":f"1px solid {BORDER}"})]),
                                html.Tr([html.Td("Approximators", style={"padding":"6px 12px","fontSize":"13px","borderBottom":f"1px solid {BORDER}"}),
                                         html.Td(", ".join(sorted(df["approximator"].dropna().unique())), style={"padding":"6px 12px","fontSize":"13px","borderBottom":f"1px solid {BORDER}"})]),
                                html.Tr([html.Td("Feature counts (n_features)", style={"padding":"6px 12px","fontSize":"13px","borderBottom":f"1px solid {BORDER}"}),
                                         html.Td(", ".join(str(int(n)) for n in sorted(df["n_features"].dropna().unique())), style={"padding":"6px 12px","fontSize":"13px","borderBottom":f"1px solid {BORDER}"})]),
                                html.Tr([html.Td("Budgets", style={"padding":"6px 12px","fontSize":"13px"}),
                                         html.Td(", ".join(str(int(b)) for b in sorted(df["budget"].dropna().unique())), style={"padding":"6px 12px","fontSize":"13px"})]),
                            ]),
                        ],
                        style={**table_style, "border": f"1px solid {BORDER}", "borderRadius": "8px", "overflow": "hidden"},
                    ),

                    h2("CSV Columns"),
                    html.Table(
                        [
                            html.Thead(html.Tr([
                                html.Th("Column", style=th_style),
                                html.Th("Type", style=th_style),
                                html.Th("Description", style=th_style),
                            ])),
                            html.Tbody(
                                [
                                    col_row("dataset", "string", "Name of the tabular dataset used for the benchmark run."),
                                    col_row("model", "string", "ML model type trained on the dataset (e.g. random_forest, gradient_boosting)."),
                                    col_row("n_features", "int", "Number of input features used in this run. Controls problem dimensionality."),
                                    col_row("n_samples", "int", "Number of samples (rows) used for SHAP value computation."),
                                    col_row("backend", "string", "Internal identifier for the backend/configuration used to run the approximation."),
                                    col_row("library", "string", "Python library that implements the approximation method (shap, shapiq, lightshap, dalex)."),
                                    col_row("computation_type", "string", '"approximation" for estimated SHAP values; "true_value" for the exact reference computation.'),
                                    col_row("approximator", "string", "Approximation strategy within the library (kernel, permutation)."),
                                    col_row("budget", "float", "Number of model evaluations allocated to the approximation. Higher = more compute, potentially better quality."),
                                    col_row("n_eval", "int", "Actual number of evaluation samples used."),
                                    col_row("runtime_s", "float", "Wall-clock runtime in seconds for the approximation."),
                                    col_row("n_model_evals", "float", "Total number of model forward passes performed."),
                                    col_row("mean_abs_diff", "float", "Mean absolute difference between approximated and exact SHAP values."),
                                    col_row("relative_mae", "float", "Mean absolute error normalised by the magnitude of the exact values. Lower is better. Values above 1.0 indicate a failed run."),
                                    col_row("sign_agreement", "float", "Fraction of features where the sign of the approximated SHAP value matches the exact value. Range 0-1, higher is better."),
                                    col_row("mean_sample_rho", "float", "Mean Spearman rank correlation per sample between approximated and exact SHAP vectors. Range -1 to 1, higher is better."),
                                    col_row("reference_backend", "string", "The backend used to compute the exact reference SHAP values that approximations are compared against."),
                                ],
                                style={"verticalAlign": "top"},
                            ),
                        ],
                        style={**table_style, "border": f"1px solid {BORDER}"},
                    ),

                    h2("Derived Metrics (computed in this dashboard)"),
                    html.Table(
                        [
                            html.Thead(html.Tr([
                                html.Th("Metric", style=th_style),
                                html.Th("Formula", style=th_style),
                                html.Th("Interpretation", style=th_style),
                            ])),
                            html.Tbody([
                                html.Tr([
                                    html.Td("quality_score", style={"fontFamily":"monospace","fontSize":"12px","padding":"8px 12px","color":ACCENT,"borderBottom":f"1px solid {BORDER}","verticalAlign":"top"}),
                                    html.Td("clip( -log10(relative_mae), 0, 12 )", style={"fontFamily":"monospace","fontSize":"12px","padding":"8px 12px","color":TEXT2,"borderBottom":f"1px solid {BORDER}","verticalAlign":"top","whiteSpace":"nowrap"}),
                                    html.Td("Transforms relative_mae into a 0-12 scale where higher is better. "
                                            "A score of 12 means near-perfect agreement; 8 is excellent; below 4 is poor. "
                                            "Failed runs (relative_mae > 1) are forced to 0.",
                                            style={"fontSize":"13px","padding":"8px 12px","color":TEXT,"borderBottom":f"1px solid {BORDER}","lineHeight":"1.5"}),
                                ]),
                                html.Tr([
                                    html.Td("is_failure", style={"fontFamily":"monospace","fontSize":"12px","padding":"8px 12px","color":ACCENT,"verticalAlign":"top"}),
                                    html.Td("relative_mae > 1 OR sign_agreement is NaN OR mean_sample_rho is NaN", style={"fontFamily":"monospace","fontSize":"12px","padding":"8px 12px","color":TEXT2,"verticalAlign":"top"}),
                                    html.Td("Boolean flag. A run is considered a failure if the approximation error exceeds 100 % "
                                            "of the true value magnitude, or if quality metrics could not be computed.",
                                            style={"fontSize":"13px","padding":"8px 12px","color":TEXT,"lineHeight":"1.5"}),
                                ]),
                            ]),
                        ],
                        style={**table_style, "border": f"1px solid {BORDER}"},
                    ),

                    h2("Leaderboard Scoring"),
                    p("Methods are ranked by a weighted combined score that aggregates multiple quality dimensions:"),
                    html.Table(
                        [
                            html.Thead(html.Tr([
                                html.Th("Component", style=th_style),
                                html.Th("Weight", style=th_style),
                                html.Th("Source column", style=th_style),
                            ])),
                            html.Tbody([
                                html.Tr([html.Td(c, style={"padding":"6px 12px","fontSize":"13px","borderBottom":f"1px solid {BORDER}"}),
                                         html.Td(w, style={"padding":"6px 12px","fontSize":"13px","borderBottom":f"1px solid {BORDER}","color":ACCENT,"fontWeight":"600"}),
                                         html.Td(s, style={"fontFamily":"monospace","fontSize":"12px","padding":"6px 12px","color":TEXT2,"borderBottom":f"1px solid {BORDER}"})])
                                for c, w, s in [
                                    ("Approximation quality", "40 %", "quality_score"),
                                    ("Sign agreement",        "25 %", "sign_agreement"),
                                    ("Rank correlation",      "20 %", "mean_sample_rho"),
                                    ("Speed (inverse runtime)","10 %", "runtime_s"),
                                    ("Reliability (no failures)","5 %", "is_failure"),
                                ]
                            ]),
                        ],
                        style={**table_style, "border": f"1px solid {BORDER}"},
                    ),
                    p("Each component is min-max normalised across the visible methods before weighting, "
                      "so scores are relative to the current filter selection."),
                ],
                style={"maxWidth": "860px", "padding": "8px 4px 40px"},
            )

        else:
            content = html.Div("Unknown tab")

        return content

    return app


# Module-level objects required by gunicorn (Procfile: benchmark_explorer:server)
app    = build_app()
server = app.server


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False,
    )
