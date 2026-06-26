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
# RQ helper chart builders
# ═══════════════════════════════════════════════════════════════════════════════

def empty_state(title: str, message: str) -> html.Div:
    """Placeholder card when a research question cannot yet be answered."""
    return html.Div(
        [
            html.Div("⚠", style={"fontSize": "32px", "marginBottom": "12px"}),
            html.Div(title, style={"fontSize": "15px", "fontWeight": "600",
                                   "color": TEXT, "marginBottom": "8px"}),
            html.Div(message, style={"fontSize": "13px", "color": TEXT2,
                                     "lineHeight": "1.7", "maxWidth": "560px"}),
        ],
        style={
            "textAlign": "center", "padding": "48px 32px",
            "background": CARD, "borderRadius": "12px",
            "border": f"1px solid {BORDER}", "marginBottom": "24px",
        },
    )


def warning_note(message: str) -> html.Div:
    """Inline amber warning banner."""
    return html.Div(
        [html.Span("⚠ ", style={"fontWeight": "700"}), message],
        style={
            "background": "#FFFBEB", "border": f"1px solid {AMBER}",
            "borderRadius": "8px", "padding": "10px 16px",
            "fontSize": "13px", "color": "#92400E", "marginBottom": "16px",
        },
    )


def rq_header(rq: str, title: str, question: str) -> html.Div:
    """Standard title/question header for each RQ tab."""
    return html.Div(
        [
            html.Div(
                [html.Span(rq + " — ", style={"color": ACCENT, "fontWeight": "700"}),
                 html.Span(title, style={"fontWeight": "600"})],
                style={"fontSize": "18px", "marginBottom": "6px"},
            ),
            html.Div(
                ["Research question: ", html.Em(question)],
                style={"fontSize": "13px", "color": TEXT2,
                       "lineHeight": "1.6", "marginBottom": "20px"},
            ),
        ],
    )


def interpretation_note(text: str) -> html.Div:
    return html.Div(
        html.Em(text, style={"fontSize": "12px", "color": TEXT2, "lineHeight": "1.7"}),
        style={"marginTop": "8px", "marginBottom": "24px"},
    )


def fig_runtime_vs_features(df: pd.DataFrame) -> go.Figure:
    """Line chart: median runtime vs n_features, one line per method."""
    sub = df[df["n_features"].notna() & df["runtime_s"].notna()]
    if sub.empty or sub["n_features"].nunique() < 2:
        return go.Figure(layout=dict(title="Not enough n_features variation", **_CHART_LAYOUT))
    grp = (
        sub.groupby(["method", "library", "n_features"])
        .agg(rt=("runtime_s", "median"))
        .reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("n_features")
        fig.add_trace(go.Scatter(
            x=mdf["n_features"], y=mdf["rt"],
            mode="lines+markers", name=method,
            line=dict(color=LIB_COLOR.get(lib, ACCENT), width=2),
            marker=dict(size=8, color=LIB_COLOR.get(lib, ACCENT),
                        line=dict(color="white", width=1.5)),
            hovertemplate=(f"<b>{method}</b><br>"
                           "n_features: %{x}<br>Runtime: %{y:.3f} s<extra></extra>"),
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Number of features", gridcolor=BORDER, zeroline=False, type="log"),
        yaxis=dict(title="Median runtime (s) — log scale",
                   gridcolor=BORDER, zeroline=False, type="log"),
    )
    return fig


def fig_quality_vs_features(df: pd.DataFrame) -> go.Figure:
    """Line chart: median quality_score vs n_features."""
    sub = df[df["n_features"].notna()]
    if sub.empty or sub["n_features"].nunique() < 2:
        return go.Figure(layout=dict(title="Not enough n_features variation", **_CHART_LAYOUT))
    grp = (
        sub.groupby(["method", "library", "n_features"])
        .agg(q=("quality_score", "median"))
        .reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("n_features")
        fig.add_trace(go.Scatter(
            x=mdf["n_features"], y=mdf["q"],
            mode="lines+markers", name=method,
            line=dict(color=LIB_COLOR.get(lib, ACCENT), width=2),
            marker=dict(size=8, color=LIB_COLOR.get(lib, ACCENT),
                        line=dict(color="white", width=1.5)),
            hovertemplate=(f"<b>{method}</b><br>"
                           "n_features: %{x}<br>Quality: %{y:.2f}<extra></extra>"),
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Number of features", gridcolor=BORDER, zeroline=False, type="log"),
        yaxis=dict(title="Median quality score (0–12)", gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_failure_heatmap_by_features(df: pd.DataFrame) -> go.Figure:
    """Heatmap: failure rate per method × n_features."""
    sub = df[df["n_features"].notna()]
    if sub.empty:
        return go.Figure(layout=dict(title="No data", **_CHART_LAYOUT))
    pivot = (
        sub.groupby(["method", "n_features"])
        .agg(fr=("is_failure", "mean"))
        .reset_index()
        .pivot(index="method", columns="n_features", values="fr")
    )
    z = pivot.values * 100
    text = [[f"{v:.0f}%" if not np.isnan(v) else "—" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(int(c)) for c in pivot.columns],
        y=list(pivot.index),
        text=text,
        texttemplate="%{text}",
        colorscale=[[0, "#D1FAE5"], [0.5, "#FEF3C7"], [1, "#FEE2E2"]],
        zmin=0, zmax=100,
        colorbar=dict(title="Failure %", thickness=14, len=0.8),
        hovertemplate=(
            "Method: <b>%{y}</b><br>"
            "n_features: <b>%{x}</b><br>"
            "Failure rate: %{z:.1f}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        height=max(260, len(pivot) * 36 + 80),
        xaxis=dict(title="Number of features", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="", gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=16, t=20, b=60),
    )
    return fig


def fig_runtime_vs_budget(df: pd.DataFrame) -> go.Figure:
    """Line chart: median runtime vs budget."""
    sub = df[df["budget"].notna() & df["runtime_s"].notna()]
    if sub.empty or sub["budget"].nunique() < 2:
        return go.Figure(layout=dict(title="Not enough budget variation", **_CHART_LAYOUT))
    grp = (
        sub.groupby(["method", "library", "budget"])
        .agg(rt=("runtime_s", "median"))
        .reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("budget")
        fig.add_trace(go.Scatter(
            x=mdf["budget"], y=mdf["rt"],
            mode="lines+markers", name=method,
            line=dict(color=LIB_COLOR.get(lib, ACCENT), width=2),
            marker=dict(size=8, color=LIB_COLOR.get(lib, ACCENT),
                        line=dict(color="white", width=1.5)),
            hovertemplate=(f"<b>{method}</b><br>"
                           "Budget: %{x}<br>Runtime: %{y:.3f} s<extra></extra>"),
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Budget (model evaluations)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median runtime (s)", gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_metric_vs_budget(df: pd.DataFrame, metric: str = "mean_sample_rho") -> go.Figure:
    """Line chart: proxy convergence metric vs budget (failed runs excluded)."""
    sub = df[df["budget"].notna() & ~df["is_failure"]].copy()
    if sub.empty or sub["budget"].nunique() < 2:
        return go.Figure(layout=dict(title="Not enough budget variation", **_CHART_LAYOUT))
    grp = (
        sub.groupby(["method", "library", "budget"])
        .agg(val=(metric, "median"))
        .reset_index()
    )
    label_map = {
        "mean_sample_rho": "Median Spearman ρ (higher = better)",
        "sign_agreement":  "Median sign agreement (higher = better)",
        "quality_score":   "Median quality score (higher = better)",
    }
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("budget")
        fig.add_trace(go.Scatter(
            x=mdf["budget"], y=mdf["val"],
            mode="lines+markers", name=method,
            line=dict(color=LIB_COLOR.get(lib, ACCENT), width=2),
            marker=dict(size=8, color=LIB_COLOR.get(lib, ACCENT),
                        line=dict(color="white", width=1.5)),
            hovertemplate=(f"<b>{method}</b><br>"
                           f"Budget: %{{x}}<br>{metric}: %{{y:.3f}}<extra></extra>"),
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=380, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Budget (model evaluations)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title=label_map.get(metric, metric), gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_runtime_by_model(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: median runtime per model, grouped by method."""
    sub = df[df["model"].notna() & df["runtime_s"].notna()]
    if sub.empty:
        return go.Figure(layout=dict(title="No data", **_CHART_LAYOUT))
    grp = (
        sub.groupby(["method", "library", "model"])
        .agg(rt=("runtime_s", "median"))
        .reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        fig.add_trace(go.Bar(
            x=mdf["model"], y=mdf["rt"], name=method,
            marker_color=LIB_COLOR.get(lib, ACCENT), opacity=0.85,
            hovertemplate=(f"<b>{method}</b><br>"
                           "Model: %{x}<br>Runtime: %{y:.3f} s<extra></extra>"),
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        barmode="group",
        xaxis=dict(title="Model", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median runtime (s)", gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_quality_by_model(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: median quality score per model, grouped by method."""
    sub = df[df["model"].notna()]
    if sub.empty:
        return go.Figure(layout=dict(title="No data", **_CHART_LAYOUT))
    grp = (
        sub.groupby(["method", "library", "model"])
        .agg(q=("quality_score", "median"))
        .reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        fig.add_trace(go.Bar(
            x=mdf["model"], y=mdf["q"], name=method,
            marker_color=LIB_COLOR.get(lib, ACCENT), opacity=0.85,
            hovertemplate=(f"<b>{method}</b><br>"
                           "Model: %{x}<br>Quality: %{y:.2f}<extra></extra>"),
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        barmode="group",
        xaxis=dict(title="Model", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median quality score (0–12)", gridcolor=BORDER, zeroline=False),
    )
    return fig


def capability_matrix_table(benchmarked_libs: set) -> html.Div:
    """Static capability comparison table for known explanation libraries."""
    th_s = {
        "fontSize": "10px", "fontWeight": "600", "color": TEXT2,
        "textTransform": "uppercase", "letterSpacing": "0.05em",
        "padding": "8px 12px", "borderBottom": f"2px solid {BORDER}",
        "textAlign": "left", "background": BG,
    }

    def td_s(extra=None):
        base = {"fontSize": "12px", "padding": "8px 12px",
                "borderBottom": f"1px solid {BORDER}"}
        return {**base, **(extra or {})}

    def yn(v):
        if v == "yes":
            return html.Span("✓", style={"color": GREEN, "fontWeight": "700"})
        if v == "no":
            return html.Span("✗", style={"color": RED})
        return html.Span(v, style={"color": TEXT2, "fontSize": "11px"})

    def status_badge(lib):
        if lib in benchmarked_libs:
            return html.Span("benchmarked", style={
                "background": "#D1FAE5", "color": "#065F46",
                "borderRadius": "4px", "padding": "1px 6px",
                "fontSize": "10px", "fontWeight": "600",
            })
        return html.Span("planned", style={
            "background": "#FEF3C7", "color": "#92400E",
            "borderRadius": "4px", "padding": "1px 6px",
            "fontSize": "10px", "fontWeight": "600",
        })

    rows_data = [
        # (lib, feat_attr, interactions, model_agnostic, graph, nn_support, nn_focus, notes)
        ("shapiq",      "yes", "yes",               "yes",     "no",  "no",      "no",  "Main benchmark focus"),
        ("shap",        "yes", "limited",            "yes",     "no",  "partial", "no",  "Model-specific & agnostic variants"),
        ("lightshap",   "yes", "no",                 "yes",     "no",  "no",      "no",  "Speed-oriented approximation"),
        ("dalex",       "yes", "no",                 "yes",     "no",  "no",      "no",  "Model-agnostic, R-inspired"),
        ("alibi",       "yes", "not main focus",     "yes",     "no",  "partial", "no",  "Planned / not yet benchmarked"),
        ("shapleyflow", "no",  "different def.",     "no",      "yes", "no",      "no",  "Requires graph assumption"),
        ("captum",      "yes", "not main focus",     "partial", "no",  "no",      "yes", "Neural-network focus (PyTorch)"),
    ]

    cols = ["Library", "Status", "Feature attr.", "Interactions",
            "Model-agnostic", "Graph/flow", "NN support", "NN focus", "Notes"]
    thead = html.Thead(html.Tr([html.Th(c, style=th_s) for c in cols]))

    tbody_rows = []
    for row in rows_data:
        lib, feat, inter, agnostic, graph, nn_sup, nn_focus, notes = row
        tbody_rows.append(html.Tr([
            html.Td(lib,             style=td_s({"fontFamily": "monospace",
                                                 "color": ACCENT, "fontWeight": "600"})),
            html.Td(status_badge(lib), style=td_s()),
            html.Td(yn(feat),        style=td_s()),
            html.Td(yn(inter),       style=td_s()),
            html.Td(yn(agnostic),    style=td_s()),
            html.Td(yn(graph),       style=td_s()),
            html.Td(yn(nn_sup),      style=td_s()),
            html.Td(yn(nn_focus),    style=td_s()),
            html.Td(notes,           style=td_s({"color": TEXT2, "fontSize": "11px"})),
        ]))

    return html.Div(
        html.Table(
            [thead, html.Tbody(tbody_rows)],
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"},
        ),
        style={
            "background": CARD, "borderRadius": "12px",
            "border": f"1px solid {BORDER}", "overflowX": "auto",
        },
    )


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
        value="overview",
        children=[
            dcc.Tab(label="Overview",               value="overview"),
            dcc.Tab(label="RQ1 — Dimensionality",   value="rq1"),
            dcc.Tab(label="RQ2 — Budget & Conv.",   value="rq2"),
            dcc.Tab(label="RQ3 — Model Complexity", value="rq3"),
            dcc.Tab(label="RQ4 — Libraries",        value="rq4"),
            dcc.Tab(label="Raw Data",               value="raw"),
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
        lb  = compute_leaderboard(fdf)

        # ── Overview ────────────────────────────────────────────────────────
        if tab == "overview":
            def rq_card(rq, color, title, desc):
                return html.Div(
                    [
                        html.Div(rq, style={"fontSize": "11px", "fontWeight": "700",
                                            "color": color, "textTransform": "uppercase",
                                            "letterSpacing": "0.07em", "marginBottom": "6px"}),
                        html.Div(title, style={"fontSize": "14px", "fontWeight": "600",
                                               "color": TEXT, "marginBottom": "6px"}),
                        html.Div(desc,  style={"fontSize": "12px", "color": TEXT2,
                                               "lineHeight": "1.6"}),
                    ],
                    style={
                        "flex": "1", "minWidth": "200px",
                        "background": CARD, "border": f"1px solid {BORDER}",
                        "borderRadius": "12px", "padding": "16px 18px",
                        "borderTop": f"3px solid {color}",
                        "boxShadow": "0 1px 3px rgba(15,23,42,0.06)",
                    },
                )

            metric_rows = [
                ("quality_score",   "clip(−log₁₀(relative_mae), 0, 12)",
                 "0–12 scale, higher = better. 12 = near-perfect; < 4 = poor; failures forced to 0."),
                ("relative_mae",    "mean |approx − exact| / mean |exact|",
                 "Lower is better. Values > 1 indicate a failed run."),
                ("sign_agreement",  "fraction of features with correct sign",
                 "0–1, higher is better."),
                ("mean_sample_rho", "mean Spearman ρ per sample",
                 "−1 to 1, higher is better. Measures ranking agreement."),
                ("is_failure",      "relative_mae > 1 OR any quality metric is NaN",
                 "Boolean flag. True = failed / unreliable run."),
            ]
            th_s = {"fontSize": "10px", "fontWeight": "600", "color": TEXT2,
                    "textTransform": "uppercase", "letterSpacing": "0.05em",
                    "padding": "8px 12px", "borderBottom": f"2px solid {BORDER}",
                    "textAlign": "left", "background": BG}
            td_s = {"fontSize": "12px", "padding": "8px 12px",
                    "borderBottom": f"1px solid {BORDER}", "verticalAlign": "top"}

            # global summary table
            disp = lb[["rank", "method", "q_median", "runtime_median",
                        "failure_rate", "n_runs"]].copy()
            disp["q_median"]      = disp["q_median"].round(2)
            disp["runtime_median"] = disp["runtime_median"].round(3)
            disp["failure_rate"]  = (disp["failure_rate"] * 100).round(1)

            content = html.Div([
                html.H1("Shapley Benchmark Explorer",
                        style={"fontSize": "22px", "fontWeight": "700",
                               "margin": "0 0 6px", "color": TEXT}),
                html.P("This dashboard evaluates Shapley-based approximation methods across "
                       "four research dimensions: dimensionality, budget / convergence, "
                       "model complexity, and library design.",
                       style={"color": TEXT2, "fontSize": "13px",
                              "lineHeight": "1.7", "margin": "0 0 24px"}),

                html.Div([
                    rq_card("RQ1", ACCENT,      "Dimensionality",
                            "How do methods scale with more features?"),
                    rq_card("RQ2", "#7C3AED",   "Budget & Convergence",
                            "How much budget is needed for stable explanations?"),
                    rq_card("RQ3", AMBER,       "Model Complexity",
                            "Do explanations change with deeper or more complex models?"),
                    rq_card("RQ4", GREEN,       "Libraries",
                            "Which libraries work best under controlled settings?"),
                ], style={"display": "flex", "gap": "12px",
                           "flexWrap": "wrap", "marginBottom": "28px"}),

                section(
                    "Global Method Summary",
                    "All methods ranked by combined score (40 % quality · "
                    "25 % sign agreement · 20 % rank correlation · "
                    "10 % speed · 5 % reliability). "
                    "Applies to current filter selection.",
                    dash_table.DataTable(
                        data=disp.to_dict("records"),
                        columns=[
                            {"name": "#",            "id": "rank"},
                            {"name": "Method",       "id": "method"},
                            {"name": "Quality (med)","id": "q_median"},
                            {"name": "Runtime (s)",  "id": "runtime_median"},
                            {"name": "Failure %",    "id": "failure_rate"},
                            {"name": "Runs",         "id": "n_runs"},
                        ],
                        sort_action="native",
                        page_size=20,
                        style_table={"overflowX": "auto"},
                        style_header={
                            "background": BG, "color": TEXT2, "fontWeight": "600",
                            "fontSize": "11px", "textTransform": "uppercase",
                            "letterSpacing": "0.05em", "border": "none",
                            "borderBottom": f"1px solid {BORDER}",
                            "padding": "10px 14px", "fontFamily": FONT,
                        },
                        style_cell={
                            "fontFamily": FONT, "fontSize": "13px",
                            "padding": "10px 14px", "border": "none",
                            "borderBottom": f"1px solid {BORDER}", "color": TEXT,
                            "background": CARD,
                        },
                        style_data_conditional=[
                            {"if": {"row_index": 0},
                             "background": "#EEF2FF", "fontWeight": "600"},
                            {"if": {"column_id": "failure_rate",
                                    "filter_query": "{failure_rate} > 20"},
                             "color": RED},
                        ],
                    ),
                ),

                section(
                    "Metric Reference",
                    "Definitions of all quality metrics used in this dashboard.",
                    html.Table(
                        [
                            html.Thead(html.Tr([
                                html.Th("Metric",          style=th_s),
                                html.Th("Formula",         style=th_s),
                                html.Th("Interpretation",  style=th_s),
                            ])),
                            html.Tbody([
                                html.Tr([
                                    html.Td(name,   style={**td_s,
                                                            "fontFamily": "monospace",
                                                            "color": ACCENT,
                                                            "whiteSpace": "nowrap"}),
                                    html.Td(formula,style={**td_s,
                                                            "fontFamily": "monospace",
                                                            "color": TEXT2,
                                                            "fontSize": "11px",
                                                            "whiteSpace": "nowrap"}),
                                    html.Td(interp, style=td_s),
                                ])
                                for name, formula, interp in metric_rows
                            ]),
                        ],
                        style={"width": "100%", "borderCollapse": "collapse"},
                    ),
                ),
            ])

        # ── RQ1 — Dimensionality ─────────────────────────────────────────────
        elif tab == "rq1":
            unique_nf = fdf["n_features"].dropna().nunique()
            max_nf    = int(fdf["n_features"].max()) if not fdf.empty else 0
            success   = 1 - fdf["is_failure"].mean() if not fdf.empty else 0
            rt_at_max = (
                fdf[fdf["n_features"] == fdf["n_features"].max()]["runtime_s"].median()
                if not fdf.empty else float("nan")
            )

            kpis = html.Div([
                kpi_card(str(unique_nf),          "Feature-count settings"),
                kpi_card(str(max_nf),             "Highest n_features benchmarked"),
                kpi_card(f"{success * 100:.0f} %","Overall success rate", GREEN if success >= 0.8 else AMBER),
                kpi_card(
                    f"{rt_at_max:.2f} s" if not np.isnan(rt_at_max) else "—",
                    f"Median runtime at n_features = {max_nf}",
                ),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "20px"})

            warns = []
            if unique_nf < 2:
                warns.append(warning_note(
                    "RQ1 needs at least two different n_features values to show scaling "
                    "behaviour. Current filter selection contains only one feature-count "
                    "setting. Remove the n_features filter to see all available settings."
                ))

            content = html.Div([
                rq_header("RQ1", "Dimensionality",
                          "How does the number of input features affect feasibility, "
                          "runtime, and stability of Shapley-based explanations?"),
                kpis,
                *warns,
                section(
                    "Runtime vs Number of Features",
                    "Median runtime (log scale) as a function of n_features. "
                    "Steep lines indicate poor scalability.",
                    dcc.Graph(figure=fig_runtime_vs_features(fdf),
                              config={"displayModeBar": False},
                              style={"padding": "8px"}),
                ),
                section(
                    "Quality vs Number of Features",
                    "Median quality score as a function of n_features. "
                    "Declining lines mean the method becomes less accurate at scale.",
                    dcc.Graph(figure=fig_quality_vs_features(fdf),
                              config={"displayModeBar": False},
                              style={"padding": "8px"}),
                ),
                section(
                    "Failure Rate Heatmap",
                    "Fraction of failed runs per method and feature count. "
                    "Red cells indicate the method is unreliable at that dimensionality.",
                    dcc.Graph(figure=fig_failure_heatmap_by_features(fdf),
                              config={"displayModeBar": False},
                              style={"padding": "8px"}),
                ),
                interpretation_note(
                    "Interpretation: methods whose runtime grows steeply with n_features "
                    "will become infeasible at high dimensionality. Look for methods that "
                    "maintain quality (stable line near the top) while keeping runtime low."
                ),
            ])

        # ── RQ2 — Budget & Convergence ───────────────────────────────────────
        elif tab == "rq2":
            has_budget   = fdf["budget"].notna()
            sub_b        = fdf[has_budget]
            unique_b     = sub_b["budget"].nunique() if not sub_b.empty else 0
            max_b        = sub_b["budget"].max()     if not sub_b.empty else float("nan")
            best_q       = lb["q_median"].max()      if not lb.empty else float("nan")
            rt_at_maxb   = (
                sub_b[sub_b["budget"] == max_b]["runtime_s"].median()
                if not sub_b.empty else float("nan")
            )

            kpis = html.Div([
                kpi_card(str(unique_b),            "Budget settings available"),
                kpi_card(
                    str(int(max_b)) if not np.isnan(max_b) else "—",
                    "Highest budget benchmarked",
                ),
                kpi_card(
                    f"{best_q:.1f} / 12" if not np.isnan(best_q) else "—",
                    "Best median quality score", GREEN,
                ),
                kpi_card(
                    f"{rt_at_maxb:.2f} s" if not np.isnan(rt_at_maxb) else "—",
                    f"Median runtime at budget = {int(max_b) if not np.isnan(max_b) else '—'}",
                ),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "20px"})

            warns = []
            if unique_b < 2:
                warns.append(warning_note(
                    "RQ2 needs at least two budget settings to show convergence. "
                    "Remove the budget filter or select all budget checkboxes to "
                    "see convergence curves."
                ))

            content = html.Div([
                rq_header("RQ2", "Budget & Convergence",
                          "How much approximation budget is needed before "
                          "explanations become stable?"),
                kpis,
                *warns,
                section(
                    "Quality vs Budget",
                    "Median quality score as budget (number of model evaluations) "
                    "increases. Failed runs excluded. "
                    "Flat lines = already converged; rising lines = still improving.",
                    dcc.Graph(figure=fig_budget(fdf),
                              config={"displayModeBar": False},
                              style={"padding": "8px"}),
                ),
                section(
                    "Runtime vs Budget",
                    "How much wall-clock time does each additional budget unit cost?",
                    dcc.Graph(figure=fig_runtime_vs_budget(fdf),
                              config={"displayModeBar": False},
                              style={"padding": "8px"}),
                ),
                section(
                    "Rank Correlation (ρ) vs Budget",
                    "Spearman ρ between approximated and exact SHAP vectors per sample. "
                    "Values closer to 1.0 indicate better convergence. "
                    "Failed runs excluded.",
                    dcc.Graph(figure=fig_metric_vs_budget(fdf, "mean_sample_rho"),
                              config={"displayModeBar": False},
                              style={"padding": "8px"}),
                ),
                html.Div(
                    [
                        html.Span("ℹ ", style={"fontWeight": "700", "color": ACCENT}),
                        "Raw attribution vectors are not stored in the current benchmark. "
                        "Therefore we cannot compute direct vector convergence such as "
                        "cosine similarity or top-k overlap. "
                        "Convergence is measured here via proxy metrics: "
                        "relative MAE, sign agreement, and Spearman rank correlation.",
                    ],
                    style={
                        "background": "#EFF6FF", "border": f"1px solid {ACCENT}",
                        "borderRadius": "8px", "padding": "10px 16px",
                        "fontSize": "13px", "color": "#1E3A8A",
                        "marginBottom": "16px", "lineHeight": "1.7",
                    },
                ),
                interpretation_note(
                    "Interpretation: look for the budget level at which quality and ρ "
                    "plateau — that is the minimum budget needed. Higher budget beyond "
                    "that point wastes compute without improving accuracy."
                ),
            ])

        # ── RQ3 — Model Complexity ───────────────────────────────────────────
        elif tab == "rq3":
            unique_models = fdf["model"].dropna().nunique() if not fdf.empty else 0
            fastest_model = (
                fdf.groupby("model")["runtime_s"].median().idxmin()
                if not fdf.empty else "—"
            )
            best_model = (
                fdf.groupby("model")["quality_score"].median().idxmax()
                if not fdf.empty else "—"
            )
            worst_fail_model = (
                fdf.groupby("model")["is_failure"].mean().idxmax()
                if not fdf.empty else "—"
            )
            worst_fail_pct = (
                fdf.groupby("model")["is_failure"].mean().max() * 100
                if not fdf.empty else 0
            )

            kpis = html.Div([
                kpi_card(str(unique_models), "Model settings available"),
                kpi_card(fastest_model,      "Fastest model (median runtime)"),
                kpi_card(best_model,         "Most accurate model (median quality)", GREEN),
                kpi_card(
                    f"{worst_fail_pct:.0f} % ({worst_fail_model})",
                    "Worst failure rate by model",
                    RED if worst_fail_pct > 10 else GREEN,
                ),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "20px"})

            warns = []
            if unique_models < 2:
                warns.append(warning_note(
                    "RQ3 needs at least two model settings to show model-complexity "
                    "effects. Current filter selection contains only one model. "
                    "Remove the Model filter to compare all models."
                ))

            future_note = html.Div(
                [
                    html.Span("TODO ", style={"fontWeight": "700", "color": AMBER}),
                    "Future benchmark runs should include tree-depth sweeps "
                    "(e.g. depth 4, 8, 15, 50) and possibly neural-network benchmarks "
                    "via Captum to fully answer model-complexity effects.",
                ],
                style={
                    "background": "#FFFBEB", "border": f"1px solid {AMBER}",
                    "borderRadius": "8px", "padding": "10px 16px",
                    "fontSize": "13px", "color": "#92400E",
                    "marginBottom": "16px", "lineHeight": "1.7",
                },
            )

            content = html.Div([
                rq_header("RQ3", "Model Complexity",
                          "How does model complexity affect explanation "
                          "runtime and attribution stability?"),
                kpis,
                *warns,
                section(
                    "Runtime by Model",
                    "Median wall-clock runtime per model type, grouped by method. "
                    "More complex models are expected to have higher runtimes.",
                    dcc.Graph(figure=fig_runtime_by_model(fdf),
                              config={"displayModeBar": False},
                              style={"padding": "8px"}),
                ),
                section(
                    "Quality by Model",
                    "Median quality score per model type. "
                    "Some approximators may degrade on non-linear models.",
                    dcc.Graph(figure=fig_quality_by_model(fdf),
                              config={"displayModeBar": False},
                              style={"padding": "8px"}),
                ),
                future_note,
                interpretation_note(
                    "Interpretation: if runtime or quality changes significantly across "
                    "model types, the approximation method is sensitive to model "
                    "complexity. Robust methods should be stable across model types."
                ),
            ])

        # ── RQ4 — Libraries ──────────────────────────────────────────────────
        elif tab == "rq4":
            n_libs    = fdf["library"].dropna().nunique() if not fdf.empty else 0
            n_methods = lb.shape[0]
            top_m     = lb.iloc[0]["method"] if not lb.empty else "—"
            top_q     = lb.iloc[0]["q_median"] if not lb.empty else float("nan")
            worst_fr  = lb["failure_rate"].max() * 100 if not lb.empty else 0
            worst_fm  = lb.loc[lb["failure_rate"].idxmax(), "method"] if not lb.empty else "—"

            kpis = html.Div([
                kpi_card(str(n_libs),    "Libraries compared"),
                kpi_card(str(n_methods), "Methods compared"),
                kpi_card(
                    f"{top_m}  ({top_q:.1f} / 12)" if not np.isnan(top_q) else top_m,
                    "Top-ranked method", ACCENT,
                ),
                kpi_card(
                    f"{worst_fr:.0f} %  ({worst_fm})",
                    "Worst failure rate",
                    RED if worst_fr > 10 else GREEN,
                ),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "20px"})

            benchmarked = set(fdf["library"].dropna().unique())
            agg_pareto = (
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

            content = html.Div([
                rq_header("RQ4", "Libraries",
                          "How do different explanation libraries compare "
                          "under the same benchmark settings?"),
                kpis,
                section(
                    "Method Ranking",
                    "Bars = median quality score. "
                    "Failure-rate labels on the right (red if > 10 %). "
                    "Color = library.",
                    dcc.Graph(figure=fig_leaderboard_bars(lb),
                              config={"displayModeBar": False},
                              style={"padding": "8px"}),
                ),
                section(
                    "Method Quality Matrix",
                    "Median quality score per library × approximator. "
                    "Darker blue = higher score = more accurate. "
                    "Failed runs scored as 0.",
                    dcc.Graph(figure=fig_matrix(fdf),
                              config={"displayModeBar": False},
                              style={"padding": "8px"}),
                ),
                section(
                    "Pareto Frontier — Speed vs Accuracy",
                    "Colored points are Pareto-optimal: no other method is both "
                    "faster AND more accurate. Gray points are dominated.",
                    dcc.Graph(figure=fig_pareto(agg_pareto),
                              config={"displayModeBar": False},
                              style={"padding": "8px"}),
                ),
                section(
                    "Library Capability Matrix",
                    "Static overview of supported explanation types per library. "
                    "Libraries not yet in the benchmark are shown as 'planned'.",
                    capability_matrix_table(benchmarked),
                ),
                interpretation_note(
                    "Interpretation: the Pareto frontier shows which library/method "
                    "combinations give the best accuracy for a given runtime budget. "
                    "The capability matrix shows which libraries support features "
                    "beyond standard feature attribution."
                ),
            ])

        # ── Raw Data ─────────────────────────────────────────────────────────
        elif tab == "raw":
            content = section(
                "Approximation Quality vs Runtime",
                "Each point is one benchmark run. "
                "Y axis = relative MAE (lower = better, log scale). "
                "X axis = runtime in seconds (log scale). "
                "Marker size = number of features. "
                "Dotted red line = failure threshold (relative MAE ≥ 1).",
                dcc.Graph(
                    figure=fig_raw_scatter(fdf),
                    config={"displayModeBar": True},
                    style={"padding": "8px"},
                ),
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
