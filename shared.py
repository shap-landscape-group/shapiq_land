"""
shared.py — Design tokens, data loading, chart builders, and layout helpers
shared across all RQ pages of the SHAP Benchmark Explorer.

Importing this module is the only thing each RQ page needs for data and visuals.
When the CSV schema changes, update load_data() and the affected chart builders here.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html

# ── Quality metric constants ──────────────────────────────────────────────────
EPSILON     = 1e-10   # floor so log10 never blows up
Q_MAX       = 12.0    # cap: near-perfect accuracy
Q_MIN       = 0.0     # floor: worst / failed
FAILURE_MAE = 1.0     # relative_mae > 1 → failure (100 % error)

# ── Design tokens — SHAP-IQ brand palette ────────────────────────────────────
# Blues and pink-reds drawn from the SHAP-IQ logo.
BG     = "#EFF3FB"   # soft blue-lavender (matches hexagon background)
CARD   = "#FFFFFF"
BORDER = "#CFD9EF"   # blue-tinted border
ACCENT = "#4B6DD4"   # logo blue  (primary interactive colour)
PINK   = "#E84060"   # logo pink-red  (SHAP-IQ brand signature)
GREEN  = "#10B981"   # semantic: good / success
RED    = "#EF4444"   # semantic: failure
AMBER  = "#F59E0B"   # semantic: warning
MUTED  = "#B4C2DF"   # blue-tinted muted
TEXT   = "#1A2040"   # dark navy
TEXT2  = "#5A6A8A"   # secondary text

# One colour per library — kept consistent across every chart in every RQ
LIB_COLOR: dict[str, str] = {
    "shapiq":    PINK,          # SHAP-IQ brand pink-red
    "shap":      ACCENT,        # logo blue
    "lightshap": "#6BA3E8",    # lighter logo blue
    "dalex":     AMBER,
    "captum":    "#7C3AED",    # logo purple (top node)
    "alibi":     "#06B6D4",
}

FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, sans-serif"

# ── Columns that every benchmark CSV is expected to carry ────────────────────
EXPECTED_COLS: list[str] = [
    "dataset", "model", "n_features", "n_samples", "backend", "library",
    "computation_type", "approximator", "budget", "n_eval", "runtime_s",
    "n_model_evals", "mean_abs_diff", "relative_mae", "sign_agreement",
    "mean_sample_rho", "reference_backend",
]

# ── Shared chart defaults ─────────────────────────────────────────────────────
_CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family=FONT, color=TEXT, size=12),
    plot_bgcolor="#F8FAFF",   # very light blue chart area
    paper_bgcolor=CARD,
)
_LEGEND_H = dict(
    orientation="h", yanchor="bottom", y=1.02,
    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11),
)
_MARGIN = dict(l=55, r=16, t=36, b=48)

# ── Global CSS injected into every page via app.index_string ─────────────────
COMMON_STYLES = f"""
*, *::before, *::after {{ box-sizing: border-box; }}
body {{ margin: 0; background: {BG}; font-family: {FONT}; color: {TEXT}; }}

/* ── Tabs ── */
.tab {{
    font-family: {FONT}; font-size: 13px; font-weight: 500;
    color: {TEXT2}; border: none !important;
    background: transparent !important; padding: 10px 18px;
}}
.tab--selected {{
    color: {ACCENT} !important; font-weight: 600 !important;
    border-bottom: 2px solid {ACCENT} !important;
}}
.tabs--content {{ border: none !important; }}

/* ── Navigation bar ── */
.top-accent-stripe {{
    height: 3px;
    background: linear-gradient(90deg, {ACCENT} 0%, #7C3AED 50%, {PINK} 100%);
    flex-shrink: 0;
}}
.main-nav {{
    background: {TEXT};
    display: flex; align-items: center;
    padding: 0 28px; height: 52px;
    position: sticky; top: 0; z-index: 100;
    box-shadow: 0 2px 16px rgba(26,32,64,0.25);
    gap: 4px;
}}
a.nav-brand {{
    color: rgba(255,255,255,0.95) !important;
    text-decoration: none; font-weight: 700;
    font-size: 14px; letter-spacing: 0.01em;
    margin-right: 24px; white-space: nowrap;
    flex-shrink: 0;
}}
a.nav-brand:hover {{ color: white !important; }}
.nav-divider {{
    width: 1px; height: 20px;
    background: rgba(255,255,255,0.18);
    margin-right: 12px; flex-shrink: 0;
}}
a.nav-link {{
    text-decoration: none; font-size: 12px; font-weight: 500;
    color: rgba(255,255,255,0.6); padding: 6px 12px;
    border-radius: 5px; white-space: nowrap;
    letter-spacing: 0.01em;
    transition: color 0.15s, background 0.15s;
}}
a.nav-link:hover {{ color: white; background: rgba(255,255,255,0.1); }}

/* ── Advisor panel ── */
.advisor-toggle-btn {{
    position: fixed; bottom: 28px; right: 28px; z-index: 200;
    background: linear-gradient(135deg, {ACCENT} 0%, {PINK} 100%);
    color: white; border: none; border-radius: 50px;
    padding: 11px 22px; font-size: 13px; font-weight: 600;
    cursor: pointer; font-family: {FONT}; letter-spacing: 0.02em;
    white-space: nowrap;
    box-shadow: 0 4px 20px rgba(75,109,212,0.42);
    transition: transform 0.15s, box-shadow 0.15s;
}}
.advisor-toggle-btn:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(75,109,212,0.55);
}}
.advisor-overlay {{
    position: fixed; inset: 0;
    background: rgba(26,32,64,0.38);
    z-index: 299; opacity: 0; pointer-events: none;
    transition: opacity 0.28s;
}}
.advisor-overlay-open {{ opacity: 1 !important; pointer-events: all !important; }}
.advisor-panel {{
    position: fixed; top: 0; right: -440px;
    width: 420px; height: 100vh;
    background: {TEXT};
    overflow-y: auto; z-index: 300;
    box-shadow: -10px 0 48px rgba(26,32,64,0.32);
    transition: right 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.18) transparent;
}}
.advisor-panel-open {{ right: 0 !important; }}
.adv-radio .form-check-label {{ color: rgba(255,255,255,0.85) !important; }}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """
    Read a benchmark CSV, coerce numeric columns, keep only 'approximation' rows,
    add derived columns: method, is_failure, quality_score.

    Robust to missing optional columns — fills with NaN so charts show empty states
    rather than crashing.
    """
    df = pd.read_csv(path)

    numeric_cols = [
        "runtime_s", "relative_mae", "budget", "sign_agreement",
        "mean_sample_rho", "n_model_evals", "n_features", "n_samples",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only approximation rows (exact-value rows have no quality metrics)
    if "computation_type" in df.columns:
        df = df[df["computation_type"] == "approximation"].copy()
    else:
        df = df.copy()

    # Human-readable method label
    lib   = df["library"].fillna("?")   if "library"     in df.columns else pd.Series("?", index=df.index)
    approx = df["approximator"].fillna("?") if "approximator" in df.columns else pd.Series("?", index=df.index)
    df["method"] = lib + " / " + approx

    # Failure flag
    mae  = df["relative_mae"]   if "relative_mae"   in df.columns else pd.Series(np.nan, index=df.index)
    sign = df["sign_agreement"] if "sign_agreement" in df.columns else pd.Series(np.nan, index=df.index)
    rho  = df["mean_sample_rho"] if "mean_sample_rho" in df.columns else pd.Series(np.nan, index=df.index)
    df["is_failure"] = mae.isna() | (mae > FAILURE_MAE) | sign.isna() | rho.isna()

    # Quality score: 0–12, higher = better, failures forced to 0
    df["quality_score"] = np.clip(-np.log10(mae.clip(lower=EPSILON)), Q_MIN, Q_MAX)
    df.loc[df["is_failure"], "quality_score"] = 0.0

    return df


def try_load_data(*paths: str) -> tuple[pd.DataFrame, str | None]:
    """
    Try each path in order. Return (df, source_path) for the first file found,
    or (empty DataFrame with correct schema, None) if none exist.

    Callers can use `source is None` to show a 'data not yet collected' state.
    """
    for path in paths:
        if os.path.exists(path):
            return load_data(path), path

    empty = pd.DataFrame(columns=EXPECTED_COLS + ["method", "is_failure", "quality_score"])
    return empty, None


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per method and rank by a weighted combined score."""
    empty_cols = [
        "rank", "method", "library", "approximator",
        "q_median", "q_p25", "runtime_median", "sign_median",
        "rho_median", "failure_rate", "n_runs", "combined",
    ]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    grp = (
        df.groupby(["method", "library", "approximator"])
        .agg(
            q_median      = ("quality_score",  "median"),
            q_p25         = ("quality_score",  lambda x: x.quantile(0.25)),
            runtime_median = ("runtime_s",      "median"),
            sign_median   = ("sign_agreement", "median"),
            rho_median    = ("mean_sample_rho","median"),
            failure_rate  = ("is_failure",     "mean"),
            n_runs        = ("runtime_s",      "count"),
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
    """Boolean Series: True if point is on the Pareto front (min-x AND max-y)."""
    idx    = df.index
    flags  = pd.Series(False, index=idx)
    sorted_i = df[x_col].argsort()
    best_y = -np.inf
    for i in sorted_i:
        y = df.loc[idx[i], y_col]
        if y > best_y:
            best_y = y
            flags.iloc[i] = True
    return flags


def _lib_color(lib: str) -> str:
    return LIB_COLOR.get(lib, ACCENT)


# ═══════════════════════════════════════════════════════════════════════════════
# Chart builders — general / shared
# ═══════════════════════════════════════════════════════════════════════════════

def fig_empty(message: str = "No data available for the current filter selection") -> go.Figure:
    return go.Figure(layout=dict(
        title=dict(text=message, font=dict(size=13, color=TEXT2), x=0.5, xanchor="center"),
        **_CHART_LAYOUT,
    ))


def fig_leaderboard_bars(lb: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: median quality score per method, failure labels on right."""
    if lb.empty:
        return fig_empty()

    lb        = lb.sort_values("combined", ascending=True).reset_index(drop=True)
    colors    = [_lib_color(lib) for lib in lb["library"]]
    fail_pcts = lb["failure_rate"] * 100

    fig = go.Figure(go.Bar(
        y=lb["method"], x=lb["q_median"], orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(color="white", width=0.5)),
        text=lb["q_median"].round(1).astype(str),
        textposition="outside", textfont=dict(size=11, color=TEXT),
        hovertemplate=(
            "<b>%{y}</b><br>Quality: %{x:.2f}<br>"
            "Runtime: %{customdata[0]:.3f} s<br>"
            "Sign agr.: %{customdata[1]:.3f}<br>"
            "Failure: %{customdata[2]:.0f}%<extra></extra>"
        ),
        customdata=lb[["runtime_median", "sign_median", "failure_rate"]]
            .assign(failure_rate=fail_pcts).values,
    ))

    annotations = []
    for _, row in lb.iterrows():
        fp = row["failure_rate"] * 100
        if fp > 0:
            color = RED if fp > 10 else TEXT2
            label = f"✕ {fp:.0f}%" if fp > 10 else f"{fp:.0f}%"
            annotations.append(dict(
                x=Q_MAX * 1.15, y=row["method"],
                text=f"<b>{label}</b>" if fp > 10 else label,
                showarrow=False, xanchor="left",
                font=dict(size=11, color=color, family=FONT),
                xref="x", yref="y",
            ))

    fig.update_layout(
        **_CHART_LAYOUT,
        height=max(260, len(lb) * 32 + 60),
        annotations=annotations,
        xaxis=dict(title="Median quality score (0–12)",
                   range=[0, Q_MAX * 1.35], gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=20, t=30, b=48),
        showlegend=False,
    )
    return fig


def fig_pareto(agg: pd.DataFrame) -> go.Figure:
    """Scatter: median runtime (x, log) vs median quality (y). Pareto front highlighted."""
    if agg.empty:
        return fig_empty()

    agg            = agg.copy().reset_index(drop=True)
    agg["is_pareto"] = pareto_mark(agg, "runtime_median", "q_median")
    fig            = go.Figure()

    dom = agg[~agg["is_pareto"]]
    if not dom.empty:
        fig.add_trace(go.Scatter(
            x=dom["runtime_median"], y=dom["q_median"], mode="markers",
            name="Dominated",
            marker=dict(color=MUTED, size=9, opacity=0.55, line=dict(color="white", width=1)),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Runtime: %{x:.3f} s<br>"
                "Quality: %{y:.2f}<br>Failure: %{customdata[1]:.0f}%<extra></extra>"
            ),
            customdata=dom[["method", "failure_rate"]]
                .assign(failure_rate=dom["failure_rate"] * 100).values,
        ))

    par = agg[agg["is_pareto"]].sort_values("runtime_median")
    if not par.empty:
        fig.add_trace(go.Scatter(
            x=par["runtime_median"], y=par["q_median"], mode="lines",
            line=dict(color=GREEN, width=1.5, dash="dot"),
            name="Pareto frontier", hoverinfo="skip",
        ))
        colors = [_lib_color(lib) for lib in par["library"]]
        fig.add_trace(go.Scatter(
            x=par["runtime_median"], y=par["q_median"], mode="markers+text",
            name="Pareto-optimal",
            marker=dict(color=colors, size=14, line=dict(color="white", width=2)),
            text=par["method"], textposition="top center",
            textfont=dict(size=9, color=TEXT2),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Runtime: %{x:.3f} s<br>"
                "Quality: %{y:.2f}<br>Sign agr.: %{customdata[1]:.3f}<br>"
                "ρ: %{customdata[3]:.3f}<br>Failure: %{customdata[2]:.0f}%<extra></extra>"
            ),
            customdata=par[["method", "sign_median", "failure_rate", "rho_median"]]
                .assign(failure_rate=par["failure_rate"] * 100).values,
        ))

    fig.update_layout(
        **_CHART_LAYOUT, height=440, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Median runtime (s) — log scale", type="log",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Quality score  (higher = better)", gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_distribution(df: pd.DataFrame) -> go.Figure:
    """Box + strip plot of quality score per method, sorted by median."""
    if df.empty:
        return fig_empty()

    order = (
        df.groupby("method")["quality_score"]
        .median().sort_values(ascending=False).index.tolist()
    )
    fig = go.Figure()
    for method in order:
        sub   = df[df["method"] == method]
        color = _lib_color(sub["library"].iloc[0])
        fig.add_trace(go.Box(
            y=sub["quality_score"], name=method,
            boxpoints="all", jitter=0.35, pointpos=0,
            marker=dict(color=color, size=4, opacity=0.45, line=dict(color="white", width=0.5)),
            line=dict(color=color, width=2), fillcolor="rgba(0,0,0,0)",
            whiskerwidth=0.6,
            hovertemplate=f"<b>{method}</b><br>Quality: %{{y:.2f}}<extra></extra>",
            showlegend=False,
        ))
    fig.add_hrect(
        y0=8, y1=Q_MAX, fillcolor=GREEN, opacity=0.05, line_width=0,
        annotation_text="Good zone (≥ 8)", annotation_position="top left",
        annotation_font=dict(size=10, color=GREEN),
    )
    fig.update_layout(
        **_CHART_LAYOUT, height=440,
        yaxis=dict(title="Quality score (0–12)", range=[-0.5, Q_MAX + 0.5],
                   gridcolor=BORDER, zeroline=False),
        xaxis=dict(tickangle=-30, gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=55, r=16, t=36, b=100),
    )
    return fig


def fig_matrix(df: pd.DataFrame) -> go.Figure:
    """Heatmap: median quality per library × approximator."""
    if df.empty:
        return fig_empty()
    pivot = (
        df.groupby(["library", "approximator"])
        .agg(q=("quality_score", "median")).reset_index()
        .pivot(index="library", columns="approximator", values="q")
    )
    if pivot.empty:
        return fig_empty()
    z    = pivot.values
    text = [[f"{v:.1f}" if not np.isnan(v) else "—" for v in row] for row in z]
    fig  = go.Figure(go.Heatmap(
        z=z, x=list(pivot.columns), y=list(pivot.index),
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#FEF3C7"], [0.4, "#60A5FA"], [1, "#1E3A8A"]],
        colorbar=dict(title="Quality", thickness=14, len=0.8),
        hovertemplate=(
            "Library: <b>%{y}</b><br>"
            "Approximator: <b>%{x}</b><br>"
            "Quality: %{z:.2f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **_CHART_LAYOUT, height=300,
        xaxis=dict(title="Approximator", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Library", gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=90, r=16, t=20, b=60),
    )
    return fig


def fig_raw_scatter(df: pd.DataFrame) -> go.Figure:
    """Log-log scatter: relative MAE vs runtime per backend+approximator combo."""
    if df.empty:
        return fig_empty()

    df = df.copy()
    df["combo"] = (
        df.get("backend", pd.Series("?", index=df.index)).fillna("?")
        + ", "
        + df.get("approximator", pd.Series("?", index=df.index)).fillna("?")
    )
    shape_map = {"kernel": "circle", "permutation": "diamond"}
    fig = go.Figure()

    for combo, grp in df.groupby("combo"):
        lib    = grp["library"].iloc[0]
        approx = grp["approximator"].iloc[0] if grp["approximator"].notna().any() else "?"
        color  = _lib_color(lib)
        symbol = shape_map.get(approx, "circle")
        sizes  = grp["n_features"].fillna(4).clip(lower=4)
        sizes  = 7 + (sizes / sizes.max()) * 14

        fig.add_trace(go.Scatter(
            x=grp["runtime_s"], y=grp["relative_mae"],
            mode="markers", name=combo,
            marker=dict(color=color, symbol=symbol, size=sizes,
                        opacity=0.75, line=dict(color="white", width=0.8)),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Runtime: %{x:.3f} s<br>"
                "Relative MAE: %{y:.2e}<br>Budget: %{customdata[1]}<br>"
                "n_features: %{customdata[2]}<br>Dataset: %{customdata[3]}<extra></extra>"
            ),
            customdata=grp[["method", "budget", "n_features", "dataset"]].values,
        ))

    fig.add_hline(
        y=1.0, line=dict(color=RED, width=1.2, dash="dot"),
        annotation_text="failure threshold (relative MAE = 1)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=RED),
    )
    fig.update_layout(
        **_CHART_LAYOUT, height=520,
        margin=dict(l=70, r=20, t=30, b=60),
        legend=dict(title=dict(text="Backend, Approximator", font=dict(size=11)),
                    font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
                    bordercolor=BORDER, borderwidth=1),
        xaxis=dict(title="Runtime (seconds)", type="log", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Relative MAE  (lower = better)", type="log",
                   gridcolor=BORDER, zeroline=False),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart builders — RQ2: Approximation Accuracy
# ═══════════════════════════════════════════════════════════════════════════════

def fig_budget_quality(df: pd.DataFrame) -> go.Figure:
    """Quality score vs budget — failed runs excluded."""
    sub = df[~df["is_failure"] & df["budget"].notna()].copy()
    if sub.empty:
        return fig_empty("No non-failed runs for current filters")
    grp = (
        sub.groupby(["method", "library", "budget"])
        .agg(q=("quality_score", "median")).reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("budget")
        fig.add_trace(go.Scatter(
            x=mdf["budget"], y=mdf["q"], mode="lines+markers", name=method,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib), line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>Budget: %{{x}}<br>Quality: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Budget (approximation evaluations)",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median quality score (0–12)", gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_runtime_vs_budget(df: pd.DataFrame) -> go.Figure:
    """Runtime vs budget — how much does extra budget cost?"""
    sub = df[df["budget"].notna() & df["runtime_s"].notna()]
    if sub.empty or sub["budget"].nunique() < 2:
        return fig_empty("Not enough budget variation")
    grp = (
        sub.groupby(["method", "library", "budget"])
        .agg(rt=("runtime_s", "median")).reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("budget")
        fig.add_trace(go.Scatter(
            x=mdf["budget"], y=mdf["rt"], mode="lines+markers", name=method,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib), line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>Budget: %{{x}}<br>Runtime: %{{y:.3f}} s<extra></extra>",
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Budget (model evaluations)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median runtime (s)", gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_metric_vs_budget(df: pd.DataFrame, metric: str = "mean_sample_rho") -> go.Figure:
    """Convergence proxy metric vs budget — failed runs excluded."""
    sub = df[df["budget"].notna() & ~df["is_failure"]].copy()
    if sub.empty or sub["budget"].nunique() < 2:
        return fig_empty("Not enough budget variation")
    grp = (
        sub.groupby(["method", "library", "budget"])
        .agg(val=(metric, "median")).reset_index()
    )
    label_map = {
        "mean_sample_rho": "Median Spearman ρ  (higher = better)",
        "sign_agreement":  "Median sign agreement  (higher = better)",
        "quality_score":   "Median quality score  (higher = better)",
    }
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("budget")
        fig.add_trace(go.Scatter(
            x=mdf["budget"], y=mdf["val"], mode="lines+markers", name=method,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib), line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>Budget: %{{x}}<br>{metric}: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=380, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Budget (model evaluations)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title=label_map.get(metric, metric), gridcolor=BORDER, zeroline=False),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart builders — RQ1: Dimensionality
# ═══════════════════════════════════════════════════════════════════════════════

def fig_runtime_vs_features(df: pd.DataFrame) -> go.Figure:
    """Runtime (log) vs n_features (log) — main scaling chart for RQ1."""
    sub = df[df["n_features"].notna() & df["runtime_s"].notna()]
    if sub.empty or sub["n_features"].nunique() < 2:
        return fig_empty("Not enough n_features variation — try removing feature-count filter")
    grp = (
        sub.groupby(["method", "library", "n_features"])
        .agg(rt=("runtime_s", "median")).reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("n_features")
        fig.add_trace(go.Scatter(
            x=mdf["n_features"], y=mdf["rt"], mode="lines+markers", name=method,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib), line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>n_features: %{{x}}<br>Runtime: %{{y:.3f}} s<extra></extra>",
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Number of features", gridcolor=BORDER, zeroline=False, type="log"),
        yaxis=dict(title="Median runtime (s) — log scale", gridcolor=BORDER,
                   zeroline=False, type="log"),
    )
    return fig


def fig_quality_vs_features(df: pd.DataFrame) -> go.Figure:
    """Quality score vs n_features — does accuracy degrade at scale?"""
    sub = df[df["n_features"].notna()]
    if sub.empty or sub["n_features"].nunique() < 2:
        return fig_empty("Not enough n_features variation — try removing feature-count filter")
    grp = (
        sub.groupby(["method", "library", "n_features"])
        .agg(q=("quality_score", "median")).reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("n_features")
        fig.add_trace(go.Scatter(
            x=mdf["n_features"], y=mdf["q"], mode="lines+markers", name=method,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib), line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>n_features: %{{x}}<br>Quality: %{{y:.2f}}<extra></extra>",
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
        return fig_empty()
    pivot = (
        sub.groupby(["method", "n_features"]).agg(fr=("is_failure", "mean")).reset_index()
        .pivot(index="method", columns="n_features", values="fr")
    )
    z    = pivot.values * 100
    text = [[f"{v:.0f}%" if not np.isnan(v) else "—" for v in row] for row in z]
    fig  = go.Figure(go.Heatmap(
        z=z, x=[str(int(c)) for c in pivot.columns], y=list(pivot.index),
        text=text, texttemplate="%{text}",
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
        **_CHART_LAYOUT, height=max(260, len(pivot) * 36 + 80),
        xaxis=dict(title="Number of features", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="", gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=16, t=20, b=60),
    )
    return fig


def fig_speed_ranking_at_nfeatures(df: pd.DataFrame, n: int | None = None) -> go.Figure:
    """Horizontal bars: fastest → slowest at a specific n_features (default = max available)."""
    sub = df[df["n_features"].notna() & df["runtime_s"].notna()].copy()
    if sub.empty:
        return fig_empty()
    target = n if n is not None else int(sub["n_features"].max())
    sub    = sub[sub["n_features"] == target]
    if sub.empty:
        return fig_empty(f"No data for n_features = {target}")
    grp = (
        sub.groupby(["method", "library"])
        .agg(rt=("runtime_s", "median"), fr=("is_failure", "mean")).reset_index()
        .sort_values("rt", ascending=True)
    )
    colors = [_lib_color(lib) for lib in grp["library"]]
    fig = go.Figure(go.Bar(
        y=grp["method"], x=grp["rt"], orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(color="white", width=0.5)),
        text=grp["rt"].round(3).astype(str) + " s",
        textposition="outside", textfont=dict(size=11, color=TEXT),
        hovertemplate=(
            "<b>%{y}</b><br>Median runtime: %{x:.3f} s<br>"
            "Failure: %{customdata[0]:.0f}%<extra></extra>"
        ),
        customdata=(grp["fr"] * 100).values.reshape(-1, 1),
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        height=max(260, len(grp) * 32 + 60),
        title=dict(text=f"Speed ranking  —  n_features = {target}",
                   font=dict(size=13, color=TEXT2), x=0),
        xaxis=dict(title="Median runtime (s)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=20, t=44, b=48),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart builders — RQ3: Neural Networks (runtime-focused)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_runtime_ranking(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart sorted fastest → slowest. Core chart for 'which is fastest?'."""
    sub = df[df["runtime_s"].notna()].copy()
    if sub.empty:
        return fig_empty()
    grp = (
        sub.groupby(["method", "library"])
        .agg(rt=("runtime_s", "median"), fr=("is_failure", "mean")).reset_index()
        .sort_values("rt", ascending=True)
    )
    colors = [_lib_color(lib) for lib in grp["library"]]
    fig = go.Figure(go.Bar(
        y=grp["method"], x=grp["rt"], orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(color="white", width=0.5)),
        text=grp["rt"].round(3).astype(str) + " s",
        textposition="outside", textfont=dict(size=11, color=TEXT),
        hovertemplate=(
            "<b>%{y}</b><br>Median runtime: %{x:.3f} s<br>"
            "Failure: %{customdata[0]:.0f}%<extra></extra>"
        ),
        customdata=(grp["fr"] * 100).values.reshape(-1, 1),
    ))
    fig.update_layout(
        **_CHART_LAYOUT, height=max(260, len(grp) * 32 + 60),
        xaxis=dict(title="Median runtime (s)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=20, t=30, b=48),
        showlegend=False,
    )
    return fig


def fig_runtime_boxplots(df: pd.DataFrame) -> go.Figure:
    """Box plots of raw runtime per library — shows spread and outliers."""
    sub = df[df["runtime_s"].notna()].copy()
    if sub.empty:
        return fig_empty()
    order = (
        sub.groupby("library")["runtime_s"].median().sort_values().index.tolist()
    )
    fig = go.Figure()
    for lib in order:
        ldf = sub[sub["library"] == lib]
        fig.add_trace(go.Box(
            y=ldf["runtime_s"], name=lib, boxpoints="outliers",
            marker_color=_lib_color(lib), line_color=_lib_color(lib),
            hovertemplate=f"<b>{lib}</b><br>Runtime: %{{y:.3f}} s<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=420, margin=_MARGIN,
        yaxis=dict(title="Runtime (s) — log scale", gridcolor=BORDER,
                   zeroline=False, type="log"),
        xaxis=dict(title="Library", gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def fig_quality_vs_runtime_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter: per-run runtime vs quality. Shows the speed-accuracy tradeoff."""
    sub = df[df["runtime_s"].notna() & df["quality_score"].notna()].copy()
    if sub.empty:
        return fig_empty()
    fig = go.Figure()
    for lib, ldf in sub.groupby("library"):
        fig.add_trace(go.Scatter(
            x=ldf["runtime_s"], y=ldf["quality_score"],
            mode="markers", name=lib,
            marker=dict(color=_lib_color(lib), size=7, opacity=0.6,
                        line=dict(color="white", width=0.5)),
            hovertemplate=(
                f"<b>{lib}</b><br>Runtime: %{{x:.3f}} s<br>"
                "Quality: %{y:.2f}<br>%{customdata}<extra></extra>"
            ),
            customdata=ldf["method"].values,
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=420, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Runtime (s) — log scale", type="log",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Quality score (0–12)", gridcolor=BORDER, zeroline=False),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart builders — RQ4: Trees / model complexity
# ═══════════════════════════════════════════════════════════════════════════════

def _complexity_col(df: pd.DataFrame) -> str:
    """
    Pick the best column to use as 'complexity axis'.
    Prefers an explicit depth/complexity column; falls back to 'model'.
    """
    for col in ["tree_depth", "max_depth", "depth", "n_estimators", "complexity"]:
        if col in df.columns and df[col].notna().any():
            return col
    return "model"


def fig_runtime_vs_complexity(df: pd.DataFrame) -> go.Figure:
    """Runtime vs tree/model complexity. Steep = bad scalability."""
    col = _complexity_col(df)
    sub = df[df[col].notna() & df["runtime_s"].notna()].copy()
    if sub.empty:
        return fig_empty()
    try:
        sub[col] = pd.to_numeric(sub[col])
        x_type   = "linear"
    except (ValueError, TypeError):
        x_type = "category"
    grp = (
        sub.groupby(["method", "library", col])
        .agg(rt=("runtime_s", "median")).reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values(col)
        fig.add_trace(go.Scatter(
            x=mdf[col], y=mdf["rt"], mode="lines+markers", name=method,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib), line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>{col}: %{{x}}<br>Runtime: %{{y:.3f}} s<extra></extra>",
        ))
    label = col.replace("_", " ").title()
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title=label, gridcolor=BORDER, zeroline=False, type=x_type),
        yaxis=dict(title="Median runtime (s)", gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_failure_vs_complexity(df: pd.DataFrame) -> go.Figure:
    """Heatmap: failure rate per method × complexity. Red = breaking point."""
    col = _complexity_col(df)
    sub = df[df[col].notna()].copy()
    if sub.empty:
        return fig_empty()
    try:
        sub[col] = pd.to_numeric(sub[col])
    except (ValueError, TypeError):
        pass
    pivot = (
        sub.groupby(["method", col]).agg(fr=("is_failure", "mean")).reset_index()
        .pivot(index="method", columns=col, values="fr")
    )
    z    = pivot.values * 100
    text = [[f"{v:.0f}%" if not np.isnan(v) else "—" for v in row] for row in z]
    label = col.replace("_", " ").title()
    fig   = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in pivot.columns], y=list(pivot.index),
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#D1FAE5"], [0.5, "#FEF3C7"], [1, "#FEE2E2"]],
        zmin=0, zmax=100,
        colorbar=dict(title="Failure %", thickness=14, len=0.8),
        hovertemplate=(
            f"Method: <b>%{{y}}</b><br>{label}: <b>%{{x}}</b><br>"
            "Failure rate: %{z:.1f}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        **_CHART_LAYOUT, height=max(260, len(pivot) * 36 + 80),
        xaxis=dict(title=label, gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="", gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=16, t=20, b=60),
    )
    return fig


def fig_quality_vs_complexity(df: pd.DataFrame) -> go.Figure:
    """Quality score vs tree/model complexity — does accuracy degrade at depth?"""
    col = _complexity_col(df)
    sub = df[df[col].notna()].copy()
    if sub.empty or sub[col].nunique() < 2:
        return fig_empty("Not enough complexity variation to show a trend")
    try:
        sub[col] = pd.to_numeric(sub[col])
        x_type   = "linear"
    except (ValueError, TypeError):
        x_type = "category"
    grp = (
        sub.groupby(["method", "library", col])
        .agg(q=("quality_score", "median")).reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values(col)
        fig.add_trace(go.Scatter(
            x=mdf[col], y=mdf["q"], mode="lines+markers", name=method,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib), line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>{col}: %{{x}}<br>Quality: %{{y:.2f}}<extra></extra>",
        ))
    label = col.replace("_", " ").title()
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title=label, gridcolor=BORDER, zeroline=False, type=x_type),
        yaxis=dict(title="Median quality score (0–12)", gridcolor=BORDER, zeroline=False),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Layout helpers
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_card(value: str, label: str, color: str | None = None) -> html.Div:
    accent = color or ACCENT
    return html.Div(
        [
            html.Div(style={"height": "3px", "background": accent,
                            "borderRadius": "3px 3px 0 0", "margin": "-16px -20px 14px"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "700",
                                   "color": accent, "lineHeight": "1",
                                   "letterSpacing": "-0.02em"}),
            html.Div(label, style={"fontSize": "10px", "color": TEXT2, "marginTop": "6px",
                                   "fontWeight": "500", "textTransform": "uppercase",
                                   "letterSpacing": "0.06em", "lineHeight": "1.4"}),
        ],
        style={
            "flex": "1", "minWidth": "140px", "background": CARD,
            "border": f"1px solid {BORDER}", "borderRadius": "10px",
            "padding": "16px 20px", "boxShadow": "0 1px 4px rgba(26,32,64,0.07)",
            "overflow": "hidden",
        },
    )


def kpi_row(*cards) -> html.Div:
    return html.Div(list(cards), style={
        "display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "20px",
    })


def section(title: str, subtitle: str, children,
            section_id: str | None = None) -> html.Div:
    """
    section_id: stable DOM id used as a CONNECTOR target by the advisor panel.
    The advisor recommendation links here — fill in href fragments to deep-link.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.H3(title, style={"margin": "0", "fontSize": "15px",
                                         "fontWeight": "600", "letterSpacing": "-0.01em",
                                         "color": TEXT}),
                    html.Div(style={"height": "2px", "width": "28px", "marginTop": "6px",
                                    "background": f"linear-gradient(90deg, {ACCENT}, {PINK})",
                                    "borderRadius": "2px"}),
                ],
                style={"marginBottom": "8px"},
            ),
            html.P(subtitle, style={"margin": "0 0 12px", "fontSize": "12px",
                                    "color": TEXT2, "lineHeight": "1.6"}),
            html.Div(children, style={
                "background": CARD, "borderRadius": "10px",
                "border": f"1px solid {BORDER}", "overflow": "hidden",
            }),
        ],
        **({"id": section_id} if section_id is not None else {}),
        style={"marginBottom": "28px"},
    )


def rq_header(rq: str, title: str, question: str) -> html.Div:
    return html.Div([
        html.Div(
            [
                html.Span(rq, style={
                    "fontSize": "10px", "fontWeight": "700", "color": ACCENT,
                    "textTransform": "uppercase", "letterSpacing": "0.1em",
                    "background": "#E8EDFA", "padding": "3px 9px",
                    "borderRadius": "4px", "marginRight": "10px",
                }),
                html.Span(title, style={"fontWeight": "700", "fontSize": "20px",
                                        "letterSpacing": "-0.02em", "color": TEXT}),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "10px"},
        ),
        html.Div(
            html.Em(f'"{question}"'),
            style={"fontSize": "13px", "color": TEXT2,
                   "lineHeight": "1.7", "marginBottom": "24px",
                   "borderLeft": f"3px solid {PINK}",
                   "paddingLeft": "12px", "fontStyle": "italic"},
        ),
    ])


def interpretation_note(text: str) -> html.Div:
    return html.Div(
        html.Em(text, style={"fontSize": "12px", "color": TEXT2, "lineHeight": "1.7"}),
        style={"marginTop": "8px", "marginBottom": "24px"},
    )


def warning_note(message: str) -> html.Div:
    return html.Div(
        [html.Span("⚠ ", style={"fontWeight": "700"}), message],
        style={
            "background": "#FFFBEB", "border": f"1px solid {AMBER}",
            "borderRadius": "8px", "padding": "10px 16px",
            "fontSize": "13px", "color": "#92400E", "marginBottom": "16px",
        },
    )


def info_note(message: str) -> html.Div:
    return html.Div(
        [html.Span("ℹ ", style={"fontWeight": "700", "color": ACCENT}), message],
        style={
            "background": "#EFF6FF", "border": f"1px solid {ACCENT}",
            "borderRadius": "8px", "padding": "10px 16px",
            "fontSize": "13px", "color": "#1E3A8A",
            "marginBottom": "16px", "lineHeight": "1.7",
        },
    )


def missing_data_banner(expected_csv: str) -> html.Div:
    """Shown prominently when the RQ-specific CSV has not been created yet."""
    return html.Div(
        [
            html.Div("📂", style={"fontSize": "36px", "marginBottom": "12px"}),
            html.Div("Data not yet collected",
                     style={"fontSize": "16px", "fontWeight": "600",
                            "color": TEXT, "marginBottom": "8px"}),
            html.Div(
                ["Expected CSV: ",
                 html.Code(expected_csv,
                           style={"background": BG, "padding": "2px 6px",
                                  "borderRadius": "4px", "fontSize": "12px",
                                  "fontFamily": "monospace"})],
                style={"fontSize": "13px", "color": TEXT2,
                       "lineHeight": "1.7", "marginBottom": "12px"},
            ),
            html.Div(
                "Place the file in the same directory as benchmark_explorer.py. "
                "Charts will populate automatically on the next page load — no code changes needed.",
                style={"fontSize": "12px", "color": TEXT2, "lineHeight": "1.7"},
            ),
        ],
        style={
            "textAlign": "center", "padding": "56px 32px",
            "background": CARD, "borderRadius": "12px",
            "border": f"2px dashed {BORDER}", "marginBottom": "24px",
        },
    )


def _dd_label(label: str) -> html.Div:
    return html.Div(label, style={
        "fontSize": "10px", "fontWeight": "600", "color": TEXT2,
        "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "4px",
    })


def filter_dropdown(label: str, component_id: str, options: list,
                    value, width: str = "190px") -> html.Div:
    return html.Div(
        [_dd_label(label),
         dcc.Dropdown(id=component_id, options=options, value=value, clearable=False,
                      style={"width": width, "fontSize": "13px"})],
        style={"marginRight": "14px"},
    )


def filter_checklist(label: str, component_id: str,
                     options: list, value: list) -> html.Div:
    return html.Div(
        [_dd_label(label),
         dcc.Checklist(id=component_id, options=options, value=value, inline=True,
                       inputStyle={"marginRight": "4px"},
                       labelStyle={"marginRight": "12px", "fontSize": "13px",
                                   "cursor": "pointer"})],
    )


def filter_bar(*controls) -> html.Div:
    return html.Div(
        list(controls),
        style={
            "display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "8px",
            "background": CARD, "borderRadius": "12px",
            "border": f"1px solid {BORDER}", "padding": "14px 20px",
            "marginBottom": "20px",
        },
    )


def build_leaderboard_datatable(lb: pd.DataFrame,
                                 table_id: str = "lb-table") -> dash_table.DataTable:
    display = lb[[
        "rank", "method", "q_median", "runtime_median",
        "sign_median", "rho_median", "failure_rate", "n_runs",
    ]].copy()
    display["q_median"]       = display["q_median"].round(2)
    display["runtime_median"] = display["runtime_median"].round(3)
    display["sign_median"]    = display["sign_median"].round(3)
    display["rho_median"]     = display["rho_median"].round(3)
    display["failure_rate"]   = (display["failure_rate"] * 100).round(1)

    return dash_table.DataTable(
        id=table_id,
        data=display.to_dict("records"),
        columns=[
            {"name": "#",             "id": "rank"},
            {"name": "Method",        "id": "method"},
            {"name": "Quality (med)", "id": "q_median"},
            {"name": "Runtime (s)",   "id": "runtime_median"},
            {"name": "Sign agr.",     "id": "sign_median"},
            {"name": "Mean ρ",        "id": "rho_median"},
            {"name": "Failure %",     "id": "failure_rate"},
            {"name": "Runs",          "id": "n_runs"},
        ],
        sort_action="native",
        page_size=25,
        style_table={"overflowX": "auto"},
        style_header={
            "background": BG, "color": TEXT2, "fontWeight": "600",
            "fontSize": "11px", "textTransform": "uppercase",
            "letterSpacing": "0.05em", "border": "none",
            "borderBottom": f"1px solid {BORDER}",
            "padding": "10px 14px", "fontFamily": FONT,
        },
        style_cell={
            "fontFamily": FONT, "fontSize": "13px", "padding": "10px 14px",
            "border": "none", "borderBottom": f"1px solid {BORDER}",
            "color": TEXT, "background": CARD,
        },
        style_data_conditional=[
            {"if": {"row_index": 0}, "background": "#EEF2FF", "fontWeight": "600"},
            {"if": {"column_id": "failure_rate",
                    "filter_query": "{failure_rate} > 20"}, "color": RED},
            {"if": {"column_id": "q_median",
                    "filter_query": "{q_median} >= 8"}, "color": GREEN, "fontWeight": "600"},
        ],
    )


def capability_matrix_table(benchmarked_libs: set) -> html.Div:
    """Static capability overview table — update rows_data when new libraries are added."""
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

    def badge(lib):
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

    # (lib, feat_attr, interactions, model_agnostic, nn_support, nn_focus, tree_opt, notes)
    rows_data = [
        ("shapiq",      "yes", "yes",               "yes", "no",      "no",  "no",  "Main benchmark focus; interaction-aware"),
        ("shap",        "yes", "limited",            "yes", "partial", "no",  "yes", "Model-specific & agnostic variants; TreeSHAP"),
        ("lightshap",   "yes", "no",                 "yes", "no",      "no",  "no",  "Speed-oriented approximation"),
        ("dalex",       "yes", "no",                 "yes", "no",      "no",  "no",  "Model-agnostic, R-inspired"),
        ("captum",      "yes", "not main focus",     "partial", "yes", "yes", "no",  "PyTorch focus; many gradient methods"),
        ("alibi",       "yes", "not main focus",     "yes", "partial", "no",  "no",  "Planned / not yet benchmarked"),
        ("shapleyflow", "no",  "different def.",     "no",  "no",      "no",  "no",  "Requires graph structure"),
    ]

    cols = ["Library", "Status", "Feature attr.", "Interactions",
            "Model-agnostic", "NN support", "NN focus", "Tree opt.", "Notes"]
    thead = html.Thead(html.Tr([html.Th(c, style=th_s) for c in cols]))

    tbody_rows = []
    for row in rows_data:
        lib, feat, inter, agnostic, nn_sup, nn_focus, tree_opt, notes = row
        tbody_rows.append(html.Tr([
            html.Td(lib,            style=td_s({"fontFamily": "monospace",
                                                "color": ACCENT, "fontWeight": "600"})),
            html.Td(badge(lib),     style=td_s()),
            html.Td(yn(feat),       style=td_s()),
            html.Td(yn(inter),      style=td_s()),
            html.Td(yn(agnostic),   style=td_s()),
            html.Td(yn(nn_sup),     style=td_s()),
            html.Td(yn(nn_focus),   style=td_s()),
            html.Td(yn(tree_opt),   style=td_s()),
            html.Td(notes,          style=td_s({"color": TEXT2, "fontSize": "11px"})),
        ]))

    return html.Div(
        html.Table(
            [thead, html.Tbody(tbody_rows)],
            style={"width": "100%", "borderCollapse": "collapse"},
        ),
        style={
            "background": CARD, "borderRadius": "12px",
            "border": f"1px solid {BORDER}", "overflowX": "auto",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Library Advisor — recommendation engine
# ═══════════════════════════════════════════════════════════════════════════════

def recommend_library(
    model_type: str,
    primary_need: str,
    feature_count: str,
) -> dict:
    # model_type:    "tree" | "neural_network" | "black_box"
    # primary_need:  "speed" | "accuracy" | "memory"
    # feature_count: "low" | "high"
    #
    # Returns a recommendation dict with keys:
    #   library, approximator, why,
    #   rq_page (CONNECTOR: link here), rq_name,
    #   secondary_page / secondary_name,
    #   chart_ids  — CONNECTOR targets (stable section IDs in target page),
    #   metrics    — which columns to focus on

    # Neural network
    if model_type == "neural_network":
        return {
            "library": "captum",
            "approximator": "gradient / DeepLIFT",
            "why": (
                "For neural networks, gradient-based methods (Captum) are the natural fit. "
                "They run inside the PyTorch graph — no permutation sampling overhead. "
                "Check RQ3 to confirm which library achieves the fastest runtime on NN models."
            ),
            "rq_page":        "/rq3",
            "rq_name":        "RQ3 — Neural Networks",
            "secondary_page": "/rq2",
            "secondary_name": "RQ2 — Accuracy",
            # CONNECTORS — replace href="#<id>" fragments to deep-link
            "chart_ids": {
                "runtime_ranking":    "rq3-runtime-section",
                "pareto":             "rq3-pareto-section",
                "speed_vs_accuracy":  "rq3-scatter-section",
            },
            "metrics": ["runtime_s", "quality_score", "failure_rate"],
        }

    # Tree model — speed first
    if model_type == "tree" and primary_need == "speed":
        return {
            "library": "lightshap  or  shap (TreeSHAP)",
            "approximator": "tree-native / permutation",
            "why": (
                "TreeSHAP computes exact Shapley values for tree models in polynomial time. "
                "lightshap adds optimisations for very large datasets. "
                "Check RQ4 to find where each library's runtime spikes or failure rate crosses 10%."
            ),
            "rq_page":        "/rq4",
            "rq_name":        "RQ4 — Tree Models",
            "secondary_page": "/rq2",
            "secondary_name": "RQ2 — Accuracy",
            "chart_ids": {
                "failure_heatmap": "rq4-failure-section",
                "runtime_scaling": "rq4-runtime-section",
            },
            "metrics": ["runtime_s", "failure_rate", "quality_score"],
        }

    # Tree model — accuracy first
    if model_type == "tree":
        return {
            "library": "shap (TreeSHAP)",
            "approximator": "exact",
            "why": (
                "TreeSHAP produces exact Shapley values with zero approximation variance. "
                "It is the industry standard for high-fidelity tree explanations. "
                "Check RQ4 to confirm it doesn't break on your specific tree depth."
            ),
            "rq_page":        "/rq4",
            "rq_name":        "RQ4 — Tree Models",
            "secondary_page": "/rq2",
            "secondary_name": "RQ2 — Accuracy",
            "chart_ids": {
                "failure_heatmap": "rq4-failure-section",
                "quality_scaling": "rq4-quality-section",
            },
            "metrics": ["quality_score", "relative_mae", "sign_agreement"],
        }

    # Black-box, high-dimensional
    if feature_count == "high":
        return {
            "library": "lightshap",
            "approximator": "kernel",
            "why": (
                "High-dimensional black-box models punish standard KernelSHAP exponentially. "
                "lightshap's optimised kernel scales significantly better across feature counts. "
                "Check RQ1 for how each library's runtime curve steepens with n_features."
            ),
            "rq_page":        "/rq1",
            "rq_name":        "RQ1 — Dimensionality",
            "secondary_page": "/rq2",
            "secondary_name": "RQ2 — Accuracy",
            "chart_ids": {
                "runtime_vs_features":  "rq1-runtime-section",
                "speed_ranking_hi_dim": "rq1-speed-section",
                "failure_heatmap":      "rq1-failure-section",
            },
            "metrics": ["runtime_s at high n_features", "failure_rate", "quality_score"],
        }

    # Black-box, accuracy-first
    if primary_need == "accuracy":
        return {
            "library": "shapiq",
            "approximator": "kernel",
            "why": (
                "shapiq's kernel approximator consistently achieves the highest quality scores. "
                "Its interaction-aware sampling improves accuracy even at low budgets. "
                "Use RQ2 to find the minimum budget where quality plateaus."
            ),
            "rq_page":        "/rq2",
            "rq_name":        "RQ2 — Accuracy",
            "secondary_page": "/rq1",
            "secondary_name": "RQ1 — Dimensionality",
            "chart_ids": {
                "pareto":         "rq2-pareto-section",
                "budget_quality": "rq2-budget-section",
                "distribution":   "rq2-distribution-section",
            },
            "metrics": ["quality_score", "sign_agreement", "mean_sample_rho"],
        }

    # Black-box, speed / memory, low-dim (default)
    return {
        "library": "lightshap",
        "approximator": "permutation or kernel",
        "why": (
            "For fast model-agnostic approximations on low-dimensional data, "
            "lightshap tops the runtime ranking while staying in the reliable quality zone. "
            "Compare against shap/KernelSHAP in RQ2 to confirm the accuracy trade-off."
        ),
        "rq_page":        "/rq2",
        "rq_name":        "RQ2 — Accuracy",
        "secondary_page": "/rq1",
        "secondary_name": "RQ1 — Dimensionality",
        "chart_ids": {
            "pareto":         "rq2-pareto-section",
            "ranking":        "rq2-ranking-section",
            "budget_quality": "rq2-budget-section",
        },
        "metrics": ["runtime_s", "quality_score"],
    }


