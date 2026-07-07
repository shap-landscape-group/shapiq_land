"""
shared/charts.py — All Plotly figure builders, one per chart type.
Add new fig_* functions here; import them in __init__.py.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .tokens import (
    BG, CARD, BORDER, ACCENT, PINK, GREEN, RED, AMBER, MUTED,
    TEXT, TEXT2, LIB_COLOR, FONT,
    FAILURE_MAE, RHO_GOOD,
    _CHART_LAYOUT, _LEGEND_H, _MARGIN,
)
from .data import pareto_mark


def _lib_color(lib: str) -> str:
    return LIB_COLOR.get(lib, ACCENT)


# ── Shared / general ──────────────────────────────────────────────────────────

def fig_empty(message: str = "No data available for the current filter selection") -> go.Figure:
    return go.Figure(layout=dict(
        title=dict(text=message, font=dict(size=13, color=TEXT2), x=0.5, xanchor="center"),
        **_CHART_LAYOUT,
    ))


def fig_leaderboard_bars(lb: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: median Spearman ρ per method, failure labels on right."""
    if lb.empty:
        return fig_empty()

    lb        = lb.sort_values("rho_median", ascending=True).reset_index(drop=True)
    colors    = [_lib_color(lib) for lib in lb["library"]]
    fail_pcts = lb["failure_rate"] * 100

    annotations = []
    for _, row in lb.iterrows():
        fp = row["failure_rate"] * 100
        if fp > 0:
            color = RED if fp > 10 else TEXT2
            label = f"✕ {fp:.0f}%" if fp > 10 else f"{fp:.0f}%"
            annotations.append(dict(
                x=1.18, y=row["method"],
                text=f"<b>{label}</b>" if fp > 10 else label,
                showarrow=False, xanchor="left",
                font=dict(size=11, color=color, family=FONT),
                xref="x", yref="y",
            ))

    fig = go.Figure(go.Bar(
        y=lb["method"], x=lb["rho_median"], orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(color="white", width=0.5)),
        text=lb["rho_median"].round(3).astype(str),
        textposition="outside", textfont=dict(size=11, color=TEXT),
        hovertemplate=(
            "<b>%{y}</b><br>Median ρ: %{x:.3f}<br>"
            "Runtime: %{customdata[0]:.3f} s<br>"
            "Rel. MAE: %{customdata[1]:.4f}<br>"
            "Failure: %{customdata[2]:.0f}%<extra></extra>"
        ),
        customdata=lb[["runtime_median", "mae_median", "failure_rate"]]
            .assign(failure_rate=fail_pcts).values,
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        height=max(260, len(lb) * 32 + 60),
        annotations=annotations,
        xaxis=dict(title="Median Spearman ρ  (0–1, higher = better)",
                   range=[0, 1.25], gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=20, t=30, b=48),
        showlegend=False,
    )
    return fig


def fig_pareto(agg: pd.DataFrame) -> go.Figure:
    """Scatter: runtime (x, log) vs quality (y). Pareto front highlighted."""
    if agg.empty:
        return fig_empty()

    agg = agg.copy().reset_index(drop=True)
    agg["is_pareto"] = pareto_mark(agg, "runtime_median", "rho_median")
    fig = go.Figure()

    dom = agg[~agg["is_pareto"]]
    if not dom.empty:
        fig.add_trace(go.Scatter(
            x=dom["runtime_median"], y=dom["rho_median"], mode="markers",
            name="Dominated",
            marker=dict(color=MUTED, size=9, opacity=0.55, line=dict(color="white", width=1)),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Runtime: %{x:.3f} s<br>"
                "Spearman ρ: %{y:.3f}<br>Failure: %{customdata[1]:.0f}%<extra></extra>"
            ),
            customdata=dom[["method", "failure_rate"]]
                .assign(failure_rate=dom["failure_rate"] * 100).values,
        ))

    par = agg[agg["is_pareto"]].sort_values("runtime_median")
    if not par.empty:
        fig.add_trace(go.Scatter(
            x=par["runtime_median"], y=par["rho_median"], mode="lines",
            line=dict(color=GREEN, width=1.5, dash="dot"),
            name="Pareto frontier", hoverinfo="skip",
        ))
        colors = [_lib_color(lib) for lib in par["library"]]
        fig.add_trace(go.Scatter(
            x=par["runtime_median"], y=par["rho_median"], mode="markers+text",
            name="Pareto-optimal",
            marker=dict(color=colors, size=14, line=dict(color="white", width=2)),
            text=par["method"], textposition="top center",
            textfont=dict(size=9, color=TEXT2),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Runtime: %{x:.3f} s<br>"
                "Spearman ρ: %{y:.3f}<br>Sign agr.: %{customdata[1]:.3f}<br>"
                "Rel. MAE: %{customdata[3]:.4f}<br>Failure: %{customdata[2]:.0f}%<extra></extra>"
            ),
            customdata=par[["method", "sign_median", "failure_rate", "mae_median"]]
                .assign(failure_rate=par["failure_rate"] * 100).values,
        ))

    fig.update_layout(
        **_CHART_LAYOUT, height=440, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Median runtime (s) — log scale", type="log",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median Spearman ρ  (higher = better)",
                   range=[0, 1.08], gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_distribution(df: pd.DataFrame) -> go.Figure:
    """Box + strip plot of Spearman ρ per method, sorted by median."""
    if df.empty:
        return fig_empty()

    order = (
        df.groupby("method")["mean_sample_rho"]
        .median().sort_values(ascending=False).index.tolist()
    )
    fig = go.Figure()
    for method in order:
        sub   = df[df["method"] == method]
        color = _lib_color(sub["library"].iloc[0])
        fig.add_trace(go.Box(
            y=sub["mean_sample_rho"], name=method,
            boxpoints="all", jitter=0.35, pointpos=0,
            marker=dict(color=color, size=4, opacity=0.45, line=dict(color="white", width=0.5)),
            line=dict(color=color, width=2), fillcolor="rgba(0,0,0,0)",
            whiskerwidth=0.6,
            hovertemplate=f"<b>{method}</b><br>Spearman ρ: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        ))
    fig.add_hrect(
        y0=RHO_GOOD, y1=1.0, fillcolor=GREEN, opacity=0.05, line_width=0,
        annotation_text=f"ρ ≥ {RHO_GOOD}", annotation_position="top left",
        annotation_font=dict(size=10, color=GREEN),
    )
    fig.add_hline(
        y=RHO_GOOD, line=dict(color=GREEN, width=1, dash="dot"),
    )
    fig.update_layout(
        **_CHART_LAYOUT, height=440,
        yaxis=dict(title="Spearman ρ  (higher = better)", range=[-0.05, 1.08],
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
        .agg(q=("mean_sample_rho", "median")).reset_index()
        .pivot(index="library", columns="approximator", values="q")
    )
    if pivot.empty:
        return fig_empty()
    z    = pivot.values
    text = [[f"{v:.3f}" if not np.isnan(v) else "—" for v in row] for row in z]
    fig  = go.Figure(go.Heatmap(
        z=z, x=list(pivot.columns), y=list(pivot.index),
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#FEE2E2"], [0.5, "#93C5FD"], [1, "#1E3A8A"]],
        zmin=0, zmax=1,
        colorbar=dict(title="Spearman ρ", thickness=14, len=0.8),
        hovertemplate=(
            "Library: <b>%{y}</b><br>"
            "Approximator: <b>%{x}</b><br>"
            "Median ρ: %{z:.3f}<extra></extra>"
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


# ── RQ2: Approximation Accuracy ───────────────────────────────────────────────

def fig_budget_rho(df: pd.DataFrame) -> go.Figure:
    """Spearman ρ vs budget — failed runs excluded."""
    sub = df[~df["is_failure"] & df["budget"].notna()].copy()
    if sub.empty:
        return fig_empty("No non-failed runs for current filters")
    grp = (
        sub.groupby(["method", "library", "budget"])
        .agg(q=("mean_sample_rho", "median")).reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("budget")
        fig.add_trace(go.Scatter(
            x=mdf["budget"], y=mdf["q"], mode="lines+markers", name=method,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib), line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>Budget: %{{x}}<br>Spearman ρ: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(
        y=RHO_GOOD, line=dict(color=GREEN, width=1.2, dash="dot"),
        annotation_text=f"ρ = {RHO_GOOD}",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=GREEN),
    )
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Budget (approximation evaluations)",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median Spearman ρ  (higher = better)",
                   range=[0, 1.08], gridcolor=BORDER, zeroline=False),
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
        "relative_mae":    "Median relative MAE  (lower = better)",
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


# ── RQ1: Dimensionality ───────────────────────────────────────────────────────

def fig_runtime_vs_features(df: pd.DataFrame) -> go.Figure:
    """Runtime (log) vs n_features (log) — main scaling chart for RQ1."""
    sub = df[df["n_features"].notna() & df["runtime_s"].notna()]
    if sub.empty or sub["n_features"].nunique() < 2:
        return fig_empty("Not enough n_features variation")
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


def fig_rho_vs_features(df: pd.DataFrame) -> go.Figure:
    """Spearman ρ vs n_features — does rank correlation hold as dimensionality grows?"""
    sub = df[df["n_features"].notna()]
    if sub.empty or sub["n_features"].nunique() < 2:
        return fig_empty("Not enough n_features variation")
    grp = (
        sub.groupby(["method", "library", "n_features"])
        .agg(q=("mean_sample_rho", "median")).reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("n_features")
        fig.add_trace(go.Scatter(
            x=mdf["n_features"], y=mdf["q"], mode="lines+markers", name=method,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib), line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>n_features: %{{x}}<br>Spearman ρ: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(
        y=RHO_GOOD, line=dict(color=GREEN, width=1.2, dash="dot"),
        annotation_text=f"ρ = {RHO_GOOD}",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=GREEN),
    )
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Number of features", gridcolor=BORDER, zeroline=False, type="log"),
        yaxis=dict(title="Median Spearman ρ  (higher = better)",
                   range=[0, 1.08], gridcolor=BORDER, zeroline=False),
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
    """Horizontal bars: fastest → slowest at a specific n_features."""
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
        xaxis=dict(title="Median runtime (s) — log scale", type="log",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=20, t=44, b=48),
        showlegend=False,
    )
    return fig


# ── RQ3: Neural Networks ──────────────────────────────────────────────────────

def fig_runtime_ranking(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart sorted fastest → slowest."""
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
    order = sub.groupby("library")["runtime_s"].median().sort_values().index.tolist()
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


def fig_rho_vs_runtime(df: pd.DataFrame) -> go.Figure:
    """Scatter: per-run runtime vs Spearman ρ. Speed-accuracy tradeoff."""
    sub = df[df["runtime_s"].notna() & df["mean_sample_rho"].notna()].copy()
    if sub.empty:
        return fig_empty()
    fig = go.Figure()
    for lib, ldf in sub.groupby("library"):
        fig.add_trace(go.Scatter(
            x=ldf["runtime_s"], y=ldf["mean_sample_rho"],
            mode="markers", name=lib,
            marker=dict(color=_lib_color(lib), size=7, opacity=0.6,
                        line=dict(color="white", width=0.5)),
            hovertemplate=(
                f"<b>{lib}</b><br>Runtime: %{{x:.3f}} s<br>"
                "Spearman ρ: %{y:.3f}<br>%{customdata}<extra></extra>"
            ),
            customdata=ldf["method"].values,
        ))
    fig.add_hline(
        y=RHO_GOOD, line=dict(color=GREEN, width=1.2, dash="dot"),
        annotation_text=f"ρ = {RHO_GOOD}",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=GREEN),
    )
    fig.update_layout(
        **_CHART_LAYOUT, height=420, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Runtime (s) — log scale", type="log",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Spearman ρ  (higher = better)",
                   range=[0, 1.08], gridcolor=BORDER, zeroline=False),
    )
    return fig


# ── RQ4: Tree / model complexity ─────────────────────────────────────────────

def _complexity_col(df: pd.DataFrame) -> str:
    """Pick the best column to use as the complexity axis."""
    for col in ["tree_depth", "max_depth", "depth", "n_estimators", "complexity"]:
        if col in df.columns and df[col].notna().any():
            return col
    return "model"


def fig_runtime_vs_complexity(df: pd.DataFrame) -> go.Figure:
    """Runtime vs tree/model complexity — steep slope = bad scalability."""
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
    z     = pivot.values * 100
    text  = [[f"{v:.0f}%" if not np.isnan(v) else "—" for v in row] for row in z]
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


def fig_rho_vs_complexity(df: pd.DataFrame) -> go.Figure:
    """Spearman ρ vs complexity — does rank correlation degrade with model complexity?"""
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
        .agg(q=("mean_sample_rho", "median")).reset_index()
    )
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values(col)
        fig.add_trace(go.Scatter(
            x=mdf[col], y=mdf["q"], mode="lines+markers", name=method,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib), line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>{col}: %{{x}}<br>Spearman ρ: %{{y:.3f}}<extra></extra>",
        ))
    label = col.replace("_", " ").title()
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title=label, gridcolor=BORDER, zeroline=False, type=x_type),
        yaxis=dict(title="Median Spearman ρ  (higher = better)",
                   range=[0, 1.08], gridcolor=BORDER, zeroline=False),
    )
    return fig
