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
        title=dict(text=message, font=dict(
            size=13, color=TEXT2), x=0.5, xanchor="center"),
        **_CHART_LAYOUT,
    ))


def fig_leaderboard_bars(lb: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: median Spearman ρ per method, failure labels on right."""
    if lb.empty:
        return fig_empty()

    lb = lb.sort_values("rho_median", ascending=True).reset_index(drop=True)
    colors = [_lib_color(lib) for lib in lb["library"]]
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
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color="white", width=0.5)),
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
            marker=dict(color=MUTED, size=9, opacity=0.55,
                        line=dict(color="white", width=1)),
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
            marker=dict(color=colors, size=14,
                        line=dict(color="white", width=2)),
            text=par["method"], textposition="top center",
            textfont=dict(size=9, color=TEXT2),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Runtime: %{x:.3f} s<br>"
                "Spearman ρ: %{y:.3f}<br>Sign agr.: %{customdata[1]:.3f}<br>"
                "Rel. MAE: %{customdata[3]:.4f}<br>Failure: %{customdata[2]:.0f}%<extra></extra>"
            ),
            customdata=par[["method", "sign_median",
                            "failure_rate", "mae_median"]]
            .assign(failure_rate=par["failure_rate"] * 100).values,
        ))

    fig.update_layout(
        **_CHART_LAYOUT, height=440, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Median runtime (s) — log scale", type="log",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median Spearman ρ  (higher = better)",
                   range=(
                       [max(0.0, float(agg["rho_median"].min()) - 0.04),
                        min(1.05, float(agg["rho_median"].max()) + 0.03)]
                       if not agg.empty else [0, 1.08]
                   ),
                   gridcolor=BORDER, zeroline=False),
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
        sub = df[df["method"] == method]
        color = _lib_color(sub["library"].iloc[0])
        fig.add_trace(go.Box(
            y=sub["mean_sample_rho"], name=method,
            boxpoints="all", jitter=0.35, pointpos=0,
            marker=dict(color=color, size=4, opacity=0.45,
                        line=dict(color="white", width=0.5)),
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
    # Tight y-range: zoom into the actual data instead of always showing 0–1
    _rho_arr = df["mean_sample_rho"].dropna().values
    _y_range = (
        [max(0.0, float(_rho_arr.min()) - 0.04), min(1.05, float(_rho_arr.max()) + 0.03)]
        if len(_rho_arr) > 0 else [-0.05, 1.08]
    )
    fig.update_layout(
        **_CHART_LAYOUT, height=440,
        yaxis=dict(title="Spearman ρ  (higher = better)", range=_y_range,
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
    z = pivot.values
    text = [[f"{v:.3f}" if not np.isnan(v) else "—" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
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


def fig_tree_case_agreement_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: pairwise agreement between libraries for the selected tree case."""
    if df.empty:
        return fig_empty()

    work = df.copy()
    work = work[work["is_failure"].fillna(False) != True]
    metric = "sign_agreement"
    if metric not in work.columns:
        metric = "mean_sample_rho" if "mean_sample_rho" in work.columns else None
    if metric is None:
        return fig_empty()

    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    if work.empty:
        return fig_empty()

    config_cols = [
        col for col in [
            "dataset", "model", "order", "n_background", "n_features",
            "n_samples", "learning_rate", "max_depth", "n_estimators",
            "budget", "seed", "computation_type", "imputer", "n_eval",
            "approximator"
        ] if col in work.columns
    ]
    if not config_cols:
        config_cols = ["dataset", "model"]

    work = work.copy()
    work["_config_key"] = work[config_cols].apply(
        lambda row: " | ".join(str(value) for value in row), axis=1
    )

    libs = sorted(work["library"].dropna().astype(str).unique())
    if len(libs) < 2:
        return fig_empty()

    pivot = (
        work.groupby(["_config_key", "library"])[metric]
        .mean()
        .reset_index()
        .pivot(index="_config_key", columns="library", values=metric)
    )
    if pivot.empty:
        return fig_empty()

    pivot = pivot.reindex(columns=libs)
    values = []
    for lib_a in libs:
        row = []
        for lib_b in libs:
            shared = pivot[[lib_a, lib_b]].dropna()
            if shared.empty or shared.shape[0] < 2:
                row.append(np.nan)
            else:
                row.append(
                    float(np.nanmean(np.abs(shared[lib_a].to_numpy() - shared[lib_b].to_numpy()))))
        values.append(row)

    z = np.array(values, dtype=float)
    text = [[f"{v:.3f}" if not np.isnan(v) else "—" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=libs, y=libs,
        text=text, texttemplate="%{text}",
        colorscale="Blues",
        zmin=0, zmax=1,
        colorbar=dict(title="Mean |Δ|", thickness=14, len=0.8),
        hovertemplate=(
            "Library A: <b>%{y}</b><br>"
            "Library B: <b>%{x}</b><br>"
            "Mean agreement gap: %{z:.3f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **_CHART_LAYOUT, height=320,
        xaxis=dict(title="Library", gridcolor="rgba(0,0,0,0)"),
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
        lib = grp["library"].iloc[0]
        approx = grp["approximator"].iloc[0] if grp["approximator"].notna(
        ).any() else "?"
        color = _lib_color(lib)
        symbol = shape_map.get(approx, "circle")
        sizes = grp["n_features"].fillna(4).clip(lower=4)
        sizes = 7 + (sizes / sizes.max()) * 14
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
            customdata=grp[["method", "budget",
                            "n_features", "dataset"]].values,
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
        xaxis=dict(title="Runtime (seconds)", type="log",
                   gridcolor=BORDER, zeroline=False),
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
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
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
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>Budget: %{{x}}<br>Runtime: %{{y:.3f}} s<extra></extra>",
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Budget (model evaluations)",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median runtime (s)",
                   gridcolor=BORDER, zeroline=False),
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
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>Budget: %{{x}}<br>{metric}: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=380, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Budget (model evaluations)",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title=label_map.get(metric, metric),
                   gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_quality_vs_cost_rq2(
    df: pd.DataFrame,
    x_metric: str = "runtime_s",
    y_metric: str = "mean_sample_rho",
) -> go.Figure:
    """Scatter: quality vs cost for RQ2.

    One point per library × approximator × budget × n_background, aggregated
    (median) over seeds, models and datasets.
    Circle = n_background 50, diamond-open = n_background 200.
    Red border = problematic combo (n_background=200, budget ≤ 64).
    """
    for col in (x_metric, y_metric):
        if col not in df.columns:
            return fig_empty(f"Column '{col}' not in data")
    if df.empty or df[x_metric].isna().all():
        return fig_empty()

    has_nbg    = "n_background" in df.columns and df["n_background"].notna().any()
    has_budget = "budget"       in df.columns and df["budget"].notna().any()

    grp_cols = ["method", "library", "approximator"]
    if has_nbg:    grp_cols.append("n_background")
    if has_budget: grp_cols.append("budget")

    sub = df[df[x_metric].notna() & df[y_metric].notna()].copy()
    if sub.empty:
        return fig_empty()

    grp = (
        sub.groupby(grp_cols, dropna=False)
        .agg(x=(x_metric, "median"), y=(y_metric, "median"), fr=("is_failure", "mean"))
        .reset_index()
    )

    x_label = {"runtime_s": "Median runtime (s)",
               "n_model_evals": "Median model evaluations"}.get(x_metric, x_metric)
    y_label = {
        "mean_sample_rho": "Median Spearman ρ  (higher = better)",
        "relative_mae":    "Median relative MAE  (lower = better)",
        "sign_agreement":  "Median sign agreement  (higher = better)",
    }.get(y_metric, y_metric)

    fig = go.Figure()
    nbg_groups = grp.groupby("n_background") if has_nbg else [(None, grp)]
    for nbg, nbg_grp in nbg_groups:
        nbg_i  = int(nbg) if nbg is not None and not pd.isna(nbg) else None
        symbol = "diamond-open" if nbg_i == 200 else "circle"

        for lib, lib_grp in nbg_grp.groupby("library"):
            if lib_grp.empty:
                continue
            color = _lib_color(lib)

            bgt_series = lib_grp["budget"] if has_budget else pd.Series(
                [float("nan")] * len(lib_grp), index=lib_grp.index)
            is_warn = (nbg_i == 200) and has_budget and (bgt_series <= 64).any()
            border_c = [
                "rgba(220,38,38,0.9)"
                if (nbg_i == 200 and has_budget and not pd.isna(b) and b <= 64)
                else "white"
                for b in bgt_series
            ]
            border_w = [2.5 if c != "white" else 1.5 for c in border_c]

            suffix = f"  (n_bg={nbg_i})" if nbg_i is not None else ""
            cd = list(zip(
                lib_grp["method"].values,
                [f"{int(b)}" if has_budget and not pd.isna(b) else "—"
                 for b in bgt_series],
                (lib_grp["fr"] * 100).values,
            ))
            fig.add_trace(go.Scatter(
                x=lib_grp["x"], y=lib_grp["y"], mode="markers",
                name=f"{lib}{suffix}",
                marker=dict(color=color, symbol=symbol, size=13, opacity=0.85,
                            line=dict(color=border_c, width=border_w)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    + (f"n_background: {nbg_i}<br>" if nbg_i is not None else "")
                    + "Budget: %{customdata[1]}<br>"
                    + f"{x_label}: %{{x:.3g}}<br>"
                    + f"{y_label}: %{{y:.4g}}<br>"
                    + "Failure: %{customdata[2]:.0f}%<extra></extra>"
                ),
                customdata=cd,
            ))

    if y_metric == "mean_sample_rho":
        fig.add_hline(y=RHO_GOOD, line=dict(color=GREEN, width=1.2, dash="dot"),
                      annotation_text=f"ρ = {RHO_GOOD}",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=GREEN))

    # Tight y-range: zoom into the actual data instead of always showing 0–1
    _y_arr    = grp["y"].dropna().values
    _bounded  = y_metric in ("mean_sample_rho", "sign_agreement")
    _y_range  = (
        [max(0.0, float(_y_arr.min()) - 0.04), min(1.05, float(_y_arr.max()) + 0.03)]
        if _bounded and len(_y_arr) > 0 else
        ([0, 1.08] if _bounded else None)
    )
    fig.update_layout(
        **_CHART_LAYOUT, height=460, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title=x_label,
                   type="log" if x_metric == "n_model_evals" else "linear",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(
            title=y_label,
            type="log" if y_metric == "relative_mae" else "linear",
            **({"range": _y_range} if _y_range is not None else {}),
            gridcolor=BORDER, zeroline=False,
        ),
    )
    return fig


def fig_budget_quality_lines(df: pd.DataFrame, metric: str = "mean_sample_rho") -> go.Figure:
    """Quality vs budget per method.

    Solid line / circle = n_background 50,
    dashed line / diamond = n_background 200 (when column is present).
    Failed runs excluded.
    """
    has_nbg = "n_background" in df.columns and df["n_background"].notna().any()
    sub = df[df["budget"].notna() & df[metric].notna() & ~df["is_failure"]].copy()
    if sub.empty or sub["budget"].nunique() < 2:
        return fig_empty("Select at least two budget values to see convergence")

    grp_cols = ["method", "library", "budget"] + (["n_background"] if has_nbg else [])
    grp = sub.groupby(grp_cols).agg(val=(metric, "median")).reset_index()

    label_map = {
        "mean_sample_rho": "Median Spearman ρ  (higher = better)",
        "relative_mae":    "Median relative MAE  (lower = better)",
        "sign_agreement":  "Median sign agreement  (higher = better)",
    }
    fig = go.Figure()
    combo_cols = ["method"] + (["n_background"] if has_nbg else [])
    for keys, mdf in grp.groupby(combo_cols):
        method = keys[0] if has_nbg else keys
        nbg    = keys[1] if has_nbg else None
        nbg_i  = int(nbg) if nbg is not None and not pd.isna(nbg) else None
        lib    = mdf["library"].iloc[0]
        color  = _lib_color(lib)
        mdf    = mdf.sort_values("budget")
        suffix = f"  n_bg={nbg_i}" if nbg_i is not None else ""
        fig.add_trace(go.Scatter(
            x=mdf["budget"], y=mdf["val"],
            mode="lines+markers",
            name=f"{method}{suffix}",
            line=dict(color=color, width=2, dash="dash" if nbg_i == 200 else "solid"),
            marker=dict(size=8, color=color,
                        symbol="diamond" if nbg_i == 200 else "circle",
                        line=dict(color="white", width=1.5)),
            hovertemplate=(
                f"<b>{method}{suffix}</b><br>"
                f"Budget: %{{x}}<br>{metric}: %{{y:.4g}}<extra></extra>"
            ),
        ))

    if metric == "mean_sample_rho":
        fig.add_hline(y=RHO_GOOD, line=dict(color=GREEN, width=1.2, dash="dot"),
                      annotation_text=f"ρ = {RHO_GOOD}",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=GREEN))

    # Tight y-range: zoom into the actual data instead of always showing 0–1
    _y_arr   = grp["val"].dropna().values
    _bounded = metric in ("mean_sample_rho", "sign_agreement")
    _y_range = (
        [max(0.0, float(_y_arr.min()) - 0.04), min(1.05, float(_y_arr.max()) + 0.03)]
        if _bounded and len(_y_arr) > 0 else
        ([0, 1.08] if _bounded else None)
    )
    fig.update_layout(
        **_CHART_LAYOUT, height=420, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Budget (model evaluations)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title=label_map.get(metric, metric),
                   **({"range": _y_range} if _y_range is not None else {}),
                   gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_pairwise_heatmap_rq2(df: pd.DataFrame, metric: str = "mean_sample_rho") -> go.Figure:
    """Cross-library agreement heatmap from pairwise_metrics JSON.

    Row = source library (whose approximation is in that row).
    Column = compared-against library / exact reference.
    Cell = mean metric across all matching runs (seeds, models, datasets, budgets).
    """
    import json as _json

    if "pairwise_metrics" not in df.columns or df.empty:
        return fig_empty("pairwise_metrics column not available")

    _key_cache: dict = {}

    def _key_to_target(k: str) -> str:
        if k not in _key_cache:
            if "true_value" in k:
                _key_cache[k] = k.replace("_true_value", "") + " (exact)"
            elif k.endswith("_approx"):
                _key_cache[k] = k[:-7]
            else:
                _key_cache[k] = k
        return _key_cache[k]

    records = []
    for _, row in df[df["pairwise_metrics"].notna()].iterrows():
        src    = row.get("library", "?")
        pm_str = row["pairwise_metrics"]
        if not isinstance(pm_str, str):
            continue
        try:
            pm = _json.loads(pm_str.replace(": NaN", ": null").replace(":NaN", ":null"))
        except Exception:
            continue
        for key, vals in pm.items():
            if not isinstance(vals, dict):
                continue
            v = vals.get(metric)
            if v is None:
                continue
            records.append({"source": src, "target": _key_to_target(key), "value": float(v)})

    if not records:
        return fig_empty("No pairwise metrics extracted for the current filters")

    pdf   = pd.DataFrame(records)
    pivot = (
        pdf.groupby(["source", "target"])["value"]
        .mean().reset_index()
        .pivot(index="source", columns="target", values="value")
    )
    # Exact-reference columns first, then alphabetical
    cols  = sorted(pivot.columns, key=lambda c: (0 if "exact" in c else 1, c))
    pivot = pivot[cols]

    z    = pivot.values
    text = [[f"{v:.3f}" if not np.isnan(v) else "—" for v in row] for row in z]

    lower_better = metric == "relative_mae"
    colorscale   = (
        [[0, "#D1FAE5"], [0.5, "#93C5FD"], [1, "#FEE2E2"]] if lower_better else
        [[0, "#FEE2E2"], [0.5, "#93C5FD"], [1, "#1E3A8A"]]
    )
    zmax = float(np.nanmax(z)) if lower_better and z.size > 0 and not np.all(np.isnan(z)) else 1.0

    fig = go.Figure(go.Heatmap(
        z=z, x=list(pivot.columns), y=list(pivot.index),
        text=text, texttemplate="%{text}",
        colorscale=colorscale, zmin=0, zmax=zmax,
        colorbar=dict(title=metric.replace("_", " "), thickness=14, len=0.8),
        hovertemplate=(
            "Source lib: <b>%{y}</b><br>"
            "Reference: <b>%{x}</b><br>"
            f"{metric}: %{{z:.3f}}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        height=max(260, len(pivot) * 60 + 110),
        xaxis=dict(title="Reference / compared against",
                   gridcolor="rgba(0,0,0,0)", side="bottom"),
        yaxis=dict(title="Source library",
                   gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=90, r=16, t=20, b=100),
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
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>n_features: %{{x}}<br>Runtime: %{{y:.3f}} s<extra></extra>",
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Number of features", gridcolor=BORDER,
                   zeroline=False, type="log"),
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
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
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
        xaxis=dict(title="Number of features", gridcolor=BORDER,
                   zeroline=False, type="log"),
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
        sub.groupby(["method", "n_features"]).agg(
            fr=("is_failure", "mean")).reset_index()
        .pivot(index="method", columns="n_features", values="fr")
    )
    z = pivot.values * 100
    text = [[f"{v:.0f}%" if not np.isnan(
        v) else "—" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
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
    sub = sub[sub["n_features"] == target]
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
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color="white", width=0.5)),
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


def fig_cost_vs_features(df: pd.DataFrame, metric: str = "runtime_s") -> go.Figure:
    """Cost (runtime_s OR n_model_evals) vs n_features — log-log scaling chart.

    Seeds are aggregated automatically (median per method × n_features).
    """
    if metric not in df.columns:
        return fig_empty(f"Column '{metric}' not found in data")
    sub = df[df["n_features"].notna() & df[metric].notna()].copy()
    if sub.empty or sub["n_features"].nunique() < 2:
        return fig_empty("Not enough n_features variation for the current filters")
    grp = (
        sub.groupby(["method", "library", "n_features"])
        .agg(val=(metric, "median")).reset_index()
    )
    label_map = {
        "runtime_s":     "Median runtime (s)",
        "n_model_evals": "Median model evaluations",
    }
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("n_features")
        fig.add_trace(go.Scatter(
            x=mdf["n_features"], y=mdf["val"], mode="lines+markers", name=method,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib), line=dict(color="white", width=1.5)),
            hovertemplate=(
                f"<b>{method}</b><br>n_features: %{{x:,}}<br>"
                f"{label_map.get(metric, metric)}: %{{y:.3g}}<extra></extra>"
            ),
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=420, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Number of features (log scale)", type="log",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title=f"{label_map.get(metric, metric)} — log scale",
                   type="log", gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_quality_vs_features(df: pd.DataFrame, metric: str = "mean_sample_rho") -> go.Figure:
    """Quality (mean_sample_rho, relative_mae, or relative_additivity_gap) vs n_features.

    Seeds are aggregated automatically (median per method × n_features).
    Log-scale metrics with values near machine precision (<1e-6) are excluded.
    """
    if metric not in df.columns:
        return fig_empty(f"Column '{metric}' not found in data")
    sub = df[df["n_features"].notna() & df[metric].notna()].copy()
    if metric in ("relative_mae", "relative_additivity_gap"):
        sub = sub[sub[metric] > 1e-6]   # exclude machine-precision noise (e.g. linear models)
    if sub.empty or sub["n_features"].nunique() < 2:
        return fig_empty("Not enough n_features variation for the current filters")
    grp = (
        sub.groupby(["method", "library", "n_features"])
        .agg(val=(metric, "median")).reset_index()
    )
    is_rho  = metric == "mean_sample_rho"
    y_label = {
        "mean_sample_rho":        "Median Spearman ρ  (higher = better)",
        "relative_mae":           "Median relative MAE  (lower = better, log scale)",
        "relative_additivity_gap": "Median relative additivity gap  (lower = better, log scale)",
    }.get(metric, f"Median {metric}")
    use_log = not is_rho
    fig = go.Figure()
    for method, mdf in grp.groupby("method"):
        lib = mdf["library"].iloc[0]
        mdf = mdf.sort_values("n_features")
        fig.add_trace(go.Scatter(
            x=mdf["n_features"], y=mdf["val"], mode="lines+markers", name=method,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib), line=dict(color="white", width=1.5)),
            hovertemplate=(
                f"<b>{method}</b><br>n_features: %{{x:,}}<br>"
                f"{metric}: %{{y:.4g}}<extra></extra>"
            ),
        ))
    if is_rho:
        fig.add_hline(
            y=RHO_GOOD, line=dict(color=GREEN, width=1.2, dash="dot"),
            annotation_text=f"ρ = {RHO_GOOD}",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color=GREEN),
        )
    fig.update_layout(
        **_CHART_LAYOUT, height=420, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Number of features (log scale)", type="log",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(
            title=y_label,
            type="linear" if is_rho else "log",
            **({"range": [0, 1.08]} if is_rho else {}),
            gridcolor=BORDER, zeroline=False,
        ),
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
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color="white", width=0.5)),
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
        xaxis=dict(title="Median runtime (s)",
                   gridcolor=BORDER, zeroline=False),
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
    order = sub.groupby("library")[
        "runtime_s"].median().sort_values().index.tolist()
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
        x_type = "linear"
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
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{method}</b><br>{col}: %{{x}}<br>Runtime: %{{y:.3f}} s<extra></extra>",
        ))
    label = col.replace("_", " ").title()
    fig.update_layout(
        **_CHART_LAYOUT, height=400, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title=label, gridcolor=BORDER, zeroline=False, type=x_type),
        yaxis=dict(title="Median runtime (s)",
                   gridcolor=BORDER, zeroline=False),
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
        sub.groupby(["method", col]).agg(
            fr=("is_failure", "mean")).reset_index()
        .pivot(index="method", columns=col, values="fr")
    )
    z = pivot.values * 100
    text = [[f"{v:.0f}%" if not np.isnan(
        v) else "—" for v in row] for row in z]
    label = col.replace("_", " ").title()
    fig = go.Figure(go.Heatmap(
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
        x_type = "linear"
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
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
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


# ── RQ4 (backend-level): richer views for tree-explanation benchmarks ─────────
#
# The tree CSVs have an empty `approximator`, so the generic `method` column
# collapses to "library / ?" and merges genuinely different algorithms
# (e.g. shap_tree_path_dependent @0.02s with shap_interaction @0.3s).
# These helpers instead key off the real `backend` column and expose the axes
# that actually vary in the data: the algorithm variant, the number of features
# (the true complexity axis), and the interaction order.

_RUNTIME_FLOOR = 1e-3   # display floor so log-scaled bars/lines survive 0.0 values


def _tree_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column if present, else a '?'/NaN-filled fallback series."""
    if name in df.columns:
        return df[name]
    fill = np.nan if name in ("runtime_s", "n_features", "order") else "?"
    return pd.Series(fill, index=df.index)


def _tree_backend_label(backend: str) -> str:
    return backend.replace("_", " ") if isinstance(backend, str) else "?"


def fig_tree_runtime_by_backend(df: pd.DataFrame) -> go.Figure:
    """Horizontal log-scale bars: median runtime per backend (the algorithm variant).

    This is the headline chart — it exposes the enormous runtime spread that the
    old library-grouped line chart hid (e.g. shapiq interactions in the tens of
    seconds vs. tree-path methods in the milliseconds). Backends that return no
    valid values (fasttreeshap here) are flagged rather than shown as "fastest".
    """
    sub = df.copy()
    sub["backend"] = _tree_col(sub, "backend")
    sub["runtime_s"] = pd.to_numeric(
        _tree_col(sub, "runtime_s"), errors="coerce")
    sub = sub[sub["runtime_s"].notna()]
    if sub.empty:
        return fig_empty()

    grp = (
        sub.groupby(["backend", "library"])
        .agg(rt=("runtime_s", "median"),
             fr=("is_failure", "mean"),
             n=("runtime_s", "count"))
        .reset_index()
        .sort_values("rt", ascending=True)
    )
    grp["failed"] = grp["fr"] >= 0.5
    grp["rt_disp"] = grp["rt"].clip(lower=_RUNTIME_FLOOR)
    grp["label"] = grp["backend"].map(_tree_backend_label)

    colors = [_lib_color(l) for l in grp["library"]]
    patterns = ["/" if f else "" for f in grp["failed"]]
    opac = [0.35 if f else 0.9 for f in grp["failed"]]
    bartext = [
        "✕ no valid output" if f else f"{rt:.3g} s"
        for f, rt in zip(grp["failed"], grp["rt"])
    ]

    fig = go.Figure(go.Bar(
        y=grp["label"], x=grp["rt_disp"], orientation="h",
        marker=dict(color=colors, opacity=opac,
                    line=dict(color="white", width=0.5),
                    pattern=dict(shape=patterns, fgcolor=RED, size=4)),
        text=bartext, textposition="outside", textfont=dict(size=11, color=TEXT),
        customdata=np.stack(
            [grp["library"], (grp["fr"] * 100), grp["n"]], axis=1),
        hovertemplate=(
            "<b>%{y}</b><br>Library: %{customdata[0]}<br>"
            "Median runtime: %{x:.4g} s<br>"
            "Failure: %{customdata[1]:.0f}%<br>"
            "Runs: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        height=max(280, len(grp) * 34 + 70),
        xaxis=dict(title="Median runtime (s) — log scale", type="log",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=110, t=20, b=48),
        showlegend=False,
    )
    return fig


def fig_tree_runtime_vs_features(df: pd.DataFrame) -> go.Figure:
    """Runtime vs number of explained features (log-log) — the true scaling axis.

    n_features is the dimensionality of the explanation problem and the axis that
    actually drives cost for tree explainers, so a steep slope here is the real
    "computational bottleneck as complexity grows" signal.
    """
    sub = df.copy()
    sub["backend"] = _tree_col(sub, "backend")
    sub["n_features"] = pd.to_numeric(
        _tree_col(sub, "n_features"), errors="coerce")
    sub["runtime_s"] = pd.to_numeric(
        _tree_col(sub, "runtime_s"),  errors="coerce")
    sub = sub[sub["n_features"].notna() & sub["runtime_s"].notna()
              & ~sub["is_failure"]]
    if sub.empty or sub["n_features"].nunique() < 2:
        return fig_empty("Not enough n_features variation to show scaling")

    grp = (
        sub.groupby(["backend", "library", "n_features"])
        .agg(rt=("runtime_s", "median")).reset_index()
    )
    fig = go.Figure()
    for backend, bdf in grp.groupby("backend"):
        lib = bdf["library"].iloc[0]
        bdf = bdf.sort_values("n_features")
        fig.add_trace(go.Scatter(
            x=bdf["n_features"], y=bdf["rt"].clip(lower=_RUNTIME_FLOOR),
            mode="lines+markers", name=_tree_backend_label(backend),
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
            hovertemplate=(
                f"<b>{_tree_backend_label(backend)}</b><br>"
                "n_features: %{x}<br>Runtime: %{y:.4g} s<extra></extra>"
            ),
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=430, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Number of features (log scale)", type="log",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median runtime (s) — log scale", type="log",
                   gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_tree_order_cost(df: pd.DataFrame) -> go.Figure:
    """Grouped bars: median runtime at interaction order 1 vs order 2, per library.

    Moving from main effects (order 1) to pairwise interactions (order 2) is the
    single biggest cost jump in the data — this makes that explosion explicit.
    """
    sub = df.copy()
    sub["order"] = pd.to_numeric(_tree_col(sub, "order"), errors="coerce")
    sub["runtime_s"] = pd.to_numeric(
        _tree_col(sub, "runtime_s"), errors="coerce")
    sub = sub[sub["order"].notna() & sub["runtime_s"].notna()
              & ~sub["is_failure"]]
    if sub.empty or sub["order"].nunique() < 2:
        return fig_empty("Need both interaction orders (1 and 2) to compare")

    grp = (
        sub.groupby(["library", "order"])
        .agg(rt=("runtime_s", "median")).reset_index()
    )
    order_vals = sorted(grp["order"].dropna().unique())
    order_name = {1: "Order 1 · main effects",
                  2: "Order 2 · pairwise interactions"}
    order_shade = {1: 0.55, 2: 1.0}
    libs = sorted(grp["library"].unique())

    fig = go.Figure()
    for o in order_vals:
        odf = grp[grp["order"] == o].set_index("library").reindex(libs)
        fig.add_trace(go.Bar(
            x=libs, y=odf["rt"].clip(lower=_RUNTIME_FLOOR).values,
            name=order_name.get(int(o), f"Order {int(o)}"),
            marker=dict(color=[_lib_color(l) for l in libs],
                        opacity=order_shade.get(int(o), 0.8),
                        line=dict(color="white", width=0.5),
                        pattern=dict(shape=["" if o == order_vals[0] else "x"] * len(libs),
                                     fgcolor="white", size=3)),
            text=[f"{v:.3g} s" if pd.notna(
                v) else "" for v in odf["rt"].values],
            textposition="outside", textfont=dict(size=10, color=TEXT),
            hovertemplate="<b>%{x}</b><br>" +
            order_name.get(int(o), f"Order {int(o)}")
                          + "<br>Median runtime: %{y:.4g} s<extra></extra>",
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=420, margin=_MARGIN, legend=_LEGEND_H,
        barmode="group",
        xaxis=dict(title="Library", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Median runtime (s) — log scale", type="log",
                   gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_tree_accuracy_vs_runtime(df: pd.DataFrame) -> go.Figure:
    """Scatter: runtime (log x) vs approximation quality (Spearman ρ) per backend.

    Answers the "efficient *without* losing accuracy" half of the question — a
    good backend sits in the upper-left (fast and faithful).
    """
    sub = df.copy()
    sub["backend"] = _tree_col(sub, "backend")
    sub["runtime_s"] = pd.to_numeric(
        _tree_col(sub, "runtime_s"),       errors="coerce")
    sub["mean_sample_rho"] = pd.to_numeric(
        _tree_col(sub, "mean_sample_rho"), errors="coerce")
    sub = sub[sub["runtime_s"].notna() & sub["mean_sample_rho"].notna()]
    if sub.empty:
        return fig_empty("No runs with both runtime and quality recorded")

    grp = (
        sub.groupby(["backend", "library"])
        .agg(rt=("runtime_s", "median"),
             rho=("mean_sample_rho", "median")).reset_index()
    )
    fig = go.Figure()
    for lib, ldf in grp.groupby("library"):
        fig.add_trace(go.Scatter(
            x=ldf["rt"].clip(lower=_RUNTIME_FLOOR), y=ldf["rho"],
            mode="markers+text", name=lib,
            marker=dict(color=_lib_color(lib), size=13,
                        line=dict(color="white", width=1.5)),
            text=[_tree_backend_label(b).replace(
                f"{lib} ", "") for b in ldf["backend"]],
            textposition="top center", textfont=dict(size=9, color=TEXT2),
            customdata=ldf["backend"].map(
                _tree_backend_label).values.reshape(-1, 1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Runtime: %{x:.4g} s<br>"
                "Spearman ρ: %{y:.3f}<extra></extra>"
            ),
        ))
    fig.add_hline(
        y=RHO_GOOD, line=dict(color=GREEN, width=1.2, dash="dot"),
        annotation_text=f"ρ = {RHO_GOOD}", annotation_position="bottom right",
        annotation_font=dict(size=10, color=GREEN),
    )
    fig.update_layout(
        **_CHART_LAYOUT, height=440, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Median runtime (s) — log scale  (left = faster)",
                   type="log", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median Spearman ρ  (higher = better)",
                   range=[0, 1.08], gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_tree_quality_ranking(df: pd.DataFrame) -> go.Figure:
    """Horizontal bars: median Spearman ρ per backend, failure flagged on the right.

    Backend-level replacement for the generic leaderboard, which can't be used
    here because the tree CSV leaves ``approximator`` empty (its groupby key
    would drop every row). Failed backends are shown at ρ = 0 with a marker.
    """
    sub = df.copy()
    sub["backend"] = _tree_col(sub, "backend")
    sub["mean_sample_rho"] = pd.to_numeric(_tree_col(sub, "mean_sample_rho"),
                                           errors="coerce")
    if sub.empty:
        return fig_empty()

    grp = (
        sub.groupby(["backend", "library"])
        .agg(rho=("mean_sample_rho", "median"),
             fr=("is_failure", "mean"))
        .reset_index()
    )
    grp["failed"] = grp["fr"] >= 0.5
    grp["rho_disp"] = grp["rho"].fillna(0.0).clip(lower=0.0)
    grp = grp.sort_values("rho_disp", ascending=True).reset_index(drop=True)
    grp["label"] = grp["backend"].map(_tree_backend_label)

    colors = [_lib_color(l) for l in grp["library"]]
    opac = [0.35 if f else 0.9 for f in grp["failed"]]
    bartext = [
        "✕ no valid output" if f else f"{r:.3f}"
        for f, r in zip(grp["failed"], grp["rho"])
    ]

    fig = go.Figure(go.Bar(
        y=grp["label"], x=grp["rho_disp"], orientation="h",
        marker=dict(color=colors, opacity=opac,
                    line=dict(color="white", width=0.5)),
        text=bartext, textposition="outside", textfont=dict(size=11, color=TEXT),
        customdata=np.stack([grp["library"], grp["fr"] * 100], axis=1),
        hovertemplate=(
            "<b>%{y}</b><br>Library: %{customdata[0]}<br>"
            "Median Spearman ρ: %{x:.3f}<br>"
            "Failure: %{customdata[1]:.0f}%<extra></extra>"
        ),
    ))
    fig.add_vline(
        x=RHO_GOOD, line=dict(color=GREEN, width=1.2, dash="dot"),
        annotation_text=f"ρ = {RHO_GOOD}", annotation_position="top",
        annotation_font=dict(size=10, color=GREEN),
    )
    fig.update_layout(
        **_CHART_LAYOUT,
        height=max(280, len(grp) * 34 + 70),
        xaxis=dict(title="Median Spearman ρ  (0–1, higher = better)",
                   range=[0, 1.15], gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=110, t=30, b=48),
        showlegend=False,
    )
    return fig


# ── RQ4 (tree-depth): the axis that actually answers the research question ────
#
# The merged tree CSV finally carries a real ``max_depth`` column (4 → 80), so
# these helpers plot cost and fidelity *as a function of tree depth* — the exact
# "which library handles extreme tree depths efficiently" question. They key off
# the ``backend`` column (the algorithm variant) because ``approximator`` is
# empty in the tree CSVs.

def _depth_col(df: pd.DataFrame) -> str:
    """Return the tree-depth column name if present, else ''."""
    for col in ("max_depth", "tree_depth", "depth"):
        if col in df.columns and df[col].notna().any():
            return col
    return ""


def fig_tree_runtime_vs_depth(df: pd.DataFrame) -> go.Figure:
    """Runtime vs tree depth (log y) — one line per backend.

    The headline RQ4 chart: a steep, upward slope means the backend gets more
    expensive as trees deepen (a scaling bottleneck), while a flat line means the
    method is depth-insensitive. Only valid (non-failing) runs are drawn.
    """
    depth = _depth_col(df)
    if not depth:
        return fig_empty("This dataset has no tree-depth column")

    sub = df.copy()
    sub["backend"] = _tree_col(sub, "backend")
    sub[depth] = pd.to_numeric(sub[depth], errors="coerce")
    sub["runtime_s"] = pd.to_numeric(
        _tree_col(sub, "runtime_s"), errors="coerce")
    sub = sub[sub[depth].notna() & sub["runtime_s"].notna()
              & ~sub["is_failure"]]
    if sub.empty or sub[depth].nunique() < 2:
        return fig_empty("Not enough tree-depth variation to show a trend")

    grp = (
        sub.groupby(["backend", "library", depth])
        .agg(rt_med=("runtime_s", "median"),
             rt_max=("runtime_s", "max")).reset_index()
    )
    fig = go.Figure()
    for backend, bdf in grp.groupby("backend"):
        lib = bdf["library"].iloc[0]
        bdf = bdf.sort_values(depth)
        med = bdf["rt_med"].clip(lower=_RUNTIME_FLOOR)
        mx = bdf["rt_max"].clip(lower=_RUNTIME_FLOOR)
        # Upper whisker reaches the slowest run at that depth — this is what
        # surfaces the "shapiq takes 100 s on deep trees" worst cases that a
        # median alone completely hides.
        fig.add_trace(go.Scatter(
            x=bdf[depth], y=med,
            mode="lines+markers", name=_tree_backend_label(backend),
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
            error_y=dict(type="data", symmetric=False,
                         array=(mx - med).values, arrayminus=[0] * len(med),
                         color=_lib_color(lib), thickness=1.2, width=4),
            customdata=np.stack([mx.values], axis=1),
            hovertemplate=(
                f"<b>{_tree_backend_label(backend)}</b><br>"
                "Tree depth: %{x}<br>Median: %{y:.4g} s<br>"
                "Worst case: %{customdata[0]:.4g} s<extra></extra>"
            ),
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=460, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Maximum tree depth",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Runtime (s) — log scale  ·  marker = median, whisker = worst case",
                   type="log", gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_tree_mean_runtime_vs_depth(df: pd.DataFrame) -> go.Figure:
    """Mean runtime vs tree depth (log y) — one line per backend.

    Companion to fig_tree_runtime_vs_depth.  The median chart shows the
    "typical" run; this chart shows the *mean*, which is pulled up by outlier
    runs and therefore better reflects real-world expected cost.  When median
    and mean diverge strongly (as for shapiq interaction at deep trees) it is a
    clear sign the distribution is right-skewed and the occasional slow run
    dominates the average.  Whiskers here span mean ± 1 standard deviation so
    you can see the spread directly.
    """
    depth = _depth_col(df)
    if not depth:
        return fig_empty("This dataset has no tree-depth column")

    sub = df.copy()
    sub["backend"] = _tree_col(sub, "backend")
    sub[depth] = pd.to_numeric(sub[depth], errors="coerce")
    sub["runtime_s"] = pd.to_numeric(
        _tree_col(sub, "runtime_s"), errors="coerce")
    sub = sub[sub[depth].notna() & sub["runtime_s"].notna()
              & ~sub["is_failure"]]
    if sub.empty or sub[depth].nunique() < 2:
        return fig_empty("Not enough tree-depth variation to show a trend")

    grp = (
        sub.groupby(["backend", "library", depth])
        .agg(rt_mean=("runtime_s", "mean"),
             rt_std=("runtime_s", "std"),
             rt_max=("runtime_s", "max")).reset_index()
    )
    # Fill NaN std (single-run groups) with 0
    grp["rt_std"] = grp["rt_std"].fillna(0.0)

    fig = go.Figure()
    for backend, bdf in grp.groupby("backend"):
        lib = bdf["library"].iloc[0]
        bdf = bdf.sort_values(depth)
        mean = bdf["rt_mean"].clip(lower=_RUNTIME_FLOOR)
        std = bdf["rt_std"]
        mx = bdf["rt_max"].clip(lower=_RUNTIME_FLOOR)

        # Upper whisker = +1 std (or clipped to max), lower whisker = -1 std
        # but never below the display floor.
        # absolute delta above mean
        upper = (mean + std - mean).clip(lower=0).values
        lower = (mean - (mean - std).clip(lower=_RUNTIME_FLOOR)
                 ).values  # absolute delta below mean

        fig.add_trace(go.Scatter(
            x=bdf[depth], y=mean,
            mode="lines+markers", name=_tree_backend_label(backend),
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
            error_y=dict(type="data", symmetric=False,
                         array=upper, arrayminus=lower,
                         color=_lib_color(lib), thickness=1.2, width=4),
            customdata=np.stack([mx.values, std.values], axis=1),
            hovertemplate=(
                f"<b>{_tree_backend_label(backend)}</b><br>"
                "Tree depth: %{x}<br>Mean: %{y:.4g} s<br>"
                "Std dev: %{customdata[1]:.4g} s<br>"
                "Worst case: %{customdata[0]:.4g} s<extra></extra>"
            ),
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=460, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Maximum tree depth",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Runtime (s) — log scale  ·  marker = mean, whisker = ±1 std dev",
                   type="log", gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_tree_depth_scaling_factor(df: pd.DataFrame) -> go.Figure:
    """Horizontal bars: runtime blow-up from the shallowest to the deepest tree.

    For each backend we divide its median runtime at the largest depth by the
    median at the smallest depth. A factor near 1× means depth-robust (ideal for
    "extreme tree depths"); a large factor flags a backend whose cost explodes as
    trees grow. This is the single clearest ranking for the RQ4 question.
    """
    depth = _depth_col(df)
    if not depth:
        return fig_empty("This dataset has no tree-depth column")

    sub = df.copy()
    sub["backend"] = _tree_col(sub, "backend")
    sub[depth] = pd.to_numeric(sub[depth], errors="coerce")
    sub["runtime_s"] = pd.to_numeric(
        _tree_col(sub, "runtime_s"), errors="coerce")
    sub = sub[sub[depth].notna() & sub["runtime_s"].notna()
              & ~sub["is_failure"]]
    if sub.empty or sub[depth].nunique() < 2:
        return fig_empty("Not enough tree-depth variation to compute scaling")

    d_lo, d_hi = sub[depth].min(), sub[depth].max()
    grp = (
        sub.groupby(["backend", "library", depth])
        .agg(rt=("runtime_s", "median")).reset_index()
    )
    rows = []
    for backend, bdf in grp.groupby("backend"):
        lib = bdf["library"].iloc[0]
        lo = bdf.loc[bdf[depth] == d_lo, "rt"]
        hi = bdf.loc[bdf[depth] == d_hi, "rt"]
        if lo.empty or hi.empty:
            continue
        lo_v = max(float(lo.iloc[0]), _RUNTIME_FLOOR)
        hi_v = max(float(hi.iloc[0]), _RUNTIME_FLOOR)
        rows.append(dict(backend=backend, library=lib,
                         factor=hi_v / lo_v, lo=lo_v, hi=hi_v))
    if not rows:
        return fig_empty("No backend spans both the shallowest and deepest tree")

    gr = pd.DataFrame(rows).sort_values("factor", ascending=True)
    gr["label"] = gr["backend"].map(_tree_backend_label)
    colors = [_lib_color(l) for l in gr["library"]]

    fig = go.Figure(go.Bar(
        y=gr["label"], x=gr["factor"], orientation="h",
        marker=dict(color=colors, opacity=0.9,
                    line=dict(color="white", width=0.5)),
        text=[f"{f:.1f}×" for f in gr["factor"]],
        textposition="outside", textfont=dict(size=11, color=TEXT),
        customdata=np.stack([gr["lo"], gr["hi"]], axis=1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"Runtime at depth {int(d_lo)}: %{{customdata[0]:.4g}} s<br>"
            f"Runtime at depth {int(d_hi)}: %{{customdata[1]:.4g}} s<br>"
            "Blow-up factor: %{x:.2f}×<extra></extra>"
        ),
    ))
    fig.add_vline(
        x=1.0, line=dict(color=GREEN, width=1.2, dash="dot"),
        annotation_text="1× · depth-robust", annotation_position="top",
        annotation_font=dict(size=10, color=GREEN),
    )
    fig.update_layout(
        **_CHART_LAYOUT,
        height=max(280, len(gr) * 34 + 80),
        xaxis=dict(title=f"Runtime blow-up, depth {int(d_lo)} → {int(d_hi)}  "
                   "(lower = handles depth better)",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=90, t=30, b=48),
        showlegend=False,
    )
    return fig


def fig_tree_quality_vs_depth(df: pd.DataFrame) -> go.Figure:
    """Spearman ρ vs tree depth — does approximation fidelity survive deep trees?

    The efficiency question is only half the story: a backend that stays fast but
    loses rank fidelity as trees deepen is still a poor choice. Lines that stay
    near the top (and above the ρ = 0.9 line) keep their explanations trustworthy
    at extreme depths.
    """
    depth = _depth_col(df)
    if not depth:
        return fig_empty("This dataset has no tree-depth column")

    sub = df.copy()
    sub["backend"] = _tree_col(sub, "backend")
    sub[depth] = pd.to_numeric(sub[depth], errors="coerce")
    sub["mean_sample_rho"] = pd.to_numeric(
        _tree_col(sub, "mean_sample_rho"), errors="coerce")
    sub = sub[sub[depth].notna() & sub["mean_sample_rho"].notna()
              & ~sub["is_failure"]]
    if sub.empty or sub[depth].nunique() < 2:
        return fig_empty("Not enough tree-depth variation to show a trend")

    grp = (
        sub.groupby(["backend", "library", depth])
        .agg(rho=("mean_sample_rho", "median")).reset_index()
    )
    fig = go.Figure()
    for backend, bdf in grp.groupby("backend"):
        lib = bdf["library"].iloc[0]
        bdf = bdf.sort_values(depth)
        fig.add_trace(go.Scatter(
            x=bdf[depth], y=bdf["rho"],
            mode="lines+markers", name=_tree_backend_label(backend),
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
            hovertemplate=(
                f"<b>{_tree_backend_label(backend)}</b><br>"
                "Tree depth: %{x}<br>Spearman ρ: %{y:.3f}<extra></extra>"
            ),
        ))
    fig.add_hline(
        y=RHO_GOOD, line=dict(color=GREEN, width=1.2, dash="dot"),
        annotation_text=f"ρ = {RHO_GOOD}", annotation_position="bottom right",
        annotation_font=dict(size=10, color=GREEN),
    )
    # Auto-zoom the y-axis to the actual data band (ρ is typically 0.9–1.0 for
    # tree explainers). A fixed 0–1 range would crush every line against the
    # top and hide the differences the user cares about; we still keep the
    # ρ = 0.9 reference line in view so "good vs. degraded" stays readable.
    y_lo = float(grp["rho"].min())
    y_hi = float(grp["rho"].max())
    pad = max((y_hi - y_lo) * 0.15, 0.005)
    y_range = [min(y_lo - pad, RHO_GOOD - 0.02), min(y_hi + pad, 1.01)]
    fig.update_layout(
        **_CHART_LAYOUT, height=440, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Maximum tree depth",
                   gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Median Spearman ρ  (higher = better, zoomed to data)",
                   range=y_range, gridcolor=BORDER, zeroline=False),
    )
    return fig


# ── RQ5: GPU vs CPU ──────────────────────────────────────────────────────────

def fig_hardware_comparison(df: pd.DataFrame) -> go.Figure:
    """Grouped horizontal bar chart comparing CPU vs GPU runtime with Q25–Q75 error bars."""
    df = df.copy()
    if "seed" not in df.columns:
        sub = df[df["runtime_median"].notna() & df["device"].notna()].copy()
        if sub.empty:
            return fig_empty()
        sub["device"] = sub["device"].astype(str).str.lower().replace({"cuda": "gpu"})
        grp = (
            sub.groupby(["method", "device"])
            .agg(
                rt=("runtime_median", "median"),
                rt_lo=("runtime_q25", "median"),
                rt_hi=("runtime_q75", "median")
            ).reset_index()
        )
    else:
        sub = df[df["runtime_s"].notna() & df["device"].notna()].copy()
        if sub.empty:
            return fig_empty()
        sub["device"] = sub["device"].astype(str).str.lower().replace({"cuda": "gpu"})
        grp = (
            sub.groupby(["method", "device"])
            .agg(
                rt=("runtime_s", "median"),
                rt_lo=("runtime_s", lambda x: x.quantile(0.25)),
                rt_hi=("runtime_s", lambda x: x.quantile(0.75))
            ).reset_index()
        )
        
    if grp.empty:
        return fig_empty()

    gpu_rts = grp[grp["device"] == "gpu"].set_index("method")["rt"]
    if gpu_rts.empty:
        gpu_rts = grp.groupby("method")["rt"].median()
        
    sorted_methods = gpu_rts.sort_values(ascending=False).index.tolist()
    
    # Calculate error boundaries
    grp["err_plus"] = grp["rt_hi"] - grp["rt"]
    grp["err_minus"] = grp["rt"] - grp["rt_lo"]
    
    fig = go.Figure()
    for dev in ["cpu", "gpu"]:
        dev_df = grp[grp["device"] == dev].set_index("method").reindex(sorted_methods).reset_index()
        fig.add_trace(go.Bar(
            y=dev_df["method"], x=dev_df["rt"], orientation="h",
            name=dev.upper(),
            marker=dict(color=PINK if dev == "gpu" else ACCENT, opacity=0.85),
            text=dev_df["rt"].apply(lambda val: f"{val:.3f} s" if pd.notna(val) else ""),
            textposition="outside",
            customdata=np.column_stack((
                dev_df["rt_lo"].values,
                dev_df["rt_hi"].values
            )),
            error_x=dict(
                type="data",
                symmetric=False,
                array=dev_df["err_plus"].values,
                arrayminus=dev_df["err_minus"].values,
                visible=True,
                color="#475569",
                thickness=1.2,
                width=4
            ),
            hovertemplate=f"<b>%{{y}} ({dev.upper()})</b><br>Median runtime: %{{x:.3f}} s <span style='font-size:11px;color:#64748B'>[%{{customdata[0]}}s - %{{customdata[1]}}s]</span><extra></extra>"
        ))
        
    fig.update_layout(
        **_CHART_LAYOUT, height=max(260, len(sorted_methods) * 45 + 80),
        barmode="group",
        xaxis=dict(title="Median runtime (s)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=20, t=30, b=48),
        legend=_LEGEND_H,
    )
    return fig


def fig_hardware_speedup(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing speedup factor of GPU over CPU (runtime_s CPU / runtime_s GPU)."""
    df = df.copy()
    if "seed" not in df.columns:
        sub = df[df["runtime_median"].notna() & df["device"].notna()].copy()
        if sub.empty:
            return fig_empty()
        sub["device"] = sub["device"].astype(str).str.lower().replace({"cuda": "gpu"})
        grp = (
            sub.groupby(["method", "device"])
            .agg(rt=("runtime_median", "median")).reset_index()
        )
        pivot = grp.pivot(index="method", columns="device", values="rt").dropna(subset=["cpu", "gpu"])
        if pivot.empty:
            return fig_empty("No matching CPU and GPU runs to compute speedup.")
        pivot["speedup"] = pivot["cpu"] / pivot["gpu"]
        pivot["speedup_lo"] = pivot["speedup"]
        pivot["speedup_hi"] = pivot["speedup"]
    else:
        sub = df[df["runtime_s"].notna() & df["device"].notna()].copy()
        if sub.empty:
            return fig_empty()
        sub["device"] = sub["device"].astype(str).str.lower().replace({"cuda": "gpu"})
        group_keys = ["dataset", "model", "library", "approximator", "method", "seed"]
        pivot_raw = sub.pivot_table(index=group_keys, columns="device", values="runtime_s").dropna(subset=["cpu", "gpu"])
        if pivot_raw.empty:
            return fig_empty("No matching CPU and GPU runs to compute speedup.")
        pivot_raw["speedup"] = pivot_raw["cpu"] / pivot_raw["gpu"]
        
        pivot = (
            pivot_raw.groupby("method")
            .agg(
                speedup=("speedup", "median"),
                speedup_lo=("speedup", lambda x: x.quantile(0.25)),
                speedup_hi=("speedup", lambda x: x.quantile(0.75)),
                cpu=("cpu", "median"),
                gpu=("gpu", "median")
            )
        )
        
    pivot = pivot.sort_values("speedup", ascending=True)
    pivot["err_plus"] = pivot["speedup_hi"] - pivot["speedup"]
    pivot["err_minus"] = pivot["speedup"] - pivot["speedup_lo"]
    
    fig = go.Figure(go.Bar(
        y=pivot.index, x=pivot["speedup"], orientation="h",
        marker=dict(color=GREEN, opacity=0.85),
        text=pivot["speedup"].apply(lambda val: f"{val:.1f}x speedup"),
        textposition="outside",
        customdata=np.column_stack((
            pivot["cpu"].values,
            pivot["gpu"].values,
            pivot["speedup_lo"].values,
            pivot["speedup_hi"].values
        )),
        error_x=dict(
            type="data",
            symmetric=False,
            array=pivot["err_plus"].values,
            arrayminus=pivot["err_minus"].values,
            visible=True,
            color="#475569",
            thickness=1.2,
            width=4
        ),
        hovertemplate="<b>%{y}</b><br>Speedup: <b>%{x:.1f}x</b> <span style='font-size:11px;color:#64748B'>[%{customdata[2]:.1f}x - %{customdata[3]:.1f}x]</span> (GPU vs CPU)<br>CPU: %{customdata[0]:.3f}s, GPU: %{customdata[1]:.3f}s<extra></extra>",
    ))
    
    fig.update_layout(
        **_CHART_LAYOUT, height=max(240, len(pivot) * 36 + 60),
        xaxis=dict(title="Speedup factor (CPU time / GPU time) — higher is better", gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=20, t=30, b=48),
        showlegend=False,
    )
    return fig


def fig_rho_vs_runtime_by_hardware(df: pd.DataFrame) -> go.Figure:
    """Scatter: runtime vs Spearman ρ, with markers colored by device, supporting seed lists."""
    df = df.copy()
    if "runtime_median" in df.columns:
        df["runtime_s"] = df["runtime_median"]
        df["mean_sample_rho"] = df["rho_median"]
        
    sub = df[df["runtime_s"].notna() & df["mean_sample_rho"].notna() & df["device"].notna()].copy()
    if sub.empty:
        return fig_empty()
    
    sub["device"] = sub["device"].astype(str).str.lower().replace({"cuda": "gpu"})
    
    fig = go.Figure()
    for (dev, lib), ldf in sub.groupby(["device", "library"]):
        name = f"{lib} ({dev.upper()})"
        color = PINK if dev == "gpu" else ACCENT
        symbol = "circle" if dev == "gpu" else "square"
        
        fig.add_trace(go.Scatter(
            x=ldf["runtime_s"], y=ldf["mean_sample_rho"],
            mode="markers", name=name,
            marker=dict(color=color, symbol=symbol, size=7, opacity=0.7,
                        line=dict(color="white", width=0.5)),
            hovertemplate=(
                f"<b>{lib} (%{{customdata[0]}})</b><br>"
                "Method: %{customdata[1]}<br>"
                "Seed: %{customdata[2]}<br>"
                "Runtime: %{x:.3f} s<br>"
                "Spearman ρ: %{y:.3f}<extra></extra>"
            ),
            customdata=np.column_stack((
                [dev.upper()] * len(ldf),
                ldf["method"].values,
                ldf["seed"].values if "seed" in ldf.columns else [0] * len(ldf)
            )),
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


# ── RQ3 Custom Charts ─────────────────────────────────────────────────────────

def fig_rq3_attribution_agreement(df: pd.DataFrame) -> go.Figure:
    """Grouped horizontal bar chart showing Spearman rho (range -1 to 1) for each explainer config, grouped by model topology."""
    df = df.copy()
    if "rho_median" in df.columns:
        grp = (
            df.groupby(["method", "model"])
            .agg(
                rho=("rho_median", "median"),
                rho_lo=("rho_q25", "median"),
                rho_hi=("rho_q75", "median"),
                mae=("relative_mae_median", "median")
            )
            .reset_index()
        )
        df = grp
    else:
        sub = df[df["mean_sample_rho"].notna()].copy()
        if sub.empty:
            return fig_empty()
        grp = (
            sub.groupby(["method", "model"])
            .agg(
                rho=("mean_sample_rho", "median"),
                rho_lo=("mean_sample_rho", lambda x: x.quantile(0.25)),
                rho_hi=("mean_sample_rho", lambda x: x.quantile(0.75)),
                mae=("relative_mae", "median")
            )
            .reset_index()
        )
        df = grp

    sub = df[df["rho"].notna()].copy()
    if sub.empty:
        return fig_empty()
    
    # Calculate error boundaries for error_x
    sub["err_plus"] = sub["rho_hi"] - sub["rho"]
    sub["err_minus"] = sub["rho"] - sub["rho_lo"]
    
    method_rhos = sub.groupby("method")["rho"].median()
    def get_lib(method_name):
        lib = method_name.split(" / ")[0]
        return "shapiq" if lib == "shapiq_proxy" else lib
    method_to_lib = {m: get_lib(m) for m in method_rhos.index}
    lib_rhos = {}
    for m, r in method_rhos.items():
        l = method_to_lib[m]
        lib_rhos[l] = max(lib_rhos.get(l, -999.0), r)
    method_order = sorted(
        method_rhos.index,
        key=lambda m: (lib_rhos[method_to_lib[m]], method_to_lib[m], method_rhos[m])
    )
    
    models = sorted(sub["model"].unique().tolist())
    model_colors = {
        "mlp": "#4B6DD4",
        "cnn_1d": "#E84060",
        "transformer": "#10B981"
    }
    
    fig = go.Figure()
    for model in models:
        model_df = sub[sub["model"] == model].set_index("method").reindex(method_order).reset_index()
        fig.add_trace(go.Bar(
            y=model_df["method"],
            x=model_df["rho"],
            name=model.upper().replace("_", "-"),
            orientation="h",
            marker=dict(color=model_colors.get(model, ACCENT), opacity=0.85),
            customdata=np.column_stack((
                model_df["mae"].values,
                model_df["rho_lo"].values,
                model_df["rho_hi"].values
            )),
            error_x=dict(
                type="data",
                symmetric=False,
                array=model_df["err_plus"].values,
                arrayminus=model_df["err_minus"].values,
                visible=True,
                color="#475569",
                thickness=1.2,
                width=4
            ),
            hovertemplate=(
                "<b>%{y}</b> (%{legendgroup})<br>"
                "Spearman ρ: <b>%{x:.3f}</b> <span style='font-size:11px;color:#64748B'>[%{customdata[1]:.3f} - %{customdata[2]:.3f}]</span><br>"
                "Relative MAE: <b>%{customdata[0]:.3f}</b><extra></extra>"
            ),
            legendgroup=model,
        ))
        
    fig.update_layout(
        **_CHART_LAYOUT,
        height=max(320, len(method_order) * 45 + 100),
        barmode="group",
        xaxis=dict(title="Spearman rank correlation coefficient (mean_sample_rho)", range=[-1.0, 1.05], gridcolor=BORDER, zeroline=True),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=20, t=30, b=48),
        legend=_LEGEND_H,
    )
    return fig


def fig_rq3_runtime_comparison(df: pd.DataFrame) -> go.Figure:
    """Vertical bar chart of median runtimes with logarithmic scale on y-axis."""
    df = df.copy()
    if "runtime_median" in df.columns:
        grp = (
            df.groupby(["method", "library"])
            .agg(
                rt=("runtime_median", "median"),
                rt_lo=("runtime_q25", "median"),
                rt_hi=("runtime_q75", "median")
            ).reset_index()
            .sort_values("rt")
        )
    else:
        sub = df[df["runtime_s"].notna()].copy()
        if sub.empty:
            return fig_empty()
        grp = (
            sub.groupby(["method", "library"])
            .agg(
                rt=("runtime_s", "median"),
                rt_lo=("runtime_s", lambda x: x.quantile(0.25)),
                rt_hi=("runtime_s", lambda x: x.quantile(0.75))
            ).reset_index()
            .sort_values("rt")
        )
    
    colors = [_lib_color(lib) for lib in grp["library"]]
    
    grp["err_plus"] = grp["rt_hi"] - grp["rt"]
    grp["err_minus"] = grp["rt"] - grp["rt_lo"]
    
    fig = go.Figure(go.Bar(
        x=grp["method"],
        y=grp["rt"],
        marker=dict(color=colors, opacity=0.85),
        text=grp["rt"].apply(lambda v: f"{v:.3f}s" if v < 1 else f"{v:.1f}s"),
        textposition="outside",
        customdata=np.column_stack((
            grp["rt_lo"].values,
            grp["rt_hi"].values
        )),
        error_y=dict(
            type="data",
            symmetric=False,
            array=grp["err_plus"].values,
            arrayminus=grp["err_minus"].values,
            visible=True,
            color="#475569",
            thickness=1.2,
            width=4
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Median runtime: <b>%{y:.3f} s</b> <span style='font-size:11px;color:#64748B'>[%{customdata[0]:.3f}s - %{customdata[1]:.3f}s]</span><extra></extra>"
        )
    ))
    
    fig.update_layout(
        **_CHART_LAYOUT,
        height=420,
        yaxis=dict(
            title="Execution time (seconds) — Log scale",
            type="log",
            gridcolor=BORDER,
            zeroline=False,
            dtick=1,
        ),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickangle=45),
        margin=dict(l=55, r=20, t=30, b=100),
        showlegend=False,
    )
    return fig


def fig_rq3_axiomatic_integrity(df: pd.DataFrame) -> go.Figure:
    """Box and whisker plot of the relative additivity gap on a logarithmic scale."""
    sub = df[df["relative_additivity_gap"].notna()].copy()
    if sub.empty:
        return fig_empty()
    
    # Use the same library-grouped sorting for consistency across charts
    method_rhos = sub.groupby("method")["mean_sample_rho"].median() if "mean_sample_rho" in sub.columns else pd.Series()
    if method_rhos.empty:
        sorted_methods = sorted(sub["method"].unique())
    else:
        def get_lib(method_name):
            lib = method_name.split(" / ")[0]
            return "shapiq" if lib == "shapiq_proxy" else lib
        method_to_lib = {m: get_lib(m) for m in method_rhos.index}
        lib_rhos = {}
        for m, r in method_rhos.items():
            l = method_to_lib[m]
            lib_rhos[l] = max(lib_rhos.get(l, -999.0), r)
        sorted_methods = sorted(
            method_rhos.index,
            key=lambda m: (lib_rhos[method_to_lib[m]], method_to_lib[m], method_rhos[m])
        )
    
    fig = go.Figure()
    for method in sorted_methods:
        mdf = sub[sub["method"] == method]
        lib = mdf["library"].iloc[0] if not mdf.empty else "shapiq"
        
        gaps = mdf["relative_additivity_gap"] + 1e-12
        
        fig.add_trace(go.Box(
            y=gaps,
            name=method,
            marker=dict(color=_lib_color(lib)),
            boxpoints="outliers",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Min: %{customdata[0]:.3e}<br>"
                "Q1: %{q1:.3e}<br>"
                "Median: %{median:.3e}<br>"
                "Q3: %{q3:.3e}<br>"
                "Max: %{customdata[1]:.3e}<extra></extra>"
            ),
            customdata=np.column_stack((
                [gaps.min()] * len(gaps),
                [gaps.max()] * len(gaps)
            ))
        ))
        
    fig.update_layout(
        **_CHART_LAYOUT,
        height=460,
        yaxis=dict(
            title="Relative additivity gap — Log scale",
            type="log",
            gridcolor=BORDER,
            zeroline=False,
            dtick=1,
        ),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickangle=45),
        margin=dict(l=55, r=20, t=30, b=100),
        showlegend=False,
    )
    return fig


def fig_rq3_scalability_wall(df: pd.DataFrame) -> go.Figure:
    """Faceted scatter plot of runtime_s (log10) vs mean_sample_rho, with 3 columns corresponding to adult_census, ames_housing, gisette."""
    from plotly.subplots import make_subplots
    df = df.copy()
    
    if "runtime_median" in df.columns:
        df["runtime_s"] = df["runtime_median"]
        df["mean_sample_rho"] = df["rho_median"]
        
    sub = df[df["runtime_s"].notna() & df["mean_sample_rho"].notna()].copy()
    if sub.empty:
        return fig_empty()
    
    sub["approximator"] = sub["approximator"].fillna("unknown")
    
    ds_list = []
    for ds in sub["dataset"].unique():
        ds_df = sub[sub["dataset"] == ds]
        m = int(ds_df["n_features"].iloc[0]) if not ds_df.empty else 0
        ds_list.append((ds, m))
    ds_list = sorted(ds_list, key=lambda x: x[1])

    datasets = [x[0] for x in ds_list]
    titles = [f"{x[0].replace('_', ' ').title()} (M={x[1]})" for x in ds_list]
    n_cols = len(datasets)
    
    if n_cols == 0:
        return fig_empty()
    
    fig = make_subplots(
        rows=1, cols=n_cols, 
        subplot_titles=titles,
        shared_yaxes=True,
        horizontal_spacing=0.04
    )
    
    if "runtime_q25" in sub.columns:
        grp = (
            sub.groupby(["dataset", "library", "approximator", "method"])
            .agg(
                rt=("runtime_median", "mean"),
                rt_lo=("runtime_q25", "mean"),
                rt_hi=("runtime_q75", "mean"),
                rho=("rho_median", "mean"),
                rho_lo=("rho_q25", "mean"),
                rho_hi=("rho_q75", "mean")
            )
            .reset_index()
        )
    else:
        grp = (
            sub.groupby(["dataset", "library", "approximator", "method"])
            .agg(
                rt=("runtime_s", "mean"),
                rt_lo=("runtime_s", lambda x: x.quantile(0.25)),
                rt_hi=("runtime_s", lambda x: x.quantile(0.75)),
                rho=("mean_sample_rho", "mean"),
                rho_lo=("mean_sample_rho", lambda x: x.quantile(0.25)),
                rho_hi=("mean_sample_rho", lambda x: x.quantile(0.75))
            )
            .reset_index()
        )
    
    methods = sorted(grp["method"].unique().tolist())
    shown_legend = set()
    
    for col_idx, ds in enumerate(datasets, start=1):
        ds_grp = grp[grp["dataset"] == ds]
        
        for method in methods:
            adf = ds_grp[ds_grp["method"] == method]
            if adf.empty:
                continue
                
            lib = adf["library"].iloc[0]
            color = _lib_color(lib)
            show_leg = method not in shown_legend
            if show_leg:
                shown_legend.add(method)
                
            fig.add_trace(
                go.Scatter(
                    x=adf["rt"],
                    y=adf["rho"],
                    mode="markers",
                    name=method,
                    marker=dict(
                        color=color, 
                        size=10, 
                        opacity=0.85,
                        line=dict(color="white", width=0.8)
                    ),
                    customdata=np.column_stack((
                        adf["library"],
                        adf["approximator"],
                        adf["method"],
                        adf["rt_lo"],
                        adf["rt_hi"],
                        adf["rho_lo"],
                        adf["rho_hi"]
                    )),
                    hovertemplate=(
                        "<b>%{customdata[2]}</b><br>"
                        "Average Runtime: <b>%{x:.3f} s</b> <span style='font-size:11px;color:#CBD5E1'>[%{customdata[3]:.3f}s - %{customdata[4]:.3f}s]</span><br>"
                        "Spearman ρ: <b>%{y:.3f}</b> <span style='font-size:11px;color:#CBD5E1'>[%{customdata[5]:.3f} - %{customdata[6]:.3f}]</span><br>"
                        "Library: %{customdata[0]}<br>"
                        "Approximator: %{customdata[1]}<extra></extra>"
                    ),
                    showlegend=show_leg,
                    legendgroup=method,
                ),
                row=1, col=col_idx
            )
            
    for i in range(1, n_cols + 1):
        fig.update_xaxes(
            title_text="Avg Runtime (s)",
            type="log",
            gridcolor=BORDER,
            zeroline=False,
            row=1, col=i
        )
    fig.update_yaxes(
        title_text="Spearman ρ (Alignment)",
        gridcolor=BORDER,
        zeroline=False,
        range=[0.35, 1.05],
        row=1, col=1
    )
    
    fig.update_layout(
        **_CHART_LAYOUT,
        height=400,
        margin=dict(l=55, r=20, t=85, b=48),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.15,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)
        ),
    )
    return fig
 
 
def fig_rq3_topology_violations(df: pd.DataFrame) -> go.Figure:
    """Grouped horizontal bar chart showing mean relative additivity gap as a percentage across network structural topologies, grouped by method."""
    df = df.copy()
    if "gap_mean" in df.columns:
        grp = (
            df.groupby(["method", "model"])
            .agg(
                gap=("gap_mean", "mean"),
                gap_lo=("gap_q25", "mean"),
                gap_hi=("gap_q75", "mean")
            ).reset_index()
        )
    else:
        sub = df[df["relative_additivity_gap"].notna()].copy()
        if sub.empty:
            return fig_empty()
        grp = (
            sub.groupby(["method", "model"])
            .agg(
                gap=("relative_additivity_gap", "mean"),
                gap_lo=("relative_additivity_gap", lambda x: x.quantile(0.25)),
                gap_hi=("relative_additivity_gap", lambda x: x.quantile(0.75))
            ).reset_index()
        )
    
    # Use the same library-grouped sorting as the Spearman Rank Alignment chart
    method_rhos = df.groupby("method")["rho_median"].median() if "rho_median" in df.columns else (
        df.groupby("method")["mean_sample_rho"].median() if "mean_sample_rho" in df.columns else pd.Series()
    )
    if method_rhos.empty:
        method_order = sorted(grp["method"].unique())
    else:
        def get_lib(method_name):
            lib = method_name.split(" / ")[0]
            return "shapiq" if lib == "shapiq_proxy" else lib
        method_to_lib = {m: get_lib(m) for m in method_rhos.index}
        lib_rhos = {}
        for m, r in method_rhos.items():
            l = method_to_lib[m]
            lib_rhos[l] = max(lib_rhos.get(l, -999.0), r)
        method_order = sorted(
            method_rhos.index,
            key=lambda m: (lib_rhos[method_to_lib[m]], method_to_lib[m], method_rhos[m])
        )
    
    # Calculate error values for horizontal bar error_x
    grp["err_plus"] = grp["gap_hi"] - grp["gap"]
    grp["err_minus"] = grp["gap"] - grp["gap_lo"]
    
    models = ["mlp", "cnn_1d", "transformer"]
    model_colors = {
        "mlp": "#4B6DD4",
        "cnn_1d": "#E84060",
        "transformer": "#10B981"
    }
    
    fig = go.Figure()
    for model in models:
        model_df = grp[grp["model"] == model].set_index("method").reindex(method_order).reset_index()
        gaps_pct = model_df["gap"] * 100
        gaps_pct_lo = model_df["gap_lo"] * 100
        gaps_pct_hi = model_df["gap_hi"] * 100
        err_plus_pct = model_df["err_plus"] * 100
        err_minus_pct = model_df["err_minus"] * 100
        
        fig.add_trace(go.Bar(
            y=model_df["method"],
            x=gaps_pct,
            name=model.upper().replace("_", "-"),
            orientation="h",
            marker=dict(color=model_colors.get(model, ACCENT), opacity=0.85),
            text=gaps_pct.apply(lambda v: f"{v:.1f}%" if v > 0.5 else ("~0%" if pd.notna(v) else "")),
            textposition="outside",
            customdata=np.column_stack((
                gaps_pct_lo.values,
                gaps_pct_hi.values
            )),
            error_x=dict(
                type="data",
                symmetric=False,
                array=err_plus_pct.values,
                arrayminus=err_minus_pct.values,
                visible=True,
                color="#475569",
                thickness=1.2,
                width=4
            ),
            hovertemplate=(
                "<b>%{y}</b> (%{legendgroup})<br>"
                "Mean violation gap: <b>%{x:.3f}%</b> <span style='font-size:11px;color:#64748B'>[%{customdata[0]:.3f}% - %{customdata[1]:.3f}%]</span><extra></extra>"
            ),
            legendgroup=model,
        ))
    fig.update_layout(
        **_CHART_LAYOUT,
        height=max(320, len(method_order) * 45 + 100),
        barmode="group",
        xaxis=dict(
            title="Mean relative additivity gap (%) — lower is better",
            gridcolor=BORDER,
            zeroline=True,
            range=[0, 105]
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=40, t=30, b=48),
        legend=_LEGEND_H,
    )
    return fig


