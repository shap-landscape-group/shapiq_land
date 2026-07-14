"""
shared/charts.py — All Plotly figure builders, one per chart type.
Add new fig_* functions here; import them in __init__.py.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .tokens import (
    BG, CARD, BORDER, ACCENT, PINK, GREEN, RED, AMBER, MUTED,
    TEXT, TEXT2, LIB_COLOR, FONT,
    FAILURE_MAE, RHO_GOOD,
    _CHART_LAYOUT, _LEGEND_H, _MARGIN,
)
from .data import pareto_mark, _backend_mode


def _lib_color(lib: str) -> str:
    return LIB_COLOR.get(lib, ACCENT)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _add_median_band_trace(fig: go.Figure, row: int, col: int, x, median: pd.Series,
                            lo: pd.Series, hi: pd.Series, color: str, name: str,
                            show_legend: bool, floor: float = 0.0,
                            x_label: str = "") -> None:
    """Add a median line + a shaded min-max ribbon (across seeds) to one subplot cell.

    Used by the small-multiple RQ4 facet charts (rows=dataset[, cols=model]): each
    cell gets its own median + range band per library, sharing a legend group across
    cells so toggling one library's legend entry hides it everywhere.
    """
    median = median.clip(lower=floor) if floor else median
    lo = lo.clip(lower=floor) if floor else lo
    fig.add_trace(go.Scatter(
        x=x, y=lo, mode="lines", line=dict(width=0),
        legendgroup=name, showlegend=False, hoverinfo="skip",
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=x, y=hi, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=_hex_to_rgba(color, 0.18),
        legendgroup=name, showlegend=False, hoverinfo="skip",
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=x, y=median, mode="lines+markers", name=name, legendgroup=name,
        showlegend=show_legend,
        line=dict(color=color, width=2),
        marker=dict(size=6, color=color, line=dict(color="white", width=1)),
        customdata=np.stack([lo.values, hi.values], axis=1),
        hovertemplate=(
            f"<b>{name}</b><br>{x_label}: %{{x}}<br>Median: %{{y:.4g}}<br>"
            "Range across seeds: %{customdata[0]:.4g} – %{customdata[1]:.4g}"
            "<extra></extra>"
        ),
    ), row=row, col=col)


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


def _tree_pairwise_matrix(sub: pd.DataFrame):
    """Build a symmetric backend x backend agreement matrix from a (pre-filtered)
    tree-CSV slice. Reads each row's own pairwise_metrics JSON directly and
    restricts comparisons to same-mode counterparts (path-dependent vs.
    path-dependent, interventional vs. interventional, interaction vs.
    interaction) — comparing across modes would conflate "different by design"
    with genuine disagreement. Returns (backends, library_labels, z) or None if
    there isn't enough data to compare at least two backends.
    """
    if sub.empty or "pairwise_metrics" not in sub.columns or "backend" not in sub.columns:
        return None
    work = sub[sub["is_failure"].fillna(False) != True]
    if work.empty:
        return None

    metric, fallback_metric = "sign_agreement", "mean_sample_rho"
    pairs = []
    for backend, metrics_str in zip(work["backend"], work["pairwise_metrics"]):
        if not isinstance(metrics_str, str) or not metrics_str.strip():
            continue
        mode = _backend_mode(backend)
        if not mode:
            continue
        try:
            clean = metrics_str.replace(": NaN", ": null").replace(":NaN", ":null")
            parsed = json.loads(clean)
        except Exception:
            continue
        for other, vals in parsed.items():
            if other == backend or not isinstance(vals, dict) \
                    or _backend_mode(other) != mode:
                continue
            val = vals.get(metric)
            if val is None:
                val = vals.get(fallback_metric)
            if val is not None:
                pairs.append((backend, other, val))
    if not pairs:
        return None

    long = pd.DataFrame(pairs, columns=["backend_a", "backend_b", "value"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    if long.empty:
        return None

    # Pool both directions of every unordered pair into one symmetric estimate.
    # Each row only records the comparison as seen from its own backend's JSON
    # entry, so (A,B) and (B,A) are independent samples of the same underlying
    # quantity (agreement between A and B) — averaging them together and
    # mirroring the result avoids the two directions drifting slightly apart
    # depending on which rows happened to be valid in each direction.
    long["_pair"] = long.apply(
        lambda r: tuple(sorted((r["backend_a"], r["backend_b"]))), axis=1)
    sym = long.groupby("_pair")["value"].mean()

    backends = sorted(set(long["backend_a"]) | set(long["backend_b"]))
    if len(backends) < 2:
        return None

    idx = {b: i for i, b in enumerate(backends)}
    z = np.full((len(backends), len(backends)), np.nan)
    for (a, b), v in sym.items():
        i, j = idx[a], idx[b]
        z[i, j] = v
        z[j, i] = v

    lib_of = work.drop_duplicates("backend").set_index("backend")["library"].to_dict()
    labels = [lib_of.get(b, _tree_backend_label(b)) for b in backends]
    return backends, labels, z


def fig_tree_pass_fail_matrix(df: pd.DataFrame) -> go.Figure:
    """Binary library x model heatmap, one small-multiple panel per dataset:
    green = every run for that (library, model, dataset) triple produced valid
    output, red = at least one failed outright.

    A single library x model matrix can't represent a failure that depends on
    the dataset *and* the model at once — e.g. woodelf only breaks on
    lightgbm/xgboost when the dataset is covertype, not on every dataset, while
    fasttreeshap breaks on xgboost regardless of dataset. Faceting by dataset
    makes that distinction visible directly: a backend that's red in every
    panel has a model-only problem; one that's red in a single panel has a
    problem tied to that specific dataset (e.g. a multi-class one) as well.
    """
    required = {"library", "model", "dataset"}
    if df.empty or not required.issubset(df.columns):
        return fig_empty()

    datasets = sorted(df["dataset"].dropna().unique())
    libs = sorted(df["library"].dropna().unique())
    models = sorted(df["model"].dropna().unique())
    if not datasets or not libs or not models:
        return fig_empty()

    fig = make_subplots(
        rows=1, cols=len(datasets), subplot_titles=[str(d) for d in datasets],
        shared_yaxes=True, horizontal_spacing=0.03,
    )
    for ci, dataset in enumerate(datasets):
        ddf = df[df["dataset"] == dataset]
        grp = (
            ddf.groupby(["library", "model"])
            .agg(fr=("is_failure", "mean"), n=("is_failure", "size"))
            .reset_index()
        )
        fr = grp.pivot(index="library", columns="model",
                        values="fr").reindex(index=libs, columns=models)
        n = grp.pivot(index="library", columns="model",
                      values="n").reindex(index=libs, columns=models)

        z = np.full(fr.shape, np.nan)
        text = [["—"] * len(models) for _ in libs]
        hover = [[""] * len(models) for _ in libs]
        for i, lib in enumerate(libs):
            for j, m in enumerate(models):
                v = fr.iloc[i, j]
                if pd.isna(v):
                    hover[i][j] = f"<b>{lib}</b> × <b>{m}</b> × {dataset}<br>no data"
                    continue
                z[i, j] = 1.0 if v > 0 else 0.0
                text[i][j] = "FAIL" if v > 0 else "ok"
                hover[i][j] = (
                    f"<b>{lib}</b> × <b>{m}</b> × {dataset}<br>"
                    f"Failure rate: {v * 100:.0f}% (n={int(n.iloc[i, j])})"
                )

        fig.add_trace(go.Heatmap(
            z=z, x=models, y=libs,
            text=text, texttemplate="%{text}",
            customdata=hover, hovertemplate="%{customdata}<extra></extra>",
            colorscale=[[0, "#D1FAE5"], [1, "#FEE2E2"]],
            zmin=0, zmax=1, showscale=False,
            xgap=3, ygap=3,
        ), row=1, col=ci + 1)

    fig.update_annotations(font_size=11)
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10), automargin=True, col=1)
    fig.update_layout(
        **_CHART_LAYOUT, height=max(220, len(libs) * 45 + 90),
        margin=dict(l=110, r=16, t=40, b=50),
    )
    return fig


def fig_tree_agreement_heatmap_by_model(df: pd.DataFrame) -> go.Figure:
    """Cross-library agreement heatmap, one small-multiple panel per model family.

    Cells = mean sign-agreement between each pair of libraries for the selected
    tree case (1.0 = perfect agreement). This is a correctness check, not a
    scaling question, so dataset/depth/seed are all pooled together within each
    panel — only model is faceted, since different model families (tree
    structures) can expose different implementation edge cases. Since these are
    exact methods, cells well below 1.0 flag a numerical/implementation bug, not
    an approximation trade-off.
    """
    if df.empty or "model" not in df.columns:
        return fig_empty()
    models = sorted(df["model"].dropna().unique())
    if not models:
        return fig_empty()

    fig = make_subplots(
        rows=1, cols=len(models),
        subplot_titles=[str(m) for m in models],
        horizontal_spacing=0.1,
    )
    any_panel = False
    for i, model in enumerate(models):
        result = _tree_pairwise_matrix(df[df["model"] == model])
        if result is None:
            continue
        _backends, labels, z = result
        text = [[f"{v:.3f}" if not np.isnan(v) else "—" for v in row] for row in z]
        fig.add_trace(go.Heatmap(
            z=z, x=labels, y=labels, text=text, texttemplate="%{text}",
            colorscale=[[0, "#FEE2E2"], [0.5, "#93C5FD"], [1, "#1E3A8A"]],
            zmin=0, zmax=1, showscale=(i == len(models) - 1),
            colorbar=dict(title="Agreement", thickness=14, len=0.8),
            hovertemplate=(
                f"Model: {model}<br>Backend A: %{{y}}<br>Backend B: %{{x}}<br>"
                "Mean agreement: %{z:.3f}<extra></extra>"
            ),
        ), row=1, col=i + 1)
        any_panel = True
    if not any_panel:
        return fig_empty()
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10), automargin=True)

    fig.update_layout(**_CHART_LAYOUT, height=340, margin=dict(l=90, r=16, t=40, b=90))
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


def fig_tree_runtime_vs_features_by_dataset(df: pd.DataFrame) -> go.Figure:
    """Runtime vs number of features (log-log), rows=dataset, color=library.

    The quadratic-blowup chart for pairwise interactions (order=2), capped at
    interaction_max_features — the main reason that cap exists.
    """
    sub = df.copy()
    sub["backend"] = _tree_col(sub, "backend")
    sub["n_features"] = pd.to_numeric(_tree_col(sub, "n_features"), errors="coerce")
    sub["runtime_s"] = pd.to_numeric(_tree_col(sub, "runtime_s"), errors="coerce")
    sub = sub[sub["n_features"].notna() & sub["runtime_s"].notna() & ~sub["is_failure"]]
    if sub.empty:
        return fig_empty("Not enough n_features variation to show scaling")

    datasets = sorted(sub["dataset"].dropna().unique()) if "dataset" in sub.columns else []
    if not datasets:
        return fig_empty()

    fig = make_subplots(rows=len(datasets), cols=1,
                         subplot_titles=[str(d) for d in datasets],
                         vertical_spacing=min(0.5 / max(len(datasets), 1), 0.15))
    seen_legend = set()
    any_panel = False
    for ri, dataset in enumerate(datasets):
        ddf = sub[sub["dataset"] == dataset]
        if ddf.empty:
            continue
        grp = (ddf.groupby(["library", "n_features"])
                  .agg(rt=("runtime_s", "median")).reset_index())
        for lib, ldf in grp.groupby("library"):
            ldf = ldf.sort_values("n_features")
            show_legend = lib not in seen_legend
            seen_legend.add(lib)
            fig.add_trace(go.Scatter(
                x=ldf["n_features"], y=ldf["rt"].clip(lower=_RUNTIME_FLOOR),
                mode="lines+markers", name=lib, legendgroup=lib,
                showlegend=show_legend,
                line=dict(color=_lib_color(lib), width=2),
                marker=dict(size=8, color=_lib_color(lib),
                            line=dict(color="white", width=1.5)),
                hovertemplate=(f"<b>{lib}</b><br>n_features: %{{x}}<br>"
                                "Runtime: %{y:.4g} s<extra></extra>"),
            ), row=ri + 1, col=1)
            any_panel = True
    if not any_panel:
        return fig_empty("Not enough n_features variation to show scaling")

    fig.update_annotations(font_size=11)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log", title_text="Runtime (s)", title_font=dict(size=10))
    fig.update_xaxes(title_text="n_features", title_font=dict(size=10),
                      row=len(datasets), col=1)
    fig.update_layout(**_CHART_LAYOUT, height=max(280, len(datasets) * 260),
                       legend=_LEGEND_H, margin=dict(l=55, r=20, t=30, b=40))
    return fig


# ── RQ4 (tree-depth): the axis that actually answers the research question ────
#
# Tree explainers here are exact methods, not approximations trading accuracy for
# speed — a quality dip signals a numerical/implementation bug, not an expected
# cost/quality trade-off. These helpers plot cost and cross-library agreement as
# a function of the *realized* tree depth (the depth the fitted tree actually
# reached, not the requested sweep target — see _depth_col). They key off the
# ``backend`` column (the algorithm variant) because ``approximator`` is empty
# in the tree CSVs.

def _depth_col(df: pd.DataFrame) -> str:
    """Return the tree-depth column name if present, else ''.

    Always the *realized* depth the fitted tree actually reached (``max_depth``),
    never the configured sweep target (``max_depth_config``) — a tree asked for
    depth 80 that only ever reaches 28 in practice should be plotted at 28, since
    that's the actual complexity the explainer had to handle.
    """
    for col in ("max_depth", "tree_depth", "depth"):
        if col in df.columns and df[col].notna().any():
            return col
    return ""


def fig_tree_runtime_vs_depth_faceted(df: pd.DataFrame, facet_model_cols: bool = True) -> go.Figure:
    """Runtime vs realized tree depth — small multiples, rows=dataset[, cols=model].

    The headline RQ4 chart: a steep, upward slope means the backend gets more
    expensive as trees deepen (a scaling bottleneck); a flat line means the
    backend is depth-robust. Each panel is one (dataset[, model]) combination so
    lines are never averaged across datasets of wildly different feature counts,
    or across model families with different depth-scaling behavior — the median
    line and shaded band are computed *only* across the (up to 10) seeds within
    that single panel. Color = library. When `facet_model_cols` is False (used
    for the interaction tab, which only has one dataset), model is left as a
    filter instead of a facet column.
    """
    depth = _depth_col(df)
    if not depth:
        return fig_empty("This dataset has no tree-depth column")

    sub = df.copy()
    sub["backend"] = _tree_col(sub, "backend")
    sub[depth] = pd.to_numeric(sub[depth], errors="coerce")
    sub["runtime_s"] = pd.to_numeric(_tree_col(sub, "runtime_s"), errors="coerce")
    sub = sub[sub[depth].notna() & sub["runtime_s"].notna() & ~sub["is_failure"]]
    if sub.empty or sub[depth].nunique() < 2:
        return fig_empty("Not enough tree-depth variation to show a trend")

    datasets = sorted(sub["dataset"].dropna().unique()) if "dataset" in sub.columns else []
    if not datasets:
        return fig_empty()
    models = (sorted(sub["model"].dropna().unique())
              if facet_model_cols and "model" in sub.columns else [None])

    n_rows, n_cols = len(datasets), len(models)
    subplot_titles = [
        (f"{d} · {m}" if m is not None else str(d))
        for d in datasets for m in models
    ]
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=subplot_titles,
        vertical_spacing=min(0.6 / max(n_rows, 1), 0.12),
        horizontal_spacing=0.06,
    )
    seen_legend = set()
    any_panel = False
    for ri, dataset in enumerate(datasets):
        for ci, model in enumerate(models):
            cell = sub[sub["dataset"] == dataset]
            if model is not None:
                cell = cell[cell["model"] == model]
            if cell.empty:
                continue
            grp = (
                cell.groupby(["backend", "library", depth])
                .agg(rt_med=("runtime_s", "median"),
                     rt_lo=("runtime_s", "min"),
                     rt_hi=("runtime_s", "max")).reset_index()
            )
            for backend, bdf in grp.groupby("backend"):
                lib = bdf["library"].iloc[0]
                bdf = bdf.sort_values(depth)
                show_legend = lib not in seen_legend
                seen_legend.add(lib)
                _add_median_band_trace(
                    fig, ri + 1, ci + 1, bdf[depth], bdf["rt_med"], bdf["rt_lo"],
                    bdf["rt_hi"], _lib_color(lib), lib, show_legend,
                    floor=_RUNTIME_FLOOR, x_label="Depth",
                )
                any_panel = True
    if not any_panel:
        return fig_empty("Not enough tree-depth variation to show a trend")

    fig.update_annotations(font_size=11)
    fig.update_yaxes(type="log")
    fig.update_xaxes(title_text="Max depth", gridcolor=BORDER, title_font=dict(size=10))
    fig.update_yaxes(title_text="Runtime (s)", gridcolor=BORDER, title_font=dict(size=10), col=1)
    fig.update_layout(
        **_CHART_LAYOUT, height=max(280, n_rows * 250), legend=_LEGEND_H,
        margin=dict(l=55, r=20, t=40, b=40),
    )
    return fig


def fig_tree_depth_scaling_factor(df: pd.DataFrame) -> go.Figure:
    """Horizontal bars: runtime blow-up from the shallowest to the deepest tree.

    The shallow -> deep ratio is computed *within* every (dataset, model) combo
    first — so a feature-heavy dataset (e.g. gisette at 1000 features) can't
    dominate the "shallow" baseline just by having a much bigger absolute runtime —
    then summarized per backend as the median ratio across combos. A factor near 1×
    means depth-robust (ideal for "extreme tree depths"); a large factor flags a
    backend whose cost explodes as trees grow.
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
    combo_cols = [c for c in ("dataset", "model") if c in sub.columns]
    grp = (
        sub.groupby(["backend", "library"] + combo_cols + [depth])
        .agg(rt=("runtime_s", "median")).reset_index()
    )
    rows = []
    for backend, bdf in grp.groupby("backend"):
        lib = bdf["library"].iloc[0]
        combo_iter = bdf.groupby(combo_cols) if combo_cols else [((), bdf)]
        ratios = []
        for _, cdf in combo_iter:
            lo = cdf.loc[cdf[depth] == d_lo, "rt"]
            hi = cdf.loc[cdf[depth] == d_hi, "rt"]
            if lo.empty or hi.empty:
                continue
            lo_v = max(float(lo.iloc[0]), _RUNTIME_FLOOR)
            hi_v = max(float(hi.iloc[0]), _RUNTIME_FLOOR)
            ratios.append(hi_v / lo_v)
        if ratios:
            rows.append(dict(backend=backend, library=lib,
                             factor=float(np.median(ratios)), n=len(ratios)))
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
        customdata=gr["n"],
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"Depth {int(d_lo)} → {int(d_hi)} blow-up<br>"
            "Median factor across %{customdata} dataset/model combo(s): "
            "%{x:.2f}×<extra></extra>"
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


def fig_tree_order_cost(df: pd.DataFrame) -> go.Figure:
    """Grouped bars: median runtime at interaction order 1 vs order 2, per library.

    Moving from main effects (order 1, path-dependent/interventional backends) to
    pairwise interactions (order 2) is the single biggest cost jump in the data —
    this makes that explosion explicit. Expects a df that mixes order-1 and
    order-2 rows for the same libraries (the interaction tab passes both modes in).
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


def fig_tree_quality_ranking(df: pd.DataFrame) -> go.Figure:
    """Horizontal bars: median Spearman ρ per backend, failure flagged on the right.

    Tree explainers are exact methods, so this ranks *cross-library agreement*
    with same-mode peers, not approximation quality — mean_sample_rho for a given
    backend is its average Spearman ρ against every *other* same-mode backend
    (a symmetric peer-consensus measure — no single library is treated as ground
    truth). A low bar means a backend's output diverges from the rest of the
    pack (a correctness concern), not that it made a deliberate speed/accuracy
    trade-off. Failed backends are shown at ρ = 0 with a marker.
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
            "Median Spearman ρ vs. same-mode peers: %{x:.3f}<br>"
            "Failure: %{customdata[1]:.0f}%<extra></extra>"
        ),
    ))
    fig.add_vline(
        x=RHO_GOOD, line=dict(color=GREEN, width=1.2, dash="dot"),
        annotation_text=f"ρ = {RHO_GOOD} · correctness threshold",
        annotation_position="top",
        annotation_font=dict(size=10, color=GREEN),
    )
    fig.update_layout(
        **_CHART_LAYOUT,
        height=max(280, len(grp) * 34 + 70),
        xaxis=dict(title="Median Spearman ρ vs. same-mode peers  (0–1, higher = better)",
                   range=[0, 1.15], gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(l=10, r=110, t=30, b=48),
        showlegend=False,
    )
    return fig


def fig_tree_runtime_vs_depth(df: pd.DataFrame) -> go.Figure:
    """Runtime vs tree depth (log y) — quick overview, one shaded-band line per backend.

    A fast global glance: a steep, upward slope means the backend gets more
    expensive as trees deepen; a flat line means the method is depth-insensitive.
    The band is ±1 std across seeds *and* across whichever datasets/models are
    currently selected — so it necessarily pools cost figures that can differ a
    lot by feature count/model family. Use the faceted "Runtime vs Max Depth"
    chart below for the dataset × model breakdown; this one is for a fast
    first look only.
    """
    depth = _depth_col(df)
    if not depth:
        return fig_empty("This dataset has no tree-depth column")

    sub = df.copy()
    sub["backend"] = _tree_col(sub, "backend")
    sub[depth] = pd.to_numeric(sub[depth], errors="coerce")
    sub["runtime_s"] = pd.to_numeric(_tree_col(sub, "runtime_s"), errors="coerce")
    sub = sub[sub[depth].notna() & sub["runtime_s"].notna() & ~sub["is_failure"]]
    if sub.empty or sub[depth].nunique() < 2:
        return fig_empty("Not enough tree-depth variation to show a trend")

    grp = (
        sub.groupby(["backend", "library", depth])
        .agg(rt_mean=("runtime_s", "mean"), rt_std=("runtime_s", "std")).reset_index()
    )
    grp["rt_std"] = grp["rt_std"].fillna(0.0)

    fig = go.Figure()
    for backend, bdf in grp.groupby("backend"):
        lib = bdf["library"].iloc[0]
        bdf = bdf.sort_values(depth)
        mean = bdf["rt_mean"].clip(lower=_RUNTIME_FLOOR)
        std = bdf["rt_std"]
        lower = (mean - std).clip(lower=_RUNTIME_FLOOR)
        upper = mean + std
        label = _tree_backend_label(backend)
        fig.add_trace(go.Scatter(
            x=bdf[depth], y=lower, mode="lines", line=dict(width=0),
            legendgroup=label, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=bdf[depth], y=upper, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_hex_to_rgba(_lib_color(lib), 0.18),
            legendgroup=label, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=bdf[depth], y=mean, mode="lines+markers", name=label,
            legendgroup=label,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
            customdata=std.values,
            hovertemplate=(
                f"<b>{label}</b><br>Depth: %{{x}}<br>"
                "Mean: %{y:.4g} s ± %{customdata:.3g}<extra></extra>"
            ),
        ))
    fig.update_layout(
        **_CHART_LAYOUT, height=420, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Maximum tree depth", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Runtime (s), log scale", type="log",
                   gridcolor=BORDER, zeroline=False),
    )
    return fig


def fig_tree_quality_vs_depth(df: pd.DataFrame) -> go.Figure:
    """Spearman ρ vs tree depth — quick overview, does cross-library agreement
    survive deep trees?

    mean_sample_rho for a backend is its average Spearman ρ against every
    *other* same-mode backend at that depth (a symmetric peer-consensus
    measure — no single library, e.g. shap, is treated as ground truth). Tree
    explainers are exact methods, so a dip is a correctness red flag (numerical
    instability or a bug at extreme depth), not an expected cost/quality
    trade-off. The band is ±1 std across seeds *and* across whichever
    datasets/models are currently selected — a fast first look, not a
    dataset/model breakdown.
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
        .agg(rho_mean=("mean_sample_rho", "mean"),
             rho_std=("mean_sample_rho", "std")).reset_index()
    )
    grp["rho_std"] = grp["rho_std"].fillna(0.0)

    fig = go.Figure()
    for backend, bdf in grp.groupby("backend"):
        lib = bdf["library"].iloc[0]
        bdf = bdf.sort_values(depth)
        mean = bdf["rho_mean"]
        std = bdf["rho_std"]
        lower = mean - std
        upper = (mean + std).clip(upper=1.0)
        label = _tree_backend_label(backend)
        fig.add_trace(go.Scatter(
            x=bdf[depth], y=lower, mode="lines", line=dict(width=0),
            legendgroup=label, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=bdf[depth], y=upper, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_hex_to_rgba(_lib_color(lib), 0.18),
            legendgroup=label, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=bdf[depth], y=mean, mode="lines+markers", name=label,
            legendgroup=label,
            line=dict(color=_lib_color(lib), width=2),
            marker=dict(size=8, color=_lib_color(lib),
                        line=dict(color="white", width=1.5)),
            customdata=std.values,
            hovertemplate=(
                f"<b>{label}</b><br>Depth: %{{x}}<br>"
                "Mean ρ vs. same-mode peers: %{y:.3f} ± %{customdata:.3f}"
                "<extra></extra>"
            ),
        ))
    fig.add_hline(
        y=RHO_GOOD, line=dict(color=GREEN, width=1.2, dash="dot"),
        annotation_text=f"ρ = {RHO_GOOD} · correctness threshold",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=GREEN),
    )
    # Auto-zoom the y-axis to the actual data band (ρ is typically 0.9–1.0 for
    # tree explainers, since these are exact methods that should closely agree).
    y_lo = float((grp["rho_mean"] - grp["rho_std"]).min())
    y_hi = float((grp["rho_mean"] + grp["rho_std"]).clip(upper=1.0).max())
    pad = max((y_hi - y_lo) * 0.15, 0.005)
    y_range = [min(y_lo - pad, RHO_GOOD - 0.02), min(y_hi + pad, 1.01)]
    fig.update_layout(
        **_CHART_LAYOUT, height=420, margin=_MARGIN, legend=_LEGEND_H,
        xaxis=dict(title="Maximum tree depth", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Spearman ρ vs. peers", range=y_range,
                   gridcolor=BORDER, zeroline=False),
    )
    return fig

