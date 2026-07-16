"""
shared/data.py — CSV loading, schema coercion, and aggregate helpers.
All data-touching logic lives here so RQ pages stay chart-and-layout only.
"""
import json
import os

import numpy as np
import pandas as pd

from .tokens import (
    EPSILON, FAILURE_MAE, EXPECTED_COLS,
)


# ── New-format helper: extract quality metrics from pairwise_metrics JSON ─────

def _extract_vs_reference(metrics_str) -> dict:
    """
    Parse the pairwise_metrics JSON cell and return metrics for this approximation
    compared against the true-value reference (the key that contains 'true_value').
    Handles NaN in the JSON string (not valid JSON but written by the benchmark).
    """
    if not isinstance(metrics_str, str) or not metrics_str.strip():
        return {}
    try:
        clean   = metrics_str.replace(": NaN", ": null").replace(":NaN", ":null")
        metrics = json.loads(clean)
        # Prefer the key that represents ground truth
        for key in metrics:
            if "true_value" in key:
                return {k: v for k, v in metrics[key].items() if v is not None}
        # Fallback: first key whose relative_mae is not 0 and not None
        for key, vals in metrics.items():
            if vals.get("relative_mae") not in (0.0, None):
                return {k: v for k, v in vals.items() if v is not None}
    except Exception:
        pass
    return {}


def _fill_quality_from_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    When the CSV uses the new pairwise_metrics column instead of direct quality
    columns, extract relative_mae / sign_agreement / mean_sample_rho / mean_abs_diff
    (compared against the "*_true_value" reference backend) so the rest of the
    pipeline can treat both formats identically.
    """
    if "pairwise_metrics" not in df.columns:
        return df
    # Only fill missing columns; don't overwrite existing data
    need = [c for c in ("relative_mae", "sign_agreement", "mean_sample_rho", "mean_abs_diff")
            if c not in df.columns]
    if not need:
        return df

    extracted = df["pairwise_metrics"].apply(_extract_vs_reference)
    for col in need:
        df[col] = extracted.apply(lambda x, c=col: x.get(c, np.nan))
    return df


# ── Main loader ───────────────────────────────────────────────────────────────

# path -> (mtime at load time, parsed DataFrame). Unbounded is fine — there are
# only a handful of results CSVs across the whole app.
_LOAD_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}


def load_data(path: str) -> pd.DataFrame:
    """
    Read a benchmark CSV (old or new format), coerce numeric columns,
    keep only 'approximation' rows, and add derived columns:
    method, is_failure.

    Cached in-process, keyed on the file's mtime: Dash re-invokes every page's
    callback on every filter/tab change, which would otherwise re-read the CSV
    from disk and re-parse the pairwise_metrics JSON for every row each time.
    Editing the CSV on disk changes its mtime, which invalidates the cache
    automatically. Returns a fresh copy each call so callers can safely mutate
    their own result without corrupting the cached original.
    """
    mtime = os.path.getmtime(path)
    cached = _LOAD_CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1].copy()

    df = _load_data_uncached(path)
    _LOAD_CACHE[path] = (mtime, df)
    return df.copy()


def _load_data_uncached(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in ["runtime_s", "relative_mae", "budget", "sign_agreement",
                "mean_sample_rho", "n_model_evals", "n_features", "n_samples",
                "additivity_gap", "relative_additivity_gap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "computation_type" in df.columns:
        approx_df = df[df["computation_type"] == "approximation"].copy()
        if approx_df.empty and not df.empty:
            # Tree / exact-method CSVs have no "approximation" rows; all methods are
            # "true_value" types being compared against a common reference.
            # Keep every row EXCEPT the primary reference itself (backend ends in _true_value).
            if "backend" in df.columns:
                df = df[~df["backend"].str.endswith("true_value", na=False)].copy()
            else:
                df = df.copy()
        else:
            df = approx_df
    else:
        df = df.copy()

    # New-format CSV: extract quality metrics from the pairwise_metrics column
    df = _fill_quality_from_pairwise(df)

    lib    = df["library"].fillna("?")      if "library"      in df.columns else pd.Series("?", index=df.index)
    approx = df["approximator"].fillna("?") if "approximator" in df.columns else pd.Series("?", index=df.index)
    df["method"] = lib + " / " + approx

    mae = df["relative_mae"] if "relative_mae" in df.columns else pd.Series(np.nan, index=df.index)
    df["is_failure"] = mae.isna() | (mae > FAILURE_MAE)

    return df


def try_load_data(*paths: str) -> tuple[pd.DataFrame, str | None]:
    """
    Try each path in order. Return (df, source_path) for the first file found,
    or (empty DataFrame with correct schema, None) if none exist.
    Use `source is None` to show a 'data not yet collected' banner.
    """
    for path in paths:
        if os.path.exists(path):
            return load_data(path), path
    empty = pd.DataFrame(columns=EXPECTED_COLS + ["method", "is_failure"])
    return empty, None


def normalize_filter_selection(selection) -> list[str] | None:
    """Topbar filter store → None means all, [] means none, else explicit list."""
    if selection is None or selection == "__all__":
        return None
    if isinstance(selection, str):
        return [selection]
    if isinstance(selection, list):
        if "__all__" in selection:
            return None
        return list(selection)
    return None


def filter_by_column(df: pd.DataFrame, column: str, selection) -> pd.DataFrame:
    selected = normalize_filter_selection(selection)
    if selected is None or column not in df.columns or df.empty:
        return df
    if not selected:
        return df.iloc[0:0]
    return df[df[column].isin(selected)]


def should_pool_dimension(selection) -> bool:
    """Pool by median when all values or more than one are selected."""
    selected = normalize_filter_selection(selection)
    if selected is None:
        return True
    return len(selected) > 1


def compute_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per method and rank by median Spearman ρ, ties broken by runtime."""
    empty_cols = ["rank", "method", "library", "approximator",
                  "rho_median", "mae_median", "sign_median",
                  "runtime_median", "failure_rate", "n_runs"]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    grp = (
        df.groupby(["method", "library", "approximator"])
        .agg(
            rho_median     = ("mean_sample_rho", "median"),
            mae_median     = ("relative_mae",    "median"),
            sign_median    = ("sign_agreement",  "median"),
            runtime_median = ("runtime_s",       "median"),
            failure_rate   = ("is_failure",      "mean"),
            n_runs         = ("runtime_s",       "count"),
        )
        .reset_index()
    )
    grp = grp.sort_values(
        ["rho_median", "runtime_median"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    grp.insert(0, "rank", range(1, len(grp) + 1))
    return grp


def pareto_mark(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    """Boolean Series: True if point is on the Pareto front (min-x AND max-y)."""
    idx     = df.index
    flags   = pd.Series(False, index=idx)
    sorted_i = df[x_col].argsort()
    best_y  = -np.inf
    for i in sorted_i:
        y = df.loc[idx[i], y_col]
        if y > best_y:
            best_y = y
            flags.iloc[i] = True
    return flags
