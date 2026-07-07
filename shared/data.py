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
    so the rest of the pipeline can treat both formats identically.
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

def load_data(path: str) -> pd.DataFrame:
    """
    Read a benchmark CSV (old or new format), coerce numeric columns,
    keep only 'approximation' rows, and add derived columns:
    method, is_failure.
    """
    df = pd.read_csv(path)

    for col in ["runtime_s", "relative_mae", "budget", "sign_agreement",
                "mean_sample_rho", "n_model_evals", "n_features", "n_samples"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "computation_type" in df.columns:
        approx_df = df[df["computation_type"] == "approximation"].copy()
        if approx_df.empty and not df.empty and "pairwise_metrics" in df.columns:
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

    mae  = df["relative_mae"]    if "relative_mae"   in df.columns else pd.Series(np.nan, index=df.index)
    sign = df["sign_agreement"]  if "sign_agreement" in df.columns else pd.Series(np.nan, index=df.index)
    rho  = df["mean_sample_rho"] if "mean_sample_rho" in df.columns else pd.Series(np.nan, index=df.index)
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
