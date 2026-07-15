"""
RQ4 results converter.

Input file:
    results/rq4_trees.csv
        Tree explainers only — every backend is an exact method (no
        approximation trade-off), so there is no single true-value reference;
        instead each row's pairwise_metrics JSON carries its agreement with
        every *other* backend that shares its computation mode
        (path-dependent / interventional / interaction).

Outputs (results/converted/):
    rq4_trees_by_seed.csv
        One row per run, with same-mode-peer-averaged quality metrics
        (mean_abs_diff, relative_mae, sign_agreement, mean_sample_rho),
        a precomputed `mode` column, and `is_failure`.
    rq4_trees_pairwise.csv
        Long format, one row per (run, same-mode peer) pair — feeds the
        cross-library agreement heatmap without re-parsing JSON at render time.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_RQ4 = os.path.join(_HERE, "results", "rq4_trees.csv")
OUT_DIR = os.path.join(_HERE, "results", "converted")

QUALITY_METRICS = ["mean_abs_diff", "relative_mae", "sign_agreement", "mean_sample_rho"]

BY_SEED_COLS = [
    "dataset", "model", "n_background", "n_features", "n_samples",
    "max_depth", "n_estimators", "max_depth_config", "order",
    "backend", "library", "mode", "seed", "imputer", "n_eval",
    "runtime_s", "additivity_gap", "relative_additivity_gap",
    "mean_abs_diff", "relative_mae", "sign_agreement", "mean_sample_rho",
    "is_failure",
]


def _backend_mode(backend) -> str:
    """Return the tree computation mode encoded in a backend name ('' if none matches)."""
    if not isinstance(backend, str):
        return ""
    b = backend.lower()
    if "path" in b and "dependent" in b:
        return "path_dependent"
    if "interventional" in b:
        return "interventional"
    if "interaction" in b:
        return "interaction"
    return ""


def _parse_pairwise(cell) -> dict:
    if not isinstance(cell, str) or not cell.strip():
        return {}
    try:
        # Benchmark sometimes writes raw NaN which is invalid in standard JSON (requires null)
        clean = cell.replace(": NaN", ": null").replace(":NaN", ":null")
        return json.loads(clean)
    except (ValueError, TypeError):
        return {}


def load_rq4_raw() -> pd.DataFrame:
    if not os.path.exists(RAW_RQ4):
        raise FileNotFoundError(f"Raw RQ4 CSV not found at {RAW_RQ4}")
    return pd.read_csv(RAW_RQ4)


def build_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single pass over the raw rows: per-row same-mode-peer quality averages
    for the by-seed output, plus the long-format same-mode pair list for the
    agreement heatmap. Failed runs (additivity_gap is null) are excluded from
    the pairwise output — a broken run has nothing meaningful to compare."""
    df = df.copy()
    df["mode"] = df["backend"].map(_backend_mode)
    lib_of_backend = df.drop_duplicates("backend").set_index("backend")["library"].to_dict()

    quality = {m: [] for m in QUALITY_METRICS}
    pairwise_rows = []

    for row in df.itertuples(index=False):
        backend, mode = row.backend, row.mode
        pm = _parse_pairwise(row.pairwise_metrics)
        same_mode = {
            key: vals for key, vals in pm.items()
            if key != backend and _backend_mode(key) == mode and isinstance(vals, dict)
        }

        for m in QUALITY_METRICS:
            vals = [v[m] for v in same_mode.values() if v.get(m) is not None]
            quality[m].append(float(np.nanmean(vals)) if vals else np.nan)

        is_failure = pd.isna(row.additivity_gap)
        if not is_failure and mode:
            for other, vals in same_mode.items():
                sa, rho = vals.get("sign_agreement"), vals.get("mean_sample_rho")
                if sa is None and rho is None:
                    # Peer produced no valid output for this run (its own JSON
                    # entry is all-null) — nothing to compare, matches the old
                    # live code's implicit `val is not None` filter.
                    continue
                pairwise_rows.append(dict(
                    dataset=row.dataset, model=row.model, seed=row.seed,
                    max_depth=row.max_depth, mode=mode,
                    backend_a=backend, backend_b=other,
                    library_a=lib_of_backend.get(backend, "?"),
                    library_b=lib_of_backend.get(other, "?"),
                    sign_agreement=sa,
                    mean_sample_rho=rho,
                ))

    for m in QUALITY_METRICS:
        df[m] = quality[m]
    df["is_failure"] = df["additivity_gap"].isna()

    by_seed = df[[c for c in BY_SEED_COLS if c in df.columns]]
    pairwise = pd.DataFrame(pairwise_rows)
    return by_seed, pairwise


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("RQ4 converter: loading raw data...")
    df = load_rq4_raw()

    print("RQ4 converter: extracting same-mode-peer quality metrics + pairwise table...")
    by_seed, pairwise = build_outputs(df)

    by_seed_path = os.path.join(OUT_DIR, "rq4_trees_by_seed.csv")
    pairwise_path = os.path.join(OUT_DIR, "rq4_trees_pairwise.csv")

    by_seed.to_csv(by_seed_path, index=False)
    print(f"  wrote {by_seed_path}: {len(by_seed)} rows")

    pairwise.to_csv(pairwise_path, index=False)
    print(f"  wrote {pairwise_path}: {len(pairwise)} rows")


if __name__ == "__main__":
    sys.exit(main())
