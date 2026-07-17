"""
RQ3 results converter.

Input file:
    results/rq3_results_config-neural-networks-cpu.csv
        3 datasets x 3 models x 10 seeds x 5 method configurations (approximation)
        1,500 total rows.

Outputs (results/converted/):
    rq3_neural_networks_by_seed.csv
    rq3_neural_networks_aggregated.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_RQ3 = os.path.join(_HERE, "results", "rq3_results_config-neural-networks-cpu.csv")
OUT_DIR = os.path.join(_HERE, "results", "converted")

REQUIRED_COLUMNS = [
    "dataset", "model", "n_features", "seed", "library", "backend",
    "approximator", "budget", "runtime_s", "n_model_evals",
    "relative_additivity_gap", "pairwise_metrics",
]

ERROR_METRICS = ["relative_mae", "mean_abs_diff", "sign_agreement", "mean_sample_rho"]


def load_rq3_raw() -> pd.DataFrame:
    if not os.path.exists(RAW_RQ3):
        raise FileNotFoundError(f"Raw RQ3 CSV not found at {RAW_RQ3}")
    
    df = pd.read_csv(RAW_RQ3)
    
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{RAW_RQ3}: missing required columns {missing}")
        
    return df


def _parse_pairwise(cell: str) -> dict:
    if not isinstance(cell, str) or not cell.strip():
        return {}
    try:
        # Benchmark sometimes writes raw NaN which is invalid in standard JSON (requires null)
        clean = cell.replace(": NaN", ": null").replace(":NaN", ":null")
        return json.loads(clean)
    except (ValueError, TypeError):
        return {}


def build_by_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Extract pairwise metrics vs reference for each run."""
    apx = df[df["computation_type"] == "approximation"].copy()
    if apx.empty:
        apx = df.copy()
        
    apx["method"] = apx["library"] + " / " + apx["approximator"]
    # Display-only rename: "shapiq_proxy / proxy" reads as a variant of shapiq_proxy
    # rather than shapiq's proxy-based SHAP method — relabel for clarity.
    apx.loc[apx["method"] == "shapiq_proxy / proxy", "method"] = "shapiq / proxyShap"

    extracted = {m: [] for m in ERROR_METRICS}
    for cell_json in apx["pairwise_metrics"]:
        pm = _parse_pairwise(cell_json)
        
        # In RQ3, there is no key containing 'true_value'.
        # We fall back to the first key whose relative_mae is not 0 (which compares against gradient_shap usually).
        ref = {}
        for key, vals in pm.items():
            if isinstance(vals, dict) and vals.get("relative_mae") not in (0.0, None):
                ref = vals
                break
        for m in ERROR_METRICS:
            extracted[m].append(ref.get(m, np.nan))
            
    for m in ERROR_METRICS:
        apx[m] = extracted[m]
        
    keep = [
        "dataset", "model", "n_features", "seed", "library", "backend",
        "approximator", "budget", "method", "device", "imputer", "n_background",
        "n_eval", "runtime_s", "n_model_evals", "relative_additivity_gap",
        "relative_mae", "mean_abs_diff", "sign_agreement", "mean_sample_rho"
    ]
    return apx[[c for c in keep if c in apx.columns]]


def aggregate_seeds(by_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the 10 seeds using median and quartiles (Q25/Q75)."""
    def q25(s): return s.quantile(0.25)
    def q75(s): return s.quantile(0.75)
    
    GROUP_KEYS = [
        "dataset", "model", "library", "approximator", "budget", "method",
        "device", "imputer", "n_background", "n_eval", "n_features"
    ]
    
    return (
        by_seed.groupby(GROUP_KEYS)
        .agg(
            seed_count=("seed", "nunique"),
            runtime_median=("runtime_s", "median"),
            runtime_q25=("runtime_s", q25),
            runtime_q75=("runtime_s", q75),
            relative_mae_median=("relative_mae", "median"),
            relative_mae_q25=("relative_mae", q25),
            relative_mae_q75=("relative_mae", q75),
            rho_median=("mean_sample_rho", "median"),
            rho_q25=("mean_sample_rho", q25),
            rho_q75=("mean_sample_rho", q75),
            gap_mean=("relative_additivity_gap", "mean"),
            gap_median=("relative_additivity_gap", "median"),
            gap_q25=("relative_additivity_gap", q25),
            gap_q75=("relative_additivity_gap", q75),
            sign_agreement_median=("sign_agreement", "median"),
        )
        .reset_index()
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("RQ3 converter: loading raw data...")
    df = load_rq3_raw()
    
    print("RQ3 converter: parsing pairwise metrics per seed...")
    by_seed = build_by_seed(df)
    
    print("RQ3 converter: aggregating seeds...")
    aggregated = aggregate_seeds(by_seed)
    
    by_seed_path = os.path.join(OUT_DIR, "rq3_neural_networks_by_seed.csv")
    aggregated_path = os.path.join(OUT_DIR, "rq3_neural_networks_aggregated.csv")
    
    by_seed.to_csv(by_seed_path, index=False)
    print(f"  wrote {by_seed_path}: {len(by_seed)} rows")
    
    aggregated.to_csv(aggregated_path, index=False)
    print(f"  wrote {aggregated_path}: {len(aggregated)} rows")


if __name__ == "__main__":
    sys.exit(main())
