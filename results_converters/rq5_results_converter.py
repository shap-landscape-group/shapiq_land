import os
import json
import pandas as pd
import numpy as np


def extract_metrics(row):
    """Extract this row's quality metrics vs. the reference method.

    Each row's `pairwise_metrics` cell holds a dict keyed by every method run in
    that (dataset, model, seed) group, e.g. "captum_approx|gradient_shap|512@cuda".
    A row's entry for its OWN key is always a trivial self-comparison (diff=0,
    rho=1), so it can't be used directly. Instead — matching the convention used
    across the rest of the app (shared/data.py:_extract_vs_reference and
    rq3_results_converter.py) — prefer the "*_true_value" reference key if present,
    otherwise fall back to the first key whose relative_mae isn't 0/None.
    """
    metrics_str = row.get("pairwise_metrics")
    if pd.isna(metrics_str):
        return pd.Series([np.nan, np.nan, np.nan], index=["mean_sample_rho", "relative_mae", "sign_agreement"])
    try:
        clean = metrics_str.replace(": NaN", ": null").replace(":NaN", ":null")
        metrics_json = json.loads(clean)
        for key, vals in metrics_json.items():
            if "true_value" in key:
                return pd.Series([
                    vals.get("mean_sample_rho"), vals.get("relative_mae"), vals.get("sign_agreement")
                ], index=["mean_sample_rho", "relative_mae", "sign_agreement"])
        for key, vals in metrics_json.items():
            if vals.get("relative_mae") not in (0.0, None):
                return pd.Series([
                    vals.get("mean_sample_rho"), vals.get("relative_mae"), vals.get("sign_agreement")
                ], index=["mean_sample_rho", "relative_mae", "sign_agreement"])
    except Exception:
        pass
    return pd.Series([np.nan, np.nan, np.nan], index=["mean_sample_rho", "relative_mae", "sign_agreement"])


def main():
    print("Converting RQ5 results...")
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)

    raw_path = os.path.join(project_root, "results", "RQ5_gpu_vs_cpu.csv")

    out_dir = os.path.join(project_root, "results", "converted")
    os.makedirs(out_dir, exist_ok=True)

    aggregated_out = os.path.join(out_dir, "rq5_gpu_cpu_comparison_aggregated.csv")
    by_seed_out = os.path.join(out_dir, "rq5_gpu_cpu_comparison_by_seed.csv")

    print(f"Loading data from {raw_path}...")
    df = pd.read_csv(raw_path)

    print("Extracting pairwise metrics...")
    metrics = df.apply(extract_metrics, axis=1)
    df = pd.concat([df, metrics], axis=1)

    # Clean devices to be uniform (cpu/gpu)
    df["device"] = df["device"].astype(str).str.lower().replace({"cuda": "gpu"})

    columns_to_keep = [
        "dataset", "model", "library", "approximator", "device", "seed", "n_features",
        "runtime_s", "relative_additivity_gap", "mean_sample_rho", "relative_mae", "sign_agreement"
    ]
    df_clean = df[[c for c in columns_to_keep if c in df.columns]].copy()
    for col in columns_to_keep:
        if col not in df_clean.columns:
            df_clean[col] = np.nan
    df_combined = df_clean[columns_to_keep].copy()

    # Create canonical method label
    df_combined["method"] = df_combined["library"] + " / " + df_combined["approximator"]
    # Display-only rename: "shapiq_proxy / proxy" reads as a variant of shapiq_proxy
    # rather than shapiq's proxy-based SHAP method — relabel for clarity.
    df_combined.loc[df_combined["method"] == "shapiq_proxy / proxy", "method"] = "shapiq / proxyShap"

    # Save seed-level file
    print(f"Saving seed-level runs to {by_seed_out}...")
    df_combined.to_csv(by_seed_out, index=False)

    # Aggregate across seeds
    print("Aggregating across seeds...")
    group_cols = ["dataset", "model", "library", "approximator", "method", "device", "n_features"]

    df_agg = (
        df_combined.groupby(group_cols, dropna=False)
        .agg(
            runtime_median=("runtime_s", "median"),
            runtime_q25=("runtime_s", lambda x: x.quantile(0.25)),
            runtime_q75=("runtime_s", lambda x: x.quantile(0.75)),
            gap_mean=("relative_additivity_gap", "mean"),
            gap_q25=("relative_additivity_gap", lambda x: x.quantile(0.25)),
            gap_q75=("relative_additivity_gap", lambda x: x.quantile(0.75)),
            rho_median=("mean_sample_rho", "median"),
            rho_q25=("mean_sample_rho", lambda x: x.quantile(0.25)),
            rho_q75=("mean_sample_rho", lambda x: x.quantile(0.75)),
            relative_mae_median=("relative_mae", "median"),
            relative_mae_q25=("relative_mae", lambda x: x.quantile(0.25)),
            relative_mae_q75=("relative_mae", lambda x: x.quantile(0.75)),
            sign_agreement_median=("sign_agreement", "median"),
            seed_count=("seed", "count")
        )
        .reset_index()
    )

    print(f"Saving aggregated runs to {aggregated_out}...")
    df_agg.to_csv(aggregated_out, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
