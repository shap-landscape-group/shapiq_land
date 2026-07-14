import os
import json
import pandas as pd
import numpy as np

def extract_nn_metrics(row):
    metrics_str = row.get("pairwise_metrics")
    if pd.isna(metrics_str):
        return pd.Series([np.nan, np.nan, np.nan], index=["mean_sample_rho", "relative_mae", "sign_agreement"])
    try:
        metrics_json = json.loads(metrics_str)
        approximator = row.get("approximator")
        target_key = None
        for k in metrics_json.keys():
            parts = k.split("|")
            if len(parts) >= 2 and parts[1] == approximator:
                target_key = k
                break
        
        if target_key and target_key in metrics_json:
            m = metrics_json[target_key]
            return pd.Series([
                m.get("mean_sample_rho"),
                m.get("relative_mae"),
                m.get("sign_agreement")
            ], index=["mean_sample_rho", "relative_mae", "sign_agreement"])
    except Exception:
        pass
    return pd.Series([np.nan, np.nan, np.nan], index=["mean_sample_rho", "relative_mae", "sign_agreement"])

def extract_tree_metrics(row):
    metrics_str = row.get("pairwise_metrics")
    if pd.isna(metrics_str):
        return pd.Series([np.nan, np.nan, np.nan], index=["mean_sample_rho", "relative_mae", "sign_agreement"])
    try:
        metrics_json = json.loads(metrics_str)
        backend = row.get("backend")
        if backend in metrics_json:
            m = metrics_json[backend]
            return pd.Series([
                m.get("mean_sample_rho"),
                m.get("relative_mae"),
                m.get("sign_agreement")
            ], index=["mean_sample_rho", "relative_mae", "sign_agreement"])
    except Exception:
        pass
    return pd.Series([np.nan, np.nan, np.nan], index=["mean_sample_rho", "relative_mae", "sign_agreement"])

def main():
    print("Converting RQ5 results...")
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    
    # Files
    nn_path = os.path.join(project_root, "results", "rq5_model_gpu-vs-cpu-shapiq-shap.csv")
    tree_path = os.path.join(project_root, "results", "rq5_tree-gpu-vs-cpu.csv")
    
    out_dir = os.path.join(project_root, "results", "converted")
    os.makedirs(out_dir, exist_ok=True)
    
    aggregated_out = os.path.join(out_dir, "rq5_gpu_cpu_comparison_aggregated.csv")
    by_seed_out = os.path.join(out_dir, "rq5_gpu_cpu_comparison_by_seed.csv")
    
    # 1. Load and process NN data
    print(f"Loading NN data from {nn_path}...")
    df_nn = pd.read_csv(nn_path)
    
    print("Extracting metrics for NN...")
    metrics_nn = df_nn.apply(extract_nn_metrics, axis=1)
    df_nn = pd.concat([df_nn, metrics_nn], axis=1)
    
    # 2. Load and process Tree data
    print(f"Loading Tree data from {tree_path}...")
    df_tree = pd.read_csv(tree_path)
    
    # Standardize Tree columns
    if "approximator" in df_tree.columns:
        df_tree["approximator"] = df_tree["approximator"].fillna(df_tree["backend"].str.replace("woodelf_", "", regex=False))
    else:
        df_tree["approximator"] = df_tree["backend"].str.replace("woodelf_", "", regex=False)
        
    print("Extracting metrics for Tree...")
    metrics_tree = df_tree.apply(extract_tree_metrics, axis=1)
    df_tree = pd.concat([df_tree, metrics_tree], axis=1)
    
    # Combine datasets
    columns_to_keep = [
        "dataset", "model", "library", "approximator", "device", "seed",
        "runtime_s", "relative_additivity_gap", "mean_sample_rho", "relative_mae", "sign_agreement"
    ]
    
    # Clean devices to be uniform (cpu/gpu)
    df_nn["device"] = df_nn["device"].astype(str).str.lower().replace({"cuda": "gpu"})
    df_tree["device"] = df_tree["device"].astype(str).str.lower().replace({"cuda": "gpu"})
    
    df_nn_clean = df_nn[[c for c in columns_to_keep if c in df_nn.columns]].copy()
    df_tree_clean = df_tree[[c for c in columns_to_keep if c in df_tree.columns]].copy()
    
    # Fill in missing columns if any
    for col in columns_to_keep:
        if col not in df_nn_clean.columns:
            df_nn_clean[col] = np.nan
        if col not in df_tree_clean.columns:
            df_tree_clean[col] = np.nan
            
    df_combined = pd.concat([df_nn_clean[columns_to_keep], df_tree_clean[columns_to_keep]], ignore_index=True)
    
    # Create canonical method label
    df_combined["method"] = df_combined["library"] + " / " + df_combined["approximator"]
    
    # Save seed-level file
    print(f"Saving seed-level runs to {by_seed_out}...")
    df_combined.to_csv(by_seed_out, index=False)
    
    # Aggregate across seeds
    print("Aggregating across seeds...")
    group_cols = ["dataset", "model", "library", "approximator", "method", "device"]
    
    df_agg = (
        df_combined.groupby(group_cols)
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
