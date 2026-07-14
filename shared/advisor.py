"""
shared/advisor.py — Library recommendation engine.

Pure logic; no Dash or Plotly imports. The advisor panel in benchmark_explorer.py
calls recommend_library() and renders the result. This module is intentionally
separate from the benchmark display code — the advisor is a navigation layer on
top of the data, not a primary feature.

To add a new path: add a branch here and add the section_id CONNECTOR target
to the relevant RQ page (via section_id= in shared.layout.section()).
"""


def recommend_library(
    model_type: str,    # "tree" | "neural_network" | "black_box"
    primary_need: str,  # "speed" | "accuracy" | "memory"
    feature_count: str, # "low" | "high"
) -> dict:
    """
    Returns:
        library      — recommended library name(s)
        approximator — recommended method
        why          — plain-English explanation
        rq_page      — path to the most relevant RQ page  (CONNECTOR: link here)
        rq_name      — human name of that page
        secondary_page / secondary_name — optional second page to consult
        chart_ids    — dict of {description: section_id} — CONNECTORS;
                       change advisor href to f'{rq_page}#{section_id}' to deep-link
        metrics      — columns worth examining in the data
    """

    # ── Neural network ────────────────────────────────────────────────────────
    if model_type == "neural_network":
        return {
            "library":        "captum",
            "approximator":   "gradient / DeepLIFT",
            "why": (
                "For neural networks, gradient-based methods (Captum) are the natural fit. "
                "They run inside the PyTorch graph — no permutation sampling overhead. "
                "Check RQ3 to see which library achieves the fastest runtime on NN models, "
                "and consult RQ5 to see how much speedup GPU acceleration provides over CPU execution."
            ),
            "rq_page":        "/rq3",
            "rq_name":        "RQ3 — Neural Networks",
            "secondary_page": "/rq5",
            "secondary_name": "RQ5 — GPU vs CPU",
            "chart_ids": {                           # CONNECTORS
                "runtime_ranking":   "rq3-runtime-section",
                "pareto":            "rq3-pareto-section",
                "speed_vs_accuracy": "rq3-scatter-section",
            },
            "metrics": ["runtime_s", "mean_sample_rho", "failure_rate"],
        }

    # ── Tree — speed first ────────────────────────────────────────────────────
    if model_type == "tree" and primary_need == "speed":
        return {
            "library":        "lightshap  or  shap (TreeSHAP)",
            "approximator":   "tree-native / permutation",
            "why": (
                "TreeSHAP computes exact Shapley values for tree models in polynomial time. "
                "lightshap adds optimisations for very large datasets. "
                "Check RQ4 to find where each library's runtime spikes or failure rate crosses 10 %."
            ),
            "rq_page":        "/rq4",
            "rq_name":        "RQ4 — Tree Models",
            "secondary_page": "/rq2",
            "secondary_name": "RQ2 — Accuracy",
            "chart_ids": {                           # CONNECTORS
                "failure_heatmap": "rq4-failure-section",
                "runtime_scaling": "rq4-runtime-section",
            },
            "metrics": ["runtime_s", "failure_rate", "mean_sample_rho"],
        }

    # ── Tree — accuracy first ─────────────────────────────────────────────────
    if model_type == "tree":
        return {
            "library":        "shap (TreeSHAP)",
            "approximator":   "exact",
            "why": (
                "TreeSHAP produces exact Shapley values with zero approximation variance. "
                "It is the industry standard for high-fidelity tree explanations. "
                "Check RQ4 to confirm it stays stable at your specific tree depth."
            ),
            "rq_page":        "/rq4",
            "rq_name":        "RQ4 — Tree Models",
            "secondary_page": "/rq2",
            "secondary_name": "RQ2 — Accuracy",
            "chart_ids": {                           # CONNECTORS
                "failure_heatmap": "rq4-failure-section",
                "quality_scaling": "rq4-quality-section",
            },
            "metrics": ["mean_sample_rho", "relative_mae", "sign_agreement"],
        }

    # ── Black-box, high-dimensional ───────────────────────────────────────────
    if feature_count == "high":
        return {
            "library":        "lightshap",
            "approximator":   "kernel",
            "why": (
                "High-dimensional black-box models punish standard KernelSHAP exponentially. "
                "lightshap's optimised kernel scales significantly better across feature counts. "
                "Check RQ1 for how each library's runtime curve steepens with n_features."
            ),
            "rq_page":        "/rq1",
            "rq_name":        "RQ1 — Dimensionality",
            "secondary_page": "/rq2",
            "secondary_name": "RQ2 — Accuracy",
            "chart_ids": {                           # CONNECTORS
                "runtime_vs_features":  "rq1-runtime-section",
                "speed_ranking_hi_dim": "rq1-speed-section",
                "failure_heatmap":      "rq1-failure-section",
            },
            "metrics": ["runtime_s at high n_features", "failure_rate", "mean_sample_rho"],
        }

    # ── Black-box, accuracy first ─────────────────────────────────────────────
    if primary_need == "accuracy":
        return {
            "library":        "shapiq",
            "approximator":   "kernel",
            "why": (
                "shapiq's kernel approximator consistently achieves the highest quality scores. "
                "Its interaction-aware sampling improves accuracy even at low budgets. "
                "Use RQ2 to find the minimum budget where quality plateaus."
            ),
            "rq_page":        "/rq2",
            "rq_name":        "RQ2 — Accuracy",
            "secondary_page": "/rq1",
            "secondary_name": "RQ1 — Dimensionality",
            "chart_ids": {                           # CONNECTORS
                "pareto":         "rq2-pareto-section",
                "budget_quality": "rq2-budget-section",
                "distribution":   "rq2-distribution-section",
            },
            "metrics": ["mean_sample_rho", "sign_agreement", "mean_sample_rho"],
        }

    # ── Black-box, speed / scale, low-dim (default) ───────────────────────────
    return {
        "library":        "lightshap",
        "approximator":   "permutation or kernel",
        "why": (
            "For fast model-agnostic approximations on low-dimensional data, "
            "lightshap tops the runtime ranking while staying in the reliable quality zone. "
            "Compare against shap/KernelSHAP in RQ2 to confirm the trade-off is acceptable."
        ),
        "rq_page":        "/rq2",
        "rq_name":        "RQ2 — Accuracy",
        "secondary_page": "/rq1",
        "secondary_name": "RQ1 — Dimensionality",
        "chart_ids": {                               # CONNECTORS
            "pareto":         "rq2-pareto-section",
            "ranking":        "rq2-ranking-section",
            "budget_quality": "rq2-budget-section",
        },
        "metrics": ["runtime_s", "mean_sample_rho"],
    }
