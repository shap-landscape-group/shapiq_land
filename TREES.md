Visualization:

Tab 1 - Path-dependent

Backends: shap_tree_path_dependent, shapiq_tree_path_dependent, woodelf_path_dependent, fasttreeshap_path_dependent

1. Cross-library agreement heatmap — cells = mean mean_abs_diff (or sign_agreement) between each pair of the 4 libraries; one heatmap per model family, model as small-multiple columns. No dataset facet (aggregate across dataset/depth/seed) since this is a correctness check. Answers: do the exact methods agree with each other.
2. Runtime vs. max_depth — x=max_depth, y=runtime_s (median line, band across the 10 seeds), color=library, rows=dataset, columns=model family (xgboost/lightgbm/random_forest). Answers: the core depth-cost sweep.
3. Runtime vs. n_features (log-log probably) — x=n_features, y=runtime_s, color=dataset (overlaid, not faceted), separate panel per library. Model family as filter. Answers: cross-dataset scaling per library.

Tab 2 - Interventional

Backends: shap_true_value (ignore it!!!), woodelf_interventional, shapiq_tree_interventional

1. Runtime vs. max_depth — same layout as Tab 1 plot 2, this backend set. Cost of interventional methods vs. depth.
2. Runtime vs. n_features (log-log probably) — same layout as Tab 1 plot 3, this backend set.

Tab 3 - Interactions

Backends: shap_interaction, shapiq_interaction, woodelf_interaction, fasttreeshap_interaction order=2

1. Cross-library agreement heatmap — same pattern as Tab 1 plot 1, order=2 pairwise values. Correctness check for interactions.
2. Runtime vs. n_features (log-log probably, capped at interaction_max_features) — x=n_features, y=runtime_s, color=library, rows=dataset. The quadratic-blowup chart — the main reason the cap exists.
3. Runtime vs. max_depth — x=max_depth, y=runtime_s, color=library, rows=dataset, model family as filter.


benchmark:
  # Single source of truth for all randomness (data subsampling, model
  # training, any stochastic approximator). Required — read unconditionally.
  seed: [0,1,2,3,4,5,6,7,8,9]
  # Shared value function ("marginal" is the only supported value). Tree
  # backends use their own path-dependent/interventional distinction (see
  # tree_modes below) instead; kept here for parity with the other RQ configs.
  imputer: marginal
  # Reference background set for the imputer; cost scales with this.
  n_background: 100
  # Instances explained per (model, dataset) cell; null = explain every row.
  n_eval: 10
  # Wall-clock budget per (library, model, depth) true-value backend run; a
  # backend exceeding this logs a [SKIP] and gets an all-NaN row instead of
  # stalling the whole sweep. Deep trees (see models.*.max_depth below) can
  # blow up tree-explainer runtime, hence the cap. null = no timeout.
  backend_timeout_s: 600
  # Model-agnostic approximation sweep is disabled here, and there's no
  # `true_backends` key either, so only the tree-specific backends below run.
  approx_backends: []
  approximators: []
  budgets: []
  # Tree-specific true-value backends, only run for tree models.
  tree_libraries: [shap_tree, shapiq_tree, woodelf, fasttreeshap]
  tree_modes: [path_dependent, interventional]
  # Pairwise (order-2) interactions, path-dependent only. "shap_tree" is
  # implicit/always-on (the oracle). interaction_max_features caps d**2 output
  # columns (quadratic — raise with care).
  interaction_libraries: [shapiq_tree, woodelf, fasttreeshap]
  interaction_max_features: 32

# max_depth below is the cap handed to the trainer (max *allowed* depth). In
# the results CSV, the max_depth column reports the depth the fitted model
# actually reached; the cap is recorded separately as max_depth_config.
models:
  xgboost:
    n_estimators: [100]
    max_depth: [4, 8, 12, 15, 20, 50, 80]
    learning_rate: [0.1]

  lightgbm:
    n_estimators: [100]
    max_depth: [4, 8, 12, 15, 20, 50, 80]
    learning_rate: [0.1]

  random_forest:
    n_estimators: [100]
    max_depth: [4, 8, 12, 15, 20, 50, 80]

datasets:
  gisette:
    n_features: [1000]
    n_samples: [1000]

  ames_housing:
    n_features: [79]
    n_samples: [1000]

  bike:
    n_features: [12]
    n_samples: [1000]

  diabetes_130:
    n_features: [47]
    n_samples: [1000]

  covertype:
    n_features: [54]
    n_samples: [1000]