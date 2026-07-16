"""
shared/__init__.py — Re-exports everything so `import shared as S` keeps working.

Sub-modules:
  tokens.py   — design tokens, COMMON_STYLES, chart-layout defaults
  data.py     — CSV loading, compute_leaderboard, pareto_mark
  charts.py   — all fig_* Plotly figure builders
  layout.py   — Dash component helpers (kpi_card, section, filter_bar, …)
"""

# Design tokens & CSS (edit tokens.py to retheme)
from .tokens import (
    EPSILON, RHO_GOOD, FAILURE_MAE,
    BG, CARD, BORDER, ACCENT, PINK, GREEN, RED, AMBER, MUTED, TEXT, TEXT2,
    LIB_COLOR, FONT, EXPECTED_COLS,
    COMMON_STYLES,
    _CHART_LAYOUT, _LEGEND_H, _MARGIN,
)

# Data helpers
from .data import load_data, try_load_data, compute_leaderboard, pareto_mark
from .data import normalize_filter_selection, filter_by_column, should_pool_dimension

# Chart builders
from .charts import (
    graph_config,
    fig_empty,
    fig_leaderboard_bars,
    fig_pareto,
    fig_distribution,
    fig_matrix,
    fig_raw_scatter,
    fig_budget_rho,
    fig_runtime_vs_budget,
    fig_metric_vs_budget,
    fig_quality_vs_cost_rq2,
    fig_budget_quality_lines,
    fig_pairwise_heatmap_rq2,
    fig_runtime_vs_features,
    fig_rho_vs_features,
    fig_cost_vs_features,
    fig_quality_vs_features,
    fig_failure_heatmap_by_features,
    fig_speed_ranking_at_nfeatures,
    fig_runtime_ranking,
    fig_runtime_boxplots,
    fig_rho_vs_runtime,
    fig_runtime_vs_complexity,
    fig_failure_vs_complexity,
    fig_rho_vs_complexity,
    fig_tree_pass_fail_matrix,
    fig_tree_failure_vs_depth,
    fig_tree_agreement_heatmap_by_model,
    fig_tree_runtime_vs_depth_faceted,
    fig_tree_depth_scaling_factor,
    fig_tree_order_cost,
    fig_tree_quality_ranking,
    fig_tree_runtime_vs_depth,
    fig_tree_quality_vs_depth,
    _depth_col,        # used by pages/rq4_trees.py as S._depth_col
    fig_hardware_comparison,
    fig_hardware_speedup,
    fig_rho_vs_runtime_by_hardware,
    fig_rq3_attribution_agreement,
    fig_rq3_runtime_comparison,
    fig_rq3_axiomatic_integrity,
    fig_rq3_scalability_wall,
    fig_rq3_topology_violations,
    fig_captum_hardware_dividends,
    fig_woodelf_depth_scaling,
)


# Layout helpers
from .layout import (
    kpi_card,
    kpi_row,
    section,
    rq_header,
    interpretation_note,
    info_content,
    provenance_line,
    warning_note,
    info_note,
    missing_data_banner,
    filter_dropdown,
    filter_checklist,
    filter_bar,
    data_summary_card,
    build_leaderboard_datatable,
    capability_matrix_table,
)
