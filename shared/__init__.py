"""
shared/__init__.py — Re-exports everything so `import shared as S` keeps working.

Sub-modules:
  tokens.py   — design tokens, COMMON_STYLES, chart-layout defaults
  data.py     — CSV loading, compute_leaderboard, pareto_mark
  charts.py   — all fig_* Plotly figure builders
  layout.py   — Dash component helpers (kpi_card, section, filter_bar, …)
  advisor.py  — recommend_library() (pure logic, no Dash imports)
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

# Chart builders
from .charts import (
    fig_empty,
    fig_leaderboard_bars,
    fig_pareto,
    fig_distribution,
    fig_matrix,
    fig_raw_scatter,
    fig_budget_rho,
    fig_runtime_vs_budget,
    fig_metric_vs_budget,
    fig_runtime_vs_features,
    fig_rho_vs_features,
    fig_failure_heatmap_by_features,
    fig_speed_ranking_at_nfeatures,
    fig_runtime_ranking,
    fig_runtime_boxplots,
    fig_rho_vs_runtime,
    fig_runtime_vs_complexity,
    fig_failure_vs_complexity,
    fig_rho_vs_complexity,
    _complexity_col,   # used by pages/rq4_trees.py as S._complexity_col
)

# Layout helpers
from .layout import (
    kpi_card,
    kpi_row,
    section,
    rq_header,
    interpretation_note,
    warning_note,
    info_note,
    missing_data_banner,
    filter_dropdown,
    filter_checklist,
    filter_bar,
    data_summary_card,
    charts_toc,
    build_leaderboard_datatable,
    capability_matrix_table,
)

# Advisor recommendation engine
from .advisor import recommend_library
