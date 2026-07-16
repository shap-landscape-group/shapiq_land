"""
shared/layout.py — Reusable Dash layout components and filter helpers.
Everything that builds html.Div / dcc.* components lives here.
"""
from dash import dash_table, dcc, html

from .tokens import (
    BG, CARD, BORDER, ACCENT, PINK, GREEN, RED, AMBER, TEXT, TEXT2, FONT,
)


# ── KPI cards ─────────────────────────────────────────────────────────────────

def kpi_card(value: str, label: str, color: str | None = None) -> html.Div:
    accent = color or ACCENT
    return html.Div(
        [
            html.Div(style={"height": "3px", "background": accent,
                            "borderRadius": "3px 3px 0 0", "margin": "-16px -20px 14px"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "700",
                                   "color": accent, "lineHeight": "1",
                                   "letterSpacing": "-0.02em"}),
            html.Div(label, style={"fontSize": "10px", "color": TEXT2, "marginTop": "6px",
                                   "fontWeight": "500", "textTransform": "uppercase",
                                   "letterSpacing": "0.06em", "lineHeight": "1.4"}),
        ],
        style={
            "flex": "1", "minWidth": "140px", "background": CARD,
            "border": f"1px solid {BORDER}", "borderRadius": "10px",
            "padding": "16px 20px", "boxShadow": "0 1px 4px rgba(26,32,64,0.07)",
            "overflow": "hidden",
        },
    )


def kpi_row(*cards) -> html.Div:
    return html.Div(list(cards), style={
        "display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "20px",
    })


# ── Section wrapper ────────────────────────────────────────────────────────────

def provenance_line(*parts: str) -> str:
    """Join provenance fragments for the info box footer."""
    return "  ·  ".join(parts)


def info_content(primary, secondary: str = "", provenance: str = "") -> html.Div:
    """Info body for a section's collapsible blue note."""
    primary_node = (primary if not isinstance(primary, str)
                    else html.Div(primary, style={
                        "fontSize": "12px", "color": TEXT, "lineHeight": "1.6",
                        "marginBottom": "4px" if (secondary or provenance) else "0",
                    }))
    children = [primary_node]
    if secondary:
        children.append(html.Div(secondary, style={
            "fontSize": "11px", "color": TEXT2, "lineHeight": "1.55",
            "marginBottom": "4px" if provenance else "0",
        }))
    if provenance:
        children.append(html.Div(provenance, style={
            "fontSize": "10px", "color": TEXT2, "fontFamily": "monospace",
            "lineHeight": "1.55", "letterSpacing": "0.01em",
        }))
    return html.Div(children)


def section(title, subtitle=None, children=None,
            section_id: str | None = None,
            info_id: str | None = None) -> html.Div:
    """
    Wraps a chart or table with an editorial header and card border.

    section_id: stable DOM id — used as a CONNECTOR target by the advisor panel.
    To deep-link from the advisor, change the advisor href to f'{rq_page}#{section_id}'.
    """
    title_node = (
        html.H3(title, style={"margin": "0", "fontSize": "15px",
                              "fontWeight": "600", "letterSpacing": "-0.01em",
                              "color": TEXT})
        if isinstance(title, str) else title
    )
    _info_icon = html.Span(
        "i",
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "justifyContent": "center",
            "width": "16px",
            "height": "16px",
            "borderRadius": "999px",
            "border": f"1px solid {ACCENT}",
            "color": ACCENT,
            "fontSize": "11px",
            "fontWeight": "700",
            "lineHeight": "16px",
            "textAlign": "center",
            "paddingTop": "0",
            "flexShrink": "0",
        },
    )
    _gradient = html.Div(style={
        "height": "2px", "width": "28px", "marginTop": "6px",
        "background": f"linear-gradient(90deg, {ACCENT}, {PINK})",
        "borderRadius": "2px",
    })
    _info_box_style = {
        "marginTop": "8px",
        "background": "#F0F4FF",
        "border": f"1px solid {BORDER}",
        "borderRadius": "6px",
        "padding": "9px 13px",
    }

    if children is None:
        children = []

    def _has_subtitle(value) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value)
        return True  # Dash components are falsy when empty — still valid info slots

    if info_id:
        info_body = html.Div(id=info_id)
    elif _has_subtitle(subtitle):
        info_body = (subtitle if not isinstance(subtitle, str)
                     else html.Div(subtitle, style={
                         "fontSize": "12px", "color": TEXT2, "lineHeight": "1.6",
                     }))
    else:
        info_body = None

    if info_body is not None:
        header = html.Div([
            html.Details([
                html.Summary(
                    [title_node, _info_icon],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "8px",
                        "cursor": "pointer",
                        "listStyle": "none",
                    },
                ),
                html.Div(info_body, style=_info_box_style),
            ]),
            _gradient,
        ], style={"marginBottom": "8px"})
    else:
        header = html.Div([title_node, _gradient], style={"marginBottom": "8px"})

    return html.Div(
        [
            header,
            html.Div(children, style={
                "background": CARD, "borderRadius": "10px",
                "border": f"1px solid {BORDER}", "overflow": "hidden",
            }),
        ],
        **({"id": section_id} if section_id is not None else {}),
        style={"marginBottom": "28px"},
    )


# ── Page-level header ─────────────────────────────────────────────────────────

def rq_header(rq: str, title: str, question: str) -> html.Div:
    return html.Div([
        html.Div(
            [
                html.Span(rq, style={
                    "fontSize": "15px", "fontWeight": "700", "color": ACCENT,
                    "textTransform": "uppercase", "letterSpacing": "0.1em",
                    "background": "#E8EDFA", "padding": "3px 9px",
                    "borderRadius": "4px", "marginRight": "10px",
                }),
                html.Span(title, style={"fontWeight": "700", "fontSize": "20px",
                                        "letterSpacing": "-0.02em", "color": TEXT}),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "10px"},
        ),
        html.Div(
            html.Em(f'"{question}"'),
            style={"fontSize": "18px", "color": TEXT2,
                   "lineHeight": "1.7", "marginBottom": "24px",
                   "borderLeft": f"3px solid {PINK}",
                   "paddingLeft": "12px", "fontStyle": "italic"},
        ),
    ])


# ── Contextual notes ──────────────────────────────────────────────────────────

def interpretation_note(text: str) -> html.Div:
    return html.Div(
        html.Em(text, style={"fontSize": "12px", "color": TEXT2, "lineHeight": "1.7"}),
        style={"marginTop": "8px", "marginBottom": "24px"},
    )


def warning_note(message: str) -> html.Div:
    return html.Div(
        [html.Span("⚠ ", style={"fontWeight": "700"}), message],
        style={
            "background": "#FFFBEB", "border": f"1px solid {AMBER}",
            "borderRadius": "8px", "padding": "10px 16px",
            "fontSize": "13px", "color": "#92400E", "marginBottom": "16px",
        },
    )


def info_note(message) -> html.Div:
    if isinstance(message, list):
        children = [html.Span("ℹ ", style={"fontWeight": "700", "color": ACCENT})] + message
    else:
        children = [html.Span("ℹ ", style={"fontWeight": "700", "color": ACCENT}), message]
    return html.Div(
        children,
        style={
            "background": "#EFF6FF", "border": f"1px solid {ACCENT}",
            "borderRadius": "8px", "padding": "10px 16px",
            "fontSize": "13px", "color": "#1E3A8A",
            "marginBottom": "16px", "lineHeight": "1.7",
        },
    )


def missing_data_banner(expected_csv: str) -> html.Div:
    """Shown when the RQ-specific CSV has not been created yet."""
    return html.Div(
        [
            html.Div("📂", style={"fontSize": "36px", "marginBottom": "12px"}),
            html.Div("Data not yet collected",
                     style={"fontSize": "16px", "fontWeight": "600",
                            "color": TEXT, "marginBottom": "8px"}),
            html.Div(
                ["Expected CSV: ",
                 html.Code(expected_csv,
                           style={"background": BG, "padding": "2px 6px",
                                  "borderRadius": "4px", "fontSize": "12px",
                                  "fontFamily": "monospace"})],
                style={"fontSize": "13px", "color": TEXT2,
                       "lineHeight": "1.7", "marginBottom": "12px"},
            ),
            html.Div(
                "Place the file in the same directory as benchmark_explorer.py. "
                "Charts will populate automatically on the next page load.",
                style={"fontSize": "12px", "color": TEXT2, "lineHeight": "1.7"},
            ),
        ],
        style={
            "textAlign": "center", "padding": "56px 32px",
            "background": CARD, "borderRadius": "12px",
            "border": f"2px dashed {BORDER}", "marginBottom": "24px",
        },
    )


# ── Filter controls ───────────────────────────────────────────────────────────

def _dd_label(label: str) -> html.Div:
    return html.Div(label, style={
        "fontSize": "10px", "fontWeight": "600", "color": TEXT2,
        "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "4px",
    })


def filter_dropdown(label: str, component_id: str, options: list,
                    value, width: str = "190px") -> html.Div:
    return html.Div(
        [_dd_label(label),
         dcc.Dropdown(id=component_id, options=options, value=value, clearable=False,
                      style={"width": width, "fontSize": "13px"})],
        style={"marginRight": "14px"},
    )


def filter_checklist(label: str, component_id: str,
                     options: list, value: list) -> html.Div:
    return html.Div(
        [_dd_label(label),
         dcc.Checklist(id=component_id, options=options, value=value, inline=True,
                       inputStyle={"marginRight": "4px"},
                       labelStyle={"marginRight": "12px", "fontSize": "13px",
                                   "cursor": "pointer"})],
    )


def filter_bar(*controls) -> html.Div:
    return html.Div(
        list(controls),
        style={
            "display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "8px",
            "background": CARD, "borderRadius": "12px",
            "border": f"1px solid {BORDER}", "padding": "14px 20px",
            "marginBottom": "20px",
        },
    )


# ── Data summary card ─────────────────────────────────────────────────────────

def data_summary_card(df) -> html.Div:
    """Compact data-passport shown below the RQ header.

    Displays row count, categorical column values, and numeric column ranges
    so you can see at a glance what is in the loaded CSV without opening it.
    """
    if df is None or df.empty:
        return html.Div()

    def _pill(text: str, bg: str = BG, color: str = TEXT2,
              border_color: str | None = None) -> html.Span:
        bc = border_color or BORDER
        return html.Span(text, style={
            "display": "inline-block", "background": bg, "color": color,
            "fontSize": "11px", "fontWeight": "500",
            "padding": "2px 9px", "borderRadius": "4px",
            "marginRight": "5px", "marginBottom": "4px",
            "border": f"1px solid {bc}",
            "whiteSpace": "nowrap",
        })

    pills: list = [_pill(f"{len(df):,} rows", "#EEF2FF", ACCENT, ACCENT + "60")]

    for col in ("library", "approximator", "dataset", "model"):
        if col in df.columns:
            vals = sorted(str(v) for v in df[col].dropna().unique())
            if vals:
                pills.append(_pill(f"{col}: {', '.join(vals)}"))

    for col in ("n_features", "budget", "runtime_s", "mean_sample_rho", "relative_mae"):
        if col in df.columns and df[col].notna().any():
            lo, hi = df[col].min(), df[col].max()
            pills.append(_pill(f"{col}: {lo:.3g} – {hi:.3g}"))

    return html.Div(
        [html.Span("Data: ", style={"fontSize": "11px", "fontWeight": "700",
                                    "color": TEXT2, "marginRight": "4px"}),
         *pills],
        style={"marginBottom": "14px", "lineHeight": "2.0"},
    )


# ── Datatable ─────────────────────────────────────────────────────────────────

def build_leaderboard_datatable(lb, table_id: str = "lb-table") -> dash_table.DataTable:
    display = lb[[
        "rank", "method", "rho_median", "mae_median",
        "sign_median", "runtime_median", "failure_rate", "n_runs",
    ]].copy()
    display["rho_median"]     = display["rho_median"].round(3)
    display["mae_median"]     = display["mae_median"].round(4)
    display["runtime_median"] = display["runtime_median"].round(3)
    display["sign_median"]    = display["sign_median"].round(3)
    display["failure_rate"]   = (display["failure_rate"] * 100).round(1)

    return dash_table.DataTable(
        id=table_id,
        data=display.to_dict("records"),
        columns=[
            {"name": "#",           "id": "rank"},
            {"name": "Method",      "id": "method"},
            {"name": "Median ρ",    "id": "rho_median"},
            {"name": "Rel. MAE",    "id": "mae_median"},
            {"name": "Sign agr.",   "id": "sign_median"},
            {"name": "Runtime (s)", "id": "runtime_median"},
            {"name": "Failure %",   "id": "failure_rate"},
            {"name": "Runs",        "id": "n_runs"},
        ],
        sort_action="native",
        page_size=25,
        style_table={"overflowX": "auto"},
        style_header={
            "background": BG, "color": TEXT2, "fontWeight": "600",
            "fontSize": "11px", "textTransform": "uppercase",
            "letterSpacing": "0.05em", "border": "none",
            "borderBottom": f"1px solid {BORDER}",
            "padding": "10px 14px", "fontFamily": FONT,
        },
        style_cell={
            "fontFamily": FONT, "fontSize": "13px", "padding": "10px 14px",
            "border": "none", "borderBottom": f"1px solid {BORDER}",
            "color": TEXT, "background": CARD,
        },
        style_data_conditional=[
            {"if": {"row_index": 0}, "background": "#EEF2FF", "fontWeight": "600"},
            {"if": {"column_id": "failure_rate",
                    "filter_query": "{failure_rate} > 20"}, "color": RED},
            {"if": {"column_id": "rho_median",
                    "filter_query": "{rho_median} >= 0.9"}, "color": GREEN, "fontWeight": "600"},
        ],
    )


# ── Capability matrix ─────────────────────────────────────────────────────────

def capability_matrix_table(benchmarked_libs: set) -> html.Div:
    """Static capability overview — update rows_data when adding new libraries."""
    th_s = {
        "fontSize": "10px", "fontWeight": "600", "color": TEXT2,
        "textTransform": "uppercase", "letterSpacing": "0.05em",
        "padding": "8px 12px", "borderBottom": f"2px solid {BORDER}",
        "textAlign": "left", "background": BG,
    }

    def td_s(extra=None):
        base = {"fontSize": "12px", "padding": "8px 12px",
                "borderBottom": f"1px solid {BORDER}"}
        return {**base, **(extra or {})}

    def yn(v):
        if v == "yes":
            return html.Span("✓", style={"color": GREEN, "fontWeight": "700"})
        if v == "no":
            return html.Span("✗", style={"color": RED})
        return html.Span(v, style={"color": TEXT2, "fontSize": "11px"})

    def badge(lib):
        if lib in benchmarked_libs:
            return html.Span("benchmarked", style={
                "background": "#D1FAE5", "color": "#065F46",
                "borderRadius": "4px", "padding": "1px 6px",
                "fontSize": "10px", "fontWeight": "600",
            })
        return html.Span("planned", style={
            "background": "#FEF3C7", "color": "#92400E",
            "borderRadius": "4px", "padding": "1px 6px",
            "fontSize": "10px", "fontWeight": "600",
        })

    rows_data = [
        ("shapiq",      "yes", "yes",           "yes",     "no",      "no",  "no",  "Main benchmark focus; interaction-aware"),
        ("shap",        "yes", "limited",        "yes",     "partial", "no",  "yes", "Model-specific & agnostic variants; TreeSHAP"),
        ("lightshap",   "yes", "no",             "yes",     "no",      "no",  "no",  "Speed-oriented approximation"),
        ("dalex",       "yes", "no",             "yes",     "no",      "no",  "no",  "Model-agnostic, R-inspired"),
        ("captum",      "yes", "not main focus", "partial", "yes",     "yes", "no",  "PyTorch focus; many gradient methods"),
        ("alibi",       "yes", "not main focus", "yes",     "partial", "no",  "no",  "Planned / not yet benchmarked"),
        ("shapleyflow", "no",  "different def.", "no",      "no",      "no",  "no",  "Requires graph structure"),
    ]

    cols = ["Library", "Status", "Feature attr.", "Interactions",
            "Model-agnostic", "NN support", "NN focus", "Tree opt.", "Notes"]
    thead = html.Thead(html.Tr([html.Th(c, style=th_s) for c in cols]))

    tbody_rows = []
    for row in rows_data:
        lib, feat, inter, agnostic, nn_sup, nn_focus, tree_opt, notes = row
        tbody_rows.append(html.Tr([
            html.Td(lib,           style=td_s({"fontFamily": "monospace",
                                               "color": ACCENT, "fontWeight": "600"})),
            html.Td(badge(lib),    style=td_s()),
            html.Td(yn(feat),      style=td_s()),
            html.Td(yn(inter),     style=td_s()),
            html.Td(yn(agnostic),  style=td_s()),
            html.Td(yn(nn_sup),    style=td_s()),
            html.Td(yn(nn_focus),  style=td_s()),
            html.Td(yn(tree_opt),  style=td_s()),
            html.Td(notes,         style=td_s({"color": TEXT2, "fontSize": "11px"})),
        ]))

    return html.Div(
        html.Table(
            [thead, html.Tbody(tbody_rows)],
            style={"width": "100%", "borderCollapse": "collapse"},
        ),
        style={
            "background": CARD, "borderRadius": "12px",
            "border": f"1px solid {BORDER}", "overflowX": "auto",
        },
    )
