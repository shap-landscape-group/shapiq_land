"""
shared/tokens.py — Design tokens, CSS, and chart-layout defaults.
Edit this file to retheme the entire app.
"""

# ── Quality metric constants ──────────────────────────────────────────────────
EPSILON = 1e-10
FAILURE_MAE = 1.0
RHO_GOOD = 0.9    # reference line: "good" Spearman ρ

# ── SHAP-IQ brand palette ─────────────────────────────────────────────────────
BG = "#F5F5F5"
CARD = "#FFFFFF"
BORDER = "#E2E2E2"
ACCENT = "#4B6DD4"   # logo blue
PINK = "#E84060"   # logo pink-red
GREEN = "#10B981"
RED = "#EF4444"
AMBER = "#F59E0B"
MUTED = "#B4C2DF"
TEXT = "#1A2040"   # dark navy
TEXT2 = "#5A6A8A"

LIB_COLOR: dict[str, str] = {
    "shapiq":       PINK,
    "shapiq_proxy": PINK,        # same pink-red — ShapIQ family
    "shap":         ACCENT,
    "shap_nn":      ACCENT,      # shap NN backend — same blue family
    "lightshap":    "#6BA3E8",
    "dalex":        AMBER,
    "captum":       "#7C3AED",
    "alibi":        "#06B6D4",
    "woodelf":      AMBER,
    "fasttreeshap": "#0E9488",
}


FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, sans-serif"

# ── Expected CSV columns ──────────────────────────────────────────────────────
EXPECTED_COLS: list[str] = [
    "dataset", "model", "n_features", "n_samples", "backend", "library",
    "computation_type", "approximator", "budget", "n_eval", "runtime_s",
    "n_model_evals", "mean_abs_diff", "relative_mae", "sign_agreement",
    "mean_sample_rho", "reference_backend",
]

# ── Plotly chart layout defaults ──────────────────────────────────────────────
_CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family=FONT, color=TEXT, size=12),
    plot_bgcolor="#F8FAFF",
    paper_bgcolor=CARD,
)
_LEGEND_H = dict(
    orientation="h", yanchor="bottom", y=1.02,
    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11),
)
_MARGIN = dict(l=55, r=16, t=36, b=48)

# ── Global CSS (injected via app.index_string) ────────────────────────────────
COMMON_STYLES = f"""
*, *::before, *::after {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; height: 100%; font-family: {FONT}; color: {TEXT}; }}

/* ── Tab component ───────────────────────────────────────────────────────── */
.tab {{
    font-family: {FONT}; font-size: 13px; font-weight: 500;
    color: {TEXT2}; border: none !important;
    background: transparent !important; padding: 10px 18px;
}}
.tab--selected {{
    color: {ACCENT} !important; font-weight: 600 !important;
    border-bottom: 2px solid {ACCENT} !important;
}}
.tabs--content {{ border: none !important; }}

/* ── Layout shell ────────────────────────────────────────────────────────── */
.page-wrapper {{ display: flex; height: 100vh; overflow: hidden; background: {BG}; }}
.app-shell {{ display: flex; flex: 1; min-width: 0; overflow: hidden; }}

/* ── Advisor strip (always-visible left edge) ────────────────────────────── */
.advisor-strip {{
    width: 26px; flex-shrink: 0; cursor: pointer; user-select: none;
    background: linear-gradient(180deg, {ACCENT} 0%, #7C3AED 50%, {PINK} 100%);
    display: flex; align-items: center; justify-content: center;
    transition: opacity 0.15s; z-index: 10;
    border: none; outline: none; padding: 0; border-radius: 0;
}}
.advisor-strip:hover {{ opacity: 0.85; }}
.advisor-strip-label {{
    writing-mode: vertical-rl; transform: rotate(180deg);
    color: white; font-size: 9px; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    white-space: nowrap; pointer-events: none;
}}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
.sidebar {{
    width: 260px; flex-shrink: 0; background: {TEXT};
    display: flex; flex-direction: column;
    overflow: hidden; transition: width 0.22s ease;
    border-right: 1px solid rgba(255,255,255,0.06);
}}
.sidebar-collapsed {{ width: 0 !important; }}
.sidebar-top {{
    height: 52px; padding: 0 10px 0 18px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.08); flex-shrink: 0;
}}
a.sidebar-brand {{
    color: rgba(255,255,255,0.95) !important; font-weight: 700; font-size: 13px;
    text-decoration: none; white-space: nowrap; letter-spacing: -0.01em; flex: 1; min-width: 0;
}}
a.sidebar-brand:hover {{ color: white !important; }}
.sidebar-collapse-btn {{
    background: none; border: none; cursor: pointer; padding: 5px 8px;
    color: rgba(255,255,255,0.38); font-size: 14px; border-radius: 4px;
    font-family: {FONT}; flex-shrink: 0; line-height: 1;
    transition: color 0.12s, background 0.12s;
}}
.sidebar-collapse-btn:hover {{ color: white; background: rgba(255,255,255,0.1); }}
.sidebar-nav {{
    flex: 1; overflow-y: auto; padding: 10px 8px;
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.12) transparent;
}}
.nav-section-label {{
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: rgba(255,255,255,0.22);
    padding: 10px 14px 4px; display: block;
}}
a.nav-card {{
    display: block; text-decoration: none;
    padding: 10px 14px; margin-bottom: 3px;
    border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);
    transition: background 0.14s, border-color 0.14s;
}}
a.nav-card:hover {{ background: rgba(255,255,255,0.07); border-color: rgba(255,255,255,0.12); }}
a.nav-card-active {{
    background: rgba(75,109,212,0.22) !important;
    border-color: rgba(75,109,212,0.45) !important;
}}
.nav-rq-tag {{
    font-size: 9px; font-weight: 700; color: rgba(165,184,248,0.85);
    text-transform: uppercase; letter-spacing: 0.1em; display: block; margin-bottom: 2px;
}}
.nav-title {{
    font-size: 13px; font-weight: 600; color: rgba(255,255,255,0.92);
    display: block; margin-bottom: 4px;
    overflow: hidden; display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}}
.nav-subtitle {{
    font-size: 11px; color: rgba(255,255,255,0.42); line-height: 1.5;
}}
.sidebar-footer {{
    padding: 8px; border-top: 1px solid rgba(255,255,255,0.07); flex-shrink: 0;
}}
a.sidebar-footer-link {{
    font-size: 11px; color: rgba(255,255,255,0.28); text-decoration: none;
    display: block; padding: 6px 14px; border-radius: 6px; transition: color 0.12s;
}}
a.sidebar-footer-link:hover {{ color: rgba(255,255,255,0.65); }}

/* ── Main content area ───────────────────────────────────────────────────── */
.main-area {{
    flex: 1; min-width: 0; display: flex; flex-direction: column;
    overflow: hidden; background: {BG};
}}
.main-topbar {{
    min-height: 46px; background: {CARD}; flex-shrink: 0;
    border-bottom: 1px solid {BORDER};
    display: flex; align-items: flex-start; flex-wrap: wrap;
    padding: 8px 20px; gap: 10px;
}}
.topbar-toggle-btn {{
    background: none; border: none; cursor: pointer; padding: 5px 8px;
    color: {TEXT2}; font-size: 15px; border-radius: 5px;
    font-family: {FONT}; line-height: 1; transition: color 0.12s, background 0.12s;
}}
.topbar-toggle-btn:hover {{ color: {TEXT}; background: {BG}; }}
.page-scroll {{ flex: 1; overflow-y: auto; }}
.page-inner {{ max-width: 1160px; margin: 0 auto; padding: 28px 24px; }}

/* ── Advisor overlay ──────────────────────────────────────────────────────── */
.advisor-overlay {{
    position: fixed; top: 0; right: 0; bottom: 0; left: 0;
    background: rgba(26,32,64,0.35); z-index: 299;
}}

/* ── Advisor panel (right side) ──────────────────────────────────────────── */
.advisor-panel {{
    position: fixed; top: 0; right: 0;
    width: 380px; height: 100vh;
    background: #2B2B2B; z-index: 300;
    overflow-y: auto;
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.18) transparent;
}}
"""
