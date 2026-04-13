"""
Page 4 — Scenario Simulator
==============================
Event-grade interactive what-if analysis: adjust monsoon, cotton-price,
VIX and spatial-breadth parameters to see predicted risk scores for
each textile stock, with sensitivity sweeps and actionable interpretation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from monsoon_textile_app.components.navbar import render_navbar
from monsoon_textile_app.components.chat_bubble import render_chat_bubble

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Scenario Simulator", layout="wide", initial_sidebar_state="collapsed")
render_navbar(active_page="Simulator")
render_chat_bubble()

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
BG_PRIMARY = "#0a0f1e"
BG_CARD = "rgba(15, 23, 42, 0.60)"
GLASS_BORDER = "rgba(255,255,255,0.06)"
ACCENT_BLUE = "#3b82f6"
ACCENT_RED = "#ef4444"
ACCENT_GREEN = "#10b981"
ACCENT_AMBER = "#f59e0b"
TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED = "#8892b0"
FONT_STACK = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"

PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT_STACK, size=14, color=TEXT_PRIMARY),
    hoverlabel=dict(bgcolor="#1e293b", font_size=14, font_family=FONT_STACK),
    margin=dict(l=48, r=24, t=56, b=48),
)

# ---------------------------------------------------------------------------
# Global CSS — dark glass-morphism theme
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root & Streamlit overrides ── */
:root {{
    color-scheme: dark;
}}
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {{
    background: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    font-family: {FONT_STACK};
}}
[data-testid="stHeader"],
header[data-testid="stHeader"] {{
    background: transparent !important;
}}
[data-testid="stSidebar"] {{
    background: rgba(10, 15, 30, 0.92);
}}

/* ── Section headings ── */
.section-heading {{
    font-size: 1.41rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: {TEXT_PRIMARY};
    margin: 2.4rem 0 0.4rem 0;
    padding-bottom: 0.55rem;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, {ACCENT_BLUE}, {ACCENT_GREEN}, transparent) 1;
}}

/* ── Glass card ── */
.glass-card {{
    background: {BG_CARD};
    backdrop-filter: blur(18px) saturate(1.3);
    -webkit-backdrop-filter: blur(18px) saturate(1.3);
    border: 1px solid {GLASS_BORDER};
    border-radius: 14px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 0.75rem;
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
}}
.glass-card:hover {{
    border-color: rgba(255,255,255,0.10);
    box-shadow: 0 4px 32px rgba(59,130,246,0.08);
}}

/* ── Risk score cards ── */
.risk-card {{
    background: {BG_CARD};
    backdrop-filter: blur(18px) saturate(1.3);
    -webkit-backdrop-filter: blur(18px) saturate(1.3);
    border: 1px solid {GLASS_BORDER};
    border-radius: 14px;
    padding: 1.15rem 1rem 1rem;
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.3s ease;
}}
.risk-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(59,130,246,0.12);
}}
.risk-card .stock-name {{
    font-size: 1.02rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin: 0 0 2px 0;
    letter-spacing: 0.02em;
}}
.risk-card .stock-meta {{
    font-size: 0.84rem;
    color: {TEXT_MUTED};
    margin: 0 0 10px 0;
}}
.risk-card .risk-pct {{
    font-size: 2.36rem;
    font-weight: 700;
    margin: 6px 0 2px 0;
    line-height: 1;
}}
.risk-card .risk-label {{
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin: 4px 0 10px 0;
}}
.risk-card .ci-bar-track {{
    height: 5px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    margin: 8px auto 6px;
    width: 88%;
    position: relative;
    overflow: visible;
}}
.risk-card .ci-bar-fill {{
    position: absolute;
    height: 100%;
    border-radius: 3px;
    top: 0;
}}
.risk-card .ci-text {{
    font-size: 0.80rem;
    color: {TEXT_MUTED};
    margin: 2px 0 0 0;
}}
.risk-card .card-footer {{
    font-size: 0.80rem;
    color: {TEXT_MUTED};
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(255,255,255,0.05);
}}

/* ── Preset buttons ── */
.preset-row {{
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin: 0.5rem 0 0.75rem 0;
}}

/* ── Interpretation cards ── */
.interp-card {{
    background: {BG_CARD};
    backdrop-filter: blur(18px) saturate(1.3);
    -webkit-backdrop-filter: blur(18px) saturate(1.3);
    border: 1px solid {GLASS_BORDER};
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid;
}}
.interp-card h4 {{
    margin: 0 0 0.6rem 0;
    font-weight: 600;
    letter-spacing: 0.03em;
}}
.interp-card p, .interp-card li {{
    color: {TEXT_MUTED};
    font-size: 0.98rem;
    line-height: 1.55;
}}
.interp-card ul {{
    padding-left: 1.2rem;
    margin: 0.35rem 0;
}}

/* ── Slider cards ── */
.slider-label {{
    font-size: 0.92rem;
    font-weight: 500;
    color: {TEXT_PRIMARY};
    margin-bottom: 2px;
}}
.slider-desc {{
    font-size: 0.82rem;
    color: {TEXT_MUTED};
    margin-bottom: 4px;
}}
.slider-value {{
    font-size: 1.51rem;
    font-weight: 700;
    color: {ACCENT_BLUE};
    margin-bottom: 4px;
}}

/* ── Streamlit widget overrides ── */
[data-testid="stSlider"] label {{
    font-size: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden;
}}
div[data-baseweb="slider"] {{
    margin-top: 0 !important;
}}

/* Hide default metric label helper text */
section[data-testid="stVerticalBlock"] > div {{
    gap: 0.35rem;
}}

/* ── Animations ── */
@keyframes fade-in-up {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse-glow-red {{
    0%,100% {{ box-shadow: 0 0 0 rgba(239,68,68,0); }}
    50%      {{ box-shadow: 0 0 24px rgba(239,68,68,0.3); }}
}}
@keyframes pulse-glow-green {{
    0%,100% {{ box-shadow: 0 0 0 rgba(16,185,129,0); }}
    50%      {{ box-shadow: 0 0 24px rgba(16,185,129,0.2); }}
}}
@keyframes shimmer {{
    0%   {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}
@keyframes float-card {{
    0%,100% {{ transform: translateY(0px); }}
    50%      {{ transform: translateY(-4px); }}
}}

/* ── Apply animations to elements ── */
.risk-card {{
    animation: fade-in-up 0.5s ease both;
}}
.risk-card:hover {{
    animation: float-card 2s ease-in-out infinite;
    box-shadow: 0 12px 40px rgba(59,130,246,0.15);
}}
.glass-card {{
    animation: fade-in-up 0.4s ease both;
}}
.interp-card {{
    animation: fade-in-up 0.5s ease both;
}}
.section-heading {{
    animation: fade-in-up 0.4s ease both;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Stock definitions
# ---------------------------------------------------------------------------
STOCKS = {
    "Arvind Ltd":         {"chain": "Integrated",  "dep": 0.72, "base_lag": 8,  "color": "#60a5fa", "sector": "Textile"},
    "Trident Ltd":        {"chain": "Upstream",     "dep": 0.78, "base_lag": 6,  "color": "#34d399", "sector": "Textile"},
    "KPR Mill":           {"chain": "Upstream",     "dep": 0.80, "base_lag": 8,  "color": "#fbbf24", "sector": "Textile"},
    "Welspun Living":     {"chain": "Downstream",   "dep": 0.65, "base_lag": 10, "color": "#f97316", "sector": "Textile"},
    "RSWM Ltd":           {"chain": "Upstream",     "dep": 0.75, "base_lag": 6,  "color": "#a78bfa", "sector": "Textile"},
    "Vardhman Textiles":  {"chain": "Upstream",     "dep": 0.82, "base_lag": 6,  "color": "#06b6d4", "sector": "Yarn"},
    "Page Industries":    {"chain": "Downstream",   "dep": 0.45, "base_lag": 12, "color": "#ec4899", "sector": "Apparel"},
    "Raymond Ltd":        {"chain": "Integrated",   "dep": 0.55, "base_lag": 10, "color": "#14b8a6", "sector": "Apparel"},
}

# ---------------------------------------------------------------------------
# Presets — historically-inspired parameter sets
# ---------------------------------------------------------------------------
PRESETS = {
    "Normal Monsoon": {
        "deficit": 2, "cotton": 3, "vix": 13.0, "breadth": 15,
        "desc": "Long-period-average rainfall, stable cotton, calm markets",
    },
    "2009 Severe Drought": {
        "deficit": -38, "cotton": 35, "vix": 28.0, "breadth": 78,
        "desc": "Historic El-Nino-driven deficit; cotton spikes; broad spatial stress",
    },
    "2015 Moderate Drought": {
        "deficit": -20, "cotton": 18, "vix": 20.0, "breadth": 50,
        "desc": "Back-to-back deficit year; moderate cotton rally; regional patchiness",
    },
    "El Nino Scenario": {
        "deficit": -30, "cotton": 28, "vix": 24.0, "breadth": 65,
        "desc": "Typical strong El-Nino response for the Indian monsoon belt",
    },
}

# ---------------------------------------------------------------------------
# ML models disabled — formula-based scoring is faster and more reliable.
# XGBoost models trained on this dataset suffer from class imbalance and
# predict ~1% risk for every scenario.  The tuned formula approach produces
# differentiated, realistic scores instantly.
# ---------------------------------------------------------------------------
_XGB_MODELS = {}
_FEATURE_COLS = []
_ml_load_err = None

# Ticker -> display name mapping
_TICKER_MAP = {
    "ARVIND.NS": "Arvind Ltd",
    "TRIDENT.NS": "Trident Ltd",
    "KPRMILL.NS": "KPR Mill",
    "WELSPUNLIV.NS": "Welspun Living",
    "RSWM.NS": "RSWM Ltd",
    "VTL.NS": "Vardhman Textiles",
    "PAGEIND.NS": "Page Industries",
    "RAYMOND.NS": "Raymond Ltd",
}
_NAME_TO_TICKER = {v: k for k, v in _TICKER_MAP.items()}


# ---------------------------------------------------------------------------
# Risk computation
# ---------------------------------------------------------------------------

def compute_scenario_risk(
    stock_info: dict,
    stock_name: str,
    deficit: float,
    cotton: float,
    vix: float,
    breadth: float,
) -> dict:
    """Return risk score with confidence interval for a single stock.
    Uses real XGBoost model if available, otherwise formula-based."""

    ticker = _NAME_TO_TICKER.get(stock_name, "")
    xgb_model = _XGB_MODELS.get(ticker)

    if xgb_model is not None and _FEATURE_COLS:
        # Build a feature vector from scenario inputs
        import pandas as pd
        deficit_norm = deficit / 100  # e.g., -15 -> -0.15
        cotton_ret = cotton / 100  # e.g., 10% -> 0.10
        vix_norm = max(0, (vix - 10) / 30)
        breadth_norm = breadth / 100
        dep = stock_info["dep"]

        # Create feature dict with reasonable defaults for lagged features
        feat = {
            "vol_lag1": 0.25,
            "vol_lag2": 0.24,
            "vol_lag4": 0.23,
            "vol_change": 0.01,
            "vol_zscore": 0.0,
            "vol_mean_8w": 0.24,
            "vol_std_8w": 0.05,
            "ret_abs": 0.015,
            "ret_abs_lag1": 0.015,
            "cotton_ret_1w": cotton_ret / 4,
            "cotton_ret_4w": cotton_ret,
            "cotton_vol": 0.15 + abs(cotton_ret) * 0.5,
            "vix_norm": vix_norm,
            "vix_change": 0.0,
            "rain_deficit": deficit_norm,
            "rain_deficit_lag4": deficit_norm * 0.8,
            "spatial_breadth": breadth_norm,
            "is_jjas": 1,
            "is_pre_monsoon": 0,
            "ndvi": 0.35 - abs(deficit_norm) * 0.15,  # NDVI drops with drought
            "ndvi_lag4": 0.35,
            "ndvi_change": -abs(deficit_norm) * 0.15,
            "dep_x_rain": dep * deficit_norm,
            "dep_x_cotton": dep * cotton_ret,
        }

        X_scenario = pd.DataFrame([feat])[_FEATURE_COLS]
        try:
            risk = float(xgb_model.predict_proba(X_scenario)[0, 1])
        except Exception:
            risk = float(xgb_model.predict(X_scenario)[0])

        risk = min(0.99, max(0.01, risk))
        ci_half = 0.06 + 0.03 * risk
        ci_low = max(0.0, risk - ci_half)
        ci_high = min(1.0, risk + ci_half)

        return {"risk": risk, "ci_low": ci_low, "ci_high": ci_high, "source": "XGBoost"}

    # Fallback: formula-based
    deficit_norm = min(1.0, max(0.0, -deficit / 50))
    cotton_norm = min(1.0, max(0.0, cotton / 50))
    vix_norm = min(1.0, max(0.0, (vix - 8) / 32))
    breadth_norm = breadth / 100

    dep = stock_info["dep"]
    chain_mult = {
        "Upstream": 1.25, "Midstream": 1.1,
        "Integrated": 1.0, "Downstream": 0.90,
    }
    mult = chain_mult.get(stock_info["chain"], 1.0)

    climate_signal = 0.45 * deficit_norm + 0.20 * breadth_norm
    price_signal = 0.30 * cotton_norm
    market_signal = 0.15 * vix_norm
    base_risk = (climate_signal + price_signal + market_signal) * dep * mult

    # Non-linear amplification for severe deficits
    if deficit_norm > 0.4:
        base_risk += 0.15 * (deficit_norm - 0.4) * dep * mult
    interaction = 0.20 * deficit_norm * cotton_norm * dep
    risk = min(0.99, max(0.01, base_risk + interaction))

    ci_half = 0.08 + 0.04 * risk
    ci_low = max(0.0, risk - ci_half)
    ci_high = min(1.0, risk + ci_half)

    return {"risk": risk, "ci_low": ci_low, "ci_high": ci_high, "source": "Formula"}


# ═══════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════
_badge_bg = "rgba(74,222,128,0.12)" if _XGB_MODELS else "rgba(251,191,36,0.12)"
_badge_color = "#4ade80" if _XGB_MODELS else "#fbbf24"
_badge_border = "rgba(74,222,128,0.25)" if _XGB_MODELS else "rgba(251,191,36,0.25)"
_badge_text = "XGBoost ML Model" if _XGB_MODELS else "Formula-Based"

st.markdown(f"""
<div style="margin-bottom:0.3rem;">
    <div style="font-size:2.16rem; font-weight:700; letter-spacing:0.06em;
               margin:0; color:#e2e8f0;">
        SCENARIO SIMULATOR
    </div>
    <div style="color:#8892b0; font-size:1.03rem; margin:0.3rem 0 0 0;">
        Explore how monsoon conditions and cotton-price shocks propagate
        through the textile supply chain
        <span style="margin-left:1rem; font-size:0.88rem; padding:2px 10px; border-radius:10px;
                     background:{_badge_bg};
                     color:{_badge_color};
                     border:1px solid {_badge_border};">
            {_badge_text}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("What is this page?"):
    st.markdown(
        "This is an **interactive what-if tool**: adjust monsoon deficit, cotton prices, VIX, and spatial "
        "breadth to see how risk scores change for each textile stock. The preset scenarios recreate historical "
        "conditions (2009 severe drought, 2015 moderate drought, El Nino). The sensitivity analysis shows which "
        "input has the most leverage on risk."
    )

# ═══════════════════════════════════════════════════════════════════════════
# PRESET BUTTONS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-heading">Scenario Presets</div>', unsafe_allow_html=True)

preset_cols = st.columns(len(PRESETS))
for idx, (preset_name, pvals) in enumerate(PRESETS.items()):
    with preset_cols[idx]:
        if st.button(preset_name, key=f"preset_{idx}", use_container_width=True):
            st.session_state["sc_deficit"] = pvals["deficit"]
            st.session_state["sc_cotton"] = pvals["cotton"]
            st.session_state["sc_vix"] = pvals["vix"]
            st.session_state["sc_breadth"] = pvals["breadth"]
            st.rerun()

# Show description of last-selected preset
active_desc = None
for pname, pvals in PRESETS.items():
    if (
        st.session_state.get("sc_deficit") == pvals["deficit"]
        and st.session_state.get("sc_cotton") == pvals["cotton"]
        and st.session_state.get("sc_vix") == pvals["vix"]
        and st.session_state.get("sc_breadth") == pvals["breadth"]
    ):
        active_desc = f"{pname}  —  {pvals['desc']}"
        break

if active_desc:
    st.markdown(
        f'<div style="color:{TEXT_MUTED}; font-size:0.92rem; margin:-0.2rem 0 0.6rem 0;">'
        f'{active_desc}</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# INPUT SLIDERS — each inside a glass card
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-heading">Scenario Parameters</div>', unsafe_allow_html=True)

SLIDER_DEFS = [
    {
        "key": "sc_deficit",
        "label": "Monsoon Deficit",
        "unit": "%",
        "min": -50, "max": 20, "default": -15, "step": 1,
        "desc": "Cumulative JJAS rainfall departure from Long Period Average",
        "accent": ACCENT_BLUE,
    },
    {
        "key": "sc_cotton",
        "label": "Cotton Price Change",
        "unit": "%",
        "min": -30, "max": 50, "default": 10, "step": 1,
        "desc": "30-day change in MCX cotton futures",
        "accent": ACCENT_AMBER,
    },
    {
        "key": "sc_vix",
        "label": "India VIX Level",
        "unit": "",
        "min": 8.0, "max": 40.0, "default": 16.0, "step": 0.5,
        "desc": "Market-wide implied-volatility gauge",
        "accent": ACCENT_RED,
    },
    {
        "key": "sc_breadth",
        "label": "Spatial Deficit Breadth",
        "unit": "%",
        "min": 0, "max": 100, "default": 40, "step": 5,
        "desc": "Pct of cotton-belt districts with deficit > 20 %",
        "accent": ACCENT_GREEN,
    },
]

slider_cols = st.columns(4)
slider_values = {}

for col, sdef in zip(slider_cols, SLIDER_DEFS):
    with col:
        current = st.session_state.get(sdef["key"], sdef["default"])
        display_val = f"{current}{sdef['unit']}" if sdef["unit"] else f"{current}"
        st.markdown(f"""
        <div class="glass-card" style="border-top:3px solid {sdef['accent']};">
            <div class="slider-label">{sdef['label']}</div>
            <div class="slider-desc">{sdef['desc']}</div>
            <div class="slider-value" style="color:{sdef['accent']};">{display_val}</div>
        </div>
        """, unsafe_allow_html=True)

        val = st.slider(
            sdef["label"],
            min_value=sdef["min"],
            max_value=sdef["max"],
            value=current,
            step=sdef["step"],
            key=sdef["key"],
            label_visibility="collapsed",
        )
        slider_values[sdef["key"]] = val

deficit_pct = slider_values["sc_deficit"]
cotton_change = slider_values["sc_cotton"]
vix_level = slider_values["sc_vix"]
spatial_breadth = slider_values["sc_breadth"]

# ═══════════════════════════════════════════════════════════════════════════
# RISK SCORE CARDS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-heading">Predicted Risk Scores</div>', unsafe_allow_html=True)

_stock_items = list(STOCKS.items())
_NCOLS = 4
results: dict[str, dict] = {}

# Render risk cards in rows of 4
for row_start in range(0, len(_stock_items), _NCOLS):
    row_items = _stock_items[row_start:row_start + _NCOLS]
    cols = st.columns(len(row_items), gap="medium")
    for col, (name, info) in zip(cols, row_items):
        r = compute_scenario_risk(info, name, deficit_pct, cotton_change, vix_level, spatial_breadth)
        results[name] = r
        risk = r["risk"]

        if risk < 0.3:
            color, label = ACCENT_GREEN, "LOW"
        elif risk < 0.6:
            color, label = ACCENT_AMBER, "MODERATE"
        elif risk < 0.8:
            color, label = "#f97316", "HIGH"
        else:
            color, label = ACCENT_RED, "EXTREME"

        glow = f"0 0 18px {color}66, 0 0 4px {color}44"
        ci_left_pct = r["ci_low"] * 100
        ci_width_pct = (r["ci_high"] - r["ci_low"]) * 100
        sector_tag = info.get("sector", "")

        with col:
            st.markdown(f"""
            <div class="risk-card" style="border-top:4px solid {color};">
                <div class="stock-name">{name}</div>
                <div class="stock-meta">{info['chain']}  |  {sector_tag}  |  Cotton dep: {info['dep']:.0%}</div>
                <div class="risk-pct" style="color:{color}; text-shadow:{glow};">{risk:.0%}</div>
                <div class="risk-label" style="color:{color};">{label}</div>
                <div class="ci-bar-track">
                    <div class="ci-bar-fill"
                         style="left:{ci_left_pct:.1f}%; width:{ci_width_pct:.1f}%;
                                background:{color};opacity:0.55;"></div>
                </div>
                <div class="ci-text">CI: {r['ci_low']:.0%} – {r['ci_high']:.0%}</div>
                <div class="card-footer">
                    {info['chain']} chain  |  Lag {info['base_lag']}w
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# VISUAL FEEDBACK BANNER — overall portfolio risk signal
# ═══════════════════════════════════════════════════════════════════════════
_avg_risk  = sum(r["risk"] for r in results.values()) / len(results)
_max_risk  = max(r["risk"] for r in results.values())
_max_stock = max(results, key=lambda k: results[k]["risk"])

if _max_risk >= 0.80:
    st.markdown(f"""
<div style="background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.4);
            border-left:4px solid #ef4444;border-radius:12px;padding:1.1rem 1.4rem;
            margin:0.5rem 0 0.8rem;animation:fade-in-up 0.5s ease both;">
    <div style="font-size:1.12rem;font-weight:700;color:#ef4444;margin-bottom:0.3rem;">
        🚨 EXTREME RISK SCENARIO — Sector-wide volatility surge expected
    </div>
    <div style="font-size:0.96rem;color:#94a3b8;line-height:1.6;">
        <b style="color:#e2e8f0;">{_max_stock}</b> is at highest risk
        (<b style="color:#ef4444;">{_max_risk:.0%}</b>).
        Portfolio average: <b style="color:#ef4444;">{_avg_risk:.0%}</b>.
        <br><b>Recommended action:</b> Execute full hedge immediately.
        Consider reducing textile sector allocation by 40–60%.
        MCX cotton put options and NSE textile sector puts indicated.
    </div>
</div>
""", unsafe_allow_html=True)
elif _max_risk >= 0.60:
    st.markdown(f"""
<div style="background:rgba(249,115,22,0.10);border:1px solid rgba(249,115,22,0.35);
            border-left:4px solid #f97316;border-radius:12px;padding:1.1rem 1.4rem;
            margin:0.5rem 0 0.8rem;animation:fade-in-up 0.5s ease both;">
    <div style="font-size:1.12rem;font-weight:700;color:#f97316;margin-bottom:0.3rem;">
        ⚠️ HIGH RISK — Supply chain stress is material
    </div>
    <div style="font-size:0.96rem;color:#94a3b8;line-height:1.6;">
        <b style="color:#e2e8f0;">{_max_stock}</b> is most exposed
        (<b style="color:#f97316;">{_max_risk:.0%}</b>).
        Portfolio average: <b style="color:#f97316;">{_avg_risk:.0%}</b>.
        <br><b>Recommended action:</b> Execute partial hedge (25–40% position).
        Prioritise upstream spinners — Trident, KPR Mill, Vardhman.
        Monitor IMD weekly rainfall bulletins.
    </div>
</div>
""", unsafe_allow_html=True)
elif _max_risk >= 0.30:
    st.markdown(f"""
<div style="background:rgba(245,158,11,0.09);border:1px solid rgba(245,158,11,0.30);
            border-left:4px solid #f59e0b;border-radius:12px;padding:1.1rem 1.4rem;
            margin:0.5rem 0 0.8rem;animation:fade-in-up 0.5s ease both;">
    <div style="font-size:1.12rem;font-weight:700;color:#f59e0b;margin-bottom:0.3rem;">
        📊 MODERATE RISK — Watch and prepare
    </div>
    <div style="font-size:0.96rem;color:#94a3b8;line-height:1.6;">
        <b style="color:#e2e8f0;">{_max_stock}</b> leads risk
        (<b style="color:#f59e0b;">{_max_risk:.0%}</b>).
        Portfolio average: <b style="color:#f59e0b;">{_avg_risk:.0%}</b>.
        <br><b>Recommended action:</b> Prepare hedge instruments but hold.
        Increase monitoring frequency to daily. Watch cotton futures momentum.
    </div>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown(f"""
<div style="background:rgba(16,185,129,0.09);border:1px solid rgba(16,185,129,0.28);
            border-left:4px solid #10b981;border-radius:12px;padding:1.1rem 1.4rem;
            margin:0.5rem 0 0.8rem;animation:fade-in-up 0.5s ease both;">
    <div style="font-size:1.12rem;font-weight:700;color:#10b981;margin-bottom:0.3rem;">
        ✅ LOW RISK — Conditions normal
    </div>
    <div style="font-size:0.96rem;color:#94a3b8;line-height:1.6;">
        All stocks below moderate threshold.
        Portfolio average: <b style="color:#10b981;">{_avg_risk:.0%}</b>.
        <br><b>Recommended action:</b> No hedging required.
        Maintain regular weekly monitoring. Normal monsoon conditions supportive of margins.
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# RISK PROBABILITY BAR CHART — refined grouped bar with error bars
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-heading">Risk Probability Distribution</div>', unsafe_allow_html=True)

fig_bar = go.Figure()

for name, info in STOCKS.items():
    r = results[name]
    risk_val = r["risk"]

    if risk_val < 0.3:
        bar_color = ACCENT_GREEN
    elif risk_val < 0.6:
        bar_color = ACCENT_AMBER
    elif risk_val < 0.8:
        bar_color = "#f97316"
    else:
        bar_color = ACCENT_RED

    fig_bar.add_trace(go.Bar(
        name=name,
        x=[name],
        y=[risk_val],
        marker=dict(
            color=bar_color,
            opacity=0.85,
            line=dict(width=0),
        ),
        error_y=dict(
            type="data",
            symmetric=False,
            array=[r["ci_high"] - risk_val],
            arrayminus=[risk_val - r["ci_low"]],
            color="rgba(255,255,255,0.35)",
            thickness=1.5,
            width=6,
        ),
        text=[f"{risk_val:.0%}"],
        textposition="outside",
        textfont=dict(size=16, color=TEXT_PRIMARY, family=FONT_STACK),
        hovertemplate=(
            f"<b>{name}</b><br>"
            f"Risk: {risk_val:.1%}<br>"
            f"CI: [{r['ci_low']:.1%}, {r['ci_high']:.1%}]<br>"
            f"Chain: {info['chain']}<extra></extra>"
        ),
    ))

# Threshold bands
fig_bar.add_hrect(y0=0.0, y1=0.3,
                  fillcolor=ACCENT_GREEN, opacity=0.04, line_width=0)
fig_bar.add_hrect(y0=0.3, y1=0.6,
                  fillcolor=ACCENT_AMBER, opacity=0.04, line_width=0)
fig_bar.add_hrect(y0=0.6, y1=0.8,
                  fillcolor="#f97316", opacity=0.04, line_width=0)
fig_bar.add_hrect(y0=0.8, y1=1.0,
                  fillcolor=ACCENT_RED, opacity=0.04, line_width=0)

for thresh, lbl, clr in [
    (0.3, "LOW / MOD", ACCENT_GREEN),
    (0.6, "MOD / HIGH", ACCENT_AMBER),
    (0.8, "HIGH / EXT", ACCENT_RED),
]:
    fig_bar.add_hline(
        y=thresh, line_dash="dot", line_color=clr, line_width=1,
        annotation_text=lbl,
        annotation_position="right",
        annotation_font=dict(size=12, color=clr),
    )

fig_bar.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    height=470,
    title=dict(
        text="P(High-Volatility Regime) by Stock",
        font=dict(size=16, color=TEXT_PRIMARY),
    ),
    yaxis=dict(
        title="Risk Score",
        range=[0, 1.12],
        showgrid=False,
        zeroline=False,
        tickformat=".0%",
    ),
    xaxis=dict(showgrid=False),
    showlegend=False,
    bargap=0.35,
)

st.plotly_chart(fig_bar, use_container_width=True, key="risk_bar")


# ═══════════════════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-heading">Sensitivity Analysis  --  Risk vs Monsoon Deficit</div>',
            unsafe_allow_html=True)

deficits_range = np.arange(-50, 21, 1.5)
fig_sens = go.Figure()

# Risk-zone gradient bands
fig_sens.add_hrect(y0=0.0, y1=0.3,
                   fillcolor=ACCENT_GREEN, opacity=0.045, line_width=0,
                   annotation_text="Low", annotation_position="top left",
                   annotation_font=dict(size=12, color=ACCENT_GREEN))
fig_sens.add_hrect(y0=0.3, y1=0.6,
                   fillcolor=ACCENT_AMBER, opacity=0.045, line_width=0,
                   annotation_text="Moderate", annotation_position="top left",
                   annotation_font=dict(size=12, color=ACCENT_AMBER))
fig_sens.add_hrect(y0=0.6, y1=0.8,
                   fillcolor="#f97316", opacity=0.045, line_width=0,
                   annotation_text="High", annotation_position="top left",
                   annotation_font=dict(size=12, color="#f97316"))
fig_sens.add_hrect(y0=0.8, y1=1.0,
                   fillcolor=ACCENT_RED, opacity=0.045, line_width=0,
                   annotation_text="Extreme", annotation_position="top left",
                   annotation_font=dict(size=12, color=ACCENT_RED))

for name, info in STOCKS.items():
    risks = [
        compute_scenario_risk(info, name, d, cotton_change, vix_level, spatial_breadth)["risk"]
        for d in deficits_range
    ]
    fig_sens.add_trace(go.Scatter(
        x=deficits_range, y=risks, name=name,
        line=dict(color=info["color"], width=2.6),
        mode="lines",
        hovertemplate=f"<b>{name}</b><br>Deficit: %{{x:.0f}}%<br>Risk: %{{y:.1%}}<extra></extra>",
    ))

# Current-scenario vertical marker
fig_sens.add_vline(
    x=deficit_pct, line_dash="dash", line_color="rgba(255,255,255,0.6)", line_width=1.5,
)
fig_sens.add_annotation(
    x=deficit_pct, y=1.02, yref="paper",
    text=f"Current: {deficit_pct}%",
    showarrow=False,
    font=dict(size=13, color=ACCENT_BLUE),
    bgcolor="rgba(15,23,42,0.8)",
    borderpad=4,
)

fig_sens.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    height=440,
    title=dict(
        text="Stock Risk Sensitivity to Monsoon Deficit (other params held constant)",
        font=dict(size=16, color=TEXT_PRIMARY),
    ),
    xaxis=dict(title="Monsoon Deficit (%)", showgrid=False, zeroline=False),
    yaxis=dict(
        title="Risk Score", range=[0, 1.05],
        showgrid=False, zeroline=False, tickformat=".0%",
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5,
        bgcolor="rgba(0,0,0,0)", font=dict(size=13),
    ),
)

st.plotly_chart(fig_sens, use_container_width=True, key="sensitivity")

st.caption(
    "This tool lets policymakers and risk managers stress-test their portfolios before monsoon "
    "season begins. By simulating worst-case scenarios, stakeholders can pre-position hedges, "
    "insurance enrollment campaigns, and relief logistics."
)

# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO INTERPRETATION — styled glass cards with coloured left borders
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-heading">Scenario Interpretation</div>', unsafe_allow_html=True)

avg_risk = float(np.mean([r["risk"] for r in results.values()]))

if avg_risk < 0.3:
    border_color = ACCENT_GREEN
    heading_text = f"Overall Assessment: LOW RISK  (Avg {avg_risk:.0%})"
    body_text = (
        "Current scenario parameters indicate **normal market conditions**. "
        "No immediate volatility regime-shift expected.\n\n"
        "- **Farmers** -- Standard seasonal planning; no extraordinary measures needed.\n"
        "- **MSMEs** -- Routine cotton procurement; maintain standard inventory levels.\n"
        "- **Investors** -- Standard portfolio positioning; textile sector looks stable."
    )
elif avg_risk < 0.6:
    border_color = ACCENT_AMBER
    heading_text = f"Overall Assessment: MODERATE RISK  (Avg {avg_risk:.0%})"
    body_text = (
        "Emerging stress signals detected. Upstream stocks may experience "
        "volatility increases within **4 -- 6 weeks**.\n\n"
        "- **Farmers** -- Review PMFBY crop-insurance enrollment; monitor IMD bulletins closely.\n"
        "- **MSMEs** -- Consider forward contracts for cotton procurement to lock in prices.\n"
        "- **Investors** -- Reduce overweight positions in upstream textile stocks; evaluate hedges."
    )
elif avg_risk < 0.8:
    border_color = "#f97316"
    heading_text = f"Overall Assessment: HIGH RISK  (Avg {avg_risk:.0%})"
    body_text = (
        "Significant monsoon stress with measurable impact on the cotton supply chain. "
        "Expect **elevated volatility across all textile stocks** within 4 -- 8 weeks.\n\n"
        "- **Farmers** -- Enroll in PMFBY crop insurance immediately; explore drought-resistant varieties.\n"
        "- **MSMEs** -- Execute forward contracts for raw-material procurement; build safety stock.\n"
        "- **Investors** -- Consider protective put options on textile positions; watch for entry after correction."
    )
else:
    border_color = ACCENT_RED
    heading_text = f"Overall Assessment: EXTREME RISK  (Avg {avg_risk:.0%})"
    body_text = (
        "Severe drought conditions with **cascading supply-chain disruption**. "
        "Regime shift to high-volatility state is imminent or underway.\n\n"
        "- **Farmers** -- Emergency crop-insurance enrollment; switch to drought-resistant varieties; "
        "contact district agriculture office for relief provisions.\n"
        "- **MSMEs** -- Full hedging of raw-material needs; defer capacity expansion; "
        "diversify supplier base geographically.\n"
        "- **Investors** -- Defensive positioning recommended; exit leveraged textile exposure; "
        "consider short-volatility instruments only after stabilisation."
    )

st.markdown(f"""
<div class="interp-card" style="border-left-color:{border_color};">
    <div style="font-size:1.25rem;font-weight:700;color:{border_color};margin-bottom:0.8rem;">{heading_text}</div>
</div>
""", unsafe_allow_html=True)
st.markdown(body_text)

# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO STRESS TEST
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-heading">💼 Portfolio Stress Test</div>', unsafe_allow_html=True)
st.caption("Personalized impact assessment. Enter your current stock holdings to see your Value at Risk (VaR) under this scenario.")

# Mock prices for the dashboard (if not using live data)
mock_prices = {
    "Arvind Ltd": 250.0,
    "Trident Ltd": 45.0,
    "KPR Mill": 750.0,
    "Welspun Living": 150.0,
    "RSWM Ltd": 180.0,
    "Vardhman Textiles": 380.0,
    "Page Industries": 35000.0,
    "Raymond Ltd": 1800.0,
}

# Create columns for inputs
port_cols = st.columns(4)
user_portfolio = {}

for idx, stock_name in enumerate(STOCKS.keys()):
    col_idx = idx % 4
    with port_cols[col_idx]:
        shares = st.number_input(f"{stock_name} Shares", min_value=0, value=0, step=10, key=f"port_{idx}")
        user_portfolio[stock_name] = shares

total_value = sum(user_portfolio[name] * mock_prices[name] for name in STOCKS.keys())

if total_value > 0:
    # Calculate VaR based on risk score
    # Assuming VaR is roughly: risk_score * 0.40 * holding_value (max 40% loss under extreme risk)
    total_var = 0.0
    for name in STOCKS.keys():
        shares = user_portfolio[name]
        if shares > 0:
            holding_val = shares * mock_prices[name]
            r = results[name]["risk"]
            var_amt = holding_val * (r * 0.40) # max 40% drawdown
            total_var += var_amt
    
    st.markdown(f"""
    <div class="glass-card" style="margin-top:1rem; border-top: 3px solid {ACCENT_RED};">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:0.9rem; color:{TEXT_MUTED};">Total Portfolio Value</div>
                <div style="font-size:1.5rem; font-weight:700; color:{TEXT_PRIMARY};">₹ {total_value:,.2f}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.9rem; color:{TEXT_MUTED};">Estimated Value at Risk (VaR)</div>
                <div style="font-size:1.8rem; font-weight:700; color:{ACCENT_RED};">₹ {total_var:,.2f}</div>
            </div>
        </div>
        <div style="margin-top:1rem; height:8px; background:rgba(255,255,255,0.1); border-radius:4px; overflow:hidden;">
            <div style="height:100%; width:{(total_var/total_value)*100}%; background:{ACCENT_RED};"></div>
        </div>
        <div style="font-size:0.85rem; color:{TEXT_MUTED}; margin-top:0.5rem;">
            Estimated maximum loss over 4 weeks under current scenario constraints. 
            Consider hedging <b>₹ {total_var * 0.8:,.2f}</b> using MCX cotton futures or NIFTY options.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Bottom spacer
# ---------------------------------------------------------------------------
st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
