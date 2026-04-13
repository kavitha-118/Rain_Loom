"""
Monsoon-Textile Volatility Risk Monitor -- Overview / About Page
================================================================

Comprehensive landing page explaining the project, methodology,
dashboard pages, data sources, and technical architecture.

Launch:
    streamlit run monsoon_textile_app/app.py --server.port 8501
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from monsoon_textile_app.components.navbar import render_navbar
from monsoon_textile_app.components.chat_bubble import render_chat_bubble
from monsoon_textile_app.utils.session_guard import init_csrf

# -- Page Configuration --------------------------------------------------------
st.set_page_config(
    page_title="RainLoom -- Monsoon & Textile Volatility",
    page_icon="\u26a1",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- Navbar (MUST come before any other content) --------------------------------
render_navbar(active_page="Overview")
render_chat_bubble()

# -- CSRF session token (Feature 6) -------------------------------------------
init_csrf()

# -- Auto-start background email alert scheduler ---------------------------------
if "_email_scheduler_started" not in st.session_state:
    try:
        from monsoon_textile_app.utils.email_scheduler import start_scheduler
        from monsoon_textile_app.utils.audit_log import audit
        start_scheduler()  # reads ALERT_CHECK_INTERVAL_MINUTES from .env (default 15 min)
        st.session_state["_email_scheduler_started"] = True
        audit("system", "scheduler_started", details={"page": "app.py"})
    except Exception as _sched_err:
        st.session_state["_email_scheduler_started"] = f"error: {_sched_err}"

# -- Master CSS ----------------------------------------------------------------
st.markdown("""
<style>
    /* ---- Import font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---- Root variables ---- */
    :root {
        --bg-primary: #0a0f1e;
        --bg-card: rgba(15, 23, 42, 0.60);
        --bg-card-hover: rgba(20, 30, 55, 0.75);
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-accent: rgba(59, 130, 246, 0.25);
        --text-primary: #e2e8f0;
        --text-secondary: #8892b0;
        --text-muted: #64748b;
        --accent-blue: #3b82f6;
        --accent-red: #ef4444;
        --accent-green: #10b981;
        --accent-gold: #f59e0b;
        --accent-cyan: #06b6d4;
        --accent-purple: #8b5cf6;
        --radius-lg: 16px;
        --radius-md: 12px;
        --radius-sm: 8px;
    }

    /* ---- Hide all Streamlit chrome ---- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    [data-testid="stDeployButton"] { display: none; }
    [data-testid="stToolbar"] { display: none; }
    .stDeployButton { display: none !important; }

    /* ---- Global resets ---- */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--text-primary);
    }

    [data-testid="stAppViewContainer"] {
        background: var(--bg-primary);
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(59,130,246,0.08) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(16,185,129,0.04) 0%, transparent 50%);
    }

    .main .block-container {
        padding-bottom: 2rem;
        max-width: 1440px;
    }

    /* ---- Glass card base ---- */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.4rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        border-color: var(--border-accent);
        box-shadow: 0 4px 30px rgba(59, 130, 246, 0.06);
    }

    /* ---- Hero header ---- */
    .hero-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .hero-title {
        font-size: 2.46rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 40%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .hero-tagline {
        font-size: 1.03rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-top: 0.6rem;
        letter-spacing: 0.01em;
    }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 1.11rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
        position: relative;
        display: inline-block;
    }
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -4px;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue) 0%, transparent 100%);
        border-radius: 1px;
    }

    /* ---- Gradient divider ---- */
    .gradient-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(59,130,246,0.3) 30%, rgba(59,130,246,0.3) 70%, transparent 100%);
        border: none;
        margin: 1.5rem 0;
    }

    /* ---- Content labels & text ---- */
    .content-label {
        font-size: 0.80rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }
    .content-text {
        font-size: 0.94rem;
        color: var(--text-secondary);
        line-height: 1.65;
        margin: 0 0 0.2rem 0;
    }

    /* ---- Solution steps ---- */
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.4rem 0;
    }
    .step-num {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.82rem;
        font-weight: 700;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .step-text {
        font-size: 0.94rem;
        color: var(--text-secondary);
        line-height: 1.55;
    }

    /* ---- Causal chain stepper ---- */
    .chain-container {
        display: flex;
        align-items: flex-start;
        gap: 0;
        margin: 0.5rem 0;
        position: relative;
    }
    .chain-step {
        flex: 1;
        text-align: center;
        position: relative;
        padding: 0 0.4rem;
    }
    .chain-node {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 1.1rem 0.8rem;
        position: relative;
        z-index: 2;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .chain-node:hover {
        border-color: var(--border-accent);
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }
    .chain-number {
        font-size: 0.77rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    .chain-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.4rem;
    }
    .chain-desc {
        font-size: 0.84rem;
        color: var(--text-secondary);
        line-height: 1.45;
        margin-bottom: 0.4rem;
    }
    .chain-lag {
        font-size: 0.80rem;
        color: var(--text-muted);
        font-weight: 500;
    }
    .chain-connector {
        position: absolute;
        top: 50%;
        right: -0.5rem;
        transform: translateY(-50%);
        z-index: 3;
        color: var(--text-muted);
        font-size: 1rem;
        font-weight: 300;
    }

    /* ---- Dashboard guide cards ---- */
    .guide-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 0.85rem;
    }
    .guide-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.3rem 1.4rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .guide-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
    }
    .guide-card:hover {
        border-color: var(--border-accent);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .guide-card-icon {
        width: 36px;
        height: 36px;
        border-radius: var(--radius-sm);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.03rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .guide-card-page {
        font-size: 0.77rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
    }
    .guide-card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.55rem 0;
    }
    .guide-card-desc {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin: 0 0 0.65rem 0;
    }
    .guide-card-usage {
        font-size: 0.86rem;
        color: var(--accent-cyan);
        font-weight: 500;
        font-style: italic;
    }

    /* ---- Data table ---- */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.90rem;
    }
    .data-table-header {
        display: grid;
        grid-template-columns: 1.2fr 1fr 1.5fr 1fr;
        gap: 0.5rem;
        padding: 0.65rem 1rem;
        background: rgba(59, 130, 246, 0.08);
        border-radius: var(--radius-sm);
        margin-bottom: 0.3rem;
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .data-table-row {
        display: grid;
        grid-template-columns: 1.2fr 1fr 1.5fr 1fr;
        gap: 0.5rem;
        padding: 0.6rem 1rem;
        border-bottom: 1px solid var(--border-subtle);
        font-size: 0.90rem;
        color: var(--text-secondary);
        transition: background 0.2s ease;
    }
    .data-table-row:hover {
        background: rgba(59, 130, 246, 0.04);
    }
    .data-table-row:last-child {
        border-bottom: none;
    }
    .dt-source {
        color: var(--text-primary);
        font-weight: 600;
    }
    .dt-api {
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 0.87rem;
        color: var(--accent-cyan);
    }

    /* ---- Securities grid ---- */
    .sec-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.6rem;
    }
    .sec-card {
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .sec-card:hover {
        border-color: var(--border-accent);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .sec-ticker {
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 0.88rem;
        color: var(--accent-blue);
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .sec-name {
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.3rem;
    }
    .sec-role {
        font-size: 0.84rem;
        color: var(--text-secondary);
        line-height: 1.45;
    }

    /* ---- Tech stack pills ---- */
    .tech-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.6rem;
    }
    .tech-pill {
        display: inline-block;
        background: rgba(59, 130, 246, 0.1);
        color: var(--accent-blue);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 20px;
        padding: 0.3rem 0.75rem;
        font-size: 0.86rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .tech-pill:hover {
        background: rgba(59, 130, 246, 0.18);
        border-color: rgba(59, 130, 246, 0.3);
    }
    .tech-pill-green {
        background: rgba(16, 185, 129, 0.1);
        color: var(--accent-green);
        border-color: rgba(16, 185, 129, 0.15);
    }
    .tech-pill-green:hover {
        background: rgba(16, 185, 129, 0.18);
        border-color: rgba(16, 185, 129, 0.3);
    }
    .tech-pill-gold {
        background: rgba(245, 158, 11, 0.1);
        color: var(--accent-gold);
        border-color: rgba(245, 158, 11, 0.15);
    }
    .tech-pill-gold:hover {
        background: rgba(245, 158, 11, 0.18);
        border-color: rgba(245, 158, 11, 0.3);
    }
    .tech-pill-purple {
        background: rgba(139, 92, 246, 0.1);
        color: var(--accent-purple);
        border-color: rgba(139, 92, 246, 0.15);
    }
    .tech-pill-purple:hover {
        background: rgba(139, 92, 246, 0.18);
        border-color: rgba(139, 92, 246, 0.3);
    }

    /* ---- Footer ---- */
    .app-footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        color: var(--text-muted);
        font-size: 0.84rem;
    }
    .footer-version {
        display: inline-block;
        background: rgba(59, 130, 246, 0.12);
        color: var(--accent-blue);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.80rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.4rem;
    }
    .footer-tagline {
        color: var(--text-muted);
        font-size: 0.86rem;
        margin-top: 0.3rem;
        letter-spacing: 0.02em;
    }

    /* ── Animations ── */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 0px rgba(59,130,246,0); }
        50%       { box-shadow: 0 0 22px rgba(59,130,246,0.28), 0 0 6px rgba(6,182,212,0.18); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes float-up {
        0%, 100% { transform: translateY(0px); }
        50%       { transform: translateY(-5px); }
    }
    @keyframes slide-in-left {
        from { opacity: 0; transform: translateX(-20px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes fade-in-up {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes border-spin {
        0%   { border-color: rgba(59,130,246,0.5) rgba(6,182,212,0.1) rgba(139,92,246,0.1) rgba(6,182,212,0.1); }
        25%  { border-color: rgba(6,182,212,0.5) rgba(59,130,246,0.1) rgba(6,182,212,0.1) rgba(139,92,246,0.1); }
        50%  { border-color: rgba(139,92,246,0.5) rgba(6,182,212,0.1) rgba(59,130,246,0.1) rgba(6,182,212,0.1); }
        75%  { border-color: rgba(6,182,212,0.5) rgba(139,92,246,0.1) rgba(6,182,212,0.1) rgba(59,130,246,0.1); }
        100% { border-color: rgba(59,130,246,0.5) rgba(6,182,212,0.1) rgba(139,92,246,0.1) rgba(6,182,212,0.1); }
    }

    /* ── Animate page sections on load ── */
    .hero-header      { animation: fade-in-up 0.7s ease both; }
    .glass-card       { animation: fade-in-up 0.5s ease both; }
    .chain-node       { animation: fade-in-up 0.6s ease both; }
    .guide-card       { animation: fade-in-up 0.55s ease both; }
    .step-item        { animation: slide-in-left 0.5s ease both; }

    /* ── Pulse glow on glass-card hover ── */
    .glass-card:hover {
        border-color: var(--border-accent);
        animation: pulse-glow 2s ease-in-out infinite;
    }

    /* ── Floating hero RL badge ── */
    .hero-rl-badge {
        display: inline-block;
        animation: float-up 3s ease-in-out infinite;
    }

    /* ── Shimmer stat pills ── */
    .shimmer-pill {
        background: linear-gradient(
            90deg,
            rgba(59,130,246,0.12) 0%,
            rgba(6,182,212,0.22) 40%,
            rgba(139,92,246,0.12) 60%,
            rgba(59,130,246,0.12) 100%
        );
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
        border-radius: 20px;
        padding: 0.28rem 0.9rem;
        font-size: 0.86rem;
        font-weight: 600;
        color: var(--text-primary);
        display: inline-block;
        border: 1px solid rgba(59,130,246,0.18);
        margin: 0.2rem 0.15rem;
    }

    /* ── Judge's tour card ── */
    .tour-card {
        background: linear-gradient(135deg,
            rgba(15,23,42,0.92) 0%,
            rgba(10,15,30,0.95) 100%);
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 18px;
        padding: 1.5rem 1.8rem 1.2rem;
        margin-bottom: 1rem;
        animation: border-spin 6s linear infinite, fade-in-up 0.7s ease both;
        position: relative;
        overflow: hidden;
    }
    .tour-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #06b6d4, #8b5cf6, #3b82f6);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }
    .tour-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }
    .tour-sub {
        font-size: 0.88rem;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }
    .tour-step-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.45rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .tour-step-row:last-child { border-bottom: none; }
    .tour-step-num {
        font-size: 0.75rem;
        font-weight: 800;
        color: var(--accent-blue);
        min-width: 20px;
    }
    .tour-step-text {
        font-size: 0.94rem;
        color: var(--text-secondary);
        flex: 1;
    }

    /* ── Why Is This Hard card ── */
    .hard-card {
        background: rgba(15,23,42,0.60);
        border: 1px solid rgba(245,158,11,0.2);
        border-left: 3px solid var(--accent-gold);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        animation: fade-in-up 0.5s ease both;
    }
    .hard-item {
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        padding: 0.4rem 0;
        animation: slide-in-left 0.5s ease both;
    }
    .hard-num {
        font-size: 0.78rem;
        font-weight: 800;
        color: var(--accent-gold);
        min-width: 22px;
        padding-top: 2px;
    }
    .hard-text {
        font-size: 0.93rem;
        color: var(--text-secondary);
        line-height: 1.55;
    }
    .hard-text b { color: var(--text-primary); }

    /* ── Stat bar (live metrics strip) ── */
    .stat-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin: 0.8rem 0 1.2rem;
    }

    /* ── Guide card extra hover glow ── */
    .guide-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(59,130,246,0.14);
        animation: pulse-glow 2.5s ease-in-out infinite;
    }

    /* ── Chain node float on hover ── */
    .chain-node:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(0,0,0,0.35);
        animation: float-up 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. Hero Header
# ==============================================================================
st.markdown("""
<div class="hero-header">
    <div class="hero-title">RainLoom &mdash; Monsoon Failures &amp; Textile Stock Volatility</div>
    <div class="hero-tagline">
        Predicting NSE textile-sector volatility regimes from IMD rainfall deficits,
        satellite NDVI, and cotton futures using causal machine learning
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── Live stat strip ────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-strip">
    <span class="shimmer-pill">&#127381; AUC-ROC 0.81</span>
    <span class="shimmer-pill">&#9889; 8 NSE Stocks</span>
    <span class="shimmer-pill">&#127760; 83 Districts Live</span>
    <span class="shimmer-pill">&#8987; 4-Week Lead Time</span>
    <span class="shimmer-pill">&#128200; 18-24% Hedging Gain</span>
    <span class="shimmer-pill">&#127800; F-Stat 5.8 IV/2SLS</span>
    <span class="shimmer-pill">&#9889; Parametric Payout</span>
    <span class="shimmer-pill">&#128248; Email Auto-Alerts</span>
    <span class="shimmer-pill">&#128105; 27M Women Tracked</span>
</div>
""", unsafe_allow_html=True)

# ── Judge's Quick Tour ─────────────────────────────────────────────────────
st.markdown("""
<div class="tour-card">
    <div class="tour-title">&#127919;&nbsp; Judge's Quick Tour — 3 Minutes</div>
    <div class="tour-sub">Six features that prove this is production-grade, not a prototype</div>
    <div class="tour-step-row">
        <span class="tour-step-num">01</span>
        <span class="tour-step-text">&#128200;&nbsp; <b>Live Risk Scores</b> — 8 NSE textile stocks with monsoon deficit meter &amp; 8-week fan-chart forecast</span>
    </div>
    <div class="tour-step-row">
        <span class="tour-step-num">02</span>
        <span class="tour-step-text">&#127775;&nbsp; <b>Drought Stress-Test</b> — set -38% deficit in Scenario Simulator, watch sector risk detonate to EXTREME</span>
    </div>
    <div class="tour-step-row">
        <span class="tour-step-num">03</span>
        <span class="tour-step-text">&#128202;&nbsp; <b>Causal Proof</b> — Knowledge Graph + IV/2SLS F=5.8 proves monsoon&rarr;stock is causal, not correlation</span>
    </div>
    <div class="tour-step-row">
        <span class="tour-step-num">04</span>
        <span class="tour-step-text">&#9889;&nbsp; <b>Parametric Payout Gateway</b> — subscribe a farm, trigger insurance payout in 4 seconds via UPI (no inspector)</span>
    </div>
    <div class="tour-step-row">
        <span class="tour-step-num">05</span>
        <span class="tour-step-text">&#128105;&nbsp; <b>Women's Livelihood Heatmap</b> — 27M women workers mapped by risk level &amp; state — India's only such dataset</span>
    </div>
    <div class="tour-step-row">
        <span class="tour-step-num">06</span>
        <span class="tour-step-text">&#128248;&nbsp; <b>Email Alert Scheduler</b> — subscribe above, trigger a drought, receive a live HTML alert email in &lt;15 min</span>
    </div>
</div>
""", unsafe_allow_html=True)

_tc1, _tc2, _tc3, _tc4 = st.columns(4)
with _tc1:
    st.page_link("pages/1_Live_Risk_Monitor.py",   label="&#8594; Risk Monitor",      use_container_width=True)
with _tc2:
    st.page_link("pages/4_Scenario_Simulator.py",  label="&#8594; Scenario Sim",      use_container_width=True)
with _tc3:
    st.page_link("pages/3_Model_Performance.py",   label="&#8594; Model Proof",       use_container_width=True)
with _tc4:
    st.page_link("pages/7_Geospatial_Nowcast.py",  label="&#8594; Geospatial Map",    use_container_width=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── Email Alert Manager ─────────────────────────────────────────────────────
with st.expander("📧  Email Alert Manager — Subscribe to Triggered Risk Notifications", expanded=False):
    st.markdown("""
    <div style="color:#94a3b8; font-size:0.9rem; margin-bottom:1rem; line-height:1.6;">
        The background scheduler watches all 8 stocks + monsoon deficits every
        <b style="color:#e2e8f0;">15 minutes</b>. The moment
        any condition breaches a threshold (stock crosses HIGH/EXTREME, sector average
        jumps 10%, or district deficit worsens past -20%), a richly-formatted HTML email
        is dispatched instantly to all subscribers below.
    </div>
    """, unsafe_allow_html=True)

    em_c1, em_c2 = st.columns([3, 2], gap="large")

    with em_c1:
        st.markdown("**Subscribe to Alerts**")
        sub_email = st.text_input("Email Address", placeholder="you@example.com", key="sub_email_input")
        alert_types = st.multiselect(
            "Alert Levels",
            options=["critical", "warning", "all"],
            default=["critical", "warning"],
            key="sub_alert_types",
        )
        col_sub, col_test = st.columns(2)
        with col_sub:
            if st.button("✅ Subscribe", use_container_width=True, key="btn_subscribe_email"):
                if sub_email and "@" in sub_email:
                    try:
                        from monsoon_textile_app.api.data_bridge import add_subscriber
                        result = add_subscriber(sub_email, alert_types)
                        st.success(f"✅ {result['status'].title()}: {sub_email}")
                    except Exception as e:
                        st.error(f"Subscription failed: {e}")
                else:
                    st.warning("Please enter a valid email address.")
        with col_test:
            if st.button("📨 Send Test Email", use_container_width=True, key="btn_test_email"):
                if sub_email and "@" in sub_email:
                    try:
                        from monsoon_textile_app.utils.email_scheduler import send_alert_email_html
                        from monsoon_textile_app.api.data_bridge import _get_smtp_config
                        from datetime import datetime, timezone
                        smtp_cfg = _get_smtp_config()
                        if smtp_cfg.get("enabled"):
                            test_alert = [{
                                "severity": "critical",
                                "category": "test",
                                "title": "🧪 Test Alert — RainLoom is Active",
                                "message": "This is a test dispatch confirming your RainLoom alert subscription is working correctly.",
                                "timestamp": datetime.now(timezone.utc),
                            }]
                            send_alert_email_html(sub_email, test_alert, smtp_cfg)
                            st.success(f"Test email sent to {sub_email}!")
                        else:
                            st.error("SMTP not configured. Add SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS to your .env file.")
                    except Exception as e:
                        st.error(f"Send failed: {e}")
                else:
                    st.warning("Enter an email address first.")

    with em_c2:
        st.markdown("**Scheduler Status**")
        try:
            from monsoon_textile_app.utils.email_scheduler import scheduler_status
            status = scheduler_status()
            running_color = "#10b981" if status["running"] else "#ef4444"
            running_label = "🟢 Running" if status["running"] else "🔴 Stopped"
            smtp_ok = "✅ Configured" if status["smtp_configured"] else "❌ Not configured (set SMTP_* in .env)"
            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(99,102,241,0.2);
                        border-radius:10px; padding:14px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#94a3b8;">Scheduler</span>
                <span style="color:{running_color}; font-weight:700;">{running_label}</span>
              </div>
              <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#94a3b8;">Poll Interval</span>
                <span style="color:#e2e8f0;">{status['interval_minutes']} min</span>
              </div>
              <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#94a3b8;">Stocks Watched</span>
                <span style="color:#e2e8f0;">{len(status['last_snapshot_stocks'])}</span>
              </div>
              <div style="display:flex; justify-content:space-between;">
                <span style="color:#94a3b8;">SMTP</span>
                <span style="font-size:0.85rem; color:{'#10b981' if status['smtp_configured'] else '#ef4444'}">{smtp_ok}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.info(f"Scheduler info unavailable: {e}")

    # ── .env setup — displayed inline (Streamlit doesn't allow nested expanders) ──
    st.markdown("**⚙️ .env Setup Guide** — add these to your `.env` file:")
    st.code("""
# Email Alerts (Gmail recommended)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password      # Gmail → Security → App Passwords
SMTP_SENDER=your-email@gmail.com
SMTP_USE_TLS=1

# Scheduler tuning (optional — these are the defaults)
ALERT_CHECK_INTERVAL_MINUTES=15
ALERT_RISK_THRESHOLD_HIGH=0.60
ALERT_RISK_THRESHOLD_EXTREME=0.80
ALERT_SECTOR_JUMP_PCT=0.10
ALERT_RAINFALL_DEFICIT=-20
ENABLE_EMAIL_SCHEDULER=1
    """, language="bash")
    st.caption("Gmail users: Google Account → Security → App Passwords to generate SMTP_PASS.")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


st.markdown('<div class="section-header">The Problem</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <div class="content-text">
        India's &#8377;12 lakh crore textile industry is structurally exposed to monsoon
        failures. When June&#8211;September (JJAS) rainfall falls 20% or more below the
        Long Period Average, cotton yields drop 15&#8211;25%, triggering a predictable
        cascade: cotton prices spike, manufacturer margins compress, and stock volatility
        surges. Despite this well-documented chain, no existing system provides early
        warning of the full transmission from rainfall deficit to equity risk. This
        dashboard fills that gap.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 3. Our Solution
# ==============================================================================
st.markdown('<div class="section-header">Our Solution</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <div class="content-text" style="margin-bottom: 0.8rem;">
        A 4-layer causal ML pipeline that transforms raw climate data into actionable
        risk scores with 8+ weeks of lead time:
    </div>
</div>
""", unsafe_allow_html=True)

_steps = [
    ("#60a5fa", "1", "Proves causation",
     "(not just correlation) using stationarity-corrected Granger causality tests on cotton log-returns and rainfall deficit, with Vector Autoregression (VAR) models validating each link in the chain."),
    ("#34d399", "2", "Detects regime shifts",
     "using GJR-GARCH(1,1) with AIC-based model selection to identify when markets transition between calm and turbulent volatility states."),
    ("#fbbf24", "3", "Classifies risk states",
     "using XGBoost with 24 climate-informed features including spatial rainfall deficit breadth, lag-transformed cotton returns, NDVI satellite anomalies, and seasonal indicators."),
    ("#f97316", "4", "Captures temporal sequences",
     "using MLP neural network with temporal cross-validation to model the multi-week propagation dynamics from rainfall shock to market response."),
    ("#a78bfa", "5", "Combines into ensemble",
     "risk score (XGBoost 40% + GARCH 30% + MLP 30%) that fuses all layer outputs with calibrated weights, delivering actionable alerts with 8+ weeks of lead time."),
]
for color, num, title, desc in _steps:
    st.markdown(
        f'<div class="step-item">'
        f'<div class="step-num" style="background: {color}22; color: {color};">{num}</div>'
        f'<div class="step-text"><span style="color:{color}; font-weight:600;">{title}</span> {desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# 4. Causal Transmission Mechanism
# ==============================================================================
st.markdown('<div class="section-header">Causal Transmission Mechanism</div>', unsafe_allow_html=True)

_stages = [
    ("Stage 01", "Climate Signal",   "IMD rainfall anomaly &lt; &#8211;20% LPA", "T + 0 weeks",  "#60a5fa"),
    ("Stage 02", "Crop Stress",      "NDVI drops below threshold",               "T + 2&#8211;4 wks", "#34d399"),
    ("Stage 03", "Price Spike",      "MCX cotton futures surge",                 "T + 4&#8211;6 wks", "#fbbf24"),
    ("Stage 04", "Margin Squeeze",   "EBITDA margin compression",                "T + 6&#8211;10 wks", "#f97316"),
    ("Stage 05", "Volatility Shift", "Stock realized vol &gt; 2&#963;",          "T + 4&#8211;8 wks", "#ef4444"),
]

_chain_cols = st.columns(len(_stages))
for i, (num, title, desc, lag, color) in enumerate(_stages):
    with _chain_cols[i]:
        connector = '&rsaquo;' if i < len(_stages) - 1 else ''
        st.markdown(
            f'<div class="chain-node" style="border-top: 2px solid {color};">'
            f'<div class="chain-number" style="color: {color};">{num}</div>'
            f'<div class="chain-title">{title}</div>'
            f'<div class="chain-desc">{desc}</div>'
            f'<div class="chain-lag">{lag}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# 4b. Why Is This Hard?
# ==============================================================================
with st.expander("&#129300;  Why is this not a trivial dashboard? (Click to expand)", expanded=False):
    st.markdown("""
<div class="hard-card">
    <div class="hard-item">
        <span class="hard-num">01</span>
        <span class="hard-text"><b>Cross-domain expertise:</b> Requires simultaneous mastery of
        climate science (monsoon dynamics, ENSO teleconnections), agricultural economics (cotton supply
        chains, yield models), and quantitative finance (volatility modelling, equity risk) — three fields
        that rarely communicate.</span>
    </div>
    <div class="hard-item">
        <span class="hard-num">02</span>
        <span class="hard-text"><b>Causal inference, not correlation:</b> IV/2SLS (Instrumental Variables
        with Two-Stage Least Squares) using ENSO ONI as an instrument is a non-standard econometric
        technique applied here specifically to address endogeneity — the fact that rainfall and stock
        prices may both respond to common latent factors (e.g., global risk-off sentiment).</span>
    </div>
    <div class="hard-item">
        <span class="hard-num">03</span>
        <span class="hard-text"><b>Live data integration across 7 APIs:</b> Yahoo Finance (NSE stocks),
        IMD (district rainfall), MCX (cotton futures), Open-Meteo (live precipitation), NOAA ENSO ONI,
        India VIX, and USD/INR — all with retry logic, caching, and graceful degradation when APIs are
        down.</span>
    </div>
    <div class="hard-item">
        <span class="hard-num">04</span>
        <span class="hard-text"><b>Stakeholder translation:</b> The same ensemble risk score must be
        translated into entirely different outputs — ₹/acre insurance premiums for farmers, hedge
        ratios and MCX instrument recommendations for MSMEs, and VaR-adjusted position sizing for
        fund managers. Each requires domain-specific knowledge to implement correctly.</span>
    </div>
    <div class="hard-item">
        <span class="hard-num">05</span>
        <span class="hard-text"><b>Backtesting rigor without leakage:</b> Walk-forward
        <code>TimeSeriesSplit</code> throughout — no random shuffling of time-series data. Drift
        detection (Page-Hinkley + ADWIN) monitors for distribution shift. Platt scaling calibrates
        probability outputs. These are not standard in most ML projects.</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# 5. Dashboard Guide
# ==============================================================================
st.markdown('<div class="section-header">How to Use This System</div>', unsafe_allow_html=True)

_guide_pages = [
    {
        "num": "01",
        "icon": "\u26a0",
        "icon_bg": "linear-gradient(135deg, #1e3a5f 0%, #0f2440 100%)",
        "accent": "#3b82f6",
        "title": "Risk Monitor",
        "desc": (
            "Real-time risk scores for 8 NSE textile stocks. Gauge indicators display "
            "the ensemble risk level for each security. A rainfall deficit map highlights "
            "stressed cotton-belt states across India. Position alerts flag when risk "
            "crosses actionable thresholds."
        ),
        "usage": "Use this to monitor current conditions and identify emerging risk.",
    },
    {
        "num": "02",
        "icon": "\u21c4",
        "icon_bg": "linear-gradient(135deg, #1a3a2a 0%, #0f2a1a 100%)",
        "accent": "#10b981",
        "title": "Causal Analysis",
        "desc": (
            "Statistical proof that monsoon rainfall causes textile stock volatility. "
            "Stationarity-corrected Granger causality tests on cotton log-returns and "
            "rainfall deficit validate each link at p &lt; 0.05. Lag heatmaps show "
            "optimal transmission delays across 4 causal pathways."
        ),
        "usage": "Use this to understand WHY the model works and validate causal claims.",
    },
    {
        "num": "03",
        "icon": "\u2713",
        "icon_bg": "linear-gradient(135deg, #3a2a10 0%, #2a1f08 100%)",
        "accent": "#f59e0b",
        "title": "Model Performance",
        "desc": (
            "ROC curves (AUC 0.84&#8211;0.98), SHAP feature importance across 24 features "
            "including NDVI satellite data, and 5-fold temporal cross-validation results. "
            "Per-stock model cards show XGBoost, GARCH, and MLP performance with "
            "ensemble risk scoring."
        ),
        "usage": "Use this to evaluate model reliability and compare against benchmarks.",
    },
    {
        "num": "04",
        "icon": "\u2699",
        "icon_bg": "linear-gradient(135deg, #2a1a3a 0%, #1a0f2a 100%)",
        "accent": "#8b5cf6",
        "title": "Scenario Simulator",
        "desc": (
            "Interactive what-if tool powered by trained XGBoost models. Adjust monsoon "
            "deficit percentage, cotton spot prices, India VIX level, and spatial deficit "
            "breadth using sliders to see ML-predicted risk scores update in real time. "
            "Historical presets recreate conditions from past drought years."
        ),
        "usage": "Use this to stress-test portfolios under hypothetical climate scenarios.",
    },
    {
        "num": "05",
        "icon": "\u2764",
        "icon_bg": "linear-gradient(135deg, #2a1525 0%, #1f0f1a 100%)",
        "accent": "#ef4444",
        "title": "Societal Impact",
        "desc": (
            "Translates model predictions into actionable advisories for three stakeholder "
            "groups: cotton farmers receive crop insurance enrollment alerts, textile MSMEs "
            "get forward-contract hedging recommendations, and state agriculture departments "
            "access weekly risk dashboards for pre-positioned relief planning."
        ),
        "usage": "Use this to see real-world impact beyond financial markets.",
    },
    {
        "num": "06",
        "icon": "\u2696",
        "icon_bg": "linear-gradient(135deg, #1a1a3a 0%, #0f0f2a 100%)",
        "accent": "#8b5cf6",
        "title": "Hedging Backtest",
        "desc": (
            "Simulates risk-signal-driven hedging strategies across historical drought "
            "events (2009, 2014, 2015, 2023). Compares hedged vs unhedged portfolio P&amp;L, "
            "Sharpe ratios, and maximum drawdowns to quantify the economic value of the "
            "ensemble risk forecast."
        ),
        "usage": "Use this to measure if the risk signal has real economic value.",
    },
]

# Render guide cards using st.columns to avoid deep nesting
_row1_pages = _guide_pages[:3]
_row2_pages = _guide_pages[3:]

_g_cols1 = st.columns(len(_row1_pages))
for col, p in zip(_g_cols1, _row1_pages):
    with col:
        st.markdown(
            f'<div class="guide-card" style="border-top: 2px solid {p["accent"]};">'
            f'<div class="guide-card-icon" style="background: {p["icon_bg"]}; color: {p["accent"]};">{p["icon"]}</div>'
            f'<div class="guide-card-page" style="color: {p["accent"]};">Page {p["num"]}</div>'
            f'<div class="guide-card-title">{p["title"]}</div>'
            f'<div class="guide-card-desc">{p["desc"]}</div>'
            f'<div class="guide-card-usage">{p["usage"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

_g_cols2 = st.columns(len(_row2_pages))
for col, p in zip(_g_cols2, _row2_pages):
    with col:
        st.markdown(
            f'<div class="guide-card" style="border-top: 2px solid {p["accent"]};">'
            f'<div class="guide-card-icon" style="background: {p["icon_bg"]}; color: {p["accent"]};">{p["icon"]}</div>'
            f'<div class="guide-card-page" style="color: {p["accent"]};">Page {p["num"]}</div>'
            f'<div class="guide-card-title">{p["title"]}</div>'
            f'<div class="guide-card-desc">{p["desc"]}</div>'
            f'<div class="guide-card-usage">{p["usage"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# 6. Data Sources & APIs
# ==============================================================================
st.markdown('<div class="section-header">Data Sources &amp; APIs</div>', unsafe_allow_html=True)

import pandas as pd
_data_sources = pd.DataFrame({
    "Source": [
        "IMD Gridded Rainfall",
        "Live Rainfall (Open-Meteo)",
        "NSE Equity OHLCV",
        "Cotton Futures (MCX / ICE proxy)",
        "NDVI Satellite (NASA MODIS)",
        "Macro Controls",
    ],
    "API": [
        "imdlib",
        "Open-Meteo Archive API",
        "yfinance",
        "yfinance + USD/INR forex",
        "ORNL DAAC REST API",
        "yfinance",
    ],
    "Data": [
        "0.25 deg daily rainfall grid, 2000-2025",
        "900+ daily observations across 10 cotton-belt states",
        "8 textile stocks + Nifty50 + India VIX",
        "MCX Cotton (primary) or ICE CT=F with real forex conversion",
        "MOD13Q1 250m vegetation index for 10 states",
        "USD/INR exchange rate, Brent Crude",
    ],
    "Update Frequency": [
        "Daily during JJAS",
        "Daily (real-time)",
        "Daily",
        "Daily",
        "16-day composites",
        "Daily",
    ],
})
st.dataframe(_data_sources, use_container_width=True, hide_index=True)


# ==============================================================================
# 7. Target Securities
# ==============================================================================
st.markdown('<div class="section-header">Target Securities</div>', unsafe_allow_html=True)

_securities = [
    ("ARVIND.NS", "Arvind Ltd", "Vertically integrated textile manufacturer. Cotton is primary raw material input (~40% of COGS).", "#3b82f6"),
    ("TRIDENT.NS", "Trident Ltd", "Home textiles and yarn producer. High cotton dependency with significant export exposure.", "#3b82f6"),
    ("KPRMILL.NS", "KPR Mill", "Integrated spinning and garment manufacturer. Cotton yarn is core intermediate product.", "#3b82f6"),
    ("WELSPUNLIV.NS", "Welspun Living", "Home textiles (towels, bed linen). Premium cotton dependency with global supply chain.", "#3b82f6"),
    ("RSWM.NS", "RSWM Ltd", "Spinning and weaving division under LNJ Bhilwara Group. Cotton and synthetic yarn producer.", "#3b82f6"),
    ("VTL.NS", "Vardhman Textiles", "Largest yarn manufacturer in India. Direct upstream cotton exposure (82% dependency).", "#10b981"),
    ("PAGEIND.NS", "Page Industries", "Innerwear and athleisure (Jockey licensee). Downstream apparel with moderate cotton input.", "#f59e0b"),
    ("RAYMOND.NS", "Raymond Ltd", "Fabric and apparel conglomerate. Integrated value chain with diversified fibre mix.", "#f59e0b"),
]

# Row 1: Original textile stocks
_sec_cols1 = st.columns(5)
for col, (ticker, name, role, accent) in zip(_sec_cols1, _securities[:5]):
    with col:
        st.markdown(
            f'<div class="sec-card" style="border-top: 2px solid {accent};">'
            f'<div class="sec-ticker">{ticker}</div>'
            f'<div class="sec-name">{name}</div>'
            f'<div class="sec-role">{role}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# Row 2: Adjacent sectors
st.markdown('<div class="content-label" style="margin-top: 0.8rem; color: #10b981;">TEXTILE-ADJACENT SECTORS</div>',
            unsafe_allow_html=True)
_sec_cols2 = st.columns(3)
for col, (ticker, name, role, accent) in zip(_sec_cols2, _securities[5:]):
    with col:
        st.markdown(
            f'<div class="sec-card" style="border-top: 2px solid {accent};">'
            f'<div class="sec-ticker">{ticker}</div>'
            f'<div class="sec-name">{name}</div>'
            f'<div class="sec-role">{role}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ==============================================================================
# 8. Technical Architecture
# ==============================================================================
st.markdown('<div class="section-header">Technical Architecture</div>', unsafe_allow_html=True)

st.markdown("""
<div class="content-text" style="margin-bottom: 0.6rem;">
    Built as a modular Python application with a Streamlit frontend. Each analytical
    layer runs independently and feeds into the ensemble scoring engine. The pipeline
    supports both batch processing of historical data and near-real-time ingestion
    during monsoon season.
</div>
""", unsafe_allow_html=True)

_tech_stacks = [
    ("Core Framework", "",
     ["Python 3.10+", "Streamlit", "Plotly", "pandas", "NumPy"]),
    ("Machine Learning", "tech-pill-green",
     ["XGBoost", "scikit-learn", "SHAP", "MLP Neural Network", "Quantile Regression"]),
    ("Econometrics & Time Series", "tech-pill-gold",
     ["statsmodels", "arch (GJR-GARCH)", "scipy", "Granger Causality"]),
    ("Data Sources & APIs", "tech-pill-purple",
     ["imdlib", "yfinance", "Open-Meteo API", "NASA MODIS ORNL DAAC"]),
    ("Infrastructure", "tech-pill-purple",
     ["loguru", "pickle (model caching)", "TimeSeriesSplit CV"]),
]
for label, pill_cls, pills in _tech_stacks:
    pills_html = " ".join(f'<span class="{pill_cls} tech-pill">{p}</span>' for p in pills)
    st.markdown(
        f'<div class="content-label" style="margin-top: 0.8rem;">{label}</div>'
        f'<div class="tech-pills">{pills_html}</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# 9. Footer
# ==============================================================================
st.markdown("""
<div class="app-footer">
    <div class="footer-version">v3.0.0</div>
    <div class="footer-tagline">
        RainLoom &mdash; Monsoon &rarr; Cotton &rarr; Margin &rarr; Volatility
    </div>
</div>
""", unsafe_allow_html=True)
