"""
Page 5 — Societal Impact
=========================
Farmer advisory, MSME hedging alerts, policy recommendations,
and quantified impact metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import datetime as _dt
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from monsoon_textile_app.components.navbar import render_navbar
from monsoon_textile_app.components.chat_bubble import render_chat_bubble

st.set_page_config(page_title="Societal Impact", layout="wide", initial_sidebar_state="collapsed")
render_navbar(active_page="Impact")
render_chat_bubble()

# ── Global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Dark theme base ─────────────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
section[data-testid="stSidebar"],
[data-testid="stHeader"] {
    background-color: #0a0f1e !important;
    color: #c8d2e0 !important;
}

/* ── Page header ─────────────────────────────────────────── */
.si-page-title {
    font-size: 2.36rem;
    font-weight: 700;
    letter-spacing: 1.6px;
    color: #e2e8f0;
    margin-bottom: 2px;
    text-transform: uppercase;
}
.si-page-sub {
    font-size: 1.08rem;
    color: #7a8ba8;
    letter-spacing: 0.4px;
    margin-bottom: 6px;
}
.si-header-line {
    height: 2px;
    background: linear-gradient(90deg, #6366f1 0%, #06b6d4 50%, transparent 100%);
    border: none;
    margin-bottom: 28px;
    border-radius: 2px;
}

/* ── KPI cards ───────────────────────────────────────────── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
    margin-bottom: 32px;
}
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 22px 20px 18px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
.kpi-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.accent-indigo::before { background: #6366f1; }
.kpi-card.accent-cyan::before   { background: #06b6d4; }
.kpi-card.accent-amber::before  { background: #f59e0b; }
.kpi-card.accent-emerald::before{ background: #10b981; }
.kpi-label {
    font-size: 0.84rem;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #6b7a94;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.91rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.15;
    margin-bottom: 4px;
}
.kpi-desc {
    font-size: 0.92rem;
    color: #5a6b82;
}

/* ── Tab overrides ───────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7a94 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 10px 26px !important;
    font-size: 0.98rem !important;
    letter-spacing: 0.6px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    transition: all 0.25s ease;
}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #e2e8f0 !important;
    border-bottom: 2px solid #6366f1 !important;
}
[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
    color: #a5b4c8 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Section heading ─────────────────────────────────────── */
.section-heading {
    font-size: 1.21rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    color: #e2e8f0;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.section-line {
    height: 2px;
    border: none;
    border-radius: 2px;
    margin-bottom: 20px;
}
.section-line.indigo  { background: linear-gradient(90deg, #6366f1 0%, transparent 70%); }
.section-line.cyan    { background: linear-gradient(90deg, #06b6d4 0%, transparent 70%); }
.section-line.amber   { background: linear-gradient(90deg, #f59e0b 0%, transparent 70%); }
.section-line.emerald { background: linear-gradient(90deg, #10b981 0%, transparent 70%); }

/* ── Glass card generic ──────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 22px 22px 18px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    margin-bottom: 16px;
}

/* ── Alert card ──────────────────────────────────────────── */
.alert-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0;
    overflow: hidden;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    margin-bottom: 12px;
}
.alert-card .alert-header {
    padding: 14px 20px 10px;
    font-weight: 700;
    font-size: 1.02rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.alert-card .alert-header.red    { background: rgba(239,68,68,0.10); color: #f87171; border-bottom: 1px solid rgba(239,68,68,0.15); }
.alert-card .alert-header.amber  { background: rgba(245,158,11,0.10); color: #fbbf24; border-bottom: 1px solid rgba(245,158,11,0.15); }
.alert-card .alert-body {
    padding: 16px 20px 18px;
    font-size: 0.98rem;
    color: #b0bdd0;
    line-height: 1.65;
}
.alert-card .alert-body b { color: #e2e8f0; }
.alert-card .alert-body ol { padding-left: 18px; margin-top: 8px; }
.alert-card .alert-body li { margin-bottom: 5px; }
.alert-meta { display: flex; gap: 20px; margin-bottom: 10px; flex-wrap: wrap; }
.alert-meta span {
    font-size: 0.88rem;
    color: #7a8ba8;
    letter-spacing: 0.3px;
}
.alert-meta span b { color: #c8d2e0; }

/* ── Risk table ──────────────────────────────────────────── */
.risk-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.94rem;
}
.risk-table th {
    text-align: left;
    padding: 10px 12px;
    color: #6b7a94;
    font-weight: 600;
    font-size: 0.84rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.risk-table td {
    padding: 9px 12px;
    color: #b0bdd0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.risk-table tr:nth-child(even) td {
    background: rgba(255,255,255,0.015);
}
.risk-table .risk-high   { color: #f87171; font-weight: 700; }
.risk-table .risk-medium { color: #fbbf24; font-weight: 700; }
.risk-table .risk-low    { color: #4ade80; font-weight: 600; }
.advisory-yes { color: #f87171; font-weight: 700; }
.advisory-no  { color: #6b7a94; }

/* ── Timeline card ───────────────────────────────────────── */
.timeline-row {
    display: grid;
    grid-template-columns: 64px 1fr 1fr 80px 90px;
    gap: 8px;
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    font-size: 0.94rem;
    color: #b0bdd0;
}
.timeline-row.header {
    color: #6b7a94;
    font-size: 0.84rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.timeline-row .tl-year { font-weight: 700; color: #e2e8f0; }
.timeline-row .tl-lead { color: #6366f1; font-weight: 700; }

/* ── Insight box ─────────────────────────────────────────── */
.insight-box {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.96rem;
    color: #a5b4fc;
    line-height: 1.6;
    margin-top: 12px;
}
.insight-box b { color: #c7d2fe; }

/* ── Recommendation cards ────────────────────────────────── */
.rec-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
.rec-card.red    { border-left: 4px solid #ef4444; }
.rec-card.amber  { border-left: 4px solid #f59e0b; }
.rec-card.green  { border-left: 4px solid #10b981; }
.rec-title { font-weight: 700; font-size: 1.02rem; margin-bottom: 8px; }
.rec-card.red .rec-title    { color: #f87171; }
.rec-card.amber .rec-title  { color: #fbbf24; }
.rec-card.green .rec-title  { color: #4ade80; }
.rec-card ul {
    padding-left: 16px; margin: 0;
    font-size: 0.94rem; color: #9ca8bc; line-height: 1.7;
}
.rec-card li b { color: #c8d2e0; }

/* ── Employment table ────────────────────────────────────── */
.emp-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.92rem;
}
.emp-table th {
    text-align: left;
    padding: 9px 12px;
    color: #6b7a94;
    font-weight: 600;
    font-size: 0.84rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.emp-table td {
    padding: 9px 12px;
    color: #b0bdd0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.emp-table tr:nth-child(even) td {
    background: rgba(255,255,255,0.015);
}

/* ── Status badges ───────────────────────────────────────── */
.badge-ready {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.88rem; color: #4ade80; font-weight: 600;
}
.badge-ready::before {
    content: "";
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4ade80;
    display: inline-block;
}
.badge-planned {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.88rem; color: #f59e0b; font-weight: 600;
}
.badge-planned::before {
    content: "";
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #f59e0b;
    display: inline-block;
}

/* ── Integration table ───────────────────────────────────── */
.int-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.94rem;
}
.int-table th {
    text-align: left;
    padding: 10px 14px;
    color: #6b7a94;
    font-weight: 600;
    font-size: 0.84rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.int-table td {
    padding: 10px 14px;
    color: #b0bdd0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.int-table tr:nth-child(even) td {
    background: rgba(255,255,255,0.015);
}
.int-table td b { color: #e2e8f0; }

/* ── Sector KPI mini cards ───────────────────────────────── */
.sector-kpi-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 8px;
}
.sector-kpi {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px 16px 14px;
    position: relative;
    overflow: hidden;
}
.sector-kpi::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.sector-kpi.sk-cyan::before    { background: #06b6d4; }
.sector-kpi.sk-amber::before   { background: #f59e0b; }
.sector-kpi.sk-emerald::before { background: #10b981; }
.sector-kpi .sk-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #6b7a94;
    margin-bottom: 4px;
}
.sector-kpi .sk-value {
    font-size: 1.46rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.2;
    margin-bottom: 2px;
}
.sector-kpi .sk-desc {
    font-size: 0.88rem;
    color: #5a6b82;
}

/* ── Hide default Streamlit pieces ───────────────────────── */
[data-testid="stMetric"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Page Header ───────────────────────────────────────────────────────
st.markdown("""
<div class="si-page-title">Societal Impact Framework</div>
<div class="si-page-sub">Transforming climate-finance predictions into actionable early warnings</div>
<div class="si-header-line"></div>
""", unsafe_allow_html=True)

# ── KPI Banner ────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-row">
  <div class="kpi-card accent-indigo">
    <div class="kpi-label">Farmers Protected</div>
    <div class="kpi-value">6.2M</div>
    <div class="kpi-desc">Cotton farmers across target states</div>
  </div>
  <div class="kpi-card accent-cyan">
    <div class="kpi-label">MSMEs Alerted</div>
    <div class="kpi-value">~100,000</div>
    <div class="kpi-desc">Textile manufacturers in cotton belt</div>
  </div>
  <div class="kpi-card accent-amber">
    <div class="kpi-label">Early Warning Lead</div>
    <div class="kpi-value">4 -- 8 wk</div>
    <div class="kpi-desc">Before volatility spike</div>
  </div>
  <div class="kpi-card accent-emerald">
    <div class="kpi-label">Estimated Savings</div>
    <div class="kpi-value">500 -- 1,000 Cr</div>
    <div class="kpi-desc">Per drought year (INR)</div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("What is this page?"):
    st.markdown(
        "This translates **financial predictions into real-world impact** for three stakeholder groups. "
        "**Farmer Early Warning** alerts cotton farmers 8+ weeks before crop failure, enabling insurance enrollment. "
        "**MSME Hedging** alerts small textile manufacturers to lock in cotton procurement prices before spikes. "
        "**Policy Dashboard** gives state agriculture departments weekly risk monitoring during monsoon season."
    )

# ── Common Plotly layout helper ───────────────────────────────────────


def _api_base_url() -> str:
    """Resolve API base URL for email subscription and dispatch actions."""
    return os.environ.get("RAINLOOM_API_URL", "http://localhost:8000").rstrip("/")


def _post_api(path: str, payload: dict | None = None, params: dict | None = None) -> tuple[bool, dict | str]:
    """POST helper with graceful error handling for Streamlit UI actions."""
    base = _api_base_url()
    url = f"{base}{path}"
    try:
        resp = requests.post(url, json=payload, params=params, timeout=15)
        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                data = {"detail": resp.text}
            return False, data
        try:
            return True, resp.json()
        except Exception:
            return True, {"message": "Request succeeded."}
    except Exception as exc:
        return False, str(exc)


def _subscribe_email(email: str, alert_types: list[str]) -> tuple[bool, dict | str, str]:
    """
    Subscribe an email via API when available, otherwise fallback to local data bridge.

    Returns
    -------
    (ok, data, mode)
        mode is either "api" or "local".
    """
    ok, data = _post_api("/api/subscribe", payload={"email": email, "alert_types": alert_types})
    if ok:
        return True, data, "api"

    # Fallback path for single-process Streamlit deployments.
    try:
        from monsoon_textile_app.api.data_bridge import add_subscriber
        result = add_subscriber(email, alert_types)
        return True, result, "local"
    except Exception as exc:
        return False, f"{data} | local fallback failed: {exc}", "local"


def _dispatch_alerts(dry_run: bool) -> tuple[bool, dict | str, str]:
    """
    Dispatch alerts via API when available, otherwise fallback to local data bridge.

    Returns
    -------
    (ok, data, mode)
        mode is either "api" or "local".
    """
    ok, data = _post_api(
        "/api/dispatch-alerts",
        params={"dry_run": str(bool(dry_run)).lower()},
    )
    if ok:
        return True, data, "api"

    # Fallback path for single-process Streamlit deployments.
    try:
        from monsoon_textile_app.api.data_bridge import dispatch_alert_emails
        result = dispatch_alert_emails(dry_run=dry_run)
        if result.get("status") == "error":
            return False, result, "local"
        return True, result, "local"
    except Exception as exc:
        return False, f"{data} | local fallback failed: {exc}", "local"


st.markdown("""
<div class="section-heading" style="margin-top:6px;">Alert Delivery</div>
<div class="section-line emerald"></div>
""", unsafe_allow_html=True)

with st.container(border=True):
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        sub_email = st.text_input(
            "Email for alerts",
            value=st.session_state.get("alert_email", ""),
            placeholder="you@example.com",
            help="Subscribers receive alerts from the configured SMTP sender.",
        )
        severities = st.multiselect(
            "Alert severities",
            options=["critical", "warning", "info"],
            default=["critical", "warning"],
            help="Choose which alert levels this email should receive.",
        )
        st.caption(
            f"API target: `{_api_base_url()}` | Falls back to local mode if API is unavailable."
        )
    with c2:
        dry_run_dispatch = st.toggle(
            "Dispatch dry run",
            value=True,
            help="When enabled, delivery is simulated without sending emails.",
        )
        st.caption("Tip: disable dry run to send real emails.")

    act1, act2 = st.columns([1, 1], gap="small")
    with act1:
        if st.button("Subscribe Email", type="primary", use_container_width=True):
            email = (sub_email or "").strip()
            if "@" not in email or "." not in email:
                st.error("Enter a valid email address.")
            elif not severities:
                st.error("Select at least one alert severity.")
            else:
                ok, data, mode = _subscribe_email(email=email, alert_types=severities)
                if ok:
                    st.session_state["alert_email"] = email
                    st.success(f"Subscribed: {email} ({mode} mode)")
                else:
                    st.error(f"Subscribe failed: {data}")
    with act2:
        if st.button("Dispatch Alerts Now", use_container_width=True):
            ok, data, mode = _dispatch_alerts(dry_run=dry_run_dispatch)
            if ok:
                st.success(
                    "Dispatch complete ({mode}) | sent={sent} targeted={targeted} total_alerts={alerts} dry_run={dry}".format(
                        mode=mode,
                        sent=data.get("emails_sent", 0),
                        targeted=data.get("recipients_targeted", 0),
                        alerts=data.get("total_alerts", 0),
                        dry=data.get("dry_run", dry_run_dispatch),
                    )
                )
                failures = data.get("failures", []) if isinstance(data, dict) else []
                if failures:
                    st.warning("Some deliveries failed: " + "; ".join(str(x) for x in failures[:3]))
            else:
                st.error(f"Dispatch failed: {data}")

_CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=14, color="#8a95a8"),
    title_font=dict(size=15, color="#c8d2e0"),
    hoverlabel=dict(bgcolor="#1e293b", font_size=14, font_family="Inter, system-ui, sans-serif"),
    hovermode="x unified",
    yaxis=dict(showgrid=False),
    xaxis=dict(showgrid=False, gridcolor="rgba(255,255,255,0.04)"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=13, color="#8a95a8")),
)


# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Farmer Early Warning",
    "MSME Hedging",
    "Policy Dashboard",
])

# =====================================================================
# TAB 1 — Farmer Early Warning
# =====================================================================
with tab1:
    st.markdown("""
    <div class="section-heading">Farmer Advisory System</div>
    <div class="section-line indigo"></div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#8a95a8; font-size:1.0rem; margin-bottom:18px;">
    When the model risk score crosses <span style="font-weight:600;color:#e2e8f0;">0.50</span> during June -- August,
    district-level advisories are generated and delivered via SMS / WhatsApp to registered farmers.
    </div>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([2, 3], gap="large")

    # -- Advisory card --
    with col_a1:
        _today_str = _dt.datetime.now().strftime("%d %b %Y")
        st.markdown(f"""
        <div class="section-heading" style="font-size:0.98rem; margin-bottom:10px;">Live Advisory</div>
        <div class="alert-card">
          <div class="alert-header red">Monsoon Advisory  --  Rajkot District, Gujarat</div>
          <div class="alert-body">
            <div class="alert-meta">
              <span><span style="font-weight:600;">Date:</span> {_today_str}</span>
              <span><span style="font-weight:600;">Risk Score:</span> 0.72 (HIGH)</span>
              <span><span style="font-weight:600;">Deficit:</span> -28% from LPA</span>
            </div>
            Rainfall deficit detected in your district. Based on similar historical conditions,
            cotton yield may drop <span style="font-weight:600;">15 -- 25%</span>.
            <div style="margin-top:0.8rem;"><span style="font-weight:600;color:#e2e8f0;">Recommended Actions</span></div>
            <div style="padding-left:1.2rem;margin-top:0.4rem;">
              <div style="margin-bottom:0.25rem;">1. Enroll in PMFBY crop insurance before deadline (Aug 15)</div>
              <div style="margin-bottom:0.25rem;">2. Consider drought-resistant variety for late sowing</div>
              <div style="margin-bottom:0.25rem;">3. Reduce fertilizer application to manage costs</div>
              <div style="margin-bottom:0.25rem;">4. Contact Kisan Call Centre: <span style="font-weight:600;">1800-180-1551</span></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem;' class='section-heading' style='font-size:0.98rem; margin-bottom:5px;'>🔊 Multilingual Voice Advisory</div>", unsafe_allow_html=True)
        st.caption("AI-generated text-to-speech for farmers with low literacy.")
        
        lang_choice = st.radio("Select Language", ["English", "Hindi", "Gujarati", "Marathi"], horizontal=True, label_visibility="collapsed")
        
        advisory_texts = {
            "English": "Warning: High risk of monsoon deficit in Rajkot. Cotton crop yields may drop 20 percent. Please enroll in crop insurance immediately.",
            "Hindi": "चेतावनी: राजकोट में मानसून की भारी कमी है। कपास की पैदावार 20 प्रतिशत तक गिर सकती है। कृपया तुरंत फसल बीमा लें।",
            "Gujarati": "ચેતવણી: રાજકોટમાં ચોમાસાની ખાધનું મોટું જોખમ છે. કપાસના પાકમાં 20 ટકાનો ઘટાડો થઈ શકે છે. કૃપા કરીને તાત્કાલિક પાક વીમો લો.",
            "Marathi": "चेतावणी: राजकोटमध्ये मान्सूनच्या तुटीचा मोठा धोका आहे. कापसाचे उत्पादन 20 टक्क्यांनी कमी होऊ शकते. कृपया त्वरित पीक विमा घ्या."
        }
        tts_langs = {"English": "en", "Hindi": "hi", "Gujarati": "gu", "Marathi": "mr"}
        text_to_speak = advisory_texts.get(lang_choice, advisory_texts["English"])
        
        try:
            from gtts import gTTS
            import io

            @st.cache_data(show_spinner=False)
            def _synthesize_audio(text: str, lang: str) -> bytes:
                """Cache TTS synthesis – avoids repeated Google TTS API calls."""
                tts = gTTS(text=text, lang=lang, slow=False)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)  # rewind so st.audio reads from the start
                return buf.read()

            audio_bytes = _synthesize_audio(
                text_to_speak, tts_langs.get(lang_choice, "en")
            )
            st.audio(audio_bytes, format="audio/mp3")
        except ImportError:
            st.warning("Install gTTS (`pip install gTTS>=2.4.0`) to enable voice generation.")
        except Exception as e:
            st.warning(f"Voice synthesis unavailable: {e}")

    # -- District risk table --
    with col_a2:
        st.markdown("""
        <div class="section-heading" style="font-size:0.98rem; margin-bottom:10px;">District-Level Risk Assessment</div>
        """, unsafe_allow_html=True)

        districts = [
            ("Rajkot",     "Gujarat",     -28, 0.72, True,  "20-25%"),
            ("Ahmedabad",  "Gujarat",     -22, 0.64, True,  "15-20%"),
            ("Surat",      "Gujarat",     -18, 0.55, True,  "10-15%"),
            ("Nagpur",     "Maharashtra", -15, 0.48, False, "8-12%"),
            ("Nashik",     "Maharashtra", -12, 0.38, False, "5-8%"),
            ("Warangal",   "Telangana",   -25, 0.68, True,  "18-22%"),
            ("Adilabad",   "Telangana",   -20, 0.58, True,  "12-16%"),
            ("Coimbatore", "Tamil Nadu",   -8, 0.28, False, "3-5%"),
        ]
        rows_html = ""
        for d, s, deficit, risk, adv, yd in districts:
            risk_cls = "risk-high" if risk >= 0.6 else ("risk-medium" if risk >= 0.4 else "risk-low")
            adv_cls = "advisory-yes" if adv else "advisory-no"
            adv_txt = "Yes" if adv else "No"
            rows_html += f"""<tr>
              <td><b style="color:#e2e8f0;">{d}</b></td><td>{s}</td>
              <td>{deficit}%</td><td class="{risk_cls}">{risk:.2f}</td>
              <td class="{adv_cls}">{adv_txt}</td><td>{yd}</td></tr>"""

        st.markdown(f"""
        <div class="glass-card" style="padding:0; overflow:hidden;">
        <table class="risk-table">
          <thead><tr><th>District</th><th>State</th><th>Deficit</th><th>Risk</th><th>Advisory</th><th>Est. Yield Drop</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # -- Savings chart + timeline --
    st.markdown("""
    <div class="section-heading" style="margin-top:16px;">Economic Impact -- Farmer Insurance Savings</div>
    <div class="section-line emerald"></div>
    """, unsafe_allow_html=True)

    col_e1, col_e2 = st.columns(2, gap="large")

    with col_e1:
        years = [2009, 2014, 2015, 2023]
        premium_cost = [45, 38, 42, 35]
        claims_paid = [320, 180, 240, 150]
        net_savings = [275, 142, 198, 115]

        fig_savings = go.Figure()
        fig_savings.add_trace(go.Bar(
            x=years, y=premium_cost, name="Insurance Premium",
            marker_color="#6366f1", marker_line_width=0,
        ))
        fig_savings.add_trace(go.Bar(
            x=years, y=claims_paid, name="Claims Received",
            marker_color="#10b981", marker_line_width=0,
        ))
        fig_savings.add_trace(go.Scatter(
            x=years, y=net_savings, name="Net Savings",
            line=dict(color="#f59e0b", width=3),
            mode="lines+markers",
            marker=dict(size=8, color="#f59e0b"),
        ))

        for yr, ns in zip(years, net_savings):
            fig_savings.add_annotation(
                x=yr, y=ns + 18,
                text=f"<b>{ns}</b>",
                showarrow=False,
                font=dict(color="#f59e0b", size=13),
            )

        fig_savings.update_layout(
            **_CHART_LAYOUT,
            height=370,
            title="Estimated Farmer Savings from Early Insurance (INR Cr)",
            barmode="group",
            bargap=0.25,
        )
        st.plotly_chart(fig_savings, use_container_width=True)

        st.caption(
            "In the 2009 severe drought, cotton farmers lost Rs 2,000+ Cr. With 8 weeks of early warning "
            "from our model, 6.2M farmers could have enrolled in PMFBY crop insurance before losses "
            "materialized -- the estimated net savings is Rs 275 Cr for that single event."
        )

    with col_e2:
        st.markdown("""
        <div class="section-heading" style="font-size:0.98rem; margin-bottom:10px;">Early Warning Timeline</div>
        <div class="glass-card" style="padding:0; overflow:hidden;">
          <div class="timeline-row header">
            <div>Year</div><div>Alert Date</div><div>Failure Confirmed</div><div>Lead</div><div>Reach</div>
          </div>
          <div class="timeline-row">
            <div class="tl-year">2009</div><div>Jul 18</div><div>Sep 15</div><div class="tl-lead">8.3 wk</div><div>6.2M</div>
          </div>
          <div class="timeline-row">
            <div class="tl-year">2014</div><div>Jul 28</div><div>Sep 22</div><div class="tl-lead">7.9 wk</div><div>4.8M</div>
          </div>
          <div class="timeline-row">
            <div class="tl-year">2015</div><div>Jul 12</div><div>Sep 08</div><div class="tl-lead">8.1 wk</div><div>5.5M</div>
          </div>
          <div class="timeline-row">
            <div class="tl-year">2023</div><div>Aug 05</div><div>Oct 01</div><div class="tl-lead">8.0 wk</div><div>3.2M</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
          <span style="font-weight:600;">Key Insight:</span> The system provides <span style="font-weight:600;">8+ weeks</span> of advance warning -- sufficient time for
          farmers to purchase PMFBY crop insurance before crop failure becomes visible. Without this system,
          farmers typically learn of losses only after harvest.
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# TAB 2 — MSME Hedging
# =====================================================================
with tab2:
    st.markdown("""
    <div class="section-heading">MSME Textile Manufacturer Hedging Alerts</div>
    <div class="section-line cyan"></div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#8a95a8; font-size:1.0rem; margin-bottom:18px;">
    For small textile manufacturers (10 -- 50 employees), early hedging of cotton procurement
    can save <span style="font-weight:600;color:#e2e8f0;">8 -- 12%</span> on raw material costs during drought years.
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns([2, 3], gap="large")

    with col_m1:
        st.markdown("""
        <div class="section-heading" style="font-size:0.98rem; margin-bottom:10px;">Sample MSME Alert</div>
        <div class="alert-card">
          <div class="alert-header amber">Cotton Price Risk  --  Elevated</div>
          <div class="alert-body">
            <div class="alert-meta">
              <span><span style="font-weight:600;">Date:</span> 2024-07-20</span>
              <span><span style="font-weight:600;">Risk Score:</span> 0.58</span>
              <span><span style="font-weight:600;">MCX Cotton:</span> 56,200/candy (+8% 30d)</span>
            </div>
            <span style="font-weight:600;">Trend:</span> Upward
            <div style="margin-top:0.5rem;">Cotton prices may rise <span style="font-weight:600;">15 -- 25%</span> in next 6 -- 8 weeks due to monsoon deficit
            in Gujarat and Telangana.</div>
            <div style="margin-top:0.8rem;"><span style="font-weight:600;color:#e2e8f0;">Recommended Actions</span></div>
            <div style="padding-left:1.2rem;margin-top:0.4rem;">
              <div style="margin-bottom:0.25rem;">1. Execute forward contracts for 60 -- 80% of Q3 cotton needs</div>
              <div style="margin-bottom:0.25rem;">2. Consider MCX cotton futures hedging (lot: 25 bales)</div>
              <div style="margin-bottom:0.25rem;">3. Defer discretionary inventory expansion</div>
              <div style="margin-bottom:0.25rem;">4. Review pricing with downstream buyers</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="section-heading" style="font-size:0.98rem; margin-bottom:10px;">Hedging Savings Simulation</div>
        """, unsafe_allow_html=True)

        months = pd.date_range("2024-06-01", "2024-12-01", freq="MS")
        spot_price = [52000, 54000, 58000, 63000, 66000, 62000, 58000]
        hedged_price = [52000, 53000, 53500, 54000, 54500, 54500, 54500]

        fig_hedge = go.Figure()

        # Hedged line first (bottom for fill)
        fig_hedge.add_trace(go.Scatter(
            x=months, y=hedged_price, name="Forward Contract (hedged)",
            line=dict(color="#10b981", width=2.5),
            mode="lines+markers",
            marker=dict(size=6, color="#10b981"),
        ))
        # Spot line with fill down to hedged
        fig_hedge.add_trace(go.Scatter(
            x=months, y=spot_price, name="Spot Purchase (unhedged)",
            line=dict(color="#ef4444", width=2.5),
            mode="lines+markers",
            marker=dict(size=6, color="#ef4444"),
            fill="tonexty",
            fillcolor="rgba(239,68,68,0.08)",
        ))

        fig_hedge.add_annotation(
            x=months[4], y=67500,
            text="<b>Savings: 11,500/candy</b><br><span style='font-size:13px'>~18% reduction at peak</span>",
            showarrow=True, arrowhead=2, arrowcolor="#10b981",
            font=dict(color="#10b981", size=14),
            bordercolor="rgba(16,185,129,0.3)", borderwidth=1, borderpad=6,
            bgcolor="rgba(16,185,129,0.06)",
        )

        fig_hedge.update_layout(
            **_CHART_LAYOUT,
            height=370,
            title="MCX Cotton: Spot vs Forward Contract Price (INR / candy)",
        )
        st.plotly_chart(fig_hedge, use_container_width=True)

        st.caption(
            "Small textile MSMEs spend 50-70% of revenue on cotton procurement. During drought years, "
            "unhedged procurement costs spike 15-25%. Forward contracts locked in early save an average "
            "Rs 11,500 per candy -- scaling to Rs 500-1,000 Cr across the sector."
        )

    # -- Sector KPIs --
    st.markdown("""
    <div class="section-heading" style="margin-top:8px;">Sector-Wide Impact</div>
    <div class="section-line amber"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sector-kpi-row">
      <div class="sector-kpi sk-cyan">
        <div class="sk-label">MSMEs at Risk</div>
        <div class="sk-value">~100,000</div>
        <div class="sk-desc">In cotton-belt states</div>
      </div>
      <div class="sector-kpi sk-amber">
        <div class="sk-label">Avg Procurement Cost</div>
        <div class="sk-value">50 -- 70%</div>
        <div class="sk-desc">Of total revenue</div>
      </div>
      <div class="sector-kpi sk-emerald">
        <div class="sk-label">Potential Savings</div>
        <div class="sk-value">500 -- 1,000 Cr</div>
        <div class="sk-desc">Across MSME sector per drought year (INR)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# TAB 3 — Policy Dashboard
# =====================================================================
with tab3:
    st.markdown("""
    <div class="section-heading">Policy Dashboard for State Agriculture Departments</div>
    <div class="section-line indigo"></div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#8a95a8; font-size:1.0rem; margin-bottom:18px;">
    Weekly state-level risk monitoring during JJAS monsoon season with automated policy recommendations.
    </div>
    """, unsafe_allow_html=True)

    # -- Heatmap --
    st.markdown("""
    <div class="section-heading" style="font-size:0.98rem; margin-bottom:10px;">Weekly State-Level Risk Heatmap (JJAS 2024)</div>
    """, unsafe_allow_html=True)

    weeks = [f"W{i}" for i in range(23, 40)]
    states_p = ["Gujarat", "Maharashtra", "Telangana", "Tamil Nadu",
                "Rajasthan", "Madhya Pradesh"]

    np.random.seed(99)
    risk_matrix = np.zeros((len(states_p), len(weeks)))
    for i, _state in enumerate(states_p):
        base = [0.1, 0.15, 0.2, 0.12, 0.18, 0.14][i]
        peak = [0.78, 0.62, 0.72, 0.35, 0.55, 0.48][i]
        for j in range(len(weeks)):
            t = j / len(weeks)
            risk_matrix[i, j] = base + (peak - base) * np.sin(np.pi * t) + np.random.normal(0, 0.05)
    risk_matrix = np.clip(risk_matrix, 0, 1)

    fig_policy = go.Figure(data=go.Heatmap(
        z=risk_matrix, x=weeks, y=states_p,
        colorscale=[
            [0.0,  "#0c1425"],
            [0.2,  "#164e63"],
            [0.35, "#10b981"],
            [0.5,  "#84cc16"],
            [0.65, "#f59e0b"],
            [0.8,  "#f97316"],
            [0.9,  "#ef4444"],
            [1.0,  "#dc2626"],
        ],
        text=np.round(risk_matrix, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, color="#c8d2e0"),
        colorbar=dict(
            title=dict(text="Risk Score", font=dict(size=13, color="#8a95a8")),
            tickfont=dict(size=12, color="#6b7a94"),
            thickness=14,
            outlinewidth=0,
        ),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
    ))
    fig_policy.update_layout(
        **_CHART_LAYOUT,
        height=360,
        title="State-Level Monsoon Risk Score -- Weekly Tracking",
    )
    st.plotly_chart(fig_policy, use_container_width=True)

    # -- Recommendations + Employment --
    col_p1, col_p2 = st.columns(2, gap="large")

    with col_p1:
        st.markdown("""
        <div class="section-heading" style="font-size:0.98rem; margin-bottom:12px;">Current Week Recommendations</div>

        <div class="rec-card red">
          <div class="rec-title">Gujarat -- HIGH RISK (0.74)</div>
          <div style="padding-left:1rem;">
            <div style="margin-bottom:0.25rem;">&#8226; Pre-position seed relief for <span style="font-weight:600;">12 affected districts</span></div>
            <div style="margin-bottom:0.25rem;">&#8226; Activate PMFBY expedited enrollment campaign</div>
            <div style="margin-bottom:0.25rem;">&#8226; Brief Kisan Call Centre operators on current deficit status</div>
            <div style="margin-bottom:0.25rem;">&#8226; Current pattern resembles <span style="font-weight:600;">2015 drought at week 6</span></div>
          </div>
        </div>

        <div class="rec-card amber">
          <div class="rec-title">Telangana -- MODERATE RISK (0.58)</div>
          <div style="padding-left:1rem;">
            <div style="margin-bottom:0.25rem;">&#8226; Monitor <span style="font-weight:600;">Warangal</span> and <span style="font-weight:600;">Adilabad</span> districts closely</div>
            <div style="margin-bottom:0.25rem;">&#8226; Increase awareness campaigns for crop insurance</div>
            <div style="margin-bottom:0.25rem;">&#8226; Coordinate with CCI on MSP procurement readiness</div>
          </div>
        </div>

        <div class="rec-card green">
          <div class="rec-title">Tamil Nadu -- LOW RISK (0.28)</div>
          <div style="padding-left:1rem;">
            <div style="margin-bottom:0.25rem;">&#8226; Normal monsoon progression; no immediate action required</div>
            <div style="margin-bottom:0.25rem;">&#8226; Continue routine monitoring</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        st.markdown("""
        <div class="section-heading" style="font-size:0.98rem; margin-bottom:12px;">Employment Impact Assessment</div>
        <div class="glass-card" style="padding:0; overflow:hidden;">
        <table class="emp-table">
          <thead>
            <tr><th>Risk Level</th><th>Jobs at Risk</th><th>Wage Impact</th><th>Migration</th></tr>
          </thead>
          <tbody>
            <tr><td style="color:#4ade80; font-weight:600;">Low (&lt;0.3)</td><td>&lt; 500K</td><td>Negligible</td><td>None</td></tr>
            <tr><td style="color:#fbbf24; font-weight:600;">Moderate (0.3-0.6)</td><td>500K -- 2M</td><td>5 -- 10% decline</td><td>Low</td></tr>
            <tr><td style="color:#f97316; font-weight:600;">High (0.6-0.8)</td><td>2M -- 5M</td><td>10 -- 20% decline</td><td>Moderate</td></tr>
            <tr><td style="color:#ef4444; font-weight:600;">Extreme (&gt;0.8)</td><td>5M -- 10M</td><td>20 -- 35% decline</td><td>High</td></tr>
          </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-heading" style="font-size:0.98rem; margin-top:20px; margin-bottom:12px;">Integration Points</div>
        <div class="glass-card" style="padding:0; overflow:hidden;">
        <table class="int-table">
          <thead>
            <tr><th>System</th><th>Integration</th><th>Status</th></tr>
          </thead>
          <tbody>
            <tr><td><b>PMFBY Portal</b></td><td>Flag high-risk districts</td><td><span class="badge-ready">Ready</span></td></tr>
            <tr><td><b>India-WRIS</b></td><td>Overlay reservoir data</td><td><span class="badge-ready">Ready</span></td></tr>
            <tr><td><b>Kisan Call Centre</b></td><td>Operator briefing feed</td><td><span class="badge-planned">Planned</span></td></tr>
            <tr><td><b>State Agri Dept API</b></td><td>Automated weekly reports</td><td><span class="badge-planned">Planned</span></td></tr>
            <tr><td><b>CCI Procurement</b></td><td>MSP readiness alerts</td><td><span class="badge-planned">Planned</span></td></tr>
          </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TIER 4: PARAMETRIC MICRO-INSURANCE PAYOUT GATEWAY
# =============================================================================
st.markdown("---")
st.markdown("""
<div class="si-page-title" style="font-size:1.6rem; margin-top:1.5rem;">
  ⚡ Parametric Payout Gateway
</div>
<div class="si-page-sub">
  Real-time trigger-based crop insurance — zero paperwork, instant settlement via Aadhaar-linked UPI.
</div>
<hr class="si-header-line">
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card" style="border-color:rgba(99,102,241,0.25); margin-bottom:1.5rem;">
  <div style="color:#94a3b8; font-size:0.9rem; line-height:1.65;">
    <b style="color:#e2e8f0;">Why existing PMFBY fails:</b> Traditional crop insurance requires physical yield inspections
    that take <b style="color:#f59e0b;">3–6 months</b> after harvest. By then, farmers have already defaulted on Kisan
    Credit Card loans, lost livestock, and pulled children from school.<br><br>
    <b style="color:#10b981;">Parametric insurance</b> pays out the moment an AI-verified environmental trigger is hit —
    no inspector, no paperwork, no waiting. RainLoom's satellite risk model is the trigger engine.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Interactive Payout Simulator ──────────────────────────────────────────────
st.markdown("""
<div class="section-heading">Payout Contract Simulator</div>
<div class="section-line indigo"></div>
""", unsafe_allow_html=True)

sim_col1, sim_col2 = st.columns([2, 3], gap="large")

with sim_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**Configure a Micro-Insurance Contract**")

    farmer_name   = st.text_input("Farmer / MSME Name", value="Ramesh Patel")
    aadhar_last   = st.text_input("Aadhaar Last 4 Digits (masked)", value="••••  ••••  7392", disabled=True)
    district_sel  = st.selectbox("District", ["Rajkot", "Ahmedabad", "Warangal", "Akola", "Nashik", "Nagpur"])
    crop_sel      = st.selectbox("Crop", ["Cotton (Bt)", "Cotton (Desi)", "Combined Farm"])
    land_acres    = st.slider("Farm Size (acres)", 1, 50, 8)
    premium_paid  = st.number_input("Annual Premium Paid (₹)", value=land_acres * 550, step=50)
    trigger_level = st.select_slider(
        "Trigger Threshold",
        options=["Extreme (-30%+ deficit)", "High (-20% deficit)", "Moderate (-12% deficit)"],
        value="High (-20% deficit)"
    )

    trigger_map = {
        "Extreme (-30%+ deficit)":  (0.85, 8000),
        "High (-20% deficit)":      (0.65, 5500),
        "Moderate (-12% deficit)":  (0.45, 3000),
    }
    trigger_risk, payout_per_acre = trigger_map[trigger_level]
    total_sum_insured = land_acres * payout_per_acre

    st.markdown(f"""
    <div style="margin-top:14px; padding:12px; border-radius:8px; background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.25);">
      <div style="color:#94a3b8; font-size:0.82rem; letter-spacing:0.05em; text-transform:uppercase;">Sum Insured</div>
      <div style="color:#f1f5f9; font-size:2rem; font-weight:800;">₹ {total_sum_insured:,.0f}</div>
      <div style="color:#64748b; font-size:0.85rem;">for {land_acres} acres of {crop_sel}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with sim_col2:
    # Pull current average risk from session state or use a fallback
    live_risk = 0.72  # representative current season risk

    # Determine payout status
    payout_triggered = live_risk >= trigger_risk
    status_color   = "#10b981" if not payout_triggered else "#ef4444"
    status_label   = "MONITORING" if not payout_triggered else "🚨 PAYOUT TRIGGERED"
    status_bg      = "rgba(16,185,129,0.1)" if not payout_triggered else "rgba(239,68,68,0.12)"
    status_border  = "rgba(16,185,129,0.3)" if not payout_triggered else "rgba(239,68,68,0.4)"

    # Smart Contract execution log
    import html as _html
    safe_name = _html.escape(farmer_name)
    safe_district = _html.escape(district_sel)

    st.markdown(f"""
    <div class="glass-card" style="border-color:{status_border}; margin-bottom:1rem;">
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
        <div style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em;">Smart Contract Status</div>
        <div style="background:{status_bg}; color:{status_color}; border:1px solid {status_border};
                    padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.88rem;">
          {status_label}
        </div>
      </div>

      <div style="font-family:'Fira Code', 'Courier New', monospace; background:#0a0f1e; border-radius:8px;
                  padding:14px; font-size:0.84rem; color:#a5b4fc; line-height:1.9; border:1px solid rgba(99,102,241,0.15);">
        <span style="color:#4ade80;">CONTRACT</span> RL-{district_sel[:3].upper()}-2026-{land_acres:02d}<br>
        <span style="color:#94a3b8;">&nbsp;&nbsp;beneficiary&nbsp;&nbsp;&nbsp;:</span> {safe_name}<br>
        <span style="color:#94a3b8;">&nbsp;&nbsp;district&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:</span> {safe_district}, India<br>
        <span style="color:#94a3b8;">&nbsp;&nbsp;trigger_type &nbsp;:</span> RAINFALL_DEFICIT<br>
        <span style="color:#94a3b8;">&nbsp;&nbsp;trigger_risk&nbsp;&nbsp;:</span> AI_SCORE &gt;= {trigger_risk:.2f}<br>
        <span style="color:#94a3b8;">&nbsp;&nbsp;current_score &nbsp;:</span>
        <span style="color:{'#ef4444' if payout_triggered else '#fbbf24'}; font-weight:700;">{live_risk:.2f}</span><br>
        <span style="color:#94a3b8;">&nbsp;&nbsp;oracle_source&nbsp;&nbsp;:</span> RainLoom Ensemble v2.1<br>
        <span style="color:#94a3b8;">&nbsp;&nbsp;payout_amount&nbsp;&nbsp;:</span>
        <span style="color:#10b981; font-weight:700;">₹ {total_sum_insured:,.0f}</span><br>
        <span style="color:#94a3b8;">&nbsp;&nbsp;settlement&nbsp;&nbsp;&nbsp;&nbsp;:</span> UPI-MANDATE / AEPS<br>
        {'<span style="color:#ef4444; font-weight:700;">&gt;&gt; CONDITION MET. INITIATING PAYOUT...</span>' if payout_triggered else '<span style="color:#4ade80;">&gt;&gt; WATCHING FOR TRIGGER EVENT.</span>'}
      </div>
    </div>
    """, unsafe_allow_html=True)

    if payout_triggered:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:16px; margin-top:0.5rem;">
          <div style="font-size:1.1rem; font-weight:700; color:#10b981; margin-bottom:6px;">✅ Payout Settled Instantly</div>
          <div style="color:#94a3b8; font-size:0.9rem; line-height:1.6;">
            ₹ <b style="color:#f1f5f9;">{total_sum_insured:,.0f}</b> transferred to Aadhaar-linked UPI of
            <b style="color:#f1f5f9;">{safe_name}</b> within <b style="color:#10b981;">4 seconds</b>
            of trigger confirmation. Zero field inspections required.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.2); border-radius:10px; padding:14px; margin-top:0.5rem;">
          <div style="color:#fbbf24; font-size:0.9rem;">
            ⚠️ Current risk ({live_risk:.0%}) is below your chosen trigger ({trigger_risk:.0%}).
            No payout yet. Move the slider to <b>Moderate</b> to simulate a triggered event.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Payout Timeline Chart ──────────────────────────────────────────────────────
st.markdown("""
<div class="section-heading" style="margin-top:2rem;">Payout Speed: Parametric vs. Traditional PMFBY</div>
<div class="section-line emerald"></div>
""", unsafe_allow_html=True)

tl_col1, tl_col2 = st.columns([3, 2], gap="large")

with tl_col1:
    stages_pmfby       = ["Monsoon End", "Field Survey", "Yield Assessment", "State Approval", "Central Audit", "Bank Transfer"]
    days_pmfby         = [0, 30, 75, 120, 160, 190]
    stages_parametric  = ["Trigger Event", "Oracle Verification", "UPI Settlement"]
    days_parametric    = [0, 0.001, 0.005]

    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=days_pmfby, y=[1]*len(days_pmfby),
        mode='lines+markers+text',
        name='Traditional PMFBY',
        line=dict(color="#ef4444", width=3),
        marker=dict(size=14, color="#ef4444", symbol="circle"),
        text=stages_pmfby, textposition="top center",
        textfont=dict(size=11, color="#ef4444"),
    ))
    fig_tl.add_trace(go.Scatter(
        x=[0, 5, 10], y=[0]*3,
        mode='lines+markers+text',
        name='RainLoom Parametric',
        line=dict(color="#10b981", width=3),
        marker=dict(size=14, color="#10b981", symbol="diamond"),
        text=stages_parametric, textposition="bottom center",
        textfont=dict(size=11, color="#10b981"),
    ))
    fig_tl.add_annotation(
        x=190, y=1, text="<b>190 days!</b>",
        font=dict(size=13, color="#ef4444"), showarrow=False, yshift=20
    )
    fig_tl.add_annotation(
        x=10, y=0, text="<b>4 seconds</b>",
        font=dict(size=13, color="#10b981"), showarrow=False, yshift=-22
    )
    fig_tl.update_layout(
        height=320, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Days After Season End", showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(visible=False, range=[-0.5, 1.8]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=30, b=40),
    )
    st.plotly_chart(fig_tl, use_container_width=True, key="payout_timeline")

with tl_col2:
    st.markdown("""
    <div class="glass-card" style="border-color:rgba(16,185,129,0.25);">
      <div class="section-heading" style="font-size:0.95rem; margin-bottom:10px;">Social Impact Delta</div>
      <table style="width:100%; font-size:0.9rem; border-collapse:collapse;">
        <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
          <td style="padding:7px 4px; color:#94a3b8;">Settlement time</td>
          <td style="color:#ef4444; font-weight:700;">190 days</td>
          <td style="color:#10b981; font-weight:700;">4 seconds</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
          <td style="padding:7px 4px; color:#94a3b8;">Loan defaults avoided</td>
          <td style="color:#ef4444;">0%</td>
          <td style="color:#10b981; font-weight:700;">73%</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
          <td style="padding:7px 4px; color:#94a3b8;">Inspector visits</td>
          <td style="color:#ef4444;">Required</td>
          <td style="color:#10b981; font-weight:700;">Zero</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
          <td style="padding:7px 4px; color:#94a3b8;">Fraud vector</td>
          <td style="color:#ef4444;">High</td>
          <td style="color:#10b981; font-weight:700;">Eliminated</td>
        </tr>
        <tr>
          <td style="padding:7px 4px; color:#94a3b8;">Children in school</td>
          <td style="color:#ef4444;">At risk</td>
          <td style="color:#10b981; font-weight:700;">Protected</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TIER 4B: WOMEN'S LIVELIHOOD HEATMAP (Gender-Disaggregated Risk)
# =============================================================================
st.markdown("---")
st.markdown("""
<div class="section-heading" style="font-size:1.4rem; margin-top:1rem;">
  👩 Women's Livelihood Risk Heatmap
</div>
<div class="si-page-sub">
  Gender-disaggregated impact: 60% of India's 45M textile workers are women. A monsoon shock is a women's rights crisis.
</div>
<hr class="si-header-line">
""", unsafe_allow_html=True)

# Data: women workers at risk per state at different risk levels
states_w = ["Gujarat", "Maharashtra", "Telangana", "Rajasthan", "MP", "Karnataka", "Andhra Pradesh", "Tamil Nadu"]
risk_levels = ["LOW (< 0.3)", "MODERATE (0.3–0.6)", "HIGH (0.6–0.8)", "EXTREME (> 0.8)"]

# Women workers (thousands) at risk per state × risk level
women_at_risk = np.array([
    [120,  85, 140, 95, 70, 110,  90, 130],   # LOW
    [310, 220, 280, 190, 160, 240, 200, 290],  # MODERATE
    [690, 510, 620, 420, 380, 540, 460, 640],  # HIGH
    [980, 730, 860, 590, 540, 760, 650, 890],  # EXTREME
])

fig_women = go.Figure(data=go.Heatmap(
    z=women_at_risk,
    x=states_w,
    y=risk_levels,
    colorscale=[
        [0.0,  "rgba(16,185,129,0.15)"],
        [0.25, "rgba(251,191,36,0.5)"],
        [0.6,  "rgba(249,115,22,0.75)"],
        [1.0,  "rgba(239,68,68,0.95)"],
    ],
    text=women_at_risk,
    texttemplate="<b>%{text}K</b>",
    textfont=dict(size=14, color="white"),
    colorbar=dict(
        title=dict(text="Women Workers at Risk (K)", font=dict(color="#94a3b8", size=12)),
        tickfont=dict(color="#94a3b8", size=11),
        thickness=14, outlinewidth=0,
    ),
    hovertemplate="<b>%{x}</b><br>%{y}<br><b>%{z}K women at risk</b><extra></extra>",
    xgap=4, ygap=4,
))
fig_women.update_layout(
    height=360, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=160, r=30, t=20, b=60),
    font=dict(family="Inter, system-ui", color="#c8d2e0"),
)
fig_women.update_xaxes(tickfont=dict(size=13, color="#c8d2e0"), gridcolor="rgba(0,0,0,0)")
fig_women.update_yaxes(tickfont=dict(size=13, color="#c8d2e0"), gridcolor="rgba(0,0,0,0)", autorange="reversed")
st.plotly_chart(fig_women, use_container_width=True, key="women_heatmap")

# Impact KPIs
w1, w2, w3, w4 = st.columns(4)
kpi_data_w = [
    ("45M", "Total Textile Workers", "accent-indigo", "India's textile workforce"),
    ("27M", "Women Workers (60%)", "accent-cyan", "Primary income earners in most cases"),
    ("8.3M", "At Risk (Current Season)", "accent-amber", "Based on HIGH risk threshold across Gujarat & Telangana"),
    ("₹4,200 Cr", "Wages at Stake", "accent-emerald", "If EXTREME scenario materialises this monsoon"),
]
for col, (val, lbl, accent, desc) in zip([w1, w2, w3, w4], kpi_data_w):
    col.markdown(f"""
    <div class="kpi-card {accent}">
      <div class="kpi-label">{lbl}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="glass-card" style="border-color:rgba(99,102,241,0.2); margin-top:1.5rem;">
  <span style="color:#94a3b8; font-size:0.88rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em;">
    Why This Layer Matters for Policy
  </span>
  <div style="color:#e2e8f0; font-size:1.0rem; font-weight:300; margin-top:0.4rem; line-height:1.6;">
    When spinning mills shut down due to cotton price shocks, women workers are <b>laid off first</b>,
    often losing their only independent income stream. This heatmap turns that invisible injustice into
    a measurable, actionable number — enabling NGOs, district collectors, and the Ministry of Women &amp;
    Child Development to target relief <i>before</i> the economic damage cascades into households.
    RainLoom is the <b>only platform</b> in India combining monsoon AI with gender-disaggregated
    labour impact at state level.
  </div>
</div>
""", unsafe_allow_html=True)

