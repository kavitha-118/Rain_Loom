"""
Page 8 — Institutional API Gateway (B2B Play)
==============================================
Developer portal showing API documentation, generated API keys,
Webhook management, and an Embed Widget.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import json
import uuid
from textwrap import dedent
from monsoon_textile_app.components.navbar import render_navbar
from monsoon_textile_app.components.chat_bubble import render_chat_bubble

st.set_page_config(page_title="Institutional API Gateway", page_icon="🏦", layout="wide", initial_sidebar_state="collapsed")
render_navbar(active_page="API Gateway")
render_chat_bubble()

# ---------------------------------------------------------------------------
# Global CSS -- dark glass-morphism theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');

.stApp { background: linear-gradient(145deg, #0a0f1e 0%, #0d1326 40%, #0f172a 100%); }
.block-container { padding-top: 2rem; }

.page-title {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 2.26rem; font-weight: 700;
    letter-spacing: 0.06em; color: #f1f5f9; margin-bottom: 0.15rem;
}
.page-subtitle {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.06rem; font-weight: 300;
    color: #64748b; margin-bottom: 0.5rem;
}
.title-rule {
    height: 3px; border: none; border-radius: 2px;
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, transparent 100%);
    margin-bottom: 2.2rem;
}
.section-heading {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.51rem; font-weight: 600;
    letter-spacing: 0.04em; color: #e2e8f0;
    margin-bottom: 0.15rem; text-transform: uppercase;
}
.section-sub {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.02rem; font-weight: 300;
    color: #94a3b8; margin-bottom: 0.7rem;
}
.heading-rule { height: 2px; border: none; border-radius: 1px; margin-bottom: 1.6rem; }
.rule-blue  { background: linear-gradient(90deg, #3b82f6 0%, transparent 80%); }

.glass-card {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(59, 130, 246, 0.12);
    border-radius: 14px; padding: 1.6rem 1.8rem;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.25); margin-bottom: 1rem;
}
.code-block {
    font-family: 'Fira Code', monospace;
    background: #0f172a !important;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px;
    padding: 1rem;
    color: #a5b4fc;
    font-size: 0.9rem;
    overflow-x: auto;
}
.kpi-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 14px; padding: 1.5rem 1.6rem;
    backdrop-filter: blur(14px); text-align: center;
}
.kpi-value { font-size: 1.91rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.15rem; }
.kpi-label { font-size: 0.96rem; color: #94a3b8; text-transform: uppercase; font-weight:500;}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# TITLE
# =========================================================================
st.markdown("""
<div class="page-title">Institutional API Gateway</div>
<div class="page-subtitle">B2B Integration Portal: Scalable Rest API, Webhooks & Embedded Widgets for Enterprise Risk Systems</div>
<hr class="title-rule">
""", unsafe_allow_html=True)

with st.expander("Why this matters for the Hackathon Business Pitch"):
    st.markdown("""
    While the dashboard proves the model's accuracy, the **B2B API Gateway** proves commercial viability.
    Instead of relying strictly on B2C subscriptions from farmers, **RainLoom** monetizes via Tier-1 banks, 
    Agri-Insurance (PMFBY integrations), and global supply chains hitting our REST API programmatically. 
    This portal allows enterprise partners to generate API keys, set up Webhooks, and grab Embed Widgets.
    """)

# =========================================================================
# DEVELOPER DASHBOARD METRICS
# =========================================================================
col1, col2, col3, col4 = st.columns(4)
col1.markdown('<div class="kpi-card"><div class="kpi-value" style="color:#60a5fa;">21,402</div><div class="kpi-label">API Calls (This Month)</div></div>', unsafe_allow_html=True)
col2.markdown('<div class="kpi-card"><div class="kpi-value" style="color:#34d399;">12</div><div class="kpi-label">Active Webhooks</div></div>', unsafe_allow_html=True)
col3.markdown('<div class="kpi-card"><div class="kpi-value" style="color:#f59e0b;">56 ms</div><div class="kpi-label">Avg P99 Latency</div></div>', unsafe_allow_html=True)
col4.markdown('<div class="kpi-card"><div class="kpi-value" style="color:#ef4444;">99.9%</div><div class="kpi-label">Uptime SLA</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================================
# TABS: API KEYS | WEBHOOKS | EMBED WIDGET
# =========================================================================
tab1, tab2, tab3 = st.tabs(["🔑 API Keys & Auth", "⚡ Webhook Rules", "🌐 B2B Embed Widget"])

with tab1:
    st.markdown("""
    <div class="section-heading">API Key Management</div>
    <div class="section-sub">Generate and rotate secure tokens for programmatic access</div>
    <hr class="heading-rule rule-blue">
    """, unsafe_allow_html=True)
    
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = [{"name": "Default Institutional Token", "key": f"rl_live_{uuid.uuid4().hex[:16]}", "created": "2024-05-10"}]

    c1, c2 = st.columns([3, 1])
    with c1:
        key_name = st.text_input("New Application Name", placeholder="e.g. ICICI Lombard Risk Engine")
    with c2:
        st.write("")
        st.write("")
        if st.button("Generate Token", type="primary", use_container_width=True) and key_name:
            st.session_state.api_keys.append({"name": key_name, "key": f"rl_live_{uuid.uuid4().hex[:16]}", "created": "Just now"})
            st.rerun()

    # Data editor to show keys
    df_keys = pd.DataFrame(st.session_state.api_keys)
    st.dataframe(df_keys, use_container_width=True, hide_index=True)
    
    st.markdown("### Example Implementation")
    st.markdown(f"""
    <div class="code-block">
curl -X GET "https://api.rainloom.io/v1/risk-scores?stock=ARVIND.NS" \\<br>
     -H "Authorization: Bearer {st.session_state.api_keys[-1]['key']}"
<br><br>
# Response<br>
{{<br>
&nbsp;&nbsp;"ticker": "ARVIND.NS",<br>
&nbsp;&nbsp;"current_risk": 0.72,<br>
&nbsp;&nbsp;"monsoon_deficit_pct": -28,<br>
&nbsp;&nbsp;"early_warning_status": "EXTREME",<br>
&nbsp;&nbsp;"causal_lag_est_weeks": 4<br>
}}
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="section-heading">Webhook Subscriptions</div>
    <div class="section-sub">Push-based event notifications for systemic shocks</div>
    <hr class="heading-rule rule-blue">
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("**Rule Builder:** `IF (Risk Metric) (Operator) (Threshold) THEN Send POST to (URL)`")
        row = st.columns([2, 1, 1, 3])
        metric = row[0].selectbox("Metric", ["Arvind Volatility", "Average Sector Risk", "District Deficit"])
        op = row[1].selectbox("Operator", [">", "<", "="])
        val = row[2].number_input("Threshold", value=0.6)
        wh = row[3].text_input("Webhook Payload URL", placeholder="https://your-server.com/webhooks")
        if st.button("Add Subscriber Workflow", use_container_width=True):
            st.success(f"Successfully registered webhook logic block: Push -> {wh}")
            
    st.markdown("### Webhook Schema Payload")
    st.markdown("""
    <div class="code-block">
POST https://your-server.com/webhooks<br>
Content-Type: application/json<br>
<br>
{<br>
&nbsp;&nbsp;"event_id": "wh_evt_8492091a",<br>
&nbsp;&nbsp;"timestamp": "2026-08-14T09:11:00Z",<br>
&nbsp;&nbsp;"event": "trigger.condition.met",<br>
&nbsp;&nbsp;"data": {<br>
&nbsp;&nbsp;&nbsp;&nbsp;"rule": "ARVIND_VOLATILITY > 0.6",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"current_value": 0.72,<br>
&nbsp;&nbsp;&nbsp;&nbsp;"recommended_action": "Execute institutional portfolio protective put hedge on NSE."<br>
&nbsp;&nbsp;}<br>
}
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="section-heading">Embed Widget (IFrame)</div>
    <div class="section-sub">Whitelabel the RainLoom risk gauge directly into an MSME's ERP or dashboard</div>
    <hr class="heading-rule rule-blue">
    """, unsafe_allow_html=True)
    
    st.markdown("Copy and paste this snippet into your external site's `<head>` or `<body>`:")
    token = st.session_state.api_keys[-1]['key']
    st.code(f"""<iframe src="https://dash.rainloom.io/widget/risk-gauge?token={token}&theme=dark" 
    width="100%" 
    height="400" 
    frameborder="0"
    style="border-radius:12px; border: 1px solid rgba(255,255,255,0.1);">
</iframe>""", language="html")
    
    st.markdown("### Live Preview:")
    st.info("The widget snippet above would render an interactive real-time dial indicating the current Causal Monsoon Risk exactly like the one on the main dashboard.")
