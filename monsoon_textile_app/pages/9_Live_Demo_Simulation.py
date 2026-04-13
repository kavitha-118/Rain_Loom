"""
Page 9 — Live Demo Simulation Mode
==================================
Interactive playback of the 2009 severe drought event to demonstrate
the model's early warning capabilities timeline for a live audience.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import datetime as _dt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from monsoon_textile_app.components.navbar import render_navbar
from monsoon_textile_app.components.chat_bubble import render_chat_bubble

st.set_page_config(page_title="Live Demo Simulator", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")
render_navbar(active_page="Simulator")
render_chat_bubble()

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.stApp { background: #0a0f1e; }
.page-title {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 2.4rem; font-weight: 800;
    color: #f8fafc; margin-bottom: 0px; text-transform: uppercase; letter-spacing: 0.05em;
}
.page-subtitle { color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem; }
.kpi-h1 { font-family: 'Inter', sans-serif; font-size: 3rem; font-weight: 800; line-height: 1; }
.glass-panel {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.5rem;
    backdrop-filter: blur(12px);
}
.urgent-red { color: #ef4444; text-shadow: 0 0 10px rgba(239,68,68,0.5); }
.calm-green { color: #4ade80; text-shadow: 0 0 10px rgba(74,222,128,0.5); }
.warn-amber { color: #f59e0b; text-shadow: 0 0 10px rgba(245,158,11,0.5); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🎬 2009 Drought Demo Replay</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Presents a live, week-by-step playback of the 2009 Monsoon failure.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Simulation State
# ---------------------------------------------------------------------------
weeks = [
    "Week 1 (June 5)", "Week 3 (June 19)", "Week 5 (July 3)", 
    "Week 7 (July 17)", "Week 9 (July 31)", "Week 11 (Aug 14)"
]
deficits = [0, -5, -12, -23, -28, -32]
vols = [0.15, 0.16, 0.17, 0.19, 0.45, 0.58]
risks = [0.20, 0.35, 0.58, 0.85, 0.95, 0.98]
cotton_prices = [32000, 32500, 33500, 38000, 42000, 48000]

if 'demo_step' not in st.session_state:
    st.session_state.demo_step = 0

col_ctrl, col_info = st.columns([1, 4])
with col_ctrl:
    if st.button("▶️ Advance Week", type="primary", use_container_width=True):
        if st.session_state.demo_step < len(weeks) - 1:
            st.session_state.demo_step += 1
        else:
            st.session_state.demo_step = 0  # Reset
with col_info:
    st.progress((st.session_state.demo_step + 1) / len(weeks), text=f"Timeline: {weeks[st.session_state.demo_step]}")

step = st.session_state.demo_step
current_week = weeks[step]
current_def = deficits[step]
current_vol = vols[step]
current_risk = risks[step]
current_cotton = cotton_prices[step]

if current_risk < 0.4:
    color_class = "calm-green"
    alert = "Normal Operating Conditions"
elif current_risk < 0.8:
    color_class = "warn-amber"
    alert = "⚠️ Early Warning Triggered! Hedging alerts sent."
else:
    color_class = "urgent-red"
    alert = "🚨 DROUGHT CONFIRMED. Panic buying in markets."

st.markdown(f'<div style="text-align:center; margin:1rem 0;"><span style="font-size:1.5rem; font-weight:800; letter-spacing:0.1em; text-transform:uppercase;" class="{color_class}">{alert}</span></div>', unsafe_allow_html=True)

col_viz1, col_viz2, col_viz3 = st.columns(3)

# 1. RISK GAUGE
with col_viz1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div style="color:#94a3b8; font-weight:600; text-transform:uppercase;">AI Risk Model Score</div>', unsafe_allow_html=True)
    gauge_color = "#4ade80" if current_risk < 0.4 else ("#f59e0b" if current_risk < 0.8 else "#ef4444")
    fig_g = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_risk * 100,
        number = dict(suffix="%", font=dict(color=gauge_color, size=40)),
        gauge = dict(
            axis=dict(range=[0, 100], visible=False),
            bar=dict(color=gauge_color, thickness=0.8),
            bgcolor="rgba(255,255,255,0.05)"
        )
    ))
    fig_g.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_g, use_container_width=True, key=f"gauge_{step}")
    st.markdown('</div>', unsafe_allow_html=True)

# 2. RAINFALL DEFICIT 
with col_viz2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#94a3b8; font-weight:600; text-transform:uppercase;">Rainfall Deficit (Gujarat)</div>', unsafe_allow_html=True)
    def_color = "#ef4444" if current_def < -20 else "#60a5fa"
    st.markdown(f'<div class="kpi-h1" style="color:{def_color}; margin-top:2rem; text-align:center;">{current_def}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 3. MCX COTTON PRICES
with col_viz3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div style="color:#94a3b8; font-weight:600; text-transform:uppercase;">MCX Cotton Futures</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-h1" style="color:#f8fafc; margin-top:2rem; text-align:center;">₹ {current_cotton:,}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
# Line historical chart showing how we got here
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=weeks[:step+1], y=[r*100 for r in risks[:step+1]], mode='lines+markers', name="RainLoom Risk %", line=dict(color="#ef4444", width=4)))
fig_line.add_trace(go.Scatter(x=weeks[:step+1], y=[(c/48000)*100 for c in cotton_prices[:step+1]], mode='lines+markers', name="Cotton Price Index", line=dict(color="#60a5fa", width=2, dash='dot')))
fig_line.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title="Timeline Trajectory (Risk precedes Price by 4 weeks)")
st.plotly_chart(fig_line, use_container_width=True, key=f"line_{step}")
