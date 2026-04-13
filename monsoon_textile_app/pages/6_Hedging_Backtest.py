"""
Page 6 — Economic Backtesting & Hedging Simulation
====================================================
Simulates hedging strategies based on ensemble risk signals.
Compares hedged vs unhedged portfolio performance across historical
drought events to quantify the economic value of risk forecasts.
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
st.set_page_config(page_title="Hedging Backtest", layout="wide", initial_sidebar_state="collapsed")
render_navbar(active_page="Hedging")
render_chat_bubble()

# ---------------------------------------------------------------------------
# Theme constants (same as other pages)
# ---------------------------------------------------------------------------
BG_PRIMARY = "#0a0f1e"
BG_CARD = "rgba(15, 23, 42, 0.60)"
GLASS_BORDER = "rgba(255,255,255,0.06)"
ACCENT_BLUE = "#3b82f6"
ACCENT_RED = "#ef4444"
ACCENT_GREEN = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_PURPLE = "#8b5cf6"
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
# CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
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
.section-heading {{
    font-size: 1.41rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: {TEXT_PRIMARY};
    margin: 2.4rem 0 0.4rem 0;
    padding-bottom: 0.55rem;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, {ACCENT_PURPLE}, {ACCENT_GREEN}, transparent) 1;
}}
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
    box-shadow: 0 4px 32px rgba(139,92,246,0.08);
}}
.metric-card {{
    background: {BG_CARD};
    border: 1px solid {GLASS_BORDER};
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
}}
.metric-value {{
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}}
.metric-label {{
    font-size: 0.84rem;
    color: {TEXT_MUTED};
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
section[data-testid="stVerticalBlock"] > div {{
    gap: 0.35rem;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=3600)
def _load_backtest_data():
    """Load stock data and ensemble risk scores for backtesting."""
    try:
        from monsoon_textile_app.data.fetch_real_data import load_all_data, STOCKS as STOCK_CFG
        data = load_all_data()
        stock_data = data.get("stock_data", {})

        # Ensemble risk: use the risk_score column from stock_data (already merged)
        # as well as ml_details if available
        ml = data.get("ml_details", {})
        ensemble_risk = ml.get("ensemble_risk", {})

        # Fallback: extract risk_score series from stock_data if ensemble_risk is empty
        if not ensemble_risk:
            for ticker, sdf in stock_data.items():
                if "risk_score" in sdf.columns:
                    ensemble_risk[ticker] = sdf["risk_score"]

        return stock_data, ensemble_risk, STOCK_CFG
    except Exception as e:
        return {}, {}, {}


_stock_data, _ensemble_risk, _STOCK_CFG = _load_backtest_data()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="margin-bottom:0.3rem;">
    <div style="font-size:2.16rem; font-weight:700; letter-spacing:0.06em;
               margin:0; color:#e2e8f0;">
        HEDGING BACKTEST
    </div>
    <div style="color:#8892b0; font-size:1.03rem; margin:0.3rem 0 0 0;">
        Economic value of risk-signal-driven hedging strategies
        across historical drought events
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("What is this page?"):
    st.markdown(
        "This page simulates a **risk-signal-driven hedging strategy**: when the ensemble risk "
        "score exceeds a threshold, the strategy reduces equity exposure and shifts to a hedge "
        "(e.g., put options / cash). We backtest this against a buy-and-hold benchmark to measure "
        "the economic value of the monsoon risk forecast."
    )

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
st.markdown('<div class="section-heading">Strategy Parameters</div>', unsafe_allow_html=True)

param_cols = st.columns(4)
with param_cols[0]:
    risk_threshold = st.slider(
        "Risk Threshold", 0.3, 0.9, 0.55, 0.05,
        help="Hedge when ensemble risk exceeds this level",
    )
with param_cols[1]:
    hedge_ratio = st.slider(
        "Hedge Ratio", 0.3, 1.0, 0.70, 0.05,
        help="Fraction of portfolio hedged when signal triggers",
    )
with param_cols[2]:
    hedge_cost_bps = st.slider(
        "Hedge Cost (bps/wk)", 0, 30, 8, 1,
        help="Weekly cost of hedge (e.g., put premium amortised)",
    )
with param_cols[3]:
    stock_select = st.selectbox(
        "Stock",
        options=[_STOCK_CFG.get(t, {}).get("name", t) for t in _stock_data.keys()]
        if _stock_data else ["No data available"],
    )

# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

def run_hedging_backtest(
    stock_returns: pd.Series,
    risk_scores: pd.Series,
    threshold: float,
    hedge_pct: float,
    cost_bps: float,
) -> dict:
    """
    Simulate hedged vs unhedged portfolio.

    When risk > threshold:
        - Reduce equity exposure by hedge_pct
        - Pay weekly hedge cost
    When risk <= threshold:
        - Full equity exposure, no hedge cost
    """
    common = stock_returns.index.intersection(risk_scores.index)
    if len(common) < 20:
        return None

    ret = stock_returns.reindex(common).fillna(0)
    risk = risk_scores.reindex(common).fillna(0.5)

    hedge_cost = cost_bps / 10000  # bps to decimal

    # --- Unhedged (buy-and-hold) ---
    unhedged_cum = (1 + ret).cumprod()

    # --- Hedged strategy ---
    hedged_ret = ret.copy()
    hedge_active = risk > threshold
    # When hedging: reduce equity exposure, pay cost
    hedged_ret[hedge_active] = ret[hedge_active] * (1 - hedge_pct) - hedge_cost
    hedged_cum = (1 + hedged_ret).cumprod()

    # --- Metrics ---
    n_weeks = len(ret)
    ann_factor = np.sqrt(52)

    def _metrics(r):
        total = float((1 + r).prod() - 1)
        ann_ret = float((1 + total) ** (52 / max(n_weeks, 1)) - 1)
        vol = float(r.std() * ann_factor)
        sharpe = ann_ret / vol if vol > 0 else 0
        # Max drawdown
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = float(dd.min())
        return {
            "total_return": total,
            "ann_return": ann_ret,
            "ann_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }

    unhedged_metrics = _metrics(ret)
    hedged_metrics = _metrics(hedged_ret)

    # Hedge activity stats
    n_hedge_weeks = int(hedge_active.sum())
    pct_hedged = n_hedge_weeks / n_weeks

    # Per-drought-year analysis
    drought_years = [2009, 2014, 2015, 2023]
    drought_analysis = []
    for yr in drought_years:
        yr_mask = ret.index.year == yr
        if yr_mask.sum() < 4:
            continue
        yr_ret = ret[yr_mask]
        yr_hedged = hedged_ret[yr_mask]
        yr_unhedged_return = float((1 + yr_ret).prod() - 1)
        yr_hedged_return = float((1 + yr_hedged).prod() - 1)
        yr_saved = yr_hedged_return - yr_unhedged_return
        drought_analysis.append({
            "year": yr,
            "unhedged": yr_unhedged_return,
            "hedged": yr_hedged_return,
            "value_saved": yr_saved,
            "weeks_hedged": int(hedge_active[yr_mask].sum()),
        })

    return {
        "unhedged_cum": unhedged_cum,
        "hedged_cum": hedged_cum,
        "unhedged_metrics": unhedged_metrics,
        "hedged_metrics": hedged_metrics,
        "hedge_active": hedge_active,
        "risk_scores": risk,
        "n_hedge_weeks": n_hedge_weeks,
        "pct_hedged": pct_hedged,
        "drought_analysis": drought_analysis,
    }


# Find the right ticker for selected stock
_selected_ticker = None
for t, cfg in _STOCK_CFG.items():
    if cfg.get("name") == stock_select:
        _selected_ticker = t
        break

if _selected_ticker and _selected_ticker in _stock_data and _selected_ticker in _ensemble_risk:
    sdf = _stock_data[_selected_ticker]
    risk_series = _ensemble_risk[_selected_ticker]
    returns = sdf.get("log_ret", pd.Series(dtype=float))

    bt = run_hedging_backtest(returns, risk_series, risk_threshold, hedge_ratio, hedge_cost_bps)

    if bt is not None:
        # ═══════════════════════════════════════════════════════════════
        # Summary Metrics
        # ═══════════════════════════════════════════════════════════════
        st.markdown('<div class="section-heading">Performance Summary</div>', unsafe_allow_html=True)

        um = bt["unhedged_metrics"]
        hm = bt["hedged_metrics"]

        metric_cols = st.columns(6)
        metric_data = [
            ("Hedged Return", f"{hm['total_return']:+.1%}", ACCENT_GREEN if hm['total_return'] > um['total_return'] else ACCENT_RED),
            ("Unhedged Return", f"{um['total_return']:+.1%}", TEXT_MUTED),
            ("Hedged Sharpe", f"{hm['sharpe']:.2f}", ACCENT_GREEN if hm['sharpe'] > um['sharpe'] else ACCENT_RED),
            ("Unhedged Sharpe", f"{um['sharpe']:.2f}", TEXT_MUTED),
            ("Max DD (Hedged)", f"{hm['max_drawdown']:.1%}", ACCENT_GREEN if hm['max_drawdown'] > um['max_drawdown'] else ACCENT_RED),
            ("Weeks Hedged", f"{bt['pct_hedged']:.0%}", ACCENT_BLUE),
        ]

        for col, (label, value, color) in zip(metric_cols, metric_data):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color};">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════
        # Cumulative P&L Chart
        # ═══════════════════════════════════════════════════════════════
        st.markdown('<div class="section-heading">Cumulative Performance</div>', unsafe_allow_html=True)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.06,
        )

        # Shade hedge-active periods
        hedge_on = bt["hedge_active"]
        idx = hedge_on.index
        in_hedge = False
        start = None
        for i, (dt, active) in enumerate(hedge_on.items()):
            if active and not in_hedge:
                start = dt
                in_hedge = True
            elif not active and in_hedge:
                fig.add_vrect(
                    x0=start, x1=dt, row=1, col=1,
                    fillcolor=ACCENT_RED, opacity=0.06, line_width=0,
                )
                in_hedge = False
        if in_hedge and start:
            fig.add_vrect(
                x0=start, x1=idx[-1], row=1, col=1,
                fillcolor=ACCENT_RED, opacity=0.06, line_width=0,
            )

        # Cumulative curves
        fig.add_trace(go.Scatter(
            x=bt["unhedged_cum"].index, y=bt["unhedged_cum"].values,
            name="Buy & Hold", line=dict(color=TEXT_MUTED, width=2, dash="dot"),
            hovertemplate="<b>Buy & Hold</b><br>%{x|%Y-%m-%d}<br>Value: %{y:.3f}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=bt["hedged_cum"].index, y=bt["hedged_cum"].values,
            name="Hedged Strategy", line=dict(color=ACCENT_GREEN, width=2.5),
            hovertemplate="<b>Hedged</b><br>%{x|%Y-%m-%d}<br>Value: %{y:.3f}<extra></extra>",
        ), row=1, col=1)

        # Risk score subplot
        fig.add_trace(go.Scatter(
            x=bt["risk_scores"].index, y=bt["risk_scores"].values,
            name="Risk Score", line=dict(color=ACCENT_BLUE, width=1.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
            hovertemplate="<b>Risk</b><br>%{x|%Y-%m-%d}<br>Score: %{y:.2f}<extra></extra>",
        ), row=2, col=1)

        fig.add_hline(
            y=risk_threshold, row=2, col=1,
            line_dash="dash", line_color=ACCENT_RED, line_width=1.5,
            annotation_text=f"Threshold: {risk_threshold:.0%}",
            annotation_position="right",
            annotation_font=dict(size=11, color=ACCENT_RED),
        )

        fig.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height=600,
            title=dict(
                text=f"Hedged vs Unhedged — {stock_select}",
                font=dict(size=16, color=TEXT_PRIMARY),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5,
                bgcolor="rgba(0,0,0,0)", font=dict(size=13),
            ),
            yaxis=dict(title="Portfolio Value (rebased)", showgrid=False),
            yaxis2=dict(title="Risk Score", range=[0, 1], showgrid=False, tickformat=".0%"),
            xaxis2=dict(showgrid=False),
        )

        st.plotly_chart(fig, use_container_width=True, key="hedging_pnl")

        # ═══════════════════════════════════════════════════════════════
        # Drought Year Analysis
        # ═══════════════════════════════════════════════════════════════
        if bt["drought_analysis"]:
            st.markdown('<div class="section-heading">Drought Year Analysis</div>', unsafe_allow_html=True)

            da = bt["drought_analysis"]
            fig_drought = go.Figure()

            years = [str(d["year"]) for d in da]
            unhedged_vals = [d["unhedged"] * 100 for d in da]
            hedged_vals = [d["hedged"] * 100 for d in da]

            fig_drought.add_trace(go.Bar(
                x=years, y=unhedged_vals, name="Unhedged",
                marker_color=ACCENT_RED, opacity=0.75,
                text=[f"{v:+.1f}%" for v in unhedged_vals],
                textposition="outside",
            ))
            fig_drought.add_trace(go.Bar(
                x=years, y=hedged_vals, name="Hedged",
                marker_color=ACCENT_GREEN, opacity=0.85,
                text=[f"{v:+.1f}%" for v in hedged_vals],
                textposition="outside",
            ))

            fig_drought.update_layout(
                **PLOTLY_LAYOUT_DEFAULTS,
                height=400,
                title=dict(
                    text="Returns During Drought Years — Hedged vs Unhedged",
                    font=dict(size=16, color=TEXT_PRIMARY),
                ),
                barmode="group",
                yaxis=dict(title="Return (%)", showgrid=False, zeroline=True,
                           zerolinecolor="rgba(255,255,255,0.15)"),
                xaxis=dict(title="Drought Year", showgrid=False),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=13),
                ),
            )

            st.plotly_chart(fig_drought, use_container_width=True, key="drought_bars")

            # Value saved table
            st.markdown("**Value Protected During Drought Events**")
            df_drought = pd.DataFrame(da)
            df_drought.columns = ["Year", "Unhedged Return", "Hedged Return", "Value Saved", "Weeks Hedged"]
            for c in ["Unhedged Return", "Hedged Return", "Value Saved"]:
                df_drought[c] = df_drought[c].map(lambda x: f"{x:+.1%}")
            st.dataframe(df_drought, use_container_width=True, hide_index=True)

        # ═══════════════════════════════════════════════════════════════
        # Strategy Interpretation
        # ═══════════════════════════════════════════════════════════════
        st.markdown('<div class="section-heading">Strategy Assessment</div>', unsafe_allow_html=True)

        sharpe_diff = hm["sharpe"] - um["sharpe"]
        dd_improvement = hm["max_drawdown"] - um["max_drawdown"]

        if sharpe_diff > 0.1:
            assessment_color = ACCENT_GREEN
            assessment = "POSITIVE"
            assessment_text = (
                f"The hedging strategy **improves risk-adjusted returns** by "
                f"{sharpe_diff:.2f} Sharpe ratio points. Max drawdown improved by "
                f"{dd_improvement:+.1%} percentage points. "
                f"The ensemble risk signal has **genuine economic value** for this stock."
            )
        elif sharpe_diff > -0.05:
            assessment_color = ACCENT_AMBER
            assessment = "NEUTRAL"
            assessment_text = (
                f"The hedging strategy produces **similar risk-adjusted returns** "
                f"(Sharpe diff: {sharpe_diff:+.2f}). The risk signal provides some "
                f"drawdown protection (DD improvement: {dd_improvement:+.1%}) but at "
                f"a cost to total returns during non-drought periods."
            )
        else:
            assessment_color = ACCENT_RED
            assessment = "NEGATIVE"
            assessment_text = (
                f"The hedging strategy **underperforms** buy-and-hold "
                f"(Sharpe diff: {sharpe_diff:+.2f}). Consider adjusting the risk "
                f"threshold higher or reducing the hedge ratio to minimize false positives."
            )

        st.markdown(f"""
        <div class="glass-card" style="border-left:4px solid {assessment_color};">
            <div style="font-size:1.15rem; font-weight:700; color:{assessment_color};
                        margin-bottom:0.5rem;">
                Strategy Value: {assessment}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(assessment_text)

    else:
        st.warning("Insufficient overlapping data between stock returns and risk scores for backtesting.")

elif not _stock_data:
    st.info("No stock data available. Please ensure data has been loaded via the main dashboard.")
else:
    st.info(f"Risk scores not available for {stock_select}. Train models first via the main dashboard.")

# ---------------------------------------------------------------------------
# Bottom spacer
# ---------------------------------------------------------------------------
st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
