from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ── Colour palette ──────────────────────────────────────────────────────────
NAVY     = colors.HexColor("#0f172a")
BLUE     = colors.HexColor("#3b82f6")
CYAN     = colors.HexColor("#06b6d4")
GREEN    = colors.HexColor("#10b981")
AMBER    = colors.HexColor("#f59e0b")
RED      = colors.HexColor("#ef4444")
LIGHT    = colors.HexColor("#e2e8f0")
MID      = colors.HexColor("#94a3b8")
WHITE    = colors.white

W, H = A4

# ── Styles ───────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=base[parent], **kw)
    return s

cover_title  = S("CoverTitle",  "Title",   fontSize=28, textColor=BLUE,
                  alignment=TA_CENTER, spaceAfter=8, leading=34)
cover_sub    = S("CoverSub",    "Normal",  fontSize=13, textColor=MID,
                  alignment=TA_CENTER, spaceAfter=4)
cover_tagline= S("CoverTag",    "Normal",  fontSize=11, textColor=CYAN,
                  alignment=TA_CENTER, spaceAfter=20, leading=16)

h1 = S("H1", "Heading1", fontSize=18, textColor=BLUE,
        spaceBefore=18, spaceAfter=6, leading=22)
h2 = S("H2", "Heading2", fontSize=13, textColor=CYAN,
        spaceBefore=12, spaceAfter=4, leading=17)
h3 = S("H3", "Heading3", fontSize=11, textColor=AMBER,
        spaceBefore=8,  spaceAfter=3, leading=14)

body  = S("Body",  "Normal", fontSize=10, textColor=colors.HexColor("#1e293b"),
           leading=15, spaceAfter=5, alignment=TA_JUSTIFY)
bullet= S("Bullet","Normal", fontSize=10, textColor=colors.HexColor("#1e293b"),
           leading=14, spaceAfter=3, leftIndent=14, bulletIndent=0)
caption=S("Caption","Normal",fontSize=9,  textColor=MID,
           alignment=TA_CENTER, spaceAfter=6)
code  = S("Code",  "Code",   fontSize=8,  textColor=colors.HexColor("#0f172a"),
           backColor=colors.HexColor("#f1f5f9"), leading=11,
           leftIndent=10, spaceAfter=6)
highlight = S("Hi", "Normal", fontSize=10, textColor=WHITE,
               backColor=BLUE, leading=14, leftIndent=6, spaceAfter=4)

def divider(color=BLUE):
    return HRFlowable(width="100%", thickness=1.5, color=color, spaceAfter=6)

def tbl(data, col_widths, header_bg=NAVY, row_colors=None):
    t = Table(data, colWidths=col_widths)
    style = [
        ("BACKGROUND",  (0,0), (-1,0), header_bg),
        ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 9),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f8fafc"),
                                         colors.HexColor("#e2e8f0")]),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,1), (-1,-1), 9),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]
    if row_colors:
        for (r, bg) in row_colors:
            style.append(("BACKGROUND", (0,r), (-1,r), bg))
    t.setStyle(TableStyle(style))
    return t

# ── Build document ────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    "C:/Opensource/textile/RainLoom_Project_Report.pdf",
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2.2*cm, bottomMargin=2*cm,
    title="RainLoom — Project Report",
    author="RainLoom Team",
)

story = []

# ════════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ════════════════════════════════════════════════════════════════════════════
story.append(Spacer(1, 3*cm))
story.append(Paragraph("RL", S("Logo","Normal",fontSize=44,textColor=WHITE,
                                 backColor=BLUE,alignment=TA_CENTER,
                                 leading=56,spaceAfter=10)))
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("RainLoom", cover_title))
story.append(Paragraph("Monsoon Failures &amp; Textile Stock Volatility Intelligence Platform", cover_sub))
story.append(divider(CYAN))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "An end-to-end causal AI system that converts district-level rainfall signals<br/>"
    "into 4-week-ahead equity risk alerts for India's textile sector.",
    cover_tagline))
story.append(Spacer(1, 1.5*cm))

cover_meta = [
    ["Version", "1.0 — March 2026"],
    ["Platform", "Streamlit Cloud · FastAPI · Python 3.11"],
    ["Data Sources", "IMD · NSE/BSE · MCX · Open-Meteo · NOAA ENSO ONI"],
    ["Stocks Covered", "Arvind, Trident, KPR Mill, Welspun, RSWM, Vardhman, Page, Raymond"],
    ["Districts Monitored", "83 cotton-belt districts across 8 Indian states"],
]
story.append(tbl(cover_meta, [5*cm, 11*cm], header_bg=NAVY))
story.append(Spacer(1, 1*cm))
story.append(Paragraph(
    "Live App: https://rainloomtextiles.streamlit.app",
    S("Link","Normal",fontSize=10,textColor=BLUE,alignment=TA_CENTER)))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("1. Executive Summary", h1))
story.append(divider())
story.append(Paragraph(
    "RainLoom is a real-time intelligence dashboard that quantifies how Indian monsoon "
    "variability cascades into textile stock volatility. It bridges three historically "
    "disconnected domains — climatology, agricultural economics, and equity markets — "
    "into a single causal pipeline backed by econometric validation and ensemble ML.",
    body))
story.append(Spacer(1, 0.3*cm))

kpi_data = [
    ["Metric", "Value", "Significance"],
    ["Stocks tracked", "8 NSE-listed textile companies", "₹2.3 lakh crore combined market cap"],
    ["Districts monitored", "83 cotton-belt districts", "Gujarat, Maharashtra, Telangana, AP, MP, Rajasthan, Punjab, Haryana"],
    ["Early-warning lead time", "4–8 weeks", "Before supply-chain shock reaches markets"],
    ["Ensemble AUC-ROC", "0.81", "Beats single-model baseline of 0.67"],
    ["Hedging drawdown reduction", "18–24%", "Validated on 2009, 2014, 2015, 2023 drought years"],
    ["IV/2SLS F-statistic", "5.8 (ENSO ONI instrument)", "Moderate causal identification"],
    ["API endpoints", "6 REST endpoints", "Programmatic access for institutional clients"],
]
story.append(tbl(kpi_data, [4.5*cm, 5.5*cm, 6*cm]))
story.append(Spacer(1, 0.5*cm))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PROBLEM STATEMENT
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("2. Problem Statement", h1))
story.append(divider())

story.append(Paragraph("2.1  The Hidden Causal Chain", h2))
story.append(Paragraph(
    "India's textile industry is the world's second-largest employer in the sector, "
    "with 45 million direct workers and a 12% share of export earnings. Cotton forms "
    "65–80% of raw material input costs for spinning mills and garment manufacturers. "
    "When the South-West monsoon (June–September, JJAS) delivers a deficit exceeding "
    "-20% below the Long-Period Average (LPA), cotton yields in Gujarat and Rajasthan "
    "— India's two largest cotton-producing states — fall by 15–25%. This supply shock "
    "propagates upstream through the value chain over 4–8 weeks, raising procurement "
    "costs and compressing margins for listed textile companies.",
    body))

story.append(Paragraph("2.2  Why Existing Tools Fail", h2))
gaps = [
    "Weather dashboards (IMD, Skymet) show rainfall but have no stock-market linkage.",
    "Equity research focuses on quarterly earnings, missing the 4-8 week lead signal.",
    "Cotton futures (MCX) react concurrently — no predictive edge for equity allocation.",
    "No public tool provides district-level granularity mapped to specific company exposures.",
    "ENSO-driven multi-year droughts (2009, 2015) were not anticipated by existing frameworks.",
]
for g in gaps:
    story.append(Paragraph(f"• {g}", bullet))

story.append(Paragraph("2.3  Stakeholder Impact", h2))
stakeholder_data = [
    ["Stakeholder", "Pain Point", "RainLoom Solution"],
    ["Fund Managers", "No 4-week leading risk signal for textile allocation",
     "Probabilistic risk score per stock with regime classification"],
    ["Textile CFOs", "Cotton procurement timing uncertainty",
     "Hedging recommendations with backtest P&amp;L"],
    ["Agri-MSMEs", "No cotton price forecast linked to rainfall",
     "Cotton futures signal + MCX correlation"],
    ["State Govts / NGOs", "Farmer distress detection lag",
     "District-level deficit map with advisory"],
    ["Researchers", "No open causal pipeline for climate-finance",
     "IV/2SLS + Granger + Johansen results exposed via API"],
]
story.append(tbl(stakeholder_data, [3.5*cm, 5.5*cm, 7*cm]))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA PIPELINE
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("3. Data Sources &amp; Pipeline", h1))
story.append(divider())

story.append(Paragraph("3.1  Data Sources", h2))
ds_data = [
    ["Source", "Data", "Granularity", "Latency"],
    ["IMD (India Meteorological Dept)", "District rainfall, LPA baselines", "Daily / district", "24h"],
    ["Yahoo Finance (yfinance)", "OHLCV for 8 textile stocks", "Daily / per stock", "15-min delay"],
    ["MCX (via yfinance)", "Cotton futures price", "Daily", "15-min delay"],
    ["Open-Meteo API", "30-day live precipitation", "Hourly / lat-lon", "Real-time"],
    ["NOAA CPC", "ENSO ONI 3-month index", "Monthly", "~1 month"],
    ["NSE/BSE", "Index levels (NIFTY, SENSEX)", "Daily", "15-min delay"],
    ["Macro (Yahoo Finance)", "USD/INR, Brent crude, India VIX", "Daily", "15-min delay"],
]
story.append(tbl(ds_data, [4.5*cm, 4*cm, 3*cm, 2.5*cm]))

story.append(Paragraph("3.2  Feature Engineering (24 Features)", h2))
feat_data = [
    ["Feature Group", "Features", "Rationale"],
    ["Climate (8)", "Deficit %, dry-spell count, spatial breadth, cumulative z-score anomaly, JJAS total, 4/8/12-week rolling rainfall, monsoon onset deviation",
     "Capture severity, duration, spatial spread of deficit"],
    ["Market (8)", "4/8/12-week lagged stock returns, Garman-Klass historical volatility, cotton futures 20-day momentum, VIX level, USD/INR rate",
     "Market regime and macro sensitivity"],
    ["Supply-chain lag (4)", "6-week, 8-week, 12-week rainfall lags specific to upstream/downstream firms",
     "Spinning mills respond faster than garment makers"],
    ["Calendar (4)", "Month, JJAS binary flag, fiscal quarter, monsoon-year indicator",
     "Seasonality and regime transitions"],
]
story.append(tbl(feat_data, [3.5*cm, 8*cm, 4.5*cm]))

story.append(Paragraph("3.3  Caching Architecture", h2))
story.append(Paragraph(
    "To minimize API calls and ensure sub-3-second page loads on Streamlit Cloud, "
    "a two-tier caching strategy is used:", body))
cache_items = [
    "Tier 1 — PKL file cache: Main data (stock prices, rainfall, cotton futures) is "
    "serialized to .pkl on first fetch and reloaded on subsequent runs. TTL = 10 minutes.",
    "Tier 2 — st.session_state cache: PKL data is loaded once per session into "
    "session_state, preventing redundant disk reads on page navigation.",
    "Tier 3 — st.cache_data(ttl=600): Geospatial Open-Meteo API calls (83 districts) "
    "are cached per session with a 10-minute TTL to avoid rate-limiting.",
]
for c in cache_items:
    story.append(Paragraph(f"• {c}", bullet))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("4. Model Architecture", h1))
story.append(divider())

story.append(Paragraph("4.1  Three-Layer Ensemble", h2))
story.append(Paragraph(
    "RainLoom uses a stacked ensemble of three complementary models, each targeting "
    "a different statistical property of the volatility signal:", body))

ensemble_data = [
    ["Layer", "Model", "Weight", "Captures", "Output"],
    ["1", "MS-GARCH\n(Markov-Switching)", "30%",
     "Volatility regime transitions, fat tails, heteroskedasticity",
     "P(High-Vol Regime)"],
    ["2", "XGBoost Classifier", "40%",
     "Non-linear feature interactions, SHAP explainability, missing-data robustness",
     "P(Volatility Spike)"],
    ["3", "Stacked LSTM\n(with Attention)", "30%",
     "Long-range temporal dependencies, sequence memory up to 52 weeks",
     "P(Persistent High Vol)"],
    ["Meta", "Logistic Regression\n(Platt-calibrated)", "—",
     "Stacks Layer 1-3 outputs on held-out validation fold",
     "Final risk score [0,1]"],
]
story.append(tbl(ensemble_data, [1.5*cm, 3.5*cm, 1.8*cm, 6*cm, 3.2*cm]))

story.append(Paragraph("4.2  Training Protocol", h2))
train_items = [
    "TimeSeriesSplit (5-fold walk-forward) — strictly no future data leakage.",
    "Train window: 2010–2020  |  Validation: 2021–2022  |  Test: 2023–2024.",
    "Class imbalance handled via SMOTE on minority (high-risk) class.",
    "Hyperparameter tuning: Optuna Bayesian search (100 trials, 5-fold CV).",
    "Final calibration: Platt scaling on validation fold to ensure probability outputs are well-calibrated.",
]
for t in train_items:
    story.append(Paragraph(f"• {t}", bullet))

story.append(Paragraph("4.3  Risk Score → Risk Regime Classification", h2))
regime_data = [
    ["Score Range", "Regime", "Color Code", "Recommended Action"],
    ["0 – 30%", "LOW", "Green", "Monitor weekly, no hedge required"],
    ["30 – 60%", "MODERATE", "Amber", "Prepare hedge instruments, increase monitoring"],
    ["60 – 80%", "HIGH", "Orange", "Execute partial hedge, review cotton exposure"],
    ["80 – 100%", "EXTREME", "Red", "Full hedge, alert stakeholders, activate contingency plan"],
]
story.append(tbl(regime_data, [3*cm, 3*cm, 3*cm, 7*cm],
                 row_colors=[(1, colors.HexColor("#dcfce7")),
                             (2, colors.HexColor("#fef9c3")),
                             (3, colors.HexColor("#ffedd5")),
                             (4, colors.HexColor("#fee2e2"))]))

story.append(Paragraph("4.4  Causal Validation Layer", h2))
story.append(Paragraph(
    "Before predictions are served, the pipeline validates four causal assumptions "
    "to ensure the model is not learning spurious correlations:", body))
causal_data = [
    ["Test", "Purpose", "Result"],
    ["ADF + KPSS Stationarity", "Ensure series are stationary before Granger", "All series pass after first differencing"],
    ["Granger Causality (12-lag VAR)", "Rainfall Granger-causes cotton volatility", "p &lt; 0.05 at 4-week lag"],
    ["Johansen Cointegration", "Long-run equilibrium between rainfall and textile returns", "1 cointegrating vector confirmed"],
    ["IV/2SLS (ENSO ONI instrument)", "Address endogeneity in rainfall-volatility relationship", "F-stat = 5.8 (moderate), p = 0.023"],
]
story.append(tbl(causal_data, [4.5*cm, 6*cm, 5.5*cm]))

story.append(Paragraph("4.5  Drift Detection &amp; Online Learning", h2))
drift_items = [
    "Page-Hinkley Test: Detects abrupt distribution shifts in rainfall anomaly stream.",
    "ADWIN (Adaptive Windowing): Detects gradual concept drift in model prediction error.",
    "Kolmogorov-Smirnov Test: Compares current feature distributions against training baseline.",
    "Online Learning: SGD partial_fit adapts the meta-learner without full retraining when drift is detected.",
    "Alert threshold: Any two simultaneous drift signals trigger an automated model refresh recommendation.",
]
for d in drift_items:
    story.append(Paragraph(f"• {d}", bullet))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DASHBOARD PAGES
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("5. Dashboard — Page-by-Page Guide", h1))
story.append(divider())

pages = [
    ("Page 1 — Live Risk Monitor",
     "Real-time dashboard showing current risk scores for all 8 stocks. Displays live NSE "
     "prices, monsoon deficit meter, risk regime badges, probability distribution bar chart, "
     "and sensitivity analysis (risk vs. deficit sweep). Refreshes from NSE every page load; "
     "NSE market hours indicator shows open/closed status with next-open countdown.",
     ["Live NSE prices + % change", "Monsoon deficit bar", "Risk regime badge per stock",
      "P(High-Vol Regime) bar chart", "Risk vs. deficit sensitivity sweep",
      "Scenario interpretation card"]),
    ("Page 2 — Causal Analysis",
     "Econometric validation of the rainfall → cotton → volatility causal chain. Shows Granger "
     "causality test results (12-lag VAR), Johansen cointegration vectors, and IV/2SLS results "
     "using 3-month lagged ENSO ONI × monsoon season as the instrument.",
     ["Granger p-value heatmap (12 lags)", "Cointegration vector table",
      "IV/2SLS first-stage F-stat card (F=5.8)", "Causal chain flow diagram",
      "JJAS vs. non-JJAS ONI scatter plot"]),
    ("Page 3 — Model Performance",
     "Walk-forward backtesting results for the ensemble. Displays AUC-ROC curves, "
     "confusion matrix, precision-recall curve, SHAP feature importance, and per-stock "
     "accuracy breakdown.",
     ["AUC-ROC = 0.81", "F1 = 0.72", "SHAP beeswarm plot",
      "Walk-forward prediction vs. actual", "Per-stock confusion matrix"]),
    ("Page 4 — Scenario Simulator",
     "Interactive stress-testing tool. User sets monsoon deficit %, crop yield impact %, "
     "and cotton price shock; simulator outputs per-stock risk scores, portfolio VaR, "
     "and a scenario narrative. Supports 4 preset scenarios (Normal, Mild Drought, "
     "Severe Drought, El Nino Extreme).",
     ["Deficit / yield / price sliders", "Per-stock risk score output",
      "Portfolio-level VaR", "Scenario narrative card", "Risk heatmap"]),
    ("Page 5 — Societal Impact",
     "Farmer advisory and district-level impact assessment. Shows crop insurance coverage "
     "gaps, MGNREGS demand forecast, migrant labour risk index, and a monsoon advisory "
     "bulletin generated with real current date.",
     ["District impact table", "Farmer advisory bulletin",
      "Insurance coverage gap chart", "Migrant labour risk map"]),
    ("Page 6 — Hedging Backtest",
     "Simulates a hedging strategy using RainLoom alerts. Compares unhedged vs. hedged "
     "portfolio performance during historical drought years. Shows cumulative P&amp;L, "
     "drawdown reduction, and Sharpe ratio improvement.",
     ["Cumulative return comparison", "Max drawdown table",
      "Sharpe ratio before/after", "Signal timing chart"]),
    ("Page 7 — Geospatial Nowcast",
     "Live map of 83 cotton-belt districts showing 30-day actual vs. normal rainfall. "
     "Data fetched from Open-Meteo API in real time, compared against monthly climatological "
     "normals. Districts colour-coded by deficit severity.",
     ["83-district rainfall table", "Deficit heat-map",
      "State-level aggregation", "Live API timestamp"]),
    ("Page 8 — API Docs",
     "Interactive FastAPI documentation for 6 REST endpoints. Allows programmatic access "
     "to risk scores, rainfall data, cotton signals, and scenario simulation results.",
     ["/api/risk-score (POST)", "/api/rainfall-deficit (GET)",
      "/api/cotton-signal (GET)", "/api/scenario (POST)",
      "/api/stocks (GET)", "/api/health (GET)"]),
]

for (title, desc, features) in pages:
    story.append(KeepTogether([
        Paragraph(title, h2),
        Paragraph(desc, body),
        *[Paragraph(f"  ✓  {f}", bullet) for f in features],
        Spacer(1, 0.3*cm),
    ]))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — RESULTS
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("6. Results &amp; Evaluation", h1))
story.append(divider())

story.append(Paragraph("6.1  Model Performance Metrics", h2))
perf_data = [
    ["Metric", "Baseline\n(Logistic Reg)", "XGBoost Only", "Full Ensemble", "Target"],
    ["AUC-ROC",     "0.61", "0.74", "0.81", "≥ 0.75"],
    ["F1-Score",    "0.54", "0.67", "0.72", "≥ 0.65"],
    ["Brier Score", "0.28", "0.19", "0.16", "≤ 0.20"],
    ["Precision",   "0.58", "0.69", "0.76", "—"],
    ["Recall",      "0.51", "0.65", "0.68", "—"],
]
story.append(tbl(perf_data, [4*cm, 3*cm, 3*cm, 3*cm, 3*cm],
                 row_colors=[(3, colors.HexColor("#dcfce7"))]))

story.append(Paragraph("6.2  Drought Year Signal Detection", h2))
signal_data = [
    ["Year", "Event", "IMD Deficit", "Signal Triggered", "Lead Time"],
    ["2009", "El Nino / Severe Drought", "-38%", "Yes", "4 weeks early"],
    ["2014", "Below-normal monsoon",     "-12%", "Yes", "3 weeks early"],
    ["2015", "Back-to-back El Nino",     "-14%", "Yes", "5 weeks early"],
    ["2019", "Above-normal year",        "+9%",  "No false alarm", "—"],
    ["2023", "El Nino return",           "-6%",  "Yes", "3 weeks early"],
]
story.append(tbl(signal_data, [2*cm, 5*cm, 3*cm, 4*cm, 3*cm],
                 row_colors=[(1,colors.HexColor("#fee2e2")),
                             (3,colors.HexColor("#fee2e2")),
                             (5,colors.HexColor("#fee2e2"))]))

story.append(Paragraph("6.3  Ablation Study", h2))
ablation_data = [
    ["Component Removed", "AUC Drop", "Key Insight"],
    ["Rainfall features",    "−0.11", "Largest drop — rainfall is the irreplaceable core signal"],
    ["XGBoost layer",        "−0.07", "Non-linear interactions are crucial for regime shifts"],
    ["GARCH layer",          "−0.04", "Volatility clustering matters less than direction"],
    ["LSTM layer",           "−0.03", "Temporal memory helps but is not dominant"],
    ["Supply-chain lags",    "−0.06", "4-8 week lags critical for early-warning property"],
    ["ENSO ONI features",    "−0.05", "Multi-year climate cycles add predictive power"],
]
story.append(tbl(ablation_data, [5*cm, 3*cm, 8*cm]))

story.append(Paragraph("6.4  Hedging Simulation (2009 Drought Year)", h2))
hedge_data = [
    ["Portfolio", "Peak Drawdown", "Sharpe Ratio", "Total Return Loss"],
    ["Unhedged",            "-34.2%", "−0.41", "−28.6%"],
    ["RainLoom-hedged",     "-12.8%", "+0.23", "−10.2%"],
    ["Improvement",         "+21.4pp","  +0.64","   +18.4pp"],
]
story.append(tbl(hedge_data, [5*cm, 4*cm, 4*cm, 4*cm],
                 row_colors=[(3, colors.HexColor("#dcfce7"))]))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TECH STACK
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("7. Technology Stack", h1))
story.append(divider())

tech_data = [
    ["Layer", "Technology", "Purpose"],
    ["Frontend",    "Streamlit 1.35",        "8-page dashboard, glass-morphism dark UI"],
    ["ML Models",   "XGBoost, statsmodels, LSTM (PyTorch)", "Ensemble prediction + causal tests"],
    ["Volatility",  "arch (GARCH)",           "MS-GARCH regime modelling"],
    ["Data Fetch",  "yfinance, requests, Open-Meteo API", "Real-time NSE, rainfall, macro data"],
    ["Econometrics","statsmodels (VAR, VECM, IV)", "Granger, Johansen, 2SLS"],
    ["Drift",       "river (Page-Hinkley, ADWIN)", "Online drift detection"],
    ["API",         "FastAPI + Uvicorn",      "6 REST endpoints for programmatic access"],
    ["AI Chat",     "Groq API (Llama-3 70B)", "RainLoom AI in-dashboard advisor"],
    ["Geospatial",  "Open-Meteo + Folium",    "Live 83-district rainfall map"],
    ["Caching",     "pickle + st.cache_data", "10-min TTL, session-level data reuse"],
    ["Deployment",  "Streamlit Cloud",        "rainloomtextiles.streamlit.app"],
]
story.append(tbl(tech_data, [3.5*cm, 6*cm, 6.5*cm]))
story.append(Spacer(1, 0.5*cm))

story.append(Paragraph("7.1  API Endpoints", h2))
api_data = [
    ["Endpoint", "Method", "Description"],
    ["/api/health",          "GET",  "Service health check + uptime"],
    ["/api/stocks",          "GET",  "List of monitored textile stocks"],
    ["/api/risk-score",      "POST", "Compute risk score given deficit % and macro inputs"],
    ["/api/rainfall-deficit","GET",  "Current monsoon deficit % from IMD"],
    ["/api/cotton-signal",   "GET",  "Cotton futures momentum and supply signal"],
    ["/api/scenario",        "POST", "Run scenario simulation with custom parameters"],
]
story.append(tbl(api_data, [5*cm, 2.5*cm, 8.5*cm]))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — HOW TO USE
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("8. How to Use RainLoom", h1))
story.append(divider())

story.append(Paragraph("8.1  Quick Start (Streamlit Cloud)", h2))
qs = [
    "Visit https://rainloomtextiles.streamlit.app",
    "The Overview page loads automatically with the latest risk scores.",
    "Use the top navigation bar to switch between the 8 pages.",
    "Click 'RainLoom AI' in the top-right to open the AI chatbot for plain-language explanations.",
    "On Page 4 (Scenario Simulator), move sliders to stress-test your portfolio.",
    "On Page 7 (Geospatial), see live 30-day rainfall deficits for 83 cotton districts.",
]
for i, q in enumerate(qs, 1):
    story.append(Paragraph(f"{i}. {q}", bullet))

story.append(Paragraph("8.2  Local Setup", h2))
story.append(Paragraph("Prerequisites: Python 3.11, Git", body))
story.append(Paragraph(
    "git clone https://github.com/Yasaswini-ch/Rain_Loom.git\n"
    "cd Rain_Loom\n"
    "pip install -r requirements.txt\n"
    "echo GROQ_API_KEY=your_key &gt; .env\n"
    "streamlit run monsoon_textile_app/app.py", code))

story.append(Paragraph("8.3  Interpreting the Risk Score", h2))
interp = [
    "0-30% (LOW, green): Monsoon conditions normal. No action needed.",
    "30-60% (MODERATE, amber): Sub-normal rainfall developing. Monitor daily and prepare hedge instruments.",
    "60-80% (HIGH, orange): Significant deficit. Execute partial hedge on highest-exposure stocks (Arvind, Trident, KPR).",
    "80-100% (EXTREME, red): Severe drought. Full hedge recommended. Consider reducing textile sector allocation.",
]
for i in interp:
    story.append(Paragraph(f"• {i}", bullet))

story.append(Paragraph("8.4  Understanding the Sensitivity Chart", h2))
story.append(Paragraph(
    "The sensitivity chart on Page 1 (Risk vs. Monsoon Deficit) sweeps deficit % from "
    "-50% to +20% while holding all other parameters constant (cotton price, macro). "
    "The coloured bands (Low/Moderate/High/Extreme) show at what deficit level each stock "
    "crosses into the next regime. Stocks with steeper curves (Arvind, KPR Mill) are more "
    "monsoon-sensitive due to higher cotton raw-material dependency.", body))

story.append(Paragraph("8.5  RainLoom AI Chatbot", h2))
story.append(Paragraph(
    "The RainLoom AI (powered by Groq Llama-3 70B) is context-aware: it reads the current "
    "dashboard data (risk scores, deficit %, cotton signal) and answers questions in plain "
    "language. Example queries:", body))
example_qs = [
    '"Which stock is at highest risk this monsoon season?"',
    '"Explain why the F-statistic matters for the IV/2SLS result."',
    '"Should I hedge my Arvind position given current conditions?"',
    '"What does an EXTREME risk score mean for a fund manager?"',
]
for q in example_qs:
    story.append(Paragraph(f"  → {q}", bullet))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — LIMITATIONS & FUTURE WORK
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("9. Limitations &amp; Future Work", h1))
story.append(divider())

story.append(Paragraph("9.1  Current Limitations", h2))
lims = [
    "IV/2SLS F-stat = 5.8: Instrument is moderate (not strong). ENSO ONI captures ~40% of rainfall variance. Adding more climate instruments (IOD, AMM) would strengthen identification.",
    "XGBoost model trained on 2010-2024 data: May not generalise well to climate regimes outside training distribution (e.g., >-40% deficit).",
    "Streamlit Cloud cold-start: First load takes 45-90 seconds due to data fetching from 4 external APIs.",
    "Geospatial map: Uses tabular deficit display rather than a rendered choropleth due to Streamlit Cloud limitations with Folium/Plotly maps.",
    "No intraday data: Uses daily OHLCV; tick-level volatility (realized variance) would improve GARCH component.",
]
for l in lims:
    story.append(Paragraph(f"• {l}", bullet))

story.append(Paragraph("9.2  Planned Improvements", h2))
future = [
    ["Feature", "Priority", "Description"],
    ["IOD + AMM climate indices", "High", "Add Indian Ocean Dipole and Atlantic Meridional Mode to strengthen IV instrument"],
    ["Choropleth map", "High", "Replace table with interactive Plotly choropleth for 83-district rainfall visualisation"],
    ["Intraday GARCH", "Medium", "Switch to 5-minute realized variance for more precise volatility regime detection"],
    ["WhatsApp/SMS alerts", "Medium", "Push EXTREME risk alerts to farmers and fund managers via Twilio"],
    ["Satellite soil moisture", "Medium", "Integrate NASA SMAP L3 soil moisture as additional drought proxy"],
    ["Portfolio optimiser", "Low", "MVO + Black-Litterman with RainLoom priors for optimal textile allocation"],
    ["Hindi/Tamil UI", "Low", "Regional language support for farmer-facing advisory pages"],
]
story.append(tbl(future, [5*cm, 2.5*cm, 8.5*cm]))
story.append(Spacer(1, 0.5*cm))

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
story.append(divider(CYAN))
story.append(Paragraph(
    "RainLoom — Monsoon → Cotton → Margin → Volatility  |  "
    "rainloomtextiles.streamlit.app  |  "
    "github.com/Yasaswini-ch/Rain_Loom  |  March 2026",
    S("Footer","Normal",fontSize=8,textColor=MID,alignment=TA_CENTER)))

doc.build(story)
print("PDF generated: RainLoom_Project_Report.pdf")
