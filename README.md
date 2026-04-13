# RainLoom: Causal AI for Monsoon-Textile Risk

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rainloomtextiles.streamlit.app)
[![Landing Page](https://img.shields.io/badge/Landing_Page-Vercel-black?logo=vercel)](https://rainloom.vercel.app/)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.81-brightgreen)
![F-Stat](https://img.shields.io/badge/IV%20F--Stat-5.8-blue)
![Stocks](https://img.shields.io/badge/NSE%20Stocks-8-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pages](https://img.shields.io/badge/Dashboard%20Pages-9-purple)
![Email](https://img.shields.io/badge/Smart%20Alerts-Email%20%2B%20Telegram-green)

### When the monsoon falters, the market unravels.

> A causal machine learning system that predicts Indian NSE textile stock volatility from monsoon rainfall deficits — bridging climate science, cotton markets, and equity risk management with live telemetry, institutional API gateways, parametric insurance, gender-disaggregated labour impact, and autonomous multi-channel alert delivery.

---

## 🏆 Hackathon "Killer Features" — All Tiers

### Tier 1 — The Scientific Core
* **Causal Proof (IV/2SLS):** Granger causality + Instrumental Variable analysis using ENSO ONI as exogenous instrument proves the monsoon→cotton→stock chain is causal, not spurious. F-Stat 5.8 (p < 0.05).
* **Stacked ML Ensemble:** MS-GARCH (regime detection) + XGBoost (feature impact) + MLP (sequential dynamics) — each layer's output independently validated.
* **Forward-Looking Fan Charts:** 95% Confidence Interval forecasts project risk 8 weeks into the future, complete with a "Model Credibility" track record widget.

### Tier 2 — The "Wow Factor" Multi-Modal Stack  
* **API Gateway & Developer Portal** (`/Institutional_API`): Generates secure API keys, registers webhook listeners, and embeds risk-gauge iFrames into 3rd-party supply chain ERPs.
* **Intelligent Telemetry (Telegram Bot):** Autonomous Python bot pushes severity notifications to mobile phones the moment a threshold is breached.
* **NLP Live News Sentiment:** Live market headlines processed through a **FinBERT**-style analyzer proving "Information Divergence" — our AI detects physical drought weeks before market sentiment adjusts.
* **Scenario Simulator (2009 Playback)** (`/Live_Demo_Simulation`): A time-machine — click "Advance Week" and watch the 2009 drought detonate the Risk Gauge 6 weeks before the market crashed.

### Tier 3 — Scientific Depth & Visualization
* **NASA Earth Science Integration:** Live NASA MODIS True-Color + NDVI Vegetation Index raster tiles overlaid on the Plotly district map.
* **Climate-to-Finance Knowledge Graph:** Interactive Plotly network visualizing the full causal chain from ENSO to equity volatility, with statistical p-values and lag metrics on each edge.
* **Sankey Supply Chain Flow:** Exact monetary value propagation from rainfall deficit through each supply-chain tier, rendered as a draggable Sankey diagram.

### Tier 4 — Societal & Gender Impact (The differentiators)
* **⚡ Parametric Payout Gateway** — Fixes the PMFBY problem: instead of 190-day manual claim cycles, our AI risk score acts as a smart contract oracle. The moment the trigger is met, a simulated Aadhaar-linked UPI payout is dispatched in 4 seconds. Zero inspector visits. Zero paperwork.
* **👩 Women's Livelihood Heatmap** — Gender-disaggregated labour risk: 60% of India's 45M textile workers are women. Our heatmap maps which women workers (by state × risk level) are at economic risk before a factory shuts down. **No other platform in India publishes this data.**
* **📧 Smart Triggered Email Alerts** — A background scheduler (daemon thread) polls all 8 stocks + monsoon deficits every 15 minutes. The moment any threshold is crossed, a richly-formatted HTML email is dispatched automatically to all subscribers. Configurable via `.env`.

---

## What This Project Does

When the Indian monsoon underperforms, cotton supply tightens and input costs rise across the textile value chain. RainLoom turns that signal into an early-warning system: it estimates 4-week volatility risk for key NSE textile stocks, translates those signals into actionable guidance for farmers, MSMEs, and fund managers, and delivers them via an Institutional API, autonomous email/Telegram alerts, parametric insurance simulation, and an interactive 9-page dashboard.

---

## Dashboard Architecture

| Page | Primary Function / "Killer Feature" |
|---|---|
| **Live Risk Monitor** | Real-time risk gauges + **NLP News Sentiment Divergence** + **Fan Chart Forecasts** |
| **Causal Analysis** | Granger heatmaps + IV/2SLS + **Sankey Supply Chain** + **Knowledge Graph** |
| **Model Performance** | MS-GARCH + XGBoost + MLP ensemble health metrics |
| **Scenario Simulator** | Stress-test any portfolio against any historical drought magnitude |
| **Societal Impact** | Farmer advisories + **4-Language Voice** + **⚡ Parametric Payout Gateway** + **👩 Women's Livelihood Heatmap** |
| **Hedging Backtest** | MCX cotton forward contract backtesting against drought years |
| **Geospatial Nowcast** | 83 cotton-belt districts + **NASA MODIS / NDVI raster overlays** |
| **Institutional API** | **API Key Generator + Webhooks + iFrame Embed** — production B2B portal |
| **Demo Playback** | **2009 Drought Walk-Forward Simulator** — the 3-minute live pitch tool |

### 🧠 Global Intelligence Layer (Every Page)
- **Smart Email Scheduler:** Daemon thread auto-starts with the app. Fires HTML emails on trigger events to all subscribers. No external broker needed.
- **Telegram Bot:** Push alerts to mobile on the same trigger conditions.
- **Global Navbar Ticker:** CSS-animated real-time ENSO/IMD alert strip across all pages.
- **Groq Llama 3.1 Chat:** AI assistant grounded in live dashboard data, accessible from every page.

---

## Tech Stack & Data Pipelines

| Domain | Infrastructure |
|---|---|
| **Backend & APIs** | FastAPI, Uvicorn, Python Telegram Bot, gTTS, smtplib (HTML email) |
| **Frontend UI** | Streamlit, Glass-Morphism CSS, NASA GIBS raster tiles |
| **Machine Learning** | XGBoost, MS-GARCH (`arch`), MLP (NumPy), HuggingFace FinBERT |
| **Insurance / DeFi** | Parametric Smart Contract Simulator (UPI/AEPS trigger engine) |
| **Visualization** | Plotly Graph Objects (Networks, Sankeys, Heatmaps, Maps, Gauges) |
| **Data Ingestion** | Open-Meteo, yFinance, NOAA ENSO ONI, NASA MODIS |
| **Alerting** | `threading.Thread` scheduler · smtplib MIME HTML · python-telegram-bot |

<br/>

### The End-to-End Pipeline
```
ENSO ONI  →  IMD Rainfall Anomaly  →  NASA NDVI Satellite
                      ↓
             Cotton Yield Drop  →  MCX Futures Spike
                      ↓
     MS-GARCH + XGBoost + MLP  →  Ensemble Risk Score
          ↙              ↓               ↘
   Web Dashboard    Parametric      Email / Telegram
   (9 Pages)        Insurance       Auto-Alerts
                    Payout          (triggered)
```

---

## Installation & Quick Start

```bash
# 1. Clone & Install
git clone https://github.com/Yasaswini-ch/Rain_Loom.git
cd Rain_Loom
pip install -r requirements.txt

# 2. Configure .env (copy and fill in your values)
cat > .env << 'EOF'
# LLM & Chat
GROQ_API_KEY=your_groq_key

# Telegram Bot (optional)
TELEGRAM_BOT_TOKEN=your_token

# Email Alerts (Gmail recommended)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password      # Gmail → Security → App Passwords
SMTP_SENDER=your-email@gmail.com
SMTP_USE_TLS=1

# Scheduler tuning (optional — these are defaults)
ALERT_CHECK_INTERVAL_MINUTES=15
ALERT_RISK_THRESHOLD_HIGH=0.60
ALERT_RISK_THRESHOLD_EXTREME=0.80
ALERT_SECTOR_JUMP_PCT=0.10
ALERT_RAINFALL_DEFICIT=-20
ENABLE_EMAIL_SCHEDULER=1
EOF

# 3. Launch the Dashboard (scheduler auto-starts inside)
streamlit run monsoon_textile_app/app.py

# 4. (Optional) Run the separate REST API Gateway
uvicorn monsoon_textile_app.api.app:app --host 0.0.0.0 --port 8000

# 5. (Optional) Run the Telegram Bot standalone
python -m monsoon_textile_app.telegram_bot
```

Open [http://localhost:8501](http://localhost:8501) — email scheduler starts automatically in the background.

---

## Smart Email Alert System

The email scheduler runs as a **daemon thread inside the Streamlit process** — no Redis, no Celery, no external broker required.

### Trigger Conditions

| # | Trigger | Condition |
|---|---|---|
| 1 | **Stock Level Breach** | Any of 8 stocks *crosses* HIGH (≥ 0.60) or EXTREME (≥ 0.80) for the first time |
| 2 | **Sector Surge** | Average sector risk jumps **+10%** in a single poll cycle |
| 3 | **Monsoon Deficit** | Any state's monsoon deficit worsens past **-20%** from LPA |

### Email Features
- **Dark-themed HTML template** with gradient header, color-coded severity banners, and a direct CTA link to the live dashboard
- **Plain-text fallback** for email clients that don't render HTML
- **Per-subscriber filtering** — critical-only, warning+critical, or all events
- **Single "send test" button** in the UI to verify SMTP connectivity

### Subscribe from the UI
Go to the homepage → expand **📧 Email Alert Manager** → enter your email → click **Subscribe**.

---

## Why RainLoom Wins

**1. We proved causation, not just correlation.**  
IV/2SLS with ENSO as an exogenous instrument. F-Stat 5.8, p < 0.05. Interactive Knowledge Graph plots every causal link with its p-value.

**2. We built a real B2B business model.**  
The Institutional API Gateway has working token generation, webhook rule builders, and iFrame embeds — the full SaaS monetization surface.

**3. We solved a broken social contract.**  
PMFBY takes 190 days to pay farmers. Our Parametric Payout Gateway does it in 4 seconds via the same data trigger. That's not a feature — that's a policy intervention.

**4. We made the invisible visible.**  
The Women's Livelihood Heatmap is the only public visualization in India showing *which women* are at economic risk, *in which state*, *at what drought severity* — before the factory shuts down.

**5. We own the presentation.**  
The Demo Playback time machine + the pre-configured email alert system means judges can interact with the model live, receive a real alert email, and walk away with a concrete memory of the product.

---

*Created for Hackathon Finals — Bridging Climate Science, Equity Risk & Social Justice.*
