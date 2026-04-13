# 🚀 RainLoom — Hackathon Summary

> **One sentence:** A causal ML system that gives fund managers, farmers, and textile MSMEs a 4-week early warning before monsoon-driven cotton shocks hit NSE stock prices.

---

## ⚡ Try It Live (2 minutes)

1. Visit **[rainloomtextiles.streamlit.app](https://rainloomtextiles.streamlit.app)**
2. Go to **Page 4 (Scenario Simulator)**
3. Click **"2009 Severe Drought"** preset → watch all 8 stocks spike to **EXTREME (80%+)**
4. Go to **Page 6 (Hedging Backtest)** → see **-34% → -13% drawdown** with RainLoom alerts
5. Go to **Page 7 (Geospatial)** → live 30-day rainfall for **83 cotton districts**

---

## 🎯 Key Numbers (Judges' Cheat Sheet)

| Metric | Value |
|---|---|
| AUC-ROC | **0.81** (baseline 0.67) |
| F1-Score | **0.72** |
| IV/2SLS F-statistic | **5.8** (ENSO ONI instrument) |
| Granger causality | **p < 0.05** at 4-week lag |
| Hedging drawdown reduction | **21.4pp** (2009 drought year) |
| Early warning lead time | **4–8 weeks** before market impact |
| Stocks monitored | **8 NSE textile companies** |
| Districts monitored | **83 cotton-belt districts** |
| Data sources | **7 live APIs** (IMD, NSE, MCX, Open-Meteo, NOAA, VIX, FX) |

---

## 🧠 Why This Is Hard

1. **Cross-domain:** Climate science + agricultural economics + quantitative finance in one pipeline
2. **Causal, not correlational:** IV/2SLS with ENSO ONI instrument addresses endogeneity — most ML projects skip this entirely
3. **7 live API integrations** with retry logic, 10-min caching, and graceful degradation
4. **Stakeholder translation:** Same risk score → insurance premiums for farmers, hedge ratios for MSMEs, VaR for fund managers
5. **No data leakage:** `TimeSeriesSplit` throughout + drift detection (Page-Hinkley + ADWIN) + Platt calibration

---

## 🏗️ Architecture (30-second version)

```
IMD rainfall + NSE stocks + MCX cotton + NOAA ENSO + Open-Meteo
            ↓
    Feature Engineering (24 features: climate, market, supply-chain lags)
            ↓
    Causal Validation: Granger (p<0.05) → Johansen → IV/2SLS (F=5.8)
            ↓
    ┌─────────┬──────────┬─────────┐
    │ GARCH   │ XGBoost  │  LSTM   │
    │  30%    │   40%    │  30%    │
    └─────────┴──────────┴─────────┘
            ↓
    Platt-calibrated ensemble → Risk Score [0,1]
            ↓
    Dashboard · Geospatial Map · AI Advisor · REST API · Email Alerts
```

---

## 📁 Key Code Locations

| File | What's interesting |
|---|---|
| `models/causal.py` | Granger + Johansen + IV/2SLS implementation |
| `models/regime.py` | MS-GARCH volatility regime detection |
| `models/drift_detector.py` | Page-Hinkley + ADWIN online drift detection |
| `pages/4_Scenario_Simulator.py` | Interactive what-if with XGBoost inference |
| `pages/7_Geospatial_Nowcast.py` | Live 83-district rainfall map (Open-Meteo API) |
| `api/routes.py` | FastAPI REST endpoints for programmatic access |
| `components/slm_engine.py` | Groq Llama-3 context-aware AI advisor |

---

## 🔑 Environment Setup

```bash
git clone https://github.com/Yasaswini-ch/Rain_Loom.git
cd Rain_Loom
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env
streamlit run monsoon_textile_app/app.py
```

RainLoom AI works without a key (template fallback). GROQ key enables Llama-3 70B responses.

---

## 📊 Ablation Study (What Matters Most)

| Component Removed | AUC Drop |
|---|---|
| Rainfall features | **−0.11** ← most important |
| XGBoost layer | −0.07 |
| Supply-chain lag features | −0.06 |
| ENSO ONI features | −0.05 |
| GARCH layer | −0.04 |
| LSTM layer | −0.03 |

---

*Built with Python 3.11 · Streamlit · XGBoost · statsmodels · FastAPI · Groq API*
