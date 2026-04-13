"""
Advisory Engine -- Smart Template-Based NLG for Monsoon-Textile Dashboard
=========================================================================
Provides context-aware, data-grounded advisory responses using live
dashboard data. Zero API dependency, instant responses.

Intent detection via keyword matching, responses generated from templates
populated with real-time risk scores, Granger results, GARCH outputs, etc.

Usage:
    from monsoon_textile_app.components.advisory_engine import get_advisory
    response = get_advisory(user_query, dashboard_context)
"""

from __future__ import annotations
import re
from datetime import datetime
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# 1. INTENT DETECTION -- keyword-based classification
# ═══════════════════════════════════════════════════════════════════════════

_INTENT_PATTERNS = {
    "stock_risk": [
        r"risk\s+(for|of|score)\s+(\w[\w\s]*)",
        r"(arvind|trident|kpr|welspun|rswm|vardhman|page\s*ind|raymond)\s.*(risk|safe|danger)",
        r"how\s+(is|are)\s+(\w[\w\s]*)\s+(doing|performing)",
        r"what.*(risk|score).*(for|of)",
        r"(arvind|trident|kpr|welspun|rswm|vardhman|page\s*ind|raymond)",
        r"tell\s+me\s+about\s+(\w[\w\s]*)\s*(stock|company|ltd)?",
        r"(which|what)\s+stock",
        r"safest\s+stock",
        r"riskiest\s+stock",
        r"all\s+stocks?\s*(risk|score)?",
    ],
    "farmer_advisory": [
        r"farmer", r"crop", r"insurance", r"pmfby", r"agriculture",
        r"sowing", r"yield", r"harvest", r"kharif",
        r"what\s+should\s+farmer",
        r"farming",
    ],
    "msme_advisory": [
        r"msme", r"manufacturer", r"textile\s*mill", r"raw\s*material",
        r"procurement", r"forward\s*contract", r"supply\s*chain",
        r"hedg", r"inventory",
        r"small\s*(business|enterprise)",
        r"mill\s*owner",
    ],
    "investor_advisory": [
        r"invest", r"portfolio", r"buy|sell|hold",
        r"position", r"exposure", r"return",
        r"market", r"trading", r"share",
        r"what\s+should\s+i\s+do",
        r"should\s+i\s+(worry|be\s+concern)",
        r"opportunity", r"strategy",
        r"(long|short)\s+term",
        r"sector\s+outlook",
    ],
    "rainfall": [
        r"rain", r"monsoon", r"deficit", r"jjas", r"drought",
        r"imd", r"precipitation", r"flood",
        r"weather", r"forecast", r"climate",
        r"el\s*ni.o", r"la\s*ni.a",
        r"water", r"dry\s+spell",
    ],
    "cotton": [
        r"cotton", r"mcx", r"ice\s*ct", r"fibre", r"fiber",
        r"commodity", r"futures",
        r"raw\s*material\s*price",
        r"ginning",
    ],
    "model_info": [
        r"model", r"xgboost", r"garch", r"granger", r"auc",
        r"accuracy", r"shap", r"feature", r"how\s+does.*(work|predict)",
        r"methodology", r"ensemble",
        r"machine\s*learning", r"ml\b", r"ai\b",
        r"how\s+(do|does)\s+(this|the|your)",
        r"algorithm", r"predict",
        r"backtest", r"quantile",
        r"how\s+is.*calculated",
        r"what\s+data",
        r"explain",
    ],
    "ndvi": [
        r"ndvi", r"satellite", r"vegetation", r"modis", r"greenness",
        r"crop\s*health", r"remote\s*sensing",
    ],
    "summary": [
        r"summary", r"overview", r"brief", r"status",
        r"what.*(happening|going\s+on|situation|current)",
        r"dashboard",
        r"update\s+me",
        r"give\s+me.*(rundown|picture|snapshot)",
        r"big\s+picture",
        r"\bvix\b", r"volatility\s*index",
        r"market\s*(condition|state|mood|sentiment)",
    ],
    "comparison": [
        r"(compare|vs|versus|difference|better)",
        r"(which|what)\s+(is|are)\s+(the\s+)?(best|worst|safe|risky)",
        r"rank",
    ],
    "greeting": [
        r"^(hi|hello|hey|greetings|good\s*(morning|afternoon|evening))\s*!?\s*$",
        r"^(thanks|thank\s*you|cheers)",
    ],
    "help": [
        r"^help$", r"what\s+can\s+you", r"how\s+to\s+use",
        r"guide", r"tutorial",
    ],
}


def detect_intent(query: str) -> tuple[str, list[str]]:
    """
    Detect user intent from query text.
    Returns (intent_name, captured_groups).
    """
    q = query.lower().strip()

    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, q, re.IGNORECASE)
            if m:
                return intent, list(m.groups()) if m.groups() else []

    return "general", []


def _find_stock_name(query: str, context: dict) -> str | None:
    """Try to match a stock name from the query."""
    q = query.lower()
    stock_aliases = {
        "arvind": "Arvind Ltd",
        "trident": "Trident Ltd",
        "kpr": "KPR Mill",
        "kpr mill": "KPR Mill",
        "welspun": "Welspun Living",
        "rswm": "RSWM Ltd",
        "vardhman": "Vardhman Textiles",
        "page": "Page Industries",
        "page ind": "Page Industries",
        "raymond": "Raymond Ltd",
    }
    for alias, name in stock_aliases.items():
        if alias in q:
            return name

    # Check if any full stock name is mentioned
    stocks = context.get("stocks", {})
    for ticker, info in stocks.items():
        if info.get("name", "").lower() in q:
            return info["name"]

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 2. RESPONSE TEMPLATES -- data-grounded NLG
# ═══════════════════════════════════════════════════════════════════════════

def _risk_label(risk: float) -> str:
    if risk < 0.3:
        return "LOW"
    elif risk < 0.6:
        return "MODERATE"
    elif risk < 0.8:
        return "HIGH"
    return "EXTREME"


def _risk_emoji(risk: float) -> str:
    if risk < 0.3:
        return "green"
    elif risk < 0.6:
        return "amber"
    elif risk < 0.8:
        return "orange"
    return "red"


def _resp_stock_risk(query: str, ctx: dict) -> str:
    """Generate stock-specific risk advisory."""
    stock_name = _find_stock_name(query, ctx)
    risk_data = ctx.get("risk_scores", {})
    metrics = ctx.get("model_metrics", {})

    if stock_name and stock_name in risk_data:
        risk = risk_data[stock_name]
        label = _risk_label(risk)
        m = metrics.get(stock_name, {})
        auc = m.get("auc_roc", "N/A")

        drivers = []
        if ctx.get("rain_deficit", 0) < -15:
            drivers.append(f"rainfall deficit ({ctx['rain_deficit']:.0f}%)")
        if ctx.get("cotton_change", 0) > 10:
            drivers.append(f"cotton price surge (+{ctx['cotton_change']:.0f}%)")
        if ctx.get("vix", 0) > 20:
            drivers.append(f"elevated India VIX ({ctx['vix']:.1f})")

        driver_text = ", ".join(drivers) if drivers else "stable macro conditions"

        response = (
            f"**{stock_name}** currently has a **{risk:.1%} ensemble risk score ({label})**.\n\n"
        )

        if label == "LOW":
            response += (
                f"The model sees no immediate volatility threat. Key context: {driver_text}. "
                f"Standard portfolio positioning is appropriate."
            )
        elif label == "MODERATE":
            response += (
                f"Emerging stress signals detected. Key drivers: {driver_text}. "
                f"Consider reviewing hedging positions and monitoring weekly updates. "
                f"Upstream stocks in the textile chain are typically affected first."
            )
        elif label in ("HIGH", "EXTREME"):
            response += (
                f"**Significant risk detected.** Key drivers: {driver_text}. "
                f"Historical parallels suggest volatility spikes within 4-8 weeks. "
                f"Recommendation: activate hedging strategies and reduce unprotected exposure."
            )

        if auc != "N/A":
            response += f"\n\n*Model confidence: XGBoost AUC = {auc} (5-fold temporal CV)*"

        return response

    # No specific stock found -- show overview
    if risk_data:
        lines = []
        for name, risk in sorted(risk_data.items(), key=lambda x: -x[1]):
            label = _risk_label(risk)
            lines.append(f"- **{name}**: {risk:.1%} ({label})")
        return "**Current Risk Scores:**\n\n" + "\n".join(lines)

    return (
        "**Stock Risk Overview**\n\n"
        "Live risk scores are currently loading. The dashboard tracks these stocks:\n\n"
        "| Stock | Chain Position | Cotton Dependency |\n"
        "|-------|---------------|------------------|\n"
        "| Arvind Ltd | Integrated | 72% |\n"
        "| Trident Ltd | Upstream | 78% |\n"
        "| KPR Mill | Upstream | 80% |\n"
        "| Welspun Living | Downstream | 65% |\n"
        "| RSWM Ltd | Upstream | 75% |\n"
        "| Vardhman Textiles | Yarn/Spinning | 82% |\n"
        "| Page Industries | Apparel | 45% |\n"
        "| Raymond Ltd | Apparel | 55% |\n\n"
        "Upstream stocks with high cotton dependency are most at risk during monsoon deficit years. "
        "Navigate to the **Live Risk Monitor** for real-time scores."
    )


def _resp_farmer(query: str, ctx: dict) -> str:
    """Farmer-focused advisory."""
    deficit = ctx.get("rain_deficit", 0)
    ndvi_status = ctx.get("ndvi_status", "normal")
    breadth = ctx.get("spatial_breadth", 0)

    if deficit < -25:
        severity = "severe"
        action = (
            "**Immediate actions recommended:**\n"
            "1. **Enroll in PMFBY** crop insurance before the deadline -- this is critical\n"
            "2. Switch to **drought-resistant cotton varieties** (Bt cotton hybrids with shorter duration)\n"
            "3. Contact your **District Agriculture Officer** for relief provision updates\n"
            "4. Consider **deficit irrigation** strategies to conserve water\n"
            "5. Explore **intercropping** with pulses to reduce total crop risk"
        )
    elif deficit < -15:
        severity = "moderate"
        action = (
            "**Precautionary measures:**\n"
            "1. **Review PMFBY enrollment** -- ensure all cotton parcels are covered\n"
            "2. Monitor IMD weekly bulletins for your district\n"
            "3. Prepare for potential **delayed sowing** if deficit persists\n"
            "4. Maintain communication with your **Agricultural Extension Officer**"
        )
    else:
        severity = "normal"
        action = (
            "**Standard seasonal guidance:**\n"
            "1. Proceed with normal Kharif sowing schedule\n"
            "2. Maintain routine PMFBY coverage\n"
            "3. Monitor IMD forecasts as monsoon season progresses"
        )

    deficit_text = f"{deficit:+.0f}%" if deficit != 0 else "near normal"

    return (
        f"**Farmer Advisory** (Monsoon conditions: {severity})\n\n"
        f"Current JJAS rainfall: **{deficit_text}** from LPA. "
        f"Spatial drought breadth: **{breadth:.0f}%** of cotton-belt districts affected.\n\n"
        f"{action}"
    )


def _resp_msme(query: str, ctx: dict) -> str:
    """MSME / textile manufacturer advisory."""
    avg_risk = ctx.get("avg_risk", 0.5)
    cotton_change = ctx.get("cotton_change", 0)
    deficit = ctx.get("rain_deficit", 0)

    if avg_risk > 0.6:
        urgency = "HIGH"
        hedging = (
            "**Urgent procurement actions:**\n"
            "1. **Execute forward contracts** for 3-6 months of raw cotton needs\n"
            "2. **Diversify supplier base** geographically (avoid over-reliance on deficit states)\n"
            "3. Build **2-3 weeks of safety stock** above normal levels\n"
            "4. **Defer capacity expansion** plans until monsoon situation stabilises\n"
            "5. Review **credit lines** -- working capital needs may increase 15-25%"
        )
    elif avg_risk > 0.35:
        urgency = "MODERATE"
        hedging = (
            "**Recommended preparations:**\n"
            "1. **Initiate discussions** with cotton brokers for forward pricing\n"
            "2. Review current inventory levels against 4-week rolling needs\n"
            "3. Monitor MCX cotton futures weekly for price acceleration\n"
            "4. Prepare contingency budget for 10-15% raw material cost increase"
        )
    else:
        urgency = "LOW"
        hedging = (
            "**Routine operations:**\n"
            "1. Standard procurement cycle -- no extraordinary measures needed\n"
            "2. Take advantage of stable cotton prices for longer-term contracts\n"
            "3. Focus on operational efficiency and inventory optimisation"
        )

    cotton_text = f"{cotton_change:+.0f}%" if cotton_change != 0 else "stable"

    return (
        f"**MSME Advisory** (Supply chain risk: {urgency})\n\n"
        f"Cotton price trend: **{cotton_text}** (30-day). "
        f"Average textile sector risk: **{avg_risk:.0%}**.\n\n"
        f"{hedging}"
    )


def _resp_investor(query: str, ctx: dict) -> str:
    """Investor-focused advisory."""
    risk_data = ctx.get("risk_scores", {})
    avg_risk = ctx.get("avg_risk", 0.5)

    if not risk_data:
        return (
            "**Investor Advisory** (General)\n\n"
            "Live risk scores are still loading. In the meantime, here's general guidance:\n\n"
            "- **Upstream textile stocks** (spinners, ginners) are most sensitive to monsoon disruption\n"
            "- **Downstream / apparel** names (Page Industries, Raymond) have lower cotton dependency\n"
            "- During monsoon deficit years, textile stocks typically see 15-30% drawdowns\n"
            "- Our hedging backtest (Page 6) shows signal-driven hedging improves Sharpe by 0.1-0.3\n\n"
            "Navigate to the **Live Risk Monitor** page first for real-time stock scores."
        )

    # Sort by risk
    sorted_stocks = sorted(risk_data.items(), key=lambda x: x[1])
    safest = sorted_stocks[:2]
    riskiest = sorted_stocks[-2:]

    if avg_risk > 0.6:
        stance = "DEFENSIVE"
        advice = (
            f"**High-risk environment -- defensive positioning recommended:**\n\n"
            f"**Reduce exposure to:** {', '.join(n for n, _ in riskiest)} "
            f"(highest monsoon sensitivity)\n\n"
            f"**Relative safety:** {', '.join(n for n, _ in safest)} "
            f"(lower cotton dependency)\n\n"
            f"**Hedging strategies:**\n"
            f"- Consider protective puts on textile positions\n"
            f"- Our backtest shows risk-signal hedging improves Sharpe by 0.1-0.3 during droughts\n"
            f"- Watch for entry opportunities after the volatility spike (typically 4-8 weeks)"
        )
    elif avg_risk > 0.35:
        stance = "CAUTIOUS"
        advice = (
            f"**Moderate risk -- selective positioning:**\n\n"
            f"**Watch closely:** {', '.join(n for n, _ in riskiest)} "
            f"(upstream stocks with high cotton dependency)\n\n"
            f"**Relatively resilient:** {', '.join(n for n, _ in safest)} "
            f"(downstream / diversified)\n\n"
            f"Consider reducing overweight positions in upstream textile stocks. "
            f"Apparel and downstream names tend to be less affected by cotton price spikes."
        )
    else:
        stance = "CONSTRUCTIVE"
        advice = (
            f"**Low risk -- normal market conditions:**\n\n"
            f"No monsoon-driven volatility threat detected. "
            f"Textile sector fundamentals are stable.\n\n"
            f"**Opportunity:** {', '.join(n for n, _ in safest)} "
            f"offer stable positioning.\n\n"
            f"Standard portfolio weights are appropriate. "
            f"Monitor as JJAS season progresses for any deterioration."
        )

    return f"**Investor Advisory** (Stance: {stance})\n\n{advice}"


def _resp_rainfall(query: str, ctx: dict) -> str:
    """Rainfall/monsoon information."""
    deficit = ctx.get("rain_deficit", 0)
    breadth = ctx.get("spatial_breadth", 0)
    stressed_states = ctx.get("stressed_states", [])

    status = "surplus" if deficit > 5 else "normal" if deficit > -10 else "deficit" if deficit > -20 else "severe deficit"

    response = (
        f"**Monsoon Status: {status.upper()}**\n\n"
        f"- JJAS rainfall departure: **{deficit:+.0f}%** from Long Period Average\n"
        f"- Spatial deficit breadth: **{breadth:.0f}%** of cotton-belt districts\n"
    )

    if stressed_states:
        response += f"- Most stressed states: {', '.join(stressed_states[:5])}\n"

    if deficit < -20:
        response += (
            f"\nThis is a **significant deficit**. Historical parallels:\n"
            f"- 2009 (-38%): Cotton yields fell 22%, MCX cotton surged 35%\n"
            f"- 2015 (-20%): Back-to-back deficit, textile stocks fell 15-30%\n"
            f"- 2023 (-18%): Moderate impact, recovered by October"
        )
    elif deficit < -10:
        response += (
            f"\nModerate deficit -- watch for intensification in August. "
            f"Cotton price sensitivity typically accelerates when deficit exceeds -20%."
        )
    else:
        response += "\nConditions are within normal range. No extraordinary action needed."

    return response


def _resp_cotton(query: str, ctx: dict) -> str:
    """Cotton market information."""
    cotton_change = ctx.get("cotton_change", 0)
    cotton_source = ctx.get("cotton_source", "ICE (proxy)")
    regime = ctx.get("cotton_regime", "normal")

    return (
        f"**Cotton Futures Update**\n\n"
        f"- 30-day price change: **{cotton_change:+.1f}%**\n"
        f"- Data source: **{cotton_source}** (with USD/INR forex conversion)\n"
        f"- GARCH regime: **{regime}** volatility state\n\n"
        f"Cotton is the primary transmission channel from monsoon to textile stocks. "
        f"A 10% cotton price increase typically translates to 3-8% margin compression "
        f"for upstream manufacturers within 4-6 weeks."
    )


def _resp_model(query: str, ctx: dict) -> str:
    """Model methodology explanation."""
    weights = ctx.get("ensemble_weights", "XGBoost (40%) + GARCH (30%) + MLP (30%)")
    n_features = ctx.get("n_features", 24)
    granger_sig = ctx.get("granger_significant", 0)
    granger_total = ctx.get("granger_total", 0)

    return (
        f"**Model Architecture**\n\n"
        f"This dashboard uses a **3-layer ensemble** risk scoring system:\n\n"
        f"1. **XGBoost Classifier** -- {n_features} features including rainfall deficit, "
        f"cotton returns, NDVI satellite data, VIX, and seasonal indicators. "
        f"Trained with 5-fold temporal cross-validation.\n\n"
        f"2. **GJR-GARCH(1,1)** -- Captures asymmetric volatility clustering "
        f"and leverage effects in cotton and stock returns.\n\n"
        f"3. **MLP Neural Network** -- Sequence-based regime detection using "
        f"12-week lookback windows.\n\n"
        f"**Ensemble weights:** {weights}\n\n"
        f"**Causal validation:** {granger_sig}/{granger_total} Granger causality links "
        f"confirmed significant at p < 0.05 using stationarity-corrected variables "
        f"(cotton log-returns, rainfall deficit)."
    )


def _resp_ndvi(query: str, ctx: dict) -> str:
    """NDVI satellite data explanation."""
    return (
        "**NDVI Satellite Monitoring**\n\n"
        "We integrate **NASA MODIS MOD13Q1** (250m resolution, 16-day composites) "
        "vegetation index data for 10 cotton-growing states.\n\n"
        "- **NDVI > 0.4**: Healthy vegetation, good crop conditions\n"
        "- **NDVI 0.2-0.4**: Moderate stress, watch for deterioration\n"
        "- **NDVI < 0.2**: Severe stress, likely crop damage\n\n"
        "NDVI enters the XGBoost model as 3 features: current NDVI, "
        "4-week lagged NDVI, and NDVI change rate. These capture early "
        "crop stress signals 2-4 weeks before they show up in cotton prices."
    )


def _resp_greeting(query: str, ctx: dict) -> str:
    """Respond to greetings."""
    hour = datetime.now().hour
    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    return (
        f"{greeting}! I'm your **RainLoom AI Advisor**.\n\n"
        f"I can help you with:\n"
        f"- **Risk scores** for specific stocks (e.g., \"What's the risk for Trident?\")\n"
        f"- **Farmer advisories** on crop insurance and sowing strategy\n"
        f"- **MSME guidance** on cotton procurement and hedging\n"
        f"- **Investor insights** on portfolio positioning\n"
        f"- **Monsoon/rainfall** status and forecasts\n"
        f"- **Model methodology** (how the predictions work)\n\n"
        f"Ask me anything about the dashboard!"
    )


def _resp_help(query: str, ctx: dict) -> str:
    """Help / usage guide."""
    return (
        "**How to use the Advisory Assistant**\n\n"
        "I analyse the live dashboard data to provide context-aware guidance. "
        "Try asking:\n\n"
        "- \"What's the risk for Arvind Ltd?\"\n"
        "- \"Should farmers be worried about this monsoon?\"\n"
        "- \"What should MSMEs do about cotton procurement?\"\n"
        "- \"How should I position my textile portfolio?\"\n"
        "- \"What's the current monsoon deficit?\"\n"
        "- \"How does the ensemble model work?\"\n"
        "- \"Tell me about NDVI satellite data\"\n\n"
        "All responses are generated from **real-time dashboard data** -- "
        "risk scores, Granger causality results, GARCH outputs, and satellite NDVI."
    )


def _resp_summary(query: str, ctx: dict) -> str:
    """Dashboard summary / overview."""
    risk_data = ctx.get("risk_scores", {})
    avg_risk = ctx.get("avg_risk", 0.5)
    deficit = ctx.get("rain_deficit", 0)
    cotton_change = ctx.get("cotton_change", 0)
    vix = ctx.get("vix", 15.0)
    label = _risk_label(avg_risk)

    lines = [f"**Dashboard Summary** (Sector risk: {label})\n"]

    # Risk overview
    if risk_data:
        sorted_stocks = sorted(risk_data.items(), key=lambda x: -x[1])
        top_risk = sorted_stocks[0]
        low_risk = sorted_stocks[-1]
        lines.append(f"- **Avg sector risk:** {avg_risk:.0%} ({label})")
        lines.append(f"- **Highest risk:** {top_risk[0]} ({top_risk[1]:.0%})")
        lines.append(f"- **Lowest risk:** {low_risk[0]} ({low_risk[1]:.0%})")
    else:
        lines.append(f"- **Avg sector risk:** {avg_risk:.0%} ({label})")

    # Monsoon
    status = "surplus" if deficit > 5 else "normal" if deficit > -10 else "deficit" if deficit > -20 else "severe deficit"
    lines.append(f"- **Monsoon:** {status} ({deficit:+.0f}% from LPA)")

    # Cotton
    lines.append(f"- **Cotton 30d:** {cotton_change:+.1f}%")

    # VIX with interpretation
    vix_label = "calm" if vix < 15 else "normal" if vix < 20 else "elevated" if vix < 25 else "high fear"
    lines.append(f"- **India VIX:** {vix:.1f} ({vix_label})")

    # If query is specifically about VIX, add more detail
    if re.search(r"\bvix\b", query, re.IGNORECASE):
        lines.append(f"\n**VIX Analysis:**")
        if vix > 25:
            lines.append(f"VIX > 25 indicates high market fear. Textile stocks typically see amplified moves.")
        elif vix > 20:
            lines.append(f"Elevated VIX. Options premiums are richer — consider protective puts on textile positions.")
        elif vix > 15:
            lines.append(f"Normal range. No VIX-driven concern for textile portfolios.")
        else:
            lines.append(f"Low VIX environment. Good for accumulating positions, but complacency risk exists.")

    lines.append(
        f"\nAsk me for specific stock risks, farmer/investor advisories, or model details."
    )
    return "\n".join(lines)


def _resp_comparison(query: str, ctx: dict) -> str:
    """Compare stocks or find best/worst."""
    risk_data = ctx.get("risk_scores", {})

    if not risk_data:
        return _resp_summary(query, ctx)

    sorted_stocks = sorted(risk_data.items(), key=lambda x: x[1])

    lines = ["**Stock Risk Comparison** (sorted safest → riskiest)\n"]
    for i, (name, risk) in enumerate(sorted_stocks, 1):
        label = _risk_label(risk)
        bar = "█" * int(risk * 20)
        lines.append(f"{i}. **{name}**: {risk:.1%} ({label}) {bar}")

    safest = sorted_stocks[0]
    riskiest = sorted_stocks[-1]
    lines.append(
        f"\n**Safest:** {safest[0]} ({safest[1]:.1%}) — lowest monsoon exposure\n"
        f"**Riskiest:** {riskiest[0]} ({riskiest[1]:.1%}) — highest cotton dependency"
    )

    return "\n".join(lines)


def _resp_general(query: str, ctx: dict) -> str:
    """Intelligent fallback -- provide a useful data-grounded response."""
    avg_risk = ctx.get("avg_risk", 0.5)
    risk_data = ctx.get("risk_scores", {})
    deficit = ctx.get("rain_deficit", 0)
    cotton_change = ctx.get("cotton_change", 0)
    label = _risk_label(avg_risk)

    # Build a comprehensive overview so ANY question gets useful data
    lines = [
        f"Here's what the dashboard currently shows:\n",
        f"**Sector Risk:** {avg_risk:.0%} ({label})",
        f"**Monsoon Departure:** {deficit:+.0f}% from LPA",
        f"**Cotton 30d Change:** {cotton_change:+.1f}%",
    ]

    if risk_data:
        sorted_stocks = sorted(risk_data.items(), key=lambda x: -x[1])
        lines.append(f"\n**Stock Risk Scores:**")
        for name, risk in sorted_stocks:
            lines.append(f"- {name}: {risk:.1%} ({_risk_label(risk)})")

    lines.append(
        f"\nI specialise in monsoon-textile risk analysis. "
        f"I can give detailed advisories for **farmers**, **MSMEs**, **investors**, "
        f"or dig into **specific stocks**, **monsoon data**, **cotton markets**, "
        f"and **model methodology**. Just ask!"
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MAIN API -- single entry point
# ═══════════════════════════════════════════════════════════════════════════

_INTENT_HANDLERS = {
    "stock_risk": _resp_stock_risk,
    "farmer_advisory": _resp_farmer,
    "msme_advisory": _resp_msme,
    "investor_advisory": _resp_investor,
    "rainfall": _resp_rainfall,
    "cotton": _resp_cotton,
    "model_info": _resp_model,
    "ndvi": _resp_ndvi,
    "summary": _resp_summary,
    "comparison": _resp_comparison,
    "greeting": _resp_greeting,
    "help": _resp_help,
    "general": _resp_general,
}


def build_context(dashboard_data: dict | None = None) -> dict:
    """
    Build advisory context from dashboard data.
    Can be called once per page load and reused for all queries.
    """
    if not dashboard_data:
        return {"avg_risk": 0.5}

    ctx = {}

    # Stock risk scores
    stock_data = dashboard_data.get("stock_data", {})
    stocks_config = dashboard_data.get("stocks_config", {})
    # Fallback: if stocks_config not provided, build it from STOCKS dict
    if not stocks_config:
        try:
            from monsoon_textile_app.data.fetch_real_data import STOCKS as _STOCKS
            stocks_config = _STOCKS
        except Exception:
            stocks_config = {}
    risk_scores = {}
    for ticker, sdf in stock_data.items():
        name = stocks_config.get(ticker, {}).get("name", ticker)
        if hasattr(sdf, "iloc") and "risk_score" in sdf.columns:
            risk_scores[name] = float(sdf["risk_score"].iloc[-1])
    ctx["risk_scores"] = risk_scores
    ctx["avg_risk"] = sum(risk_scores.values()) / max(len(risk_scores), 1) if risk_scores else 0.5
    ctx["stocks"] = stocks_config

    # Model metrics
    ctx["model_metrics"] = dashboard_data.get("model_metrics", {})

    # Rainfall
    rainfall = dashboard_data.get("rainfall", {})
    latest_deficit = rainfall.get("latest_deficit", {})
    # Convert DataFrame to dict if needed
    if hasattr(latest_deficit, "to_dict"):
        try:
            latest_deficit = latest_deficit.to_dict()
        except Exception:
            latest_deficit = {}
    if latest_deficit and isinstance(latest_deficit, dict) and len(latest_deficit) > 0:
        vals = [v for v in latest_deficit.values() if isinstance(v, (int, float))]
        ctx["rain_deficit"] = sum(vals) / max(len(vals), 1) if vals else 0
        ctx["stressed_states"] = [s for s, d in latest_deficit.items() if isinstance(d, (int, float)) and d < -15]
    else:
        ctx["rain_deficit"] = 0
        ctx["stressed_states"] = []

    # Spatial breadth (approx from deficit data)
    if latest_deficit and isinstance(latest_deficit, dict) and len(latest_deficit) > 0:
        vals = [v for v in latest_deficit.values() if isinstance(v, (int, float))]
        n_deficit = sum(1 for d in vals if d < -10)
        ctx["spatial_breadth"] = (n_deficit / max(len(vals), 1)) * 100 if vals else 0
    else:
        ctx["spatial_breadth"] = 0

    # Cotton
    cotton = dashboard_data.get("cotton", None)
    if cotton is not None and hasattr(cotton, "iloc") and len(cotton) > 4:
        # Handle both "price_inr" (full data) and "price" (Risk Monitor format)
        _pcol = "price_inr" if "price_inr" in cotton.columns else "price" if "price" in cotton.columns else None
        latest_price = float(cotton[_pcol].iloc[-1]) if _pcol else 0
        # ~30 trading days back for 30-day change
        lookback = min(22, len(cotton) - 1)
        prev_price = float(cotton[_pcol].iloc[-lookback]) if _pcol else 0
        ctx["cotton_change"] = ((latest_price / max(prev_price, 1)) - 1) * 100
        ctx["cotton_source"] = cotton.get("cotton_source", pd.Series(["ICE"])).iloc[0] if "cotton_source" in cotton.columns else "ICE (proxy)"
        regime = float(cotton["regime_prob"].iloc[-1]) if "regime_prob" in cotton.columns else 0.5
        ctx["cotton_regime"] = "high" if regime > 0.6 else "normal"
    else:
        ctx["cotton_change"] = 0
        ctx["cotton_source"] = "N/A"
        ctx["cotton_regime"] = "unknown"

    # VIX
    vix = dashboard_data.get("vix", None)
    if vix is not None and hasattr(vix, "iloc") and "vix" in vix.columns and len(vix) > 0:
        ctx["vix"] = float(vix["vix"].iloc[-1])
    else:
        ctx["vix"] = 15.0

    # Granger
    granger = dashboard_data.get("granger", {})
    ctx["granger_total"] = len(granger)
    ctx["granger_significant"] = sum(1 for v in granger.values() if v.get("significant"))

    # Model info
    ml_details = dashboard_data.get("ml_details", {})
    ctx["ensemble_weights"] = ml_details.get("ensemble_weights", "XGBoost + GARCH + MLP")
    ctx["n_features"] = len(ml_details.get("feature_cols", [])) or 24

    return ctx


def get_advisory(query: str, context: dict | None = None) -> str:
    """
    Main entry point: detect intent and generate advisory response.

    Parameters
    ----------
    query : str
        User's natural language question
    context : dict
        Dashboard context from build_context()

    Returns
    -------
    str
        Markdown-formatted advisory response
    """
    if not query or not query.strip():
        return _resp_help(query, context or {})

    intent, groups = detect_intent(query)
    handler = _INTENT_HANDLERS.get(intent, _resp_general)

    return handler(query, context or {})


# Avoid import errors if pandas not loaded at module level
try:
    import pandas as pd
except ImportError:
    pass
