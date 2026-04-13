"""
SLM Advisory Engine -- Groq-powered conversational assistant
=============================================================
Uses Groq's free API tier (Llama 3.1 8B) for natural language
responses grounded in live dashboard data. Falls back to the
template-based advisory_engine when no API key is available.

Setup:  set env var  GROQ_API_KEY=<your-free-key>
        (get one free at https://console.groq.com)
"""

from __future__ import annotations
import os
import json

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
_MODEL = "llama-3.1-8b-instant"  # Free tier, fast
_MAX_TOKENS = 512

_SYSTEM_PROMPT = """\
You are the RainLoom AI Advisor, an AI assistant embedded in a live \
dashboard that tracks how Indian monsoon failures cascade into textile-stock \
volatility.

You answer questions about:
- Stock-specific risk scores (ensemble of XGBoost + GJR-GARCH + MLP)
- Farmer advisories (crop insurance, sowing strategy, PMFBY)
- MSME / textile-mill procurement guidance (cotton hedging, inventory)
- Investor positioning (portfolio strategy, sector outlook)
- Monsoon / rainfall status and historical drought parallels
- Cotton futures and GARCH regime analysis
- Model methodology (24 features, Granger causality, quantile regression)
- NDVI satellite vegetation monitoring

Rules:
1. Ground every answer in the LIVE DATA provided below. Cite numbers.
2. Keep responses concise (3-8 sentences). Use markdown bold for emphasis.
3. If the data shows default/placeholder values (avg_risk=0.5, rain=0), say \
   "data is still loading" and give general guidance instead.
4. Never fabricate risk scores or statistics not present in the data.
5. For farmer queries, always mention PMFBY crop insurance.
6. For investor queries, reference the hedging backtest on Page 6.
7. Be professional but accessible. Avoid jargon unless explaining methodology.

--- LIVE DASHBOARD DATA ---
{context_block}
--- END DATA ---
"""


def _format_context(ctx: dict) -> str:
    """Convert advisory context dict into a readable text block for the LLM."""
    lines = []

    # Risk scores
    risk_scores = ctx.get("risk_scores", {})
    if risk_scores:
        lines.append("STOCK RISK SCORES (0-1 scale, higher = riskier):")
        for name, score in sorted(risk_scores.items(), key=lambda x: -x[1]):
            label = "LOW" if score < 0.3 else "MODERATE" if score < 0.6 else "HIGH" if score < 0.8 else "EXTREME"
            lines.append(f"  {name}: {score:.1%} ({label})")
        avg = ctx.get("avg_risk", 0.5)
        lines.append(f"  Average sector risk: {avg:.1%}")
    else:
        lines.append("STOCK RISK SCORES: not yet loaded")

    # Monsoon
    deficit = ctx.get("rain_deficit", 0)
    breadth = ctx.get("spatial_breadth", 0)
    stressed = ctx.get("stressed_states", [])
    lines.append(f"\nMONSOON: JJAS rainfall departure = {deficit:+.1f}% from LPA")
    lines.append(f"  Spatial deficit breadth: {breadth:.0f}% of cotton-belt districts")
    if stressed:
        lines.append(f"  Stressed states: {', '.join(stressed[:5])}")

    # Cotton
    cotton_chg = ctx.get("cotton_change", 0)
    regime = ctx.get("cotton_regime", "unknown")
    lines.append(f"\nCOTTON: 30-day price change = {cotton_chg:+.1f}%, GARCH regime = {regime}")

    # VIX
    vix = ctx.get("vix", 15.0)
    lines.append(f"INDIA VIX: {vix:.1f}")

    # Model info
    weights = ctx.get("ensemble_weights", "XGBoost 40% + GARCH 30% + MLP 30%")
    n_feat = ctx.get("n_features", 24)
    g_sig = ctx.get("granger_significant", 0)
    g_tot = ctx.get("granger_total", 0)
    lines.append(f"\nMODEL: Ensemble weights = {weights}")
    lines.append(f"  Features: {n_feat}, Granger links: {g_sig}/{g_tot} significant")

    return "\n".join(lines)


def is_available() -> bool:
    """Check if SLM engine can be used (API key present)."""
    return bool(_GROQ_API_KEY)


def get_slm_response(
    query: str,
    context: dict,
    chat_history: list[dict] | None = None,
) -> str:
    """
    Get a response from the Groq SLM.

    Parameters
    ----------
    query : str
        User's question.
    context : dict
        Advisory context from build_context().
    chat_history : list of dicts, optional
        Previous messages [{"role": "user"/"assistant", "content": "..."}].

    Returns
    -------
    str
        Model response text, or empty string on failure.
    """
    if not _GROQ_API_KEY:
        return ""

    try:
        from groq import Groq

        client = Groq(api_key=_GROQ_API_KEY)

        # Build messages
        context_block = _format_context(context)
        system_msg = _SYSTEM_PROMPT.format(context_block=context_block)

        messages = [{"role": "system", "content": system_msg}]

        # Add recent chat history (last 6 messages for context)
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Add current query
        messages.append({"role": "user", "content": query})

        response = client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            max_tokens=_MAX_TOKENS,
            temperature=0.4,
            top_p=0.9,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[SLM] Groq API error: {e}")
        return ""
