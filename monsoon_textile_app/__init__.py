"""
Monsoon Failures & Textile Stock Volatility Prediction System
=============================================================

Causal ML pipeline predicting NSE textile-sector stock volatility
from IMD monsoon rainfall deficits.

Architecture:
    Layer 1 — Granger Causality & VAR (causal proof)
    Layer 2 — Markov-Switching GARCH (regime detection)
    Layer 3 — XGBoost Ensemble (prediction engine)
    Layer 4 — LSTM Sequence Model (temporal patterns)
"""

__version__ = "2.0.0"
