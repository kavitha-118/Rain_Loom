"""
Ensemble Risk Scoring Engine for Monsoon-Textile Volatility System.

Combines MS-GARCH regime probabilities, XGBoost classification scores,
and LSTM sequence predictions into a calibrated composite risk score
for cotton price volatility forecasting.

Author: Monsoon-Textile Volatility Research Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


_DEFAULT_WEIGHTS: Dict[str, float] = {
    "ms_garch": 0.30,
    "xgboost": 0.40,
    "lstm": 0.30,
}

_RISK_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "LOW": (0.0, 0.3),
    "MODERATE": (0.3, 0.6),
    "HIGH": (0.6, 0.8),
    "EXTREME": (0.8, 1.0),
}


class EnsembleRiskScorer:
    """Weighted ensemble risk scoring across three model families.

    The scorer aggregates probabilistic outputs from a Markov-Switching GARCH
    model, an XGBoost classifier, and an LSTM sequence model into a single
    calibrated risk score in [0, 1].  Weights can be set manually or optimised
    via a stacking (logistic-regression) procedure on held-out validation data.

    Parameters
    ----------
    weights : dict, optional
        Mapping of model name to weight.  Keys must be a subset of
        ``{'ms_garch', 'xgboost', 'lstm'}``.  Defaults to
        ``{ms_garch: 0.30, xgboost: 0.40, lstm: 0.30}``.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights: Dict[str, float] = dict(weights or _DEFAULT_WEIGHTS)
        self._validate_weights()
        self._stacking_model: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None
        logger.info(
            "EnsembleRiskScorer initialised | weights={weights}",
            weights=self.weights,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_weights(self) -> None:
        """Ensure weights are non-negative and sum to ~1."""
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            logger.warning(
                "Weights sum to {total:.4f}; normalising to 1.0", total=total
            )
            for k in self.weights:
                self.weights[k] /= total

        for k, v in self.weights.items():
            if v < 0:
                raise ValueError(f"Weight for '{k}' must be non-negative, got {v}")

    @staticmethod
    def _clip_probability(value: float, name: str = "prob") -> float:
        """Clip a scalar probability to [0, 1] with a warning on out-of-range."""
        if value < 0.0 or value > 1.0:
            logger.warning(
                "{name} out of range ({value:.4f}); clipping to [0, 1]",
                name=name,
                value=value,
            )
        return float(np.clip(value, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def compute_risk_score(
        self,
        ms_garch_prob: float,
        xgb_prob: float,
        lstm_prob: float,
    ) -> float:
        """Compute a weighted ensemble risk score from three model outputs.

        Parameters
        ----------
        ms_garch_prob : float
            Probability of high-volatility regime from MS-GARCH (0-1).
        xgb_prob : float
            Predicted probability from the XGBoost classifier (0-1).
        lstm_prob : float
            Predicted probability from the LSTM model (0-1).

        Returns
        -------
        float
            Composite risk score in [0, 1].
        """
        ms_garch_prob = self._clip_probability(ms_garch_prob, "ms_garch_prob")
        xgb_prob = self._clip_probability(xgb_prob, "xgb_prob")
        lstm_prob = self._clip_probability(lstm_prob, "lstm_prob")

        score: float = (
            self.weights.get("ms_garch", 0.0) * ms_garch_prob
            + self.weights.get("xgboost", 0.0) * xgb_prob
            + self.weights.get("lstm", 0.0) * lstm_prob
        )
        score = float(np.clip(score, 0.0, 1.0))
        logger.debug(
            "Risk score={score:.4f} | garch={g:.3f} xgb={x:.3f} lstm={l:.3f}",
            score=score,
            g=ms_garch_prob,
            x=xgb_prob,
            l=lstm_prob,
        )
        return score

    def classify_risk(self, score: float) -> str:
        """Map a risk score to a categorical risk level.

        Parameters
        ----------
        score : float
            Composite risk score in [0, 1].

        Returns
        -------
        str
            One of ``'LOW'``, ``'MODERATE'``, ``'HIGH'``, ``'EXTREME'``.
        """
        score = float(np.clip(score, 0.0, 1.0))
        for label, (lo, hi) in _RISK_THRESHOLDS.items():
            if lo <= score < hi:
                return label
        # Edge case: score == 1.0
        return "EXTREME"

    # ------------------------------------------------------------------
    # Weight optimisation via stacking
    # ------------------------------------------------------------------

    def optimize_weights(
        self,
        ms_garch_probs: np.ndarray,
        xgb_probs: np.ndarray,
        lstm_probs: np.ndarray,
        y_true: np.ndarray,
        *,
        regularization: float = 1.0,
    ) -> Dict[str, float]:
        """Optimise ensemble weights using a stacking logistic regression.

        A logistic regression is fitted on the three model probability streams
        against the binary ground-truth labels.  The fitted coefficients are
        normalised to produce updated ensemble weights.

        Parameters
        ----------
        ms_garch_probs : np.ndarray
            Array of MS-GARCH regime probabilities (validation set).
        xgb_probs : np.ndarray
            Array of XGBoost predicted probabilities (validation set).
        lstm_probs : np.ndarray
            Array of LSTM predicted probabilities (validation set).
        y_true : np.ndarray
            Binary ground-truth labels (1 = spike, 0 = no spike).
        regularization : float
            Inverse regularisation strength for logistic regression.

        Returns
        -------
        dict
            Updated weights mapping.
        """
        X = np.column_stack([
            np.asarray(ms_garch_probs, dtype=np.float64),
            np.asarray(xgb_probs, dtype=np.float64),
            np.asarray(lstm_probs, dtype=np.float64),
        ])
        y = np.asarray(y_true, dtype=np.int32).ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Feature rows ({X.shape[0]}) != label length ({y.shape[0]})"
            )

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._stacking_model = LogisticRegression(
            C=regularization,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        self._stacking_model.fit(X_scaled, y)

        raw_coefs = np.abs(self._stacking_model.coef_.ravel())
        normalised = raw_coefs / raw_coefs.sum()

        self.weights = {
            "ms_garch": float(normalised[0]),
            "xgboost": float(normalised[1]),
            "lstm": float(normalised[2]),
        }

        logger.info(
            "Optimised weights via stacking | ms_garch={g:.3f} xgboost={x:.3f} lstm={l:.3f}",
            g=self.weights["ms_garch"],
            x=self.weights["xgboost"],
            l=self.weights["lstm"],
        )
        return dict(self.weights)

    # ------------------------------------------------------------------
    # Batch / time-series scoring
    # ------------------------------------------------------------------

    def compute_batch_scores(
        self,
        ms_garch_series: pd.Series,
        xgb_series: pd.Series,
        lstm_series: pd.Series,
    ) -> pd.Series:
        """Compute risk scores for aligned time-series of model probabilities.

        All three series must share a compatible ``DatetimeIndex``.

        Parameters
        ----------
        ms_garch_series : pd.Series
            MS-GARCH regime probabilities indexed by date.
        xgb_series : pd.Series
            XGBoost predicted probabilities indexed by date.
        lstm_series : pd.Series
            LSTM predicted probabilities indexed by date.

        Returns
        -------
        pd.Series
            Risk scores with ``DatetimeIndex``.
        """
        # Align on common index
        combined = pd.DataFrame(
            {
                "ms_garch": ms_garch_series,
                "xgboost": xgb_series,
                "lstm": lstm_series,
            }
        ).dropna()

        if combined.empty:
            logger.warning("No overlapping dates found across input series")
            return pd.Series(dtype=np.float64)

        scores: np.ndarray = (
            self.weights.get("ms_garch", 0.0) * combined["ms_garch"].values
            + self.weights.get("xgboost", 0.0) * combined["xgboost"].values
            + self.weights.get("lstm", 0.0) * combined["lstm"].values
        )
        scores = np.clip(scores, 0.0, 1.0)

        result = pd.Series(scores, index=combined.index, name="risk_score")
        logger.info(
            "Batch scoring complete | n={n} | mean={mean:.3f} | max={mx:.3f}",
            n=len(result),
            mean=float(result.mean()),
            mx=float(result.max()),
        )
        return result

    # ------------------------------------------------------------------
    # Risk summary / analytics
    # ------------------------------------------------------------------

    def risk_summary(self, scores_series: pd.Series) -> Dict[str, Any]:
        """Produce a summary report from a time-series of risk scores.

        Parameters
        ----------
        scores_series : pd.Series
            Risk scores indexed by date.

        Returns
        -------
        dict
            Summary containing:
            - ``current_risk``: latest risk score and classification.
            - ``trend_4w``: change over the most recent 4 weeks (28 days).
            - ``max_risk``: maximum risk score observed in the period.
            - ``days_by_level``: count of days spent in each risk level.
        """
        if scores_series.empty:
            logger.warning("Empty scores series passed to risk_summary")
            return {
                "current_risk": {"score": None, "level": None, "date": None},
                "trend_4w": None,
                "max_risk": None,
                "days_by_level": {level: 0 for level in _RISK_THRESHOLDS},
            }

        scores_sorted = scores_series.sort_index()
        current_score = float(scores_sorted.iloc[-1])
        current_date = scores_sorted.index[-1]

        # 4-week trend
        four_weeks_ago = current_date - pd.Timedelta(weeks=4)
        past_scores = scores_sorted.loc[scores_sorted.index <= four_weeks_ago]
        if not past_scores.empty:
            trend_4w = current_score - float(past_scores.iloc[-1])
        else:
            trend_4w = None

        # Days in each risk level
        levels = scores_sorted.apply(self.classify_risk)
        days_by_level: Dict[str, int] = {level: 0 for level in _RISK_THRESHOLDS}
        level_counts = levels.value_counts()
        for level_name in days_by_level:
            days_by_level[level_name] = int(level_counts.get(level_name, 0))

        summary: Dict[str, Any] = {
            "current_risk": {
                "score": round(current_score, 4),
                "level": self.classify_risk(current_score),
                "date": str(current_date.date()) if hasattr(current_date, "date") else str(current_date),
            },
            "trend_4w": round(trend_4w, 4) if trend_4w is not None else None,
            "max_risk": round(float(scores_sorted.max()), 4),
            "days_by_level": days_by_level,
        }

        logger.info(
            "Risk summary | current={level} ({score:.3f}) | 4w-trend={trend} | max={mx:.3f}",
            level=summary["current_risk"]["level"],
            score=current_score,
            trend=summary["trend_4w"],
            mx=summary["max_risk"],
        )
        return summary
