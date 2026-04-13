"""
Backtesting & Validation Framework for Monsoon-Textile Volatility System.

Provides walk-forward backtesting against historical drought years, model
comparison via Diebold-Mariano tests, and economic-value analysis of the
ensemble early-warning system for cotton price volatility.

Author: Monsoon-Textile Volatility Research Team
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


# ======================================================================
# Drought-Year Backtester
# ======================================================================


class DroughtYearBacktester:
    """Backtest the ensemble risk model against known drought and normal years.

    The backtester evaluates whether the risk-scoring system correctly
    issues early warnings ahead of realised cotton-price volatility spikes
    during drought years, and avoids false positives in normal monsoon years.

    Parameters
    ----------
    drought_years : list[int]
        Calendar years classified as drought/deficit monsoon years.
    normal_years : list[int]
        Calendar years classified as normal/surplus monsoon years.
    signal_threshold : float
        Risk score above which an early-warning signal is deemed issued.
    spike_threshold_pct : float
        Percentage increase in realised volatility that constitutes a spike.
    """

    def __init__(
        self,
        drought_years: Optional[List[int]] = None,
        normal_years: Optional[List[int]] = None,
        signal_threshold: float = 0.6,
        spike_threshold_pct: float = 20.0,
    ) -> None:
        self.drought_years: List[int] = drought_years or [2009, 2014, 2015, 2023]
        self.normal_years: List[int] = normal_years or [2010, 2013, 2016, 2019]
        self.signal_threshold = signal_threshold
        self.spike_threshold_pct = spike_threshold_pct
        logger.info(
            "DroughtYearBacktester initialised | drought={d} normal={n}",
            d=self.drought_years,
            n=self.normal_years,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_year(series: pd.Series, year: int) -> pd.Series:
        """Extract observations for a given calendar year."""
        return series.loc[series.index.year == year]

    def _first_signal_date(self, risk_scores_year: pd.Series) -> Optional[pd.Timestamp]:
        """Return the first date the risk score exceeds the signal threshold."""
        above = risk_scores_year.loc[risk_scores_year >= self.signal_threshold]
        if above.empty:
            return None
        return above.index[0]

    def _first_spike_date(
        self, actual_vol_year: pd.Series
    ) -> Optional[pd.Timestamp]:
        """Return the first date realised volatility spikes above the threshold.

        A spike is defined as the first date where the rolling 4-week change
        in annualised volatility exceeds ``spike_threshold_pct`` percent.
        """
        if actual_vol_year.empty:
            return None
        pct_change = actual_vol_year.pct_change(periods=20) * 100  # ~4 trading weeks
        spike_dates = pct_change.loc[pct_change >= self.spike_threshold_pct]
        if spike_dates.empty:
            return None
        return spike_dates.index[0]

    # ------------------------------------------------------------------
    # Per-year backtests
    # ------------------------------------------------------------------

    def backtest_drought_year(
        self,
        year: int,
        risk_scores: pd.Series,
        actual_vol: pd.Series,
        deficit_series: pd.Series,
    ) -> Dict[str, Any]:
        """Backtest a single drought year.

        Parameters
        ----------
        year : int
            The drought year to test.
        risk_scores : pd.Series
            Full ensemble risk score series (DatetimeIndex).
        actual_vol : pd.Series
            Realised cotton-price volatility series (DatetimeIndex).
        deficit_series : pd.Series
            Monsoon rainfall deficit percentages (DatetimeIndex).

        Returns
        -------
        dict
            Results including signal date, spike date, lead time (weeks),
            correctness flag, and the risk score at signal date.
        """
        rs_year = self._filter_year(risk_scores, year)
        vol_year = self._filter_year(actual_vol, year)
        def_year = self._filter_year(deficit_series, year)

        signal_date = self._first_signal_date(rs_year)
        spike_date = self._first_spike_date(vol_year)
        avg_deficit = float(def_year.mean()) if not def_year.empty else None

        lead_time_weeks: Optional[float] = None
        correct = False
        risk_at_signal: Optional[float] = None

        if signal_date is not None and spike_date is not None:
            lead_days = (spike_date - signal_date).days
            lead_time_weeks = lead_days / 7.0
            correct = lead_days > 0  # signal must precede spike
            risk_at_signal = float(rs_year.loc[signal_date])
        elif signal_date is not None:
            risk_at_signal = float(rs_year.loc[signal_date])

        result: Dict[str, Any] = {
            "year": year,
            "type": "drought",
            "deficit_pct": round(avg_deficit, 2) if avg_deficit is not None else None,
            "signal_date": str(signal_date.date()) if signal_date else None,
            "actual_spike_date": str(spike_date.date()) if spike_date else None,
            "lead_time_weeks": round(lead_time_weeks, 1) if lead_time_weeks is not None else None,
            "correct": correct,
            "risk_score_at_signal": round(risk_at_signal, 4) if risk_at_signal is not None else None,
        }
        logger.info("Drought backtest {year} | correct={c} | lead={lt}w", year=year, c=correct, lt=lead_time_weeks)
        return result

    def backtest_normal_year(
        self,
        year: int,
        risk_scores: pd.Series,
    ) -> Dict[str, Any]:
        """Analyse false-positive signals in a normal monsoon year.

        Parameters
        ----------
        year : int
            The normal year to test.
        risk_scores : pd.Series
            Full ensemble risk score series (DatetimeIndex).

        Returns
        -------
        dict
            False-positive analysis including count of false signals,
            maximum risk score, and average risk.
        """
        rs_year = self._filter_year(risk_scores, year)
        false_signals = rs_year.loc[rs_year >= self.signal_threshold]
        max_risk = float(rs_year.max()) if not rs_year.empty else 0.0
        avg_risk = float(rs_year.mean()) if not rs_year.empty else 0.0

        result: Dict[str, Any] = {
            "year": year,
            "type": "normal",
            "deficit_pct": None,
            "false_signal_days": int(len(false_signals)),
            "false_signal_pct": round(len(false_signals) / max(len(rs_year), 1) * 100, 2),
            "max_risk_score": round(max_risk, 4),
            "avg_risk_score": round(avg_risk, 4),
            "signal_date": str(false_signals.index[0].date()) if not false_signals.empty else None,
            "correct": len(false_signals) == 0,
        }
        logger.info(
            "Normal backtest {year} | false_signals={fs}d ({pct}%)",
            year=year,
            fs=result["false_signal_days"],
            pct=result["false_signal_pct"],
        )
        return result

    def backtest_all(
        self,
        risk_scores: pd.Series,
        actual_vol: pd.Series,
        deficit_series: pd.Series,
    ) -> pd.DataFrame:
        """Run backtests across all drought and normal years.

        Parameters
        ----------
        risk_scores : pd.Series
            Full ensemble risk score series.
        actual_vol : pd.Series
            Realised cotton-price volatility series.
        deficit_series : pd.Series
            Monsoon rainfall deficit percentages.

        Returns
        -------
        pd.DataFrame
            Consolidated results with one row per year.
        """
        results: List[Dict[str, Any]] = []

        for year in self.drought_years:
            results.append(
                self.backtest_drought_year(year, risk_scores, actual_vol, deficit_series)
            )

        for year in self.normal_years:
            results.append(self.backtest_normal_year(year, risk_scores))

        df = pd.DataFrame(results).sort_values("year").reset_index(drop=True)
        logger.info(
            "Backtest complete | {n} years | accuracy={acc:.1%}",
            n=len(df),
            acc=df["correct"].mean(),
        )
        return df

    # ------------------------------------------------------------------
    # Lead-time analysis
    # ------------------------------------------------------------------

    def early_warning_lead_time(
        self,
        risk_scores: pd.Series,
        actual_vol: pd.Series,
        threshold: float = 0.6,
    ) -> Optional[float]:
        """Compute weeks between first signal and first realised spike.

        Parameters
        ----------
        risk_scores : pd.Series
            Risk score series.
        actual_vol : pd.Series
            Realised volatility series.
        threshold : float
            Risk score threshold for signal issuance.

        Returns
        -------
        float or None
            Lead time in weeks, or ``None`` if signal/spike not found.
        """
        original_threshold = self.signal_threshold
        self.signal_threshold = threshold
        signal_date = self._first_signal_date(risk_scores)
        spike_date = self._first_spike_date(actual_vol)
        self.signal_threshold = original_threshold

        if signal_date is None or spike_date is None:
            logger.debug("Lead time not computable (signal={s}, spike={sp})", s=signal_date, sp=spike_date)
            return None

        lead_weeks = (spike_date - signal_date).days / 7.0
        logger.info("Early warning lead time: {lw:.1f} weeks", lw=lead_weeks)
        return round(lead_weeks, 1)

    # ------------------------------------------------------------------
    # Walk-forward backtest
    # ------------------------------------------------------------------

    def walk_forward_backtest(
        self,
        model_trainer: Callable[[pd.DataFrame, pd.DataFrame], Any],
        features_df: pd.DataFrame,
        start_year: int = 2009,
    ) -> pd.DataFrame:
        """Expanding-window walk-forward validation.

        For each year *T* from ``start_year`` onward, the model is trained on
        data from 2005 through *T-1* and tested on year *T*.

        Parameters
        ----------
        model_trainer : callable
            ``model_trainer(train_df, test_df)`` -> dict with at least
            ``{'predictions': array, 'actuals': array, 'metrics': dict}``.
        features_df : pd.DataFrame
            Full feature DataFrame with a ``DatetimeIndex``.
        start_year : int
            First out-of-sample year.

        Returns
        -------
        pd.DataFrame
            Per-year out-of-sample metrics.
        """
        results: List[Dict[str, Any]] = []
        all_years = sorted(features_df.index.year.unique())
        test_years = [y for y in all_years if y >= start_year]

        for test_year in test_years:
            train_mask = features_df.index.year < test_year
            test_mask = features_df.index.year == test_year

            train_df = features_df.loc[train_mask]
            test_df = features_df.loc[test_mask]

            if train_df.empty or test_df.empty:
                logger.warning("Skipping {year}: insufficient data", year=test_year)
                continue

            logger.info(
                "Walk-forward fold | train=2005-{te} ({n_tr} rows) | test={te} ({n_te} rows)",
                te=test_year - 1,
                n_tr=len(train_df),
                n_te=len(test_df),
            )

            fold_result = model_trainer(train_df, test_df)
            fold_result["test_year"] = test_year
            fold_result["train_size"] = len(train_df)
            fold_result["test_size"] = len(test_df)
            results.append(fold_result)

        return pd.DataFrame(results)


# ======================================================================
# Model Comparator
# ======================================================================


class ModelComparator:
    """Statistical comparison of forecast models.

    Implements the Diebold-Mariano test for predictive accuracy and
    economic value analysis of the risk-scoring strategy.
    """

    @staticmethod
    def diebold_mariano_test(
        errors_model1: np.ndarray,
        errors_model2: np.ndarray,
        *,
        loss: str = "MSE",
    ) -> Dict[str, float]:
        """Perform the Diebold-Mariano test for equal predictive accuracy.

        Parameters
        ----------
        errors_model1 : np.ndarray
            Forecast errors from the first model.
        errors_model2 : np.ndarray
            Forecast errors from the second model.
        loss : str
            Loss function: ``'MSE'`` (squared) or ``'MAE'`` (absolute).

        Returns
        -------
        dict
            ``{'dm_statistic': float, 'p_value': float}``.
        """
        e1 = np.asarray(errors_model1, dtype=np.float64)
        e2 = np.asarray(errors_model2, dtype=np.float64)

        if len(e1) != len(e2):
            raise ValueError("Error arrays must have equal length")

        if loss.upper() == "MSE":
            d = e1 ** 2 - e2 ** 2
        elif loss.upper() == "MAE":
            d = np.abs(e1) - np.abs(e2)
        else:
            raise ValueError(f"Unsupported loss '{loss}'; use 'MSE' or 'MAE'")

        n = len(d)
        d_bar = d.mean()
        d_var = np.var(d, ddof=1)

        if d_var < 1e-15:
            logger.warning("Near-zero variance in loss differential")
            return {"dm_statistic": 0.0, "p_value": 1.0}

        dm_stat = d_bar / np.sqrt(d_var / n)
        p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

        logger.info("DM test | stat={s:.3f} p={p:.4f}", s=dm_stat, p=p_value)
        return {"dm_statistic": round(float(dm_stat), 4), "p_value": round(float(p_value), 6)}

    def compare_vs_baselines(
        self,
        our_predictions: np.ndarray,
        y_true: np.ndarray,
        naive_predictions: np.ndarray,
        garch_only_predictions: np.ndarray,
        rf_predictions: np.ndarray,
    ) -> pd.DataFrame:
        """Compare the ensemble model against multiple baselines.

        Parameters
        ----------
        our_predictions : np.ndarray
            Ensemble model predictions.
        y_true : np.ndarray
            Ground-truth values.
        naive_predictions : np.ndarray
            Naive/random-walk baseline predictions.
        garch_only_predictions : np.ndarray
            Standalone GARCH model predictions.
        rf_predictions : np.ndarray
            Random-forest model predictions.

        Returns
        -------
        pd.DataFrame
            Comparison table with MSE, MAE, and DM test results.
        """
        y = np.asarray(y_true, dtype=np.float64)
        our_err = np.asarray(our_predictions, dtype=np.float64) - y

        baselines = {
            "naive": np.asarray(naive_predictions, dtype=np.float64) - y,
            "garch_only": np.asarray(garch_only_predictions, dtype=np.float64) - y,
            "random_forest": np.asarray(rf_predictions, dtype=np.float64) - y,
        }

        rows: List[Dict[str, Any]] = []

        # Ensemble model row
        rows.append({
            "model": "ensemble",
            "mse": round(float(np.mean(our_err ** 2)), 6),
            "mae": round(float(np.mean(np.abs(our_err))), 6),
            "dm_statistic_vs_ensemble": None,
            "dm_p_value_vs_ensemble": None,
            "significant_at_5pct": None,
        })

        for name, baseline_err in baselines.items():
            dm = self.diebold_mariano_test(our_err, baseline_err)
            rows.append({
                "model": name,
                "mse": round(float(np.mean(baseline_err ** 2)), 6),
                "mae": round(float(np.mean(np.abs(baseline_err))), 6),
                "dm_statistic_vs_ensemble": dm["dm_statistic"],
                "dm_p_value_vs_ensemble": dm["p_value"],
                "significant_at_5pct": dm["p_value"] < 0.05,
            })

        df = pd.DataFrame(rows)
        logger.info("Baseline comparison complete | {n} models", n=len(df))
        return df

    @staticmethod
    def economic_value_analysis(
        risk_scores: pd.Series,
        returns: pd.Series,
        threshold: float = 0.6,
        *,
        put_cost_pct: float = 2.0,
        annualisation_factor: float = 252.0,
    ) -> Dict[str, Any]:
        """Evaluate economic value of a put-buying hedging strategy.

        When the risk score exceeds ``threshold``, a protective put is
        purchased.  The strategy P&L and Sharpe ratio are compared to
        a simple buy-and-hold benchmark.

        Parameters
        ----------
        risk_scores : pd.Series
            Risk score series (DatetimeIndex).
        returns : pd.Series
            Daily returns of the underlying asset (DatetimeIndex).
        threshold : float
            Risk score above which hedging is activated.
        put_cost_pct : float
            Cost of the put as a percentage of the position.
        annualisation_factor : float
            Trading days per year for Sharpe annualisation.

        Returns
        -------
        dict
            Strategy P&L, Sharpe ratios, and hedging statistics.
        """
        aligned = pd.DataFrame({"risk": risk_scores, "return": returns}).dropna()

        if aligned.empty:
            return {"error": "No overlapping data between risk scores and returns"}

        hedge_active = aligned["risk"] >= threshold
        daily_returns = aligned["return"].values.copy()

        # Strategy: on hedge days, cap downside at -put_cost_pct / 252
        strategy_returns = daily_returns.copy()
        hedge_mask = hedge_active.values
        put_daily_cost = put_cost_pct / 100.0 / annualisation_factor

        for i in range(len(strategy_returns)):
            if hedge_mask[i]:
                if strategy_returns[i] < 0:
                    strategy_returns[i] = -put_daily_cost
                else:
                    strategy_returns[i] -= put_daily_cost

        # Cumulative P&L
        bh_cumulative = float(np.sum(daily_returns))
        strat_cumulative = float(np.sum(strategy_returns))

        # Sharpe ratios
        bh_sharpe = (
            float(np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(annualisation_factor))
            if np.std(daily_returns, ddof=1) > 1e-10
            else 0.0
        )
        strat_sharpe = (
            float(np.mean(strategy_returns) / np.std(strategy_returns, ddof=1) * np.sqrt(annualisation_factor))
            if np.std(strategy_returns, ddof=1) > 1e-10
            else 0.0
        )

        result: Dict[str, Any] = {
            "strategy_pnl": round(strat_cumulative, 6),
            "buy_hold_pnl": round(bh_cumulative, 6),
            "strategy_sharpe": round(strat_sharpe, 4),
            "buy_hold_sharpe": round(bh_sharpe, 4),
            "hedge_days": int(hedge_mask.sum()),
            "total_days": len(aligned),
            "hedge_pct": round(float(hedge_mask.sum()) / len(aligned) * 100, 2),
            "total_put_cost": round(float(hedge_mask.sum()) * put_daily_cost, 6),
        }
        logger.info(
            "Economic value | strat_sharpe={ss:.2f} vs bh_sharpe={bs:.2f} | hedge_days={hd}",
            ss=result["strategy_sharpe"],
            bs=result["buy_hold_sharpe"],
            hd=result["hedge_days"],
        )
        return result


# ======================================================================
# Validation Reporter
# ======================================================================


class ValidationReporter:
    """Format and aggregate backtesting and cross-validation results.

    Produces publication-ready tables for the Monsoon-Textile Volatility
    system's validation report.
    """

    @staticmethod
    def metrics_table(results: pd.DataFrame) -> pd.DataFrame:
        """Format backtest results into a standardised metrics table.

        Parameters
        ----------
        results : pd.DataFrame
            Output from ``DroughtYearBacktester.backtest_all``.

        Returns
        -------
        pd.DataFrame
            Formatted table: Year | Type | Deficit% | Signal Date |
            Spike Date | Lead Time | Correct? | Risk Score.
        """
        columns_map = {
            "year": "Year",
            "type": "Type",
            "deficit_pct": "Deficit %",
            "signal_date": "Signal Date",
            "actual_spike_date": "Spike Date",
            "lead_time_weeks": "Lead Time (wks)",
            "correct": "Correct?",
            "risk_score_at_signal": "Risk Score",
        }

        available_cols = [c for c in columns_map if c in results.columns]
        table = results[available_cols].rename(columns=columns_map)
        return table

    @staticmethod
    def performance_summary(cv_scores: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate metrics across cross-validation or walk-forward folds.

        Parameters
        ----------
        cv_scores : pd.DataFrame
            Per-fold results with a ``metrics`` column containing dicts,
            or numeric metric columns directly.

        Returns
        -------
        dict
            Aggregated mean, std, min, max for each metric.
        """
        if "metrics" in cv_scores.columns:
            metrics_df = pd.json_normalize(cv_scores["metrics"])
        else:
            numeric_cols = cv_scores.select_dtypes(include=[np.number]).columns
            metrics_df = cv_scores[numeric_cols]

        summary: Dict[str, Any] = {}
        for col in metrics_df.columns:
            values = metrics_df[col].dropna()
            if values.empty:
                continue
            summary[col] = {
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std()), 4),
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
                "n_folds": int(len(values)),
            }

        logger.info("Performance summary across {n} metrics", n=len(summary))
        return summary

    def generate_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive validation report.

        Parameters
        ----------
        all_results : dict
            Dictionary containing:
            - ``backtest_df``: DataFrame from ``backtest_all``.
            - ``cv_scores``: DataFrame from walk-forward or CV.
            - ``comparison``: DataFrame from ``compare_vs_baselines``.
            - ``economic``: dict from ``economic_value_analysis``.

        Returns
        -------
        dict
            Comprehensive report with formatted tables, summaries, and
            key performance indicators.
        """
        report: Dict[str, Any] = {"generated_at": str(pd.Timestamp.now())}

        # Backtest metrics
        if "backtest_df" in all_results:
            bt_df = all_results["backtest_df"]
            report["backtest_table"] = self.metrics_table(bt_df).to_dict(orient="records")
            drought_rows = bt_df.loc[bt_df["type"] == "drought"]
            normal_rows = bt_df.loc[bt_df["type"] == "normal"]

            report["backtest_summary"] = {
                "overall_accuracy": round(float(bt_df["correct"].mean()), 4),
                "drought_accuracy": round(float(drought_rows["correct"].mean()), 4) if not drought_rows.empty else None,
                "normal_accuracy": round(float(normal_rows["correct"].mean()), 4) if not normal_rows.empty else None,
                "avg_lead_time_weeks": (
                    round(float(drought_rows["lead_time_weeks"].dropna().mean()), 1)
                    if "lead_time_weeks" in drought_rows.columns and not drought_rows["lead_time_weeks"].dropna().empty
                    else None
                ),
            }

        # Cross-validation summary
        if "cv_scores" in all_results:
            report["cv_summary"] = self.performance_summary(all_results["cv_scores"])

        # Model comparison
        if "comparison" in all_results:
            report["model_comparison"] = all_results["comparison"].to_dict(orient="records")

        # Economic value
        if "economic" in all_results:
            report["economic_value"] = all_results["economic"]

        logger.info(
            "Validation report generated | sections={secs}",
            secs=list(report.keys()),
        )
        return report
