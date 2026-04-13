"""
Markov-Switching GARCH Regime Detection Module
===============================================

Provides GARCH family volatility modelling, Markov-switching regime
detection, and regime-overlay visualisation utilities for the
monsoon-textile volatility transmission system.

Classes:
    GARCHModeler           - GARCH(p,q), GJR-GARCH, model comparison.
    MarkovSwitchingDetector - Regime identification via MS models and
                              simplified GARCH-dummy fallbacks.
    RegimeAnalyzer         - Plotly overlays and backtesting of regimes
                              against known drought years.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# GARCH Modelling
# ---------------------------------------------------------------------------

class GARCHModeler:
    """Fit and compare GARCH-family models on return series.

    Parameters
    ----------
    config : dict, optional
        Recognised keys:

        * ``p`` (int) -- GARCH lag order, default ``1``.
        * ``q`` (int) -- ARCH lag order, default ``1``.
        * ``dist`` (str) -- error distribution (``'normal'``, ``'t'``,
          ``'skewt'``, ``'ged'``).  Default ``'t'``.
        * ``rescale`` (bool) -- whether to let ``arch`` auto-rescale.
          Default ``True``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.p: int = cfg.get("p", 1)
        self.q: int = cfg.get("q", 1)
        self.dist: str = cfg.get("dist", "t")
        self.rescale: bool = cfg.get("rescale", True)

    # -- Standard GARCH -----------------------------------------------------

    def fit_garch(
        self,
        returns: pd.Series,
        p: Optional[int] = None,
        q: Optional[int] = None,
        dist: Optional[str] = None,
    ) -> Tuple[Any, pd.Series]:
        """Fit a GARCH(p, q) model.

        Parameters
        ----------
        returns : pd.Series
            Return series (percentage or log returns).
        p, q : int, optional
            Override instance defaults.
        dist : str, optional
            Override instance default distribution.

        Returns
        -------
        tuple[ARCHModelResult, pd.Series]
            Fitted result object and the conditional volatility series.
        """
        from arch import arch_model

        p = p or self.p
        q = q or self.q
        dist = dist or self.dist

        returns = returns.dropna()
        if len(returns) < 50:
            logger.warning("GARCH fit: only {} observations -- estimates may be unreliable.", len(returns))

        am = arch_model(returns, vol="Garch", p=p, q=q, dist=dist, rescale=self.rescale)
        try:
            res = am.fit(disp="off", show_warning=False)
        except Exception as exc:
            logger.error("GARCH({},{}) fit failed: {}", p, q, exc)
            raise

        cond_vol = res.conditional_volatility
        logger.info(
            "GARCH({},{}) fitted | AIC={:.2f} BIC={:.2f} LogLik={:.2f}",
            p, q, res.aic, res.bic, res.loglikelihood,
        )
        return res, cond_vol

    # -- GJR-GARCH ----------------------------------------------------------

    def fit_gjr_garch(self, returns: pd.Series) -> Tuple[Any, pd.Series]:
        """Fit a GJR-GARCH(1,1) model capturing leverage effects.

        Parameters
        ----------
        returns : pd.Series
            Return series.

        Returns
        -------
        tuple[ARCHModelResult, pd.Series]
            Fitted result and conditional volatility.
        """
        from arch import arch_model

        returns = returns.dropna()
        am = arch_model(returns, vol="Garch", p=1, o=1, q=1, dist=self.dist, rescale=self.rescale)
        try:
            res = am.fit(disp="off", show_warning=False)
        except Exception as exc:
            logger.error("GJR-GARCH fit failed: {}", exc)
            raise

        cond_vol = res.conditional_volatility
        logger.info(
            "GJR-GARCH(1,1,1) fitted | AIC={:.2f} BIC={:.2f}",
            res.aic, res.bic,
        )
        return res, cond_vol

    # -- Model comparison ---------------------------------------------------

    def compare_models(self, returns: pd.Series) -> pd.DataFrame:
        """Fit several GARCH variants and return an AIC / BIC comparison.

        Models compared: GARCH(1,1), GARCH(2,1), GARCH(1,2),
        GJR-GARCH(1,1), EGARCH(1,1).

        Parameters
        ----------
        returns : pd.Series
            Return series.

        Returns
        -------
        pd.DataFrame
            Columns: ``model``, ``aic``, ``bic``, ``log_likelihood``.
        """
        from arch import arch_model

        returns = returns.dropna()
        specs: List[Dict[str, Any]] = [
            {"label": "GARCH(1,1)", "vol": "Garch", "p": 1, "o": 0, "q": 1},
            {"label": "GARCH(2,1)", "vol": "Garch", "p": 2, "o": 0, "q": 1},
            {"label": "GARCH(1,2)", "vol": "Garch", "p": 1, "o": 0, "q": 2},
            {"label": "GJR-GARCH(1,1)", "vol": "Garch", "p": 1, "o": 1, "q": 1},
            {"label": "EGARCH(1,1)", "vol": "EGARCH", "p": 1, "o": 1, "q": 1},
        ]

        rows: List[Dict[str, Any]] = []
        for spec in specs:
            try:
                am = arch_model(
                    returns,
                    vol=spec["vol"],
                    p=spec["p"],
                    o=spec["o"],
                    q=spec["q"],
                    dist=self.dist,
                    rescale=self.rescale,
                )
                res = am.fit(disp="off", show_warning=False)
                rows.append(
                    {
                        "model": spec["label"],
                        "aic": res.aic,
                        "bic": res.bic,
                        "log_likelihood": res.loglikelihood,
                    }
                )
            except Exception as exc:
                logger.warning("Could not fit {}: {}", spec["label"], exc)
                rows.append(
                    {
                        "model": spec["label"],
                        "aic": np.nan,
                        "bic": np.nan,
                        "log_likelihood": np.nan,
                    }
                )

        df = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
        logger.info("Model comparison complete; best by AIC: {}", df.iloc[0]["model"] if len(df) else "N/A")
        return df

    # -- Convenience --------------------------------------------------------

    @staticmethod
    def conditional_volatility(fitted: Any) -> pd.Series:
        """Extract the conditional volatility series from a fitted GARCH.

        Parameters
        ----------
        fitted : ARCHModelResult
            Result object from ``arch``.

        Returns
        -------
        pd.Series
            Conditional standard deviation indexed by date.
        """
        return fitted.conditional_volatility


# ---------------------------------------------------------------------------
# Markov-Switching Regime Detection
# ---------------------------------------------------------------------------

class MarkovSwitchingDetector:
    """Identify volatility regimes via Markov-switching models.

    When the full Markov-switching regression fails to converge (common for
    small samples), a simplified GARCH + regime-dummy approach is available
    as a fallback via :meth:`simple_regime_detection`.

    Parameters
    ----------
    config : dict, optional
        Recognised keys:

        * ``k_regimes`` (int) -- number of hidden states, default ``2``.
        * ``max_iter`` (int) -- EM maximum iterations, default ``500``.
        * ``significance`` (float) -- threshold for classifying the
          high-volatility regime.  Default ``0.5``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.k_regimes: int = cfg.get("k_regimes", 2)
        self.max_iter: int = cfg.get("max_iter", 500)
        self.significance: float = cfg.get("significance", 0.5)

    # -- Full MS model ------------------------------------------------------

    def fit(
        self,
        returns: pd.Series,
        exog_features: Optional[pd.DataFrame] = None,
        k_regimes: Optional[int] = None,
    ) -> Tuple[Any, pd.DataFrame]:
        """Fit a Markov-switching dynamic regression.

        Parameters
        ----------
        returns : pd.Series
            Return or volatility series to model.
        exog_features : pd.DataFrame, optional
            Exogenous regressors (e.g. rainfall deficit, cotton returns).
        k_regimes : int, optional
            Override instance default.

        Returns
        -------
        tuple[MarkovRegressionResults, pd.DataFrame]
            Fitted model and a DataFrame of smoothed regime probabilities.
        """
        import statsmodels.api as sm

        k = k_regimes or self.k_regimes
        returns = returns.dropna()

        if len(returns) < 50:
            logger.warning("MS model: only {} observations; consider the simplified fallback.", len(returns))

        endog = returns.copy()
        exog = None
        if exog_features is not None:
            exog = exog_features.reindex(endog.index).dropna()
            endog = endog.loc[exog.index]
            exog = sm.add_constant(exog)

        try:
            ms_model = sm.tsa.MarkovRegression(
                endog,
                k_regimes=k,
                trend="c",
                switching_variance=True,
                exog=exog,
            )
            fitted = ms_model.fit(maxiter=self.max_iter, em_iter=self.max_iter)
        except Exception as exc:
            logger.error("Markov-switching model failed to converge: {}", exc)
            raise

        regime_probs = self.regime_probabilities(fitted)
        logger.info("Markov-switching model fitted with {} regimes.", k)
        return fitted, regime_probs

    # -- Regime probabilities -----------------------------------------------

    @staticmethod
    def regime_probabilities(fitted: Any) -> pd.DataFrame:
        """Extract smoothed regime probabilities.

        Parameters
        ----------
        fitted : MarkovRegressionResults
            Fitted Markov-switching model.

        Returns
        -------
        pd.DataFrame
            Columns ``P(regime_0)``, ``P(regime_1)``, ... indexed by date.
        """
        probs = fitted.smoothed_marginal_probabilities
        if isinstance(probs, pd.DataFrame):
            probs.columns = [f"P(regime_{i})" for i in range(probs.shape[1])]
            return probs
        # ndarray fallback
        n_regimes = probs.shape[1] if probs.ndim == 2 else 1
        cols = [f"P(regime_{i})" for i in range(n_regimes)]
        return pd.DataFrame(probs, columns=cols, index=fitted.model.data.dates)

    # -- Regime statistics --------------------------------------------------

    @staticmethod
    def regime_statistics(fitted: Any) -> Dict[str, Any]:
        """Compute summary statistics for each regime.

        Returns
        -------
        dict
            Keys: ``transition_matrix`` (np.ndarray), ``regime_means``
            (list), ``regime_variances`` (list), ``expected_durations``
            (list of expected sojourn times in periods).
        """
        trans = np.array(fitted.params).reshape(-1)  # raw params
        # Use the model's transition matrix
        try:
            P = fitted.predicted_joint_probabilities
        except AttributeError:
            P = None

        transition_matrix = np.atleast_2d(fitted.regime_transition)

        k = transition_matrix.shape[0]
        expected_durations: List[float] = []
        for i in range(k):
            p_stay = transition_matrix[i, i]
            dur = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else np.inf
            expected_durations.append(float(dur))

        # Regime-specific means and variances from the parameter vector
        regime_means: List[float] = []
        regime_variances: List[float] = []
        try:
            for i in range(k):
                regime_means.append(float(fitted.params.get(f"const[{i}]", np.nan)))
                regime_variances.append(float(fitted.params.get(f"sigma2[{i}]", np.nan)))
        except Exception:
            regime_means = [np.nan] * k
            regime_variances = [np.nan] * k

        result = {
            "transition_matrix": transition_matrix,
            "regime_means": regime_means,
            "regime_variances": regime_variances,
            "expected_durations": expected_durations,
        }
        logger.info("Regime statistics: durations={}", expected_durations)
        return result

    # -- Simplified fallback ------------------------------------------------

    def simple_regime_detection(
        self,
        returns: pd.Series,
        deficit_series: pd.Series,
        threshold: float = -15.0,
    ) -> Tuple[Any, pd.DataFrame]:
        """Simplified regime detection using GARCH with a drought dummy.

        This fallback is useful when the Markov-switching model fails to
        converge.  A binary *drought* indicator (1 when *deficit_series* <
        *threshold*) is included as an exogenous mean regressor in a
        GARCH(1,1).

        Parameters
        ----------
        returns : pd.Series
            Return series.
        deficit_series : pd.Series
            Rainfall deficit (negative = below normal).
        threshold : float
            Deficit value below which a drought regime is flagged.

        Returns
        -------
        tuple[ARCHModelResult, pd.DataFrame]
            Fitted GARCH result and a DataFrame with columns
            ``conditional_vol``, ``drought_regime``, ``regime_label``.
        """
        from arch import arch_model

        common_idx = returns.dropna().index.intersection(deficit_series.dropna().index)
        if len(common_idx) < 50:
            logger.warning("Simple regime detection: only {} overlapping observations.", len(common_idx))

        ret = returns.loc[common_idx]
        deficit = deficit_series.loc[common_idx]
        drought_dummy = (deficit < threshold).astype(float)
        drought_dummy.name = "drought"

        exog = pd.DataFrame({"drought": drought_dummy})

        am = arch_model(ret, vol="Garch", p=1, q=1, dist=self.dist if hasattr(self, "dist") else "t", rescale=True)
        am_with_x = arch_model(ret, x=exog, vol="Garch", p=1, q=1, dist="t", rescale=True)

        try:
            res = am_with_x.fit(disp="off", show_warning=False)
        except Exception as exc:
            logger.warning("GARCH with exog failed ({}); fitting plain GARCH.", exc)
            res = am.fit(disp="off", show_warning=False)

        cond_vol = res.conditional_volatility
        regime_df = pd.DataFrame(
            {
                "conditional_vol": cond_vol,
                "drought_regime": drought_dummy.reindex(cond_vol.index, fill_value=0),
                "regime_label": np.where(
                    drought_dummy.reindex(cond_vol.index, fill_value=0) == 1,
                    "Drought / High-Vol",
                    "Normal",
                ),
            },
            index=cond_vol.index,
        )
        logger.info(
            "Simple regime detection complete; {} drought periods identified.",
            int(regime_df["drought_regime"].sum()),
        )
        return res, regime_df


# ---------------------------------------------------------------------------
# Regime Visualisation & Backtesting
# ---------------------------------------------------------------------------

class RegimeAnalyzer:
    """Overlay detected regimes on price / rainfall charts and backtest
    against historical drought events.

    Parameters
    ----------
    config : dict, optional
        Recognised keys:

        * ``high_vol_regime`` (int) -- index of the high-volatility
          regime in the probability DataFrame.  Default ``1``.
        * ``prob_threshold`` (float) -- probability above which a
          period is classified as high-vol.  Default ``0.5``.
        * ``template`` (str) -- Plotly template.  Default
          ``"plotly_white"``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.high_vol_regime: int = cfg.get("high_vol_regime", 1)
        self.prob_threshold: float = cfg.get("prob_threshold", 0.5)
        self.template: str = cfg.get("template", "plotly_white")

    # -- Price overlay ------------------------------------------------------

    def overlay_regimes_on_price(
        self,
        regime_probs: pd.DataFrame,
        price_series: pd.Series,
    ) -> go.Figure:
        """Plotly figure with price series and shaded high-vol regime bands.

        Parameters
        ----------
        regime_probs : pd.DataFrame
            Smoothed probabilities from :class:`MarkovSwitchingDetector`.
        price_series : pd.Series
            Price series to plot on the primary y-axis.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price with Regime Overlay", "Regime Probability"),
        )

        # Price trace
        fig.add_trace(
            go.Scatter(
                x=price_series.index,
                y=price_series.values,
                name="Price",
                line=dict(color="#1f77b4", width=1.5),
            ),
            row=1,
            col=1,
        )

        # Shade high-vol periods
        high_vol_col = f"P(regime_{self.high_vol_regime})"
        if high_vol_col in regime_probs.columns:
            prob = regime_probs[high_vol_col].reindex(price_series.index, method="nearest")
            self._add_regime_shading(fig, prob, row=1)

            # Probability trace
            fig.add_trace(
                go.Scatter(
                    x=prob.index,
                    y=prob.values,
                    name="P(High-Vol Regime)",
                    line=dict(color="#d62728", width=1.2),
                    fill="tozeroy",
                    fillcolor="rgba(214,39,40,0.15)",
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            template=self.template,
            height=600,
            showlegend=True,
            title_text="Regime Overlay on Price Series",
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Probability", range=[0, 1], row=2, col=1)
        return fig

    # -- Rainfall overlay ---------------------------------------------------

    def overlay_regimes_on_rainfall(
        self,
        regime_probs: pd.DataFrame,
        deficit_series: pd.Series,
    ) -> go.Figure:
        """Plotly figure with rainfall deficit and regime shading.

        Parameters
        ----------
        regime_probs : pd.DataFrame
            Smoothed regime probabilities.
        deficit_series : pd.Series
            Rainfall deficit (%) series.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=("Rainfall Deficit with Regime Overlay", "Regime Probability"),
        )

        # Deficit bar chart
        colors = [
            "#d62728" if v < -15 else ("#ff7f0e" if v < 0 else "#2ca02c")
            for v in deficit_series.values
        ]
        fig.add_trace(
            go.Bar(
                x=deficit_series.index,
                y=deficit_series.values,
                name="Rainfall Deficit %",
                marker_color=colors,
            ),
            row=1,
            col=1,
        )

        # Shade high-vol periods
        high_vol_col = f"P(regime_{self.high_vol_regime})"
        if high_vol_col in regime_probs.columns:
            prob = regime_probs[high_vol_col].reindex(deficit_series.index, method="nearest")
            self._add_regime_shading(fig, prob, row=1)

            fig.add_trace(
                go.Scatter(
                    x=prob.index,
                    y=prob.values,
                    name="P(High-Vol Regime)",
                    line=dict(color="#d62728", width=1.2),
                    fill="tozeroy",
                    fillcolor="rgba(214,39,40,0.15)",
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            template=self.template,
            height=600,
            showlegend=True,
            title_text="Regime Overlay on Rainfall Deficit",
        )
        fig.update_yaxes(title_text="Deficit %", row=1, col=1)
        fig.update_yaxes(title_text="Probability", range=[0, 1], row=2, col=1)
        return fig

    # -- Backtesting --------------------------------------------------------

    def backtest_regime_detection(
        self,
        regime_probs: pd.DataFrame,
        drought_years: List[int],
    ) -> Dict[str, Any]:
        """Evaluate how well the detected regimes align with known droughts.

        Parameters
        ----------
        regime_probs : pd.DataFrame
            Smoothed regime probabilities (must have a DatetimeIndex).
        drought_years : list[int]
            Calendar years of known drought events.

        Returns
        -------
        dict
            Keys: ``hit_rate`` (fraction of drought years correctly
            identified), ``false_alarm_rate``, ``missed_years``,
            ``false_alarm_years``, ``timeline`` (pd.DataFrame with
            year, drought_actual, regime_detected columns).
        """
        high_vol_col = f"P(regime_{self.high_vol_regime})"
        if high_vol_col not in regime_probs.columns:
            available = list(regime_probs.columns)
            logger.error("Column '{}' not found. Available: {}", high_vol_col, available)
            return {
                "hit_rate": np.nan,
                "false_alarm_rate": np.nan,
                "missed_years": drought_years,
                "false_alarm_years": [],
                "timeline": pd.DataFrame(),
            }

        probs = regime_probs[high_vol_col].copy()
        if not isinstance(probs.index, pd.DatetimeIndex):
            try:
                probs.index = pd.to_datetime(probs.index)
            except Exception:
                logger.error("Cannot convert regime_probs index to DatetimeIndex.")
                return {
                    "hit_rate": np.nan,
                    "false_alarm_rate": np.nan,
                    "missed_years": drought_years,
                    "false_alarm_years": [],
                    "timeline": pd.DataFrame(),
                }

        # Annual maximum regime probability
        annual_max = probs.groupby(probs.index.year).max()
        all_years = sorted(annual_max.index)

        detected_years = set(annual_max[annual_max >= self.prob_threshold].index)
        drought_set = set(drought_years)

        hits = drought_set & detected_years
        misses = drought_set - detected_years
        false_alarms = detected_years - drought_set
        non_drought_years = set(all_years) - drought_set

        hit_rate = len(hits) / len(drought_set) if drought_set else np.nan
        false_alarm_rate = (
            len(false_alarms) / len(non_drought_years) if non_drought_years else np.nan
        )

        timeline_rows: List[Dict[str, Any]] = []
        for yr in all_years:
            timeline_rows.append(
                {
                    "year": yr,
                    "drought_actual": yr in drought_set,
                    "regime_detected": yr in detected_years,
                    "max_regime_prob": float(annual_max.get(yr, np.nan)),
                    "correct": (yr in drought_set) == (yr in detected_years),
                }
            )

        result = {
            "hit_rate": float(hit_rate) if not np.isnan(hit_rate) else np.nan,
            "false_alarm_rate": float(false_alarm_rate) if not np.isnan(false_alarm_rate) else np.nan,
            "missed_years": sorted(misses),
            "false_alarm_years": sorted(false_alarms),
            "timeline": pd.DataFrame(timeline_rows),
        }
        logger.info(
            "Backtest: hit_rate={:.1%}  false_alarm_rate={:.1%}  missed={}  false_alarms={}",
            hit_rate if not np.isnan(hit_rate) else 0.0,
            false_alarm_rate if not np.isnan(false_alarm_rate) else 0.0,
            sorted(misses),
            sorted(false_alarms),
        )
        return result

    # -- Internal helpers ---------------------------------------------------

    def _add_regime_shading(
        self,
        fig: go.Figure,
        prob: pd.Series,
        row: int,
    ) -> None:
        """Add semi-transparent vertical rectangles for high-vol periods."""
        in_regime = False
        start = None
        for dt, p in prob.items():
            if p >= self.prob_threshold and not in_regime:
                in_regime = True
                start = dt
            elif p < self.prob_threshold and in_regime:
                in_regime = False
                fig.add_vrect(
                    x0=start,
                    x1=dt,
                    fillcolor="rgba(214,39,40,0.12)",
                    layer="below",
                    line_width=0,
                    row=row,
                    col=1,
                )
        # Close open regime at end
        if in_regime and start is not None:
            fig.add_vrect(
                x0=start,
                x1=prob.index[-1],
                fillcolor="rgba(214,39,40,0.12)",
                layer="below",
                line_width=0,
                row=row,
                col=1,
            )
