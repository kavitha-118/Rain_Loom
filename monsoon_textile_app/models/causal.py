"""
Granger Causality, VAR & Instrumental Variable Analysis Module
===============================================================

Provides tools for testing causal linkages in the monsoon-textile volatility
transmission chain: rainfall -> cotton prices -> textile stock volatility.

Classes:
    StationarityTester  - ADF, KPSS, and auto-differencing utilities.
    GrangerCausalityAnalyzer - Pairwise and full-chain Granger causality tests.
    VARAnalyzer         - VAR estimation, IRFs, FEVD, and cointegration.
    InstrumentalVariableAnalyzer - IV/2SLS regression with ENSO ONI as instrument.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ---------------------------------------------------------------------------
# Stationarity Testing
# ---------------------------------------------------------------------------

class StationarityTester:
    """Test and enforce stationarity for time-series inputs.

    Provides ADF and KPSS unit-root tests as well as an automatic
    differencing routine that iterates until the series is stationary.

    Parameters
    ----------
    config : dict, optional
        Overrides for default significance level and maximum differencing
        order.  Recognised keys:

        * ``significance`` (float) -- threshold p-value for ADF/KPSS
          decisions.  Default ``0.05``.
        * ``max_diff`` (int) -- maximum differencing order allowed by
          :meth:`auto_difference`.  Default ``2``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.significance: float = cfg.get("significance", 0.05)
        self.max_diff: int = cfg.get("max_diff", 2)

    # -- ADF ----------------------------------------------------------------

    def adf_test(self, series: pd.Series) -> Dict[str, Any]:
        """Augmented Dickey-Fuller unit-root test.

        Parameters
        ----------
        series : pd.Series
            Univariate time series (must not contain NaN).

        Returns
        -------
        dict
            Keys: ``statistic``, ``p_value``, ``used_lag``,
            ``n_obs``, ``critical_values``, ``is_stationary``.
        """
        series = series.dropna()
        if len(series) < 20:
            logger.warning("ADF test: series length ({}) may be too short for reliable results.", len(series))

        try:
            stat, p_val, used_lag, n_obs, crit, _ = adfuller(series, autolag="AIC")
        except Exception as exc:
            logger.error("ADF test failed: {}", exc)
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "used_lag": np.nan,
                "n_obs": len(series),
                "critical_values": {},
                "is_stationary": False,
            }

        result: Dict[str, Any] = {
            "statistic": float(stat),
            "p_value": float(p_val),
            "used_lag": int(used_lag),
            "n_obs": int(n_obs),
            "critical_values": {k: float(v) for k, v in crit.items()},
            "is_stationary": bool(p_val < self.significance),
        }
        logger.debug("ADF | stat={:.4f}  p={:.4f}  stationary={}", stat, p_val, result["is_stationary"])
        return result

    # -- KPSS ---------------------------------------------------------------

    def kpss_test(self, series: pd.Series) -> Dict[str, Any]:
        """KPSS stationarity test (null = stationary).

        Parameters
        ----------
        series : pd.Series
            Univariate time series.

        Returns
        -------
        dict
            Keys: ``statistic``, ``p_value``, ``used_lag``,
            ``critical_values``, ``is_stationary``.
        """
        series = series.dropna()
        try:
            stat, p_val, used_lag, crit = kpss(series, regression="c", nlags="auto")
        except Exception as exc:
            logger.error("KPSS test failed: {}", exc)
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "used_lag": np.nan,
                "critical_values": {},
                "is_stationary": False,
            }

        result: Dict[str, Any] = {
            "statistic": float(stat),
            "p_value": float(p_val),
            "used_lag": int(used_lag),
            "critical_values": {k: float(v) for k, v in crit.items()},
            # KPSS: reject stationarity when p < alpha
            "is_stationary": bool(p_val >= self.significance),
        }
        logger.debug("KPSS | stat={:.4f}  p={:.4f}  stationary={}", stat, p_val, result["is_stationary"])
        return result

    # -- Summary ------------------------------------------------------------

    def test_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run ADF and KPSS on every numeric column of *df*.

        Returns
        -------
        pd.DataFrame
            One row per column with ADF/KPSS statistics and verdicts.
        """
        rows: List[Dict[str, Any]] = []
        for col in df.select_dtypes(include=[np.number]).columns:
            adf = self.adf_test(df[col])
            kpss_res = self.kpss_test(df[col])
            rows.append(
                {
                    "variable": col,
                    "adf_statistic": adf["statistic"],
                    "adf_p_value": adf["p_value"],
                    "adf_stationary": adf["is_stationary"],
                    "kpss_statistic": kpss_res["statistic"],
                    "kpss_p_value": kpss_res["p_value"],
                    "kpss_stationary": kpss_res["is_stationary"],
                    "conclusion": self._conclude(adf["is_stationary"], kpss_res["is_stationary"]),
                }
            )
        return pd.DataFrame(rows).set_index("variable")

    # -- Auto-differencing --------------------------------------------------

    def auto_difference(self, series: pd.Series) -> Tuple[pd.Series, int]:
        """Difference *series* until it passes the ADF test.

        Parameters
        ----------
        series : pd.Series
            Raw (possibly non-stationary) series.

        Returns
        -------
        tuple[pd.Series, int]
            The differenced series and the number of differences applied
            (0 if already stationary).
        """
        current = series.dropna().copy()
        for d in range(self.max_diff + 1):
            if self.adf_test(current)["is_stationary"]:
                logger.info("Series stationary after d={} differences.", d)
                return current, d
            if d < self.max_diff:
                current = current.diff().dropna()
        logger.warning("Series not stationary after {} differences.", self.max_diff)
        return current, self.max_diff

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _conclude(adf_stationary: bool, kpss_stationary: bool) -> str:
        if adf_stationary and kpss_stationary:
            return "Stationary"
        if not adf_stationary and not kpss_stationary:
            return "Non-stationary"
        if adf_stationary and not kpss_stationary:
            return "Trend-stationary"
        return "Difference-stationary"


# ---------------------------------------------------------------------------
# Granger Causality
# ---------------------------------------------------------------------------

class GrangerCausalityAnalyzer:
    """Pairwise and full-chain Granger causality testing.

    The canonical transmission chain tested by :meth:`test_full_chain` is::

        rainfall  -->  cotton_price  -->  stock_volatility
        rainfall  ----------------------> stock_volatility

    Parameters
    ----------
    config : dict, optional
        Recognised keys:

        * ``significance`` (float) -- default ``0.05``.
        * ``max_lag`` (int) -- default ``12``.
        * ``rainfall_col`` (str) -- default ``"rainfall"``.
        * ``cotton_col`` (str) -- default ``"cotton_price"``.
        * ``volatility_col`` (str) -- default ``"stock_volatility"``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.significance: float = cfg.get("significance", 0.05)
        self.max_lag: int = cfg.get("max_lag", 12)
        self.rainfall_col: str = cfg.get("rainfall_col", "rainfall")
        self.cotton_col: str = cfg.get("cotton_col", "cotton_price")
        self.volatility_col: str = cfg.get("volatility_col", "stock_volatility")
        self._results: List[Dict[str, Any]] = []

    # -- Pairwise test ------------------------------------------------------

    def test_pairwise(
        self,
        df: pd.DataFrame,
        cause_col: str,
        effect_col: str,
        max_lag: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Granger causality test for *cause_col* -> *effect_col*.

        Uses the statsmodels ``grangercausalitytests`` wrapper which reports
        four test variants per lag (ssr_ftest, ssr_chi2test, lrtest,
        params_ftest).  The returned dict aggregates p-values from the
        ``ssr_ftest`` for each lag.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain both *cause_col* and *effect_col*.
        cause_col, effect_col : str
            Column names.
        max_lag : int, optional
            Override the instance default.

        Returns
        -------
        dict
            Keys: ``cause``, ``effect``, ``p_values`` (dict lag->p),
            ``optimal_lag``, ``min_p_value``, ``is_significant``.
        """
        max_lag = max_lag or self.max_lag
        data = df[[effect_col, cause_col]].dropna()

        if len(data) < max_lag + 10:
            logger.warning(
                "Insufficient observations ({}) for max_lag={}; reducing max_lag.",
                len(data),
                max_lag,
            )
            max_lag = max(1, len(data) - 10)

        try:
            gc_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        except Exception as exc:
            logger.error("Granger test failed ({}->{}) : {}", cause_col, effect_col, exc)
            return {
                "cause": cause_col,
                "effect": effect_col,
                "p_values": {},
                "optimal_lag": np.nan,
                "min_p_value": np.nan,
                "is_significant": False,
            }

        p_values: Dict[int, float] = {}
        for lag in range(1, max_lag + 1):
            if lag in gc_results:
                # ssr_ftest returns (F-stat, p-value, df_denom, df_num)
                p_values[lag] = float(gc_results[lag][0]["ssr_ftest"][1])

        if not p_values:
            optimal_lag = np.nan
            min_p = np.nan
        else:
            optimal_lag = int(min(p_values, key=p_values.get))  # type: ignore[arg-type]
            min_p = float(min(p_values.values()))

        result: Dict[str, Any] = {
            "cause": cause_col,
            "effect": effect_col,
            "p_values": p_values,
            "optimal_lag": optimal_lag,
            "min_p_value": min_p,
            "is_significant": bool(min_p < self.significance) if not np.isnan(min_p) else False,
        }
        self._results.append(result)
        logger.info(
            "Granger {} -> {} | optimal_lag={} min_p={:.4g} sig={}",
            cause_col,
            effect_col,
            optimal_lag,
            min_p if not np.isnan(min_p) else float("nan"),
            result["is_significant"],
        )
        return result

    # -- Full chain ---------------------------------------------------------

    def test_full_chain(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Test the three canonical links of the transmission chain.

        Returns
        -------
        dict
            Keys ``'rainfall_to_cotton'``, ``'cotton_to_volatility'``,
            ``'rainfall_to_volatility'`` each mapping to pairwise results.
        """
        self._results.clear()
        chain: Dict[str, Dict[str, Any]] = {}
        links = [
            ("rainfall_to_cotton", self.rainfall_col, self.cotton_col),
            ("cotton_to_volatility", self.cotton_col, self.volatility_col),
            ("rainfall_to_volatility", self.rainfall_col, self.volatility_col),
        ]
        for key, cause, effect in links:
            if cause not in df.columns or effect not in df.columns:
                logger.warning("Column(s) missing for {}: need {} and {}.", key, cause, effect)
                continue
            chain[key] = self.test_pairwise(df, cause, effect)
        return chain

    # -- Toda-Yamamoto robust test ------------------------------------------

    def toda_yamamoto_test(
        self,
        df: pd.DataFrame,
        cause_col: str,
        effect_col: str,
    ) -> Dict[str, Any]:
        """Toda-Yamamoto modified Granger causality test.

        Unlike the standard Granger test this procedure is valid regardless of
        the integration order of the series.  The idea is to fit a VAR(k+d_max)
        where *d_max* is the maximum integration order, then test zero
        restrictions on the first *k* lags only.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain *cause_col* and *effect_col*.
        cause_col, effect_col : str
            Column names.

        Returns
        -------
        dict
            Keys: ``cause``, ``effect``, ``optimal_lag_k``, ``d_max``,
            ``wald_statistic``, ``p_value``, ``is_significant``.
        """
        from scipy.stats import chi2 as chi2_dist
        from statsmodels.tsa.api import VAR as VARModel

        data = df[[effect_col, cause_col]].dropna()

        # Determine d_max via ADF
        tester = StationarityTester()
        _, d_effect = tester.auto_difference(data[effect_col])
        _, d_cause = tester.auto_difference(data[cause_col])
        d_max = max(d_effect, d_cause, 1)

        # Select optimal lag k using AIC on a standard VAR
        try:
            var_tmp = VARModel(data.values)
            lag_order = var_tmp.select_order(maxlags=min(self.max_lag, len(data) // 3 - 1))
            k = max(lag_order.aic, 1)
        except Exception:
            k = 2
            logger.warning("Lag selection failed; defaulting k={}.", k)

        total_lags = k + d_max
        if len(data) <= total_lags + 5:
            logger.error("Toda-Yamamoto: not enough data (n={}, k+d_max={}).", len(data), total_lags)
            return {
                "cause": cause_col,
                "effect": effect_col,
                "optimal_lag_k": k,
                "d_max": d_max,
                "wald_statistic": np.nan,
                "p_value": np.nan,
                "is_significant": False,
            }

        try:
            var_model = VARModel(data.values)
            fitted = var_model.fit(maxlags=total_lags, trend="c")
        except Exception as exc:
            logger.error("Toda-Yamamoto VAR fit failed: {}", exc)
            return {
                "cause": cause_col,
                "effect": effect_col,
                "optimal_lag_k": k,
                "d_max": d_max,
                "wald_statistic": np.nan,
                "p_value": np.nan,
                "is_significant": False,
            }

        # Wald test: coefficients of cause_col for lags 1..k in effect equation
        # effect_col is column index 0 in the VAR
        params = fitted.params  # shape (1 + n_vars*total_lags, n_vars)

        # Identify the indices of the cause variable for lags 1..k
        # params layout: const, L1.y1, L1.y2, L2.y1, L2.y2, ...
        cause_idx_in_var = 1  # cause_col is the second variable
        indices: List[int] = []
        for lag in range(1, k + 1):
            # offset: 1 (const) + (lag-1)*2 + cause_idx_in_var
            idx = 1 + (lag - 1) * 2 + cause_idx_in_var
            if idx < params.shape[0]:
                indices.append(idx)

        if not indices:
            logger.error("Toda-Yamamoto: could not locate cause coefficients.")
            return {
                "cause": cause_col,
                "effect": effect_col,
                "optimal_lag_k": k,
                "d_max": d_max,
                "wald_statistic": np.nan,
                "p_value": np.nan,
                "is_significant": False,
            }

        # Build restriction matrix R such that R * beta = 0
        n_params = params.shape[0]
        R = np.zeros((len(indices), n_params))
        for i, idx in enumerate(indices):
            R[i, idx] = 1.0

        # Effect equation coefficients and covariance
        beta_effect = params[:, 0]
        cov_beta = fitted.cov_params  # full covariance (n_vars*n_params x n_vars*n_params)

        # Extract the block for equation 0
        eq_cov = np.zeros((n_params, n_params))
        for i in range(n_params):
            for j in range(n_params):
                row = i
                col = j
                if row < cov_beta.shape[0] and col < cov_beta.shape[1]:
                    eq_cov[i, j] = cov_beta[row, col]

        try:
            Rb = R @ beta_effect
            middle = R @ eq_cov @ R.T
            wald = float(Rb.T @ np.linalg.inv(middle) @ Rb)
            p_value = float(1.0 - chi2_dist.cdf(wald, df=len(indices)))
        except np.linalg.LinAlgError:
            logger.error("Toda-Yamamoto: singular matrix in Wald computation.")
            wald = np.nan
            p_value = np.nan

        result: Dict[str, Any] = {
            "cause": cause_col,
            "effect": effect_col,
            "optimal_lag_k": k,
            "d_max": d_max,
            "wald_statistic": wald,
            "p_value": p_value,
            "is_significant": bool(p_value < self.significance) if not np.isnan(p_value) else False,
        }
        self._results.append(result)
        logger.info(
            "Toda-Yamamoto {} -> {} | k={} d_max={} Wald={:.4f} p={:.4g} sig={}",
            cause_col,
            effect_col,
            k,
            d_max,
            wald if not np.isnan(wald) else float("nan"),
            p_value if not np.isnan(p_value) else float("nan"),
            result["is_significant"],
        )
        return result

    # -- Summary table ------------------------------------------------------

    def summary_table(self) -> pd.DataFrame:
        """Return a publication-ready DataFrame of all recorded test results.

        Each row corresponds to one directional test, with columns for the
        cause, effect, test statistic / p-value, and significance.
        """
        if not self._results:
            logger.warning("No Granger test results recorded yet.")
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for r in self._results:
            row: Dict[str, Any] = {
                "Cause": r.get("cause", ""),
                "Effect": r.get("effect", ""),
            }
            if "min_p_value" in r:
                row["Test"] = "Granger"
                row["Optimal Lag"] = r.get("optimal_lag")
                row["p-value"] = r.get("min_p_value")
            elif "wald_statistic" in r:
                row["Test"] = "Toda-Yamamoto"
                row["Optimal Lag"] = r.get("optimal_lag_k")
                row["Wald Statistic"] = r.get("wald_statistic")
                row["p-value"] = r.get("p_value")
            row["Significant"] = r.get("is_significant", False)
            rows.append(row)

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# VAR Analysis
# ---------------------------------------------------------------------------

class VARAnalyzer:
    """Vector Autoregression modelling, impulse-response functions, FEVD,
    and Johansen cointegration testing.

    Parameters
    ----------
    config : dict, optional
        Recognised keys:

        * ``max_lags`` (int) -- default ``8``.
        * ``ic`` (str) -- information criterion for lag selection
          (``'aic'``, ``'bic'``, ``'hqic'``, ``'fpe'``).  Default ``'aic'``.
        * ``trend`` (str) -- deterministic trend term (``'n'``, ``'c'``,
          ``'ct'``).  Default ``'c'``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.max_lags: int = cfg.get("max_lags", 8)
        self.ic: str = cfg.get("ic", "aic")
        self.trend: str = cfg.get("trend", "c")

    # -- Fit ----------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        variables: List[str],
        max_lags: Optional[int] = None,
        ic: Optional[str] = None,
    ) -> Any:
        """Estimate a VAR model on the selected *variables*.

        Parameters
        ----------
        df : pd.DataFrame
            Source data (should be stationary).
        variables : list[str]
            Column names to include.
        max_lags : int, optional
            Override the instance default.
        ic : str, optional
            Override the instance default information criterion.

        Returns
        -------
        statsmodels.tsa.vector_ar.var_model.VARResults
            Fitted VAR model object.
        """
        max_lags = max_lags or self.max_lags
        ic = ic or self.ic

        data = df[variables].dropna()
        if len(data) < max_lags + 10:
            logger.warning("Limited data (n={}); reducing max_lags from {} to {}.", len(data), max_lags, max(1, len(data) - 10))
            max_lags = max(1, len(data) - 10)

        model = VAR(data)
        try:
            lag_order = model.select_order(maxlags=max_lags)
            selected_lag = getattr(lag_order, ic, 1)
            selected_lag = max(selected_lag, 1)
            logger.info("VAR lag selection ({}): {} lags.", ic.upper(), selected_lag)
        except Exception as exc:
            logger.warning("Lag selection failed ({}); defaulting to 1 lag.", exc)
            selected_lag = 1

        try:
            fitted = model.fit(maxlags=selected_lag, trend=self.trend)
        except Exception as exc:
            logger.error("VAR estimation failed: {}", exc)
            raise

        logger.info(
            "VAR({}) fitted on {} variables, {} observations.",
            fitted.k_ar,
            len(variables),
            fitted.nobs,
        )
        return fitted

    # -- Impulse-Response Functions ------------------------------------------

    def impulse_response(
        self,
        fitted: Any,
        impulse: str,
        response: str,
        periods: int = 20,
    ) -> Dict[str, Any]:
        """Compute orthogonalised impulse-response function.

        Parameters
        ----------
        fitted : VARResults
            Fitted VAR model from :meth:`fit`.
        impulse : str
            Shocking variable name.
        response : str
            Responding variable name.
        periods : int
            Horizon.

        Returns
        -------
        dict
            Keys: ``impulse``, ``response``, ``periods``,
            ``irf_values`` (np.ndarray), ``lower`` (np.ndarray),
            ``upper`` (np.ndarray).
        """
        try:
            irf: IRAnalysis = fitted.irf(periods=periods)
        except Exception as exc:
            logger.error("IRF computation failed: {}", exc)
            raise

        var_names = list(fitted.names)
        imp_idx = var_names.index(impulse)
        res_idx = var_names.index(response)

        irf_vals = irf.orth_irfs[:, res_idx, imp_idx]

        # Confidence bands (if available)
        try:
            irf_err = irf.orth_err_band(svar=False)
            lower = irf_err[:, res_idx, imp_idx, 0] if irf_err.ndim == 4 else irf_vals
            upper = irf_err[:, res_idx, imp_idx, 1] if irf_err.ndim == 4 else irf_vals
        except Exception:
            lower = irf_vals.copy()
            upper = irf_vals.copy()

        logger.info("IRF computed: {} -> {} over {} periods.", impulse, response, periods)
        return {
            "impulse": impulse,
            "response": response,
            "periods": list(range(periods + 1)),
            "irf_values": irf_vals,
            "lower": lower,
            "upper": upper,
        }

    # -- FEVD ---------------------------------------------------------------

    def forecast_error_variance_decomposition(
        self,
        fitted: Any,
        periods: int = 20,
    ) -> Dict[str, pd.DataFrame]:
        """Forecast error variance decomposition.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping from each endogenous variable name to a DataFrame whose
            columns are the contributing variables and rows are the forecast
            horizon.
        """
        try:
            fevd = fitted.fevd(periods=periods)
        except Exception as exc:
            logger.error("FEVD computation failed: {}", exc)
            raise

        var_names = list(fitted.names)
        result: Dict[str, pd.DataFrame] = {}
        for i, name in enumerate(var_names):
            decomp = fevd.decomp[i]  # shape (periods, n_vars)
            result[name] = pd.DataFrame(
                decomp,
                columns=var_names,
                index=pd.RangeIndex(1, periods + 1, name="horizon"),
            )
        logger.info("FEVD computed for {} variables over {} periods.", len(var_names), periods)
        return result

    # -- Johansen cointegration ---------------------------------------------

    def johansen_cointegration(
        self,
        df: pd.DataFrame,
        variables: List[str],
    ) -> Dict[str, Any]:
        """Johansen cointegration test.

        Parameters
        ----------
        df : pd.DataFrame
            Level (non-differenced) data.
        variables : list[str]
            Columns to test.

        Returns
        -------
        dict
            Keys: ``trace_stat``, ``trace_crit_90_95_99``,
            ``max_eig_stat``, ``max_eig_crit_90_95_99``,
            ``n_cointegrating`` (number of cointegrating relations at 5 %).
        """
        data = df[variables].dropna()
        if len(data) < 30:
            logger.warning("Johansen test: only {} observations, results may be unreliable.", len(data))

        try:
            jres = coint_johansen(data, det_order=0, k_ar_diff=2)
        except Exception as exc:
            logger.error("Johansen test failed: {}", exc)
            raise

        # Count cointegrating relations using the trace statistic at 95 %
        n_coint = int(np.sum(jres.lr1 > jres.cvt[:, 1]))

        result: Dict[str, Any] = {
            "trace_stat": jres.lr1.tolist(),
            "trace_crit_90_95_99": jres.cvt.tolist(),
            "max_eig_stat": jres.lr2.tolist(),
            "max_eig_crit_90_95_99": jres.cvm.tolist(),
            "n_cointegrating": n_coint,
        }
        logger.info("Johansen test: {} cointegrating relation(s) at 5%%.", n_coint)
        return result


# ---------------------------------------------------------------------------
# Instrumental Variable / 2SLS Analysis
# ---------------------------------------------------------------------------

class InstrumentalVariableAnalyzer:
    """Two-Stage Least Squares (2SLS) regression using ENSO ONI as an
    instrumental variable to strengthen causal claims beyond Granger.

    The identification strategy:
        - **Endogenous variable**: Rainfall deficit (potentially correlated
          with unobserved confounders affecting both rainfall and volatility).
        - **Instrument**: ENSO ONI index (Oceanic Niño Index). ONI drives
          Indian monsoon variability via Walker-circulation teleconnections
          but has no plausible *direct* effect on textile stock volatility.
        - **Outcome**: Stock volatility or cotton price change.

    Diagnostics:
        - **First-stage F-statistic** (> 10 for instrument relevance)
        - **Hausman test** (OLS vs IV endogeneity check)
        - **Wu-Hausman** (equivalent F-form)

    Parameters
    ----------
    config : dict, optional
        Recognised keys:

        * ``significance`` (float) -- default ``0.05``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.significance: float = cfg.get("significance", 0.05)
        self._results: List[Dict[str, Any]] = []

    def run_2sls(
        self,
        df: pd.DataFrame,
        endog_col: str,
        exog_col: str,
        instrument_col: str,
        control_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run manual 2SLS IV regression.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all named columns.
        endog_col : str
            Endogenous regressor (e.g. rainfall deficit).
        exog_col : str
            Outcome variable (e.g. stock volatility).
        instrument_col : str
            Instrument (e.g. ONI index).
        control_cols : list[str], optional
            Additional exogenous controls included in both stages.

        Returns
        -------
        dict
            Keys: first_stage, second_stage, hausman, diagnostics.
        """
        import statsmodels.api as sm
        from scipy.stats import chi2 as chi2_dist, f as f_dist

        controls = control_cols or []
        cols_needed = [endog_col, exog_col, instrument_col] + controls
        data = df[cols_needed].dropna()

        if len(data) < 30:
            logger.warning("IV/2SLS: only {} observations, results may be unreliable.", len(data))

        y = data[exog_col].values
        x_endog = data[endog_col].values
        z = data[instrument_col].values

        # Build control matrix
        if controls:
            W = sm.add_constant(data[controls].values)
        else:
            W = np.ones((len(data), 1))

        # ── First Stage: endog_col = alpha + beta*instrument + gamma*controls + e ──
        X_first = np.column_stack([W, z])
        first_stage = sm.OLS(x_endog, X_first).fit()
        x_hat = first_stage.fittedvalues

        # F-statistic for instrument (last coefficient)
        f_stat_instrument = float(first_stage.fvalue)
        f_pval_instrument = float(first_stage.f_pvalue)

        # Partial F-stat for the instrument alone (exclude controls)
        # Test H0: coefficient on instrument = 0
        t_stat_z = float(first_stage.tvalues[-1])
        partial_f = t_stat_z ** 2
        partial_f_pval = float(2 * (1 - f_dist.cdf(abs(t_stat_z), 1, first_stage.df_resid)))

        first_stage_result = {
            "f_statistic": f_stat_instrument,
            "f_pvalue": f_pval_instrument,
            "partial_f_instrument": float(partial_f),
            "partial_f_pvalue": partial_f_pval,
            "r_squared": float(first_stage.rsquared),
            "instrument_coeff": float(first_stage.params[-1]),
            "instrument_tstat": float(first_stage.tvalues[-1]),
            "instrument_pvalue": float(first_stage.pvalues[-1]),
            "n_obs": int(first_stage.nobs),
            "strong_instrument": bool(partial_f > 10),
        }

        # ── Second Stage: outcome = alpha + beta*x_hat + gamma*controls + u ──
        X_second = np.column_stack([W, x_hat])
        second_stage = sm.OLS(y, X_second).fit()

        iv_coeff = float(second_stage.params[-1])
        iv_se = float(second_stage.bse[-1])
        iv_tstat = float(second_stage.tvalues[-1])
        iv_pvalue = float(second_stage.pvalues[-1])

        second_stage_result = {
            "iv_coefficient": iv_coeff,
            "iv_std_error": iv_se,
            "iv_tstatistic": iv_tstat,
            "iv_pvalue": iv_pvalue,
            "r_squared": float(second_stage.rsquared),
            "significant": bool(iv_pvalue < self.significance),
        }

        # ── OLS (for comparison) ──
        X_ols = np.column_stack([W, x_endog])
        ols_model = sm.OLS(y, X_ols).fit()
        ols_coeff = float(ols_model.params[-1])
        ols_se = float(ols_model.bse[-1])
        ols_pvalue = float(ols_model.pvalues[-1])

        ols_result = {
            "ols_coefficient": ols_coeff,
            "ols_std_error": ols_se,
            "ols_pvalue": ols_pvalue,
        }

        # ── Hausman Test: H0 = OLS is consistent (no endogeneity) ──
        # If rejected, IV is preferred over OLS
        coeff_diff = iv_coeff - ols_coeff
        var_diff = max(iv_se**2 - ols_se**2, 1e-12)
        hausman_stat = coeff_diff**2 / var_diff
        hausman_pvalue = float(1 - chi2_dist.cdf(hausman_stat, df=1))

        hausman_result = {
            "hausman_statistic": float(hausman_stat),
            "hausman_pvalue": hausman_pvalue,
            "reject_ols": bool(hausman_pvalue < self.significance),
            "interpretation": (
                "Endogeneity detected — IV preferred over OLS"
                if hausman_pvalue < self.significance
                else "No evidence of endogeneity — OLS may be sufficient"
            ),
        }

        result = {
            "endog": endog_col,
            "outcome": exog_col,
            "instrument": instrument_col,
            "n_obs": int(len(data)),
            "first_stage": first_stage_result,
            "second_stage": second_stage_result,
            "ols_comparison": ols_result,
            "hausman": hausman_result,
        }

        self._results.append(result)

        logger.info(
            "IV/2SLS {} -> {} (Z={}) | 1st-stage F={:.1f} (strong={}) | "
            "IV coeff={:.4f} p={:.4g} | Hausman p={:.4g}",
            endog_col, exog_col, instrument_col,
            partial_f, first_stage_result["strong_instrument"],
            iv_coeff, iv_pvalue,
            hausman_pvalue,
        )
        return result

    def run_full_analysis(
        self,
        df: pd.DataFrame,
        instrument_col: str = "oni_value",
        rainfall_col: str = "rainfall",
        volatility_col: str = "stock_volatility",
        cotton_col: str = "cotton_price",
    ) -> Dict[str, Dict[str, Any]]:
        """Run IV/2SLS for the two key causal links:

        1. ONI -> Rainfall -> Stock Volatility
        2. ONI -> Rainfall -> Cotton Price

        Parameters
        ----------
        df : pd.DataFrame
            Merged weekly data with ONI, rainfall, volatility, cotton columns.

        Returns
        -------
        dict
            Keys: 'rainfall_to_volatility', 'rainfall_to_cotton'.
        """
        results: Dict[str, Dict[str, Any]] = {}

        # Link 1: Rainfall -> Stock Volatility (instrumented by ONI)
        if rainfall_col in df.columns and volatility_col in df.columns and instrument_col in df.columns:
            results["rainfall_to_volatility"] = self.run_2sls(
                df, endog_col=rainfall_col, exog_col=volatility_col,
                instrument_col=instrument_col,
            )

        # Link 2: Rainfall -> Cotton Price (instrumented by ONI)
        if rainfall_col in df.columns and cotton_col in df.columns and instrument_col in df.columns:
            results["rainfall_to_cotton"] = self.run_2sls(
                df, endog_col=rainfall_col, exog_col=cotton_col,
                instrument_col=instrument_col,
            )

        return results

    def summary_table(self) -> pd.DataFrame:
        """Return a summary DataFrame of all IV/2SLS results."""
        if not self._results:
            return pd.DataFrame()

        rows = []
        for r in self._results:
            fs = r["first_stage"]
            ss = r["second_stage"]
            ols = r["ols_comparison"]
            hm = r["hausman"]
            rows.append({
                "Endogenous": r["endog"],
                "Outcome": r["outcome"],
                "Instrument": r["instrument"],
                "N": r["n_obs"],
                "1st-Stage F": fs["partial_f_instrument"],
                "Strong Instr.": fs["strong_instrument"],
                "IV Coeff": ss["iv_coefficient"],
                "IV p-value": ss["iv_pvalue"],
                "OLS Coeff": ols["ols_coefficient"],
                "OLS p-value": ols["ols_pvalue"],
                "Hausman p": hm["hausman_pvalue"],
                "IV Preferred": hm["reject_ols"],
            })
        return pd.DataFrame(rows)
