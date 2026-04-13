"""Feature engineering module for the Monsoon-Textile Volatility system.

Provides climate, agricultural, and market feature builders plus an
end-to-end pipeline that merges all sources into a weekly modelling dataset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division that returns 0.0 where denominator is zero."""
    return num.div(den.replace(0, np.nan)).fillna(0.0)


# ---------------------------------------------------------------------------
# Climate features
# ---------------------------------------------------------------------------

class ClimateFeatureBuilder:
    """Builds monsoon-rainfall features for cotton-belt districts.

    Parameters
    ----------
    config : dict
        Must contain keys:
        - ``lpa_jjas`` : float – Long Period Average JJAS rainfall (mm).
        - ``cotton_belt_districts`` : list[str] – District identifiers in the
          cotton belt.
        - ``normal_onset_doy`` : int – Day-of-year for normal monsoon onset.
        - ``june_lpa`` : float – LPA rainfall for June (mm).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.lpa_jjas: float = config["lpa_jjas"]
        self.cotton_belt_districts: List[str] = config["cotton_belt_districts"]
        self.normal_onset_doy: int = config["normal_onset_doy"]
        self.june_lpa: float = config["june_lpa"]
        logger.info("ClimateFeatureBuilder initialised (LPA JJAS={:.1f} mm)", self.lpa_jjas)

    # -- core features -------------------------------------------------------

    def deficit_pct_cumulative(self, actual_cumul: pd.Series) -> pd.Series:
        """Weekly cumulative deficit percentage vs LPA JJAS."""
        return (actual_cumul - self.lpa_jjas) / self.lpa_jjas * 100.0

    def deficit_pct_rolling4w(self, weekly_rain: pd.Series) -> pd.Series:
        """4-week rolling rainfall deficit as a percentage of LPA."""
        rolling_sum = weekly_rain.rolling(window=4, min_periods=4).sum()
        lpa_4w = self.lpa_jjas / 18.0 * 4.0  # ~18 JJAS weeks
        return (rolling_sum - lpa_4w) / lpa_4w * 100.0

    def deficit_severity(self, deficit_pct: pd.Series) -> pd.Series:
        """Categorical severity: Normal / Deficient / Scanty."""
        return pd.cut(
            deficit_pct,
            bins=[-np.inf, -25.0, -10.0, np.inf],
            labels=["Scanty", "Deficient", "Normal"],
            right=True,
            ordered=False,
        )

    def spatial_deficit_breadth(self, district_deficits: pd.DataFrame) -> pd.Series:
        """Percentage of cotton-belt districts with deficit > 20 %.

        Parameters
        ----------
        district_deficits : DataFrame
            Columns are district names, rows indexed by date/week.
            Values are deficit percentages (negative = shortfall).
        """
        cols = [c for c in self.cotton_belt_districts if c in district_deficits.columns]
        if not cols:
            logger.warning("No cotton-belt district columns found in input.")
            return pd.Series(0.0, index=district_deficits.index, name="spatial_deficit_breadth")
        subset = district_deficits[cols]
        return (subset < -20.0).sum(axis=1) / len(cols) * 100.0

    def dry_spell_length(self, daily_rain: pd.Series) -> pd.Series:
        """Consecutive days with rainfall < 2 mm (run-length encoded)."""
        is_dry = (daily_rain < 2.0).astype(int)
        # Reset cumsum at every wet day
        groups = is_dry.ne(is_dry.shift()).cumsum()
        run_lengths = is_dry.groupby(groups).cumsum()
        return run_lengths.rename("dry_spell_length")

    def onset_delay_days(self, onset_doy: pd.Series) -> pd.Series:
        """Days monsoon onset is delayed versus normal onset DOY."""
        return (onset_doy - self.normal_onset_doy).clip(lower=0).rename("onset_delay_days")

    def june_deficit_flag(self, june_rain: pd.Series) -> pd.Series:
        """Binary flag: 1 if June rainfall < 70 % of June LPA."""
        return (june_rain < 0.70 * self.june_lpa).astype(int).rename("june_deficit_flag")


# ---------------------------------------------------------------------------
# Agricultural features
# ---------------------------------------------------------------------------

class AgriculturalFeatureBuilder:
    """Builds crop-health and commodity features.

    Parameters
    ----------
    config : dict
        Optional keys:
        - ``ndvi_clim_mean`` : float – Climatological mean NDVI.
        - ``ndvi_clim_std`` : float – Climatological std dev of NDVI.
        - ``futures_annualise`` : int – Trading days per year (default 252).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.ndvi_mean: float = config.get("ndvi_clim_mean", 0.45)
        self.ndvi_std: float = config.get("ndvi_clim_std", 0.08)
        self.ann_factor: int = config.get("futures_annualise", 252)
        logger.info("AgriculturalFeatureBuilder initialised")

    def ndvi_anomaly(self, ndvi: pd.Series) -> pd.Series:
        """Standardised NDVI anomaly."""
        return ((ndvi - self.ndvi_mean) / self.ndvi_std).rename("ndvi_anomaly")

    def ndvi_trend_8wk(self, ndvi: pd.Series) -> pd.Series:
        """Slope of NDVI over a rolling 8-week window (OLS)."""
        window = 8
        x = np.arange(window, dtype=float)
        x_demean = x - x.mean()
        denom = (x_demean ** 2).sum()

        def _slope(arr: np.ndarray) -> float:
            if len(arr) < window:
                return np.nan
            return np.dot(x_demean, arr - arr.mean()) / denom

        return (
            ndvi.rolling(window=window, min_periods=window)
            .apply(_slope, raw=True)
            .rename("ndvi_trend_8wk")
        )

    def cotton_futures_ret_30d(self, price: pd.Series) -> pd.Series:
        """30-day log return of MCX cotton."""
        return np.log(price / price.shift(30)).rename("cotton_futures_ret_30d")

    def cotton_futures_vol_20d(self, price: pd.Series) -> pd.Series:
        """20-day annualised realised volatility of cotton futures."""
        log_ret = np.log(price / price.shift(1))
        return (
            log_ret.rolling(window=20, min_periods=20).std() * np.sqrt(self.ann_factor)
        ).rename("cotton_futures_vol_20d")

    def cotton_basis(self, mcx_price: pd.Series, ice_price: pd.Series) -> pd.Series:
        """Local premium: MCX cotton minus ICE cotton."""
        return (mcx_price - ice_price).rename("cotton_basis")

    def reservoir_pct(self, current_storage: pd.Series, full_capacity: pd.Series) -> pd.Series:
        """Reservoir storage as percentage of full capacity."""
        return (_safe_div(current_storage, full_capacity) * 100.0).rename("reservoir_pct")


# ---------------------------------------------------------------------------
# Market features
# ---------------------------------------------------------------------------

class MarketFeatureBuilder:
    """Builds equity / volatility features for textile stocks.

    Parameters
    ----------
    config : dict
        Optional keys:
        - ``ann_factor`` : int – Trading days per year (default 252).
        - ``vol_window`` : int – Window for realised vol (default 20).
        - ``vol_of_vol_window`` : int – Outer window for vol-of-vol (default 60).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.ann_factor: int = config.get("ann_factor", 252)
        self.vol_window: int = config.get("vol_window", 20)
        self.vov_window: int = config.get("vol_of_vol_window", 60)
        logger.info("MarketFeatureBuilder initialised (ann_factor={})", self.ann_factor)

    def log_return(self, close: pd.Series) -> pd.Series:
        """Daily log return."""
        return np.log(close / close.shift(1)).rename("log_return")

    def realized_vol_20d(self, close: pd.Series) -> pd.Series:
        """20-day annualised realised volatility from log returns."""
        lr = np.log(close / close.shift(1))
        return (
            lr.rolling(window=self.vol_window, min_periods=self.vol_window).std()
            * np.sqrt(self.ann_factor)
        ).rename("realized_vol_20d")

    def garman_klass_vol(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Garman-Klass volatility estimator (20-day rolling, annualised)."""
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_) ** 2
        gk_daily = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
        return (
            np.sqrt(
                gk_daily.rolling(window=self.vol_window, min_periods=self.vol_window).mean()
                * self.ann_factor
            )
        ).rename("garman_klass_vol")

    def volume_ratio(self, volume: pd.Series) -> pd.Series:
        """Volume relative to its 20-day moving average."""
        avg20 = volume.rolling(window=20, min_periods=20).mean()
        return _safe_div(volume, avg20).rename("volume_ratio")

    def rel_strength_nifty(
        self, stock_ret: pd.Series, nifty_ret: pd.Series
    ) -> pd.Series:
        """Relative strength vs Nifty50."""
        return (stock_ret - nifty_ret).rename("rel_strength_nifty")

    def iv_percentile(self, iv: pd.Series) -> pd.Series:
        """Current IV rank over trailing 252-day window."""

        def _pctile(arr: np.ndarray) -> float:
            if len(arr) < 252:
                return np.nan
            current = arr[-1]
            return np.sum(arr[:-1] <= current) / (len(arr) - 1) * 100.0

        return (
            iv.rolling(window=252, min_periods=252)
            .apply(_pctile, raw=True)
            .rename("iv_percentile")
        )

    def vol_of_vol(self, close: pd.Series) -> pd.Series:
        """Std dev of 20-day realised vol over a 60-day outer window."""
        rvol = self.realized_vol_20d(close)
        return (
            rvol.rolling(window=self.vov_window, min_periods=self.vov_window).std()
        ).rename("vol_of_vol")


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """Orchestrates all feature builders into a single weekly dataset.

    Parameters
    ----------
    config : dict
        Merged configuration passed through to each sub-builder.  Must contain
        the keys required by ``ClimateFeatureBuilder``.  Optional keys for the
        other builders are passed transparently.
    """

    LAG_WEEKS: Sequence[int] = (2, 4, 6, 8)

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.climate = ClimateFeatureBuilder(config)
        self.agri = AgriculturalFeatureBuilder(config)
        self.market = MarketFeatureBuilder(config)
        logger.info("FeaturePipeline initialised")

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _to_weekly(df: pd.DataFrame, agg: str = "last") -> pd.DataFrame:
        """Resample a daily-indexed DataFrame to weekly frequency."""
        return df.resample("W").agg(agg)

    @staticmethod
    def _add_lags(
        df: pd.DataFrame, cols: List[str], lags: Sequence[int]
    ) -> pd.DataFrame:
        """Create lagged columns for *cols* at each lag in *lags*."""
        parts: List[pd.DataFrame] = [df]
        for lag in lags:
            shifted = df[cols].shift(lag)
            shifted.columns = [f"{c}_lag{lag}w" for c in cols]
            parts.append(shifted)
        return pd.concat(parts, axis=1)

    # -- public interface ----------------------------------------------------

    def build_all_features(
        self,
        rainfall_df: pd.DataFrame,
        stock_df: pd.DataFrame,
        cotton_df: pd.DataFrame,
        ndvi_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the full weekly feature matrix and binary target.

        Parameters
        ----------
        rainfall_df : DataFrame
            Daily rainfall with columns ``actual_cumul``, ``weekly_rain``,
            ``daily_rain``, ``onset_doy``, ``june_rain``.  DatetimeIndex.
        stock_df : DataFrame
            Daily OHLCV with ``Open``, ``High``, ``Low``, ``Close``, ``Volume``
            and ``nifty_ret``.  DatetimeIndex.
        cotton_df : DataFrame
            Daily MCX / ICE cotton prices: ``mcx_close``, ``ice_close``.
            DatetimeIndex.
        ndvi_df : DataFrame
            Weekly NDVI: column ``ndvi``.  DatetimeIndex.
        macro_df : DataFrame
            Reservoir data: ``current_storage``, ``full_capacity``.
            DatetimeIndex.

        Returns
        -------
        DataFrame
            Weekly features with ``high_vol_regime`` target column and no NaNs.
        """
        logger.info("Building full feature set …")

        # --- climate features (daily → weekly) ------------------------------
        clim: Dict[str, pd.Series] = {}
        clim["deficit_pct_cumulative"] = self.climate.deficit_pct_cumulative(
            rainfall_df["actual_cumul"]
        )
        clim["deficit_pct_rolling4w"] = self.climate.deficit_pct_rolling4w(
            rainfall_df["weekly_rain"]
        )
        clim["deficit_severity"] = self.climate.deficit_severity(
            clim["deficit_pct_cumulative"]
        )
        clim["dry_spell_length"] = self.climate.dry_spell_length(
            rainfall_df["daily_rain"]
        )
        clim["onset_delay_days"] = self.climate.onset_delay_days(
            rainfall_df["onset_doy"]
        )
        clim["june_deficit_flag"] = self.climate.june_deficit_flag(
            rainfall_df["june_rain"]
        )

        clim_df = pd.DataFrame(clim)
        # encode severity as ordinal for modelling
        severity_map = {"Normal": 0, "Deficient": 1, "Scanty": 2}
        clim_df["deficit_severity"] = (
            clim_df["deficit_severity"].map(severity_map).fillna(0).astype(int)
        )
        clim_weekly = self._to_weekly(clim_df)

        # --- agricultural features ------------------------------------------
        agri: Dict[str, pd.Series] = {}
        agri["ndvi_anomaly"] = self.agri.ndvi_anomaly(ndvi_df["ndvi"])
        agri["ndvi_trend_8wk"] = self.agri.ndvi_trend_8wk(ndvi_df["ndvi"])
        agri["cotton_futures_ret_30d"] = self.agri.cotton_futures_ret_30d(
            cotton_df["mcx_close"]
        )
        agri["cotton_futures_vol_20d"] = self.agri.cotton_futures_vol_20d(
            cotton_df["mcx_close"]
        )
        agri["cotton_basis"] = self.agri.cotton_basis(
            cotton_df["mcx_close"], cotton_df["ice_close"]
        )
        agri["reservoir_pct"] = self.agri.reservoir_pct(
            macro_df["current_storage"], macro_df["full_capacity"]
        )

        agri_df = pd.DataFrame(agri)
        agri_weekly = self._to_weekly(agri_df)

        # --- market features ------------------------------------------------
        mkt: Dict[str, pd.Series] = {}
        mkt["log_return"] = self.market.log_return(stock_df["Close"])
        mkt["realized_vol_20d"] = self.market.realized_vol_20d(stock_df["Close"])
        mkt["garman_klass_vol"] = self.market.garman_klass_vol(
            stock_df["Open"], stock_df["High"], stock_df["Low"], stock_df["Close"]
        )
        mkt["volume_ratio"] = self.market.volume_ratio(stock_df["Volume"])
        mkt["rel_strength_nifty"] = self.market.rel_strength_nifty(
            self.market.log_return(stock_df["Close"]),
            stock_df["nifty_ret"],
        )
        mkt["vol_of_vol"] = self.market.vol_of_vol(stock_df["Close"])

        # IV percentile only if iv column exists
        if "iv" in stock_df.columns:
            mkt["iv_percentile"] = self.market.iv_percentile(stock_df["iv"])

        mkt_df = pd.DataFrame(mkt)
        mkt_weekly = self._to_weekly(mkt_df)

        # --- merge all on weekly index --------------------------------------
        merged = (
            clim_weekly
            .join(agri_weekly, how="outer")
            .join(mkt_weekly, how="outer")
        )

        # --- lagged climate features ----------------------------------------
        climate_cols = [
            "deficit_pct_cumulative",
            "deficit_pct_rolling4w",
            "deficit_severity",
            "dry_spell_length",
            "onset_delay_days",
            "june_deficit_flag",
        ]
        present_clim_cols = [c for c in climate_cols if c in merged.columns]
        merged = self._add_lags(merged, present_clim_cols, self.LAG_WEEKS)

        # --- target: high-volatility regime ---------------------------------
        if "realized_vol_20d" in merged.columns:
            rolling_med = merged["realized_vol_20d"].rolling(window=52, min_periods=26).median()
            rolling_std = merged["realized_vol_20d"].rolling(window=52, min_periods=26).std()
            merged["high_vol_regime"] = (
                (merged["realized_vol_20d"] > rolling_med + rolling_std).astype(int)
            )
        else:
            logger.warning("realized_vol_20d missing; target column set to 0")
            merged["high_vol_regime"] = 0

        # --- drop NaN rows --------------------------------------------------
        n_before = len(merged)
        merged = merged.dropna()
        n_after = len(merged)
        logger.info(
            "Feature matrix ready: {} rows ({} dropped due to NaN)",
            n_after,
            n_before - n_after,
        )

        return merged
