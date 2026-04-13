"""
Monsoon-Textile Volatility System — Data Collection Pipeline.

Production-grade pipeline that ingests IMD gridded rainfall, NSE equity OHLCV,
MCX/ICE cotton futures, and macro indicators, then merges everything into a
single weekly-frequency Parquet dataset with quality diagnostics.
"""

from __future__ import annotations

import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from statsmodels.tsa.stattools import adfuller

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "settings.yaml"
_OUTPUT_DIR = _PROJECT_ROOT / "monsoon_textile_app" / "data" / "output"

_TRADING_DAYS_PER_YEAR = 252
_WEEKS_PER_YEAR = 52


# ---------------------------------------------------------------------------
# Retry decorator for network calls
# ---------------------------------------------------------------------------
def _retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable:
    """Exponential-backoff retry decorator."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    wait = backoff_factor ** attempt
                    logger.warning(
                        "{} failed (attempt {}/{}): {} — retrying in {:.1f}s",
                        fn.__name__,
                        attempt,
                        max_retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
            raise RuntimeError(
                f"{fn.__name__} failed after {max_retries} retries"
            ) from last_exc

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    """Load and expose settings from a YAML configuration file."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else _DEFAULT_CONFIG
        if not self._path.exists():
            raise FileNotFoundError(f"Config not found: {self._path}")
        with open(self._path, "r", encoding="utf-8") as fh:
            self._raw: Dict[str, Any] = yaml.safe_load(fh)
        logger.info("Config loaded from {}", self._path)

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested key access: ``cfg.get('data', 'imd', 'start_year')``."""
        node: Any = self._raw
        for k in keys:
            if isinstance(node, dict):
                node = node.get(k)
            else:
                return default
            if node is None:
                return default
        return node


# ---------------------------------------------------------------------------
# IMD Rainfall Loader
# ---------------------------------------------------------------------------
class IMDDataLoader:
    """Download/load IMD gridded rainfall and derive monsoon deficit features."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._start_year: int = cfg.get("data", "imd", "start_year", default=2000)
        self._end_year: int = cfg.get("data", "imd", "end_year", default=2025)
        self._resolution: float = cfg.get("data", "imd", "resolution", default=0.25)
        self._monsoon_months: List[int] = cfg.get(
            "data", "imd", "monsoon_months", default=[6, 7, 8, 9]
        )
        self._districts = self._parse_districts(
            cfg.get("data", "imd", "key_districts", default={})
        )
        self._dry_spell_threshold: float = cfg.get(
            "features", "climate", "dry_spell_threshold_mm", default=2.0
        )
        self._deficit_threshold_pct: float = cfg.get(
            "features", "climate", "deficit_threshold_pct", default=-20.0
        )

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _parse_districts(raw: Dict[str, List]) -> List[Dict[str, Any]]:
        """Convert the flat config list into structured district records."""
        districts: List[Dict[str, Any]] = []
        for state, vals in raw.items():
            i = 0
            while i < len(vals):
                lat = float(vals[i])
                lon = float(vals[i + 1])
                name = str(vals[i + 2])
                districts.append(
                    {"state": state, "name": name, "lat": lat, "lon": lon}
                )
                i += 3
        return districts

    @_retry(max_retries=3)
    def _download_imd_data(self) -> Any:
        """Download IMD gridded rainfall using imdlib."""
        import imdlib as imd

        logger.info(
            "Downloading IMD rain data {}-{} @ {}°",
            self._start_year,
            self._end_year,
            self._resolution,
        )
        data = imd.get_data(
            "rain",
            self._start_year,
            self._end_year,
            fn_format="yearwise",
            file_dir=str(_OUTPUT_DIR / "imd_cache"),
        )
        return data

    def _extract_district_series(
        self, data: Any
    ) -> Dict[str, pd.Series]:
        """Extract daily rainfall series for each cotton-belt district."""
        import imdlib as imd

        district_series: Dict[str, pd.Series] = {}
        ds = data.get_xarray()

        for d in self._districts:
            key = f"{d['state']}_{d['name']}"
            try:
                ts = ds.sel(lat=d["lat"], lon=d["lon"], method="nearest")
                s = ts["rain"].to_series()
                s.name = key
                s = s.replace(-999.0, np.nan)
                district_series[key] = s
                logger.debug("Extracted {} ({} records)", key, len(s))
            except Exception as exc:
                logger.error("Failed extracting {}: {}", key, exc)
        return district_series

    def _compute_weekly_totals(
        self, daily: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Resample daily rainfall to weekly totals."""
        frames = []
        for key, s in daily.items():
            weekly = s.resample("W-SUN").sum(min_count=5)
            weekly.name = f"rain_weekly_{key}"
            frames.append(weekly)
        return pd.concat(frames, axis=1)

    def _compute_jjas_deficit(
        self, daily: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Compute cumulative JJAS deficit % relative to LPA for each district."""
        records: List[pd.DataFrame] = []
        for key, s in daily.items():
            jjas = s[s.index.month.isin(self._monsoon_months)]
            annual_total = jjas.resample("YS-JUN").sum()
            lpa = annual_total.mean()
            if lpa == 0 or np.isnan(lpa):
                logger.warning("LPA is zero/NaN for {}, skipping deficit", key)
                continue
            deficit_pct = ((annual_total - lpa) / lpa) * 100
            deficit_pct.name = f"jjas_deficit_pct_{key}"
            records.append(deficit_pct.to_frame())
        if not records:
            return pd.DataFrame()
        return pd.concat(records, axis=1)

    def _compute_cumulative_jjas_deficit_weekly(
        self, daily: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Running cumulative deficit % within each JJAS season, at weekly freq."""
        frames: List[pd.Series] = []
        for key, s in daily.items():
            jjas = s[s.index.month.isin(self._monsoon_months)].copy()
            if jjas.empty:
                continue
            jjas_year = jjas.groupby(jjas.index.year)
            lpa_daily = jjas.groupby(jjas.index.dayofyear).mean()

            cumul_list = []
            for year, grp in jjas_year:
                actual_cum = grp.cumsum()
                doys = grp.index.dayofyear
                expected_cum = lpa_daily.reindex(doys).cumsum().values
                expected_cum_safe = np.where(expected_cum == 0, np.nan, expected_cum)
                deficit_pct = ((actual_cum.values - expected_cum) / expected_cum_safe) * 100
                cumul_list.append(
                    pd.Series(deficit_pct, index=grp.index, name=key)
                )
            if cumul_list:
                full = pd.concat(cumul_list)
                weekly = full.resample("W-SUN").last()
                weekly.name = f"cum_jjas_deficit_{key}"
                frames.append(weekly)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    def _compute_spatial_deficit_breadth(
        self, weekly_rain: pd.DataFrame
    ) -> pd.Series:
        """Fraction of districts with weekly rainfall below the dry-spell threshold."""
        n_districts = weekly_rain.shape[1]
        if n_districts == 0:
            return pd.Series(dtype=float, name="spatial_deficit_breadth")
        below = (weekly_rain < self._dry_spell_threshold).sum(axis=1) / n_districts
        below.name = "spatial_deficit_breadth"
        return below

    def _compute_dry_spell_lengths(
        self, daily: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Maximum consecutive dry days (< threshold) ending in each week, per district."""
        frames: List[pd.Series] = []
        for key, s in daily.items():
            is_dry = (s < self._dry_spell_threshold).astype(int)
            streaks = is_dry.groupby((is_dry != is_dry.shift()).cumsum()).cumsum()
            weekly_max = streaks.resample("W-SUN").max()
            weekly_max.name = f"dry_spell_{key}"
            frames.append(weekly_max)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    def load(self) -> pd.DataFrame:
        """Run the full IMD loading and feature extraction pipeline."""
        logger.info("IMDDataLoader: starting load")
        data = self._download_imd_data()
        daily = self._extract_district_series(data)

        if not daily:
            logger.error("No district series extracted — returning empty DataFrame")
            return pd.DataFrame()

        weekly_rain = self._compute_weekly_totals(daily)
        cum_deficit = self._compute_cumulative_jjas_deficit_weekly(daily)
        breadth = self._compute_spatial_deficit_breadth(weekly_rain)
        dry_spells = self._compute_dry_spell_lengths(daily)

        result = pd.concat(
            [weekly_rain, cum_deficit, breadth, dry_spells], axis=1
        )
        result.index.name = "date"
        logger.info("IMDDataLoader: produced {} rows x {} cols", *result.shape)
        return result


# ---------------------------------------------------------------------------
# NSE Equity Loader
# ---------------------------------------------------------------------------
class NSEDataLoader:
    """Download NSE textile stocks and compute volatility / return features."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._stocks: Dict[str, Dict[str, Any]] = cfg.get(
            "data", "nse", "stocks", default={}
        )
        self._benchmark: str = cfg.get("data", "nse", "benchmark", default="^NSEI")
        self._vix: str = cfg.get("data", "nse", "vix", default="^INDIAVIX")
        self._start: str = cfg.get("data", "nse", "start_date", default="2005-01-01")
        self._rv_window: int = cfg.get(
            "features", "volatility", "realized_window", default=20
        )
        self._gk_window: int = cfg.get(
            "features", "volatility", "garman_klass_window", default=20
        )

    @_retry(max_retries=3)
    def _download(self, ticker: str) -> pd.DataFrame:
        """Download OHLCV via yfinance."""
        import yfinance as yf

        logger.debug("Downloading {}", ticker)
        df = yf.download(
            ticker,
            start=self._start,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def _log_returns(close: pd.Series) -> pd.Series:
        return np.log(close / close.shift(1))

    def _realized_vol(self, log_ret: pd.Series, window: int) -> pd.Series:
        return log_ret.rolling(window).std() * np.sqrt(_TRADING_DAYS_PER_YEAR)

    @staticmethod
    def _garman_klass_vol(
        df: pd.DataFrame, window: int
    ) -> pd.Series:
        """Garman-Klass (1980) intraday volatility estimator."""
        log_hl = (np.log(df["High"] / df["Low"])) ** 2
        log_co = (np.log(df["Close"] / df["Open"])) ** 2
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return gk.rolling(window).mean().apply(np.sqrt) * np.sqrt(_TRADING_DAYS_PER_YEAR)

    @staticmethod
    def _volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
        return volume / volume.rolling(window).mean()

    def _process_stock(
        self, ticker: str, bench_close: pd.Series
    ) -> pd.DataFrame:
        """Download and compute all features for a single stock."""
        df = self._download(ticker)
        prefix = ticker.replace(".", "_").replace("^", "").replace("=", "")

        log_ret = self._log_returns(df["Close"])
        rv = self._realized_vol(log_ret, self._rv_window)
        gk = self._garman_klass_vol(df, self._gk_window)
        vr = self._volume_ratio(df["Volume"], self._rv_window)

        bench_ret = self._log_returns(bench_close).reindex(log_ret.index)
        rs = log_ret.rolling(self._rv_window).mean() - bench_ret.rolling(self._rv_window).mean()

        features = pd.DataFrame(
            {
                f"{prefix}_close": df["Close"],
                f"{prefix}_log_ret": log_ret,
                f"{prefix}_rv20": rv,
                f"{prefix}_gk_vol": gk,
                f"{prefix}_vol_ratio": vr,
                f"{prefix}_rel_strength": rs,
            },
            index=df.index,
        )
        return features

    def load(self) -> pd.DataFrame:
        """Load all NSE tickers and return a combined daily DataFrame."""
        logger.info("NSEDataLoader: starting load")

        bench_df = self._download(self._benchmark)
        bench_close = bench_df["Close"].squeeze()

        vix_df = self._download(self._vix)
        vix_series = vix_df["Close"].squeeze()
        vix_series.name = "india_vix"

        frames: List[pd.DataFrame] = []
        for ticker in self._stocks:
            try:
                frames.append(self._process_stock(ticker, bench_close))
            except Exception as exc:
                logger.error("Skipping {}: {}", ticker, exc)

        bench_features = pd.DataFrame(
            {
                "nifty_close": bench_close,
                "nifty_log_ret": self._log_returns(bench_close),
            },
            index=bench_df.index,
        )
        frames.append(bench_features)
        frames.append(vix_series.to_frame())

        result = pd.concat(frames, axis=1)
        result.index.name = "date"
        logger.info("NSEDataLoader: produced {} rows x {} cols", *result.shape)
        return result


# ---------------------------------------------------------------------------
# MCX Cotton Futures Loader
# ---------------------------------------------------------------------------
class MCXCottonLoader:
    """Download ICE Cotton No.2 (proxy for MCX) and compute vol features."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._symbol: str = cfg.get("data", "mcx", "global_alt_symbol", default="CT=F")
        self._rv_window: int = cfg.get(
            "features", "volatility", "realized_window", default=20
        )

    @_retry(max_retries=3)
    def _download(self) -> pd.DataFrame:
        import yfinance as yf

        logger.debug("Downloading cotton futures: {}", self._symbol)
        df = yf.download(self._symbol, start="2005-01-01", auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data for {self._symbol}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def load(self) -> pd.DataFrame:
        """Return cotton futures features at daily frequency."""
        logger.info("MCXCottonLoader: starting load")
        df = self._download()
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        rv = log_ret.rolling(self._rv_window).std() * np.sqrt(_TRADING_DAYS_PER_YEAR)
        log_ret_30 = np.log(df["Close"] / df["Close"].shift(30))

        result = pd.DataFrame(
            {
                "cotton_close": df["Close"],
                "cotton_log_ret": log_ret,
                "cotton_log_ret_30d": log_ret_30,
                "cotton_rv20": rv,
            },
            index=df.index,
        )
        result.index.name = "date"
        logger.info("MCXCottonLoader: produced {} rows x {} cols", *result.shape)
        return result


# ---------------------------------------------------------------------------
# Macro Data Loader
# ---------------------------------------------------------------------------
class MacroDataLoader:
    """Download macro indicators: USD/INR, Brent Crude, Nifty50, India VIX."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._symbols: Dict[str, str] = cfg.get(
            "data", "macro", "symbols", default={}
        )

    @_retry(max_retries=3)
    def _download(self, ticker: str, name: str) -> pd.Series:
        import yfinance as yf

        logger.debug("Downloading macro: {} ({})", name, ticker)
        df = yf.download(ticker, start="2005-01-01", auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        close = df["Close"].squeeze()
        close.name = name
        return close

    def load(self) -> pd.DataFrame:
        """Return macro close prices at daily frequency."""
        logger.info("MacroDataLoader: starting load")
        series: List[pd.Series] = []
        for name, ticker in self._symbols.items():
            try:
                series.append(self._download(ticker, f"macro_{name}"))
            except Exception as exc:
                logger.error("Skipping macro {}: {}", name, exc)

        if not series:
            return pd.DataFrame()
        result = pd.concat(series, axis=1)
        result.index.name = "date"
        logger.info("MacroDataLoader: produced {} rows x {} cols", *result.shape)
        return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class DataPipeline:
    """Orchestrate all loaders, merge to weekly frequency, validate, and persist."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._output_dir = _OUTPUT_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self.imd_loader = IMDDataLoader(cfg)
        self.nse_loader = NSEDataLoader(cfg)
        self.mcx_loader = MCXCottonLoader(cfg)
        self.macro_loader = MacroDataLoader(cfg)

    # ---- merge ------------------------------------------------------------
    @staticmethod
    def _to_weekly(df: pd.DataFrame, agg: str = "last") -> pd.DataFrame:
        """Resample a daily-frequency DataFrame to weekly (Sunday end)."""
        if df.empty:
            return df
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        if agg == "last":
            return df.resample("W-SUN").last()
        return df.resample("W-SUN").mean()

    def _merge(
        self,
        imd: pd.DataFrame,
        nse: pd.DataFrame,
        mcx: pd.DataFrame,
        macro: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge all sources on weekly DatetimeIndex."""
        nse_w = self._to_weekly(nse)
        mcx_w = self._to_weekly(mcx)
        macro_w = self._to_weekly(macro)

        merged = pd.concat([imd, nse_w, mcx_w, macro_w], axis=1)
        merged.index.name = "date"
        merged.sort_index(inplace=True)
        logger.info("Merged dataset: {} rows x {} cols", *merged.shape)
        return merged

    # ---- quality checks ---------------------------------------------------
    @staticmethod
    def _missing_report(df: pd.DataFrame) -> pd.DataFrame:
        """Return per-column missing value statistics."""
        total = len(df)
        missing = df.isnull().sum()
        pct = (missing / total * 100).round(2)
        report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
        report = report[report.missing_count > 0].sort_values(
            "missing_pct", ascending=False
        )
        return report

    @staticmethod
    def _adf_stationarity(
        df: pd.DataFrame, significance: float = 0.05
    ) -> pd.DataFrame:
        """Run Augmented Dickey-Fuller on each numeric column."""
        results: List[Dict[str, Any]] = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 20:
                continue
            try:
                stat, pval, usedlag, nobs, crit, _ = adfuller(series, maxlag=12)
                results.append(
                    {
                        "column": col,
                        "adf_stat": round(stat, 4),
                        "p_value": round(pval, 6),
                        "used_lag": usedlag,
                        "n_obs": nobs,
                        "stationary": pval < significance,
                    }
                )
            except Exception as exc:
                logger.warning("ADF failed for {}: {}", col, exc)
        return pd.DataFrame(results)

    def run_quality_checks(self, df: pd.DataFrame) -> None:
        """Log missing-value and stationarity diagnostics."""
        logger.info("--- Data Quality Report ---")

        miss = self._missing_report(df)
        if miss.empty:
            logger.info("No missing values detected")
        else:
            logger.info("Missing values:\n{}", miss.to_string())

        adf = self._adf_stationarity(df)
        if adf.empty:
            logger.warning("ADF tests could not be run (insufficient data)")
        else:
            non_stat = adf[~adf["stationary"]]
            logger.info(
                "ADF stationarity: {}/{} columns stationary at 5%",
                adf["stationary"].sum(),
                len(adf),
            )
            if not non_stat.empty:
                logger.info(
                    "Non-stationary columns:\n{}",
                    non_stat[["column", "adf_stat", "p_value"]].to_string(index=False),
                )

        # Persist diagnostics alongside the dataset
        miss.to_csv(self._output_dir / "quality_missing.csv")
        adf.to_csv(self._output_dir / "quality_adf.csv", index=False)

    # ---- run --------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Execute the full pipeline: load, merge, check quality, save."""
        logger.info("=== DataPipeline: starting ===")

        imd_df = self.imd_loader.load()
        nse_df = self.nse_loader.load()
        mcx_df = self.mcx_loader.load()
        macro_df = self.macro_loader.load()

        merged = self._merge(imd_df, nse_df, mcx_df, macro_df)
        self.run_quality_checks(merged)

        out_path = self._output_dir / "monsoon_textile_weekly.parquet"
        merged.to_parquet(out_path, engine="pyarrow")
        logger.info("Saved merged dataset to {}", out_path)

        logger.info("=== DataPipeline: complete ===")
        return merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(config_path: Optional[str] = None) -> pd.DataFrame:
    """Run the full data pipeline end-to-end.

    Args:
        config_path: Override path to settings.yaml. Uses default if None.

    Returns:
        Merged weekly DataFrame.
    """
    import sys

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    logger.add(
        _OUTPUT_DIR / "pipeline.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
    )

    cfg_path = Path(config_path) if config_path else None
    cfg = Config(cfg_path)
    pipeline = DataPipeline(cfg)
    return pipeline.run()


if __name__ == "__main__":
    main()
