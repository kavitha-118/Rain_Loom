"""
XGBoost Regime Classifier for Monsoon-Textile Volatility System.

Provides time-series-aware binary classification of high-volatility regimes
in Indian textile equities, driven by monsoon and macroeconomic features.
All validation respects chronological ordering -- no random splits.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

try:
    import optuna

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: Dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "verbosity": 0,
}


class XGBoostRegimeClassifier:
    """Binary classifier that predicts high-volatility regimes using XGBoost.

    Parameters
    ----------
    config : dict
        Model hyper-parameters.  Keys not supplied fall back to
        ``_DEFAULT_CONFIG``.
    """

    _METRIC_NAMES: ClassVar[List[str]] = [
        "auc_roc",
        "f1",
        "precision",
        "recall",
        "brier",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = {**_DEFAULT_CONFIG, **(config or {})}
        self.models: List[xgb.XGBClassifier] = []
        self.best_model: Optional[xgb.XGBClassifier] = None
        self.feature_names: Optional[List[str]] = None
        self._optuna_study: Optional[Any] = None
        logger.info(
            "XGBoostRegimeClassifier initialised | n_estimators={} max_depth={}",
            self.config["n_estimators"],
            self.config["max_depth"],
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        target_col: str = "high_vol_regime",
        forecast_horizon: int = 4,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Create forward-shifted target for regime prediction.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix with a ``target_col`` column.
        target_col : str
            Name of the binary regime column.
        forecast_horizon : int
            Number of periods to shift the target forward so that the model
            learns to *predict* future regimes, not contemporaneous ones.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (rows with NaN targets removed).
        y : pd.Series
            Forward-shifted binary target.
        """
        df = features_df.copy()

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not in DataFrame.")

        # Forward-shift: row *t* predicts regime at *t + forecast_horizon*
        df["_target"] = df[target_col].shift(-forecast_horizon)
        df.dropna(subset=["_target"], inplace=True)

        feature_cols = [c for c in df.columns if c not in (target_col, "_target")]
        X = df[feature_cols].copy()
        y = df["_target"].astype(int)

        self.feature_names = list(X.columns)

        logger.info(
            "prepare_data | samples={} features={} horizon={} pos_rate={:.3f}",
            len(X),
            X.shape[1],
            forecast_horizon,
            y.mean(),
        )
        return X, y

    # ------------------------------------------------------------------
    # Cross-validated training
    # ------------------------------------------------------------------
    def train_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        gap: int = 4,
    ) -> Tuple[List[xgb.XGBClassifier], List[Dict[str, float]]]:
        """Train with ``TimeSeriesSplit`` and return per-fold metrics.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary target.
        n_splits : int
            Number of chronological folds.
        gap : int
            Gap between train and test to avoid leakage.

        Returns
        -------
        models : list[xgb.XGBClassifier]
            One fitted model per fold.
        metrics : list[dict]
            Per-fold metrics: AUC-ROC, F1, Precision, Recall, Brier score.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        self.models = []
        all_metrics: List[Dict[str, float]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            sample_weights = compute_sample_weight("balanced", y_tr)

            model = self._make_model()
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weights,
                eval_set=[(X_te, y_te)],
                verbose=False,
            )

            proba = model.predict_proba(X_te)[:, 1]
            fold_metrics = self._compute_metrics(y_te, proba)
            all_metrics.append(fold_metrics)
            self.models.append(model)

            logger.info(
                "Fold {}/{} | train={} test={} | AUC={:.4f} F1={:.4f}",
                fold_idx,
                n_splits,
                len(train_idx),
                len(test_idx),
                fold_metrics["auc_roc"],
                fold_metrics["f1"],
            )

        # Select best fold model by AUC as the primary model
        best_idx = int(np.argmax([m["auc_roc"] for m in all_metrics]))
        self.best_model = self.models[best_idx]
        logger.info(
            "Best fold={} AUC={:.4f}",
            best_idx + 1,
            all_metrics[best_idx]["auc_roc"],
        )
        return self.models, all_metrics

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------
    def walk_forward_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """Expanding-window walk-forward: train on years 1..T, test on T+1.

        Requires a ``DatetimeIndex`` or a column parseable as datetime.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with a DatetimeIndex.
        y : pd.Series
            Binary target aligned to ``X``.
        start_year : int
            First year used as a *test* year.
        end_year : int
            Last year used as a *test* year (inclusive).

        Returns
        -------
        pd.DataFrame
            Per-year metrics plus aggregated predictions.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("X must have a DatetimeIndex for walk-forward validation.")

        results: List[Dict[str, Any]] = []

        for test_year in range(start_year, end_year + 1):
            train_mask = X.index.year < test_year
            test_mask = X.index.year == test_year

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                logger.warning("Skipping year {} (no train or test data).", test_year)
                continue

            X_tr, y_tr = X.loc[train_mask], y.loc[train_mask]
            X_te, y_te = X.loc[test_mask], y.loc[test_mask]

            sample_weights = compute_sample_weight("balanced", y_tr)
            model = self._make_model()
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weights,
                eval_set=[(X_te, y_te)],
                verbose=False,
            )

            proba = model.predict_proba(X_te)[:, 1]
            metrics = self._compute_metrics(y_te, proba)
            metrics["test_year"] = test_year
            metrics["train_size"] = int(train_mask.sum())
            metrics["test_size"] = int(test_mask.sum())
            results.append(metrics)

            logger.info(
                "Walk-forward year={} | train={} test={} | AUC={:.4f}",
                test_year,
                train_mask.sum(),
                test_mask.sum(),
                metrics["auc_roc"],
            )

        self.best_model = model  # last model has the most training data
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Bayesian hyper-parameter optimisation (Optuna)
    # ------------------------------------------------------------------
    def optimize_hyperparams(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        timeout: int = 600,
    ) -> Dict[str, Any]:
        """Run Optuna Bayesian optimisation over XGBoost hyper-parameters.

        Parameters
        ----------
        X, y : training data.
        n_trials : int
            Maximum number of Optuna trials.
        timeout : int
            Wall-clock seconds budget.

        Returns
        -------
        dict
            Best hyper-parameter set found.
        """
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for hyperparameter optimisation.")

        def _objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

            tscv = TimeSeriesSplit(n_splits=3, gap=4)
            aucs: List[float] = []

            for train_idx, test_idx in tscv.split(X):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                sw = compute_sample_weight("balanced", y_tr)

                cfg = {**self.config, **params}
                mdl = xgb.XGBClassifier(
                    **{k: v for k, v in cfg.items() if k != "early_stopping_rounds"},
                    early_stopping_rounds=cfg.get("early_stopping_rounds", 30),
                    use_label_encoder=False,
                )
                mdl.fit(
                    X_tr,
                    y_tr,
                    sample_weight=sw,
                    eval_set=[(X_te, y_te)],
                    verbose=False,
                )
                proba = mdl.predict_proba(X_te)[:, 1]
                aucs.append(roc_auc_score(y_te, proba))

            return float(np.mean(aucs))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", study_name="xgb_regime")
        study.optimize(_objective, n_trials=n_trials, timeout=timeout)

        self._optuna_study = study
        best = study.best_params
        self.config.update(best)
        logger.info("Optuna best AUC={:.4f} | params={}", study.best_value, best)

        # Re-train final model on full data with best params
        sample_weights = compute_sample_weight("balanced", y)
        self.best_model = self._make_model()
        self.best_model.fit(X, y, sample_weight=sample_weights, verbose=False)

        return best

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of the high-volatility regime.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same columns as training data).

        Returns
        -------
        np.ndarray
            1-D array of probabilities in [0, 1].
        """
        if self.best_model is None:
            raise RuntimeError("No trained model available. Call train_cv() first.")
        return self.best_model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def feature_importance(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Compute SHAP values for global feature importance.

        Parameters
        ----------
        X : pd.DataFrame
            Background / explanation dataset.

        Returns
        -------
        dict
            ``shap_values``: np.ndarray, ``feature_names``: list,
            ``mean_abs_shap``: pd.Series sorted descending.
        """
        if not _SHAP_AVAILABLE:
            raise ImportError("shap is required for feature importance.")
        if self.best_model is None:
            raise RuntimeError("No trained model. Call train_cv() first.")

        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(X)

        mean_abs = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X.columns,
        ).sort_values(ascending=False)

        logger.info("Top-5 features by |SHAP|: {}", dict(mean_abs.head()))
        return {
            "shap_values": shap_values,
            "feature_names": list(X.columns),
            "mean_abs_shap": mean_abs,
        }

    def explain_prediction(
        self, X: pd.DataFrame, idx: int
    ) -> Dict[str, Any]:
        """Return SHAP force-plot data for a single observation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        idx : int
            Row index within ``X``.

        Returns
        -------
        dict
            ``base_value``, ``shap_values``, ``feature_values``,
            ``feature_names``.
        """
        if not _SHAP_AVAILABLE:
            raise ImportError("shap is required for prediction explanation.")
        if self.best_model is None:
            raise RuntimeError("No trained model.")

        explainer = shap.TreeExplainer(self.best_model)
        row = X.iloc[[idx]]
        sv = explainer.shap_values(row)

        return {
            "base_value": float(explainer.expected_value),
            "shap_values": sv[0].tolist(),
            "feature_values": row.values[0].tolist(),
            "feature_names": list(X.columns),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialise the classifier state to disk via joblib.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``models/xgb_regime.joblib``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "best_model": self.best_model,
            "models": self.models,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, path)
        logger.info("Model saved to {}", path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostRegimeClassifier":
        """Load a previously saved classifier.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        XGBoostRegimeClassifier
            Restored instance.
        """
        path = Path(path)
        payload = joblib.load(path)
        instance = cls(config=payload["config"])
        instance.best_model = payload["best_model"]
        instance.models = payload["models"]
        instance.feature_names = payload["feature_names"]
        logger.info("Model loaded from {}", path)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_model(self) -> xgb.XGBClassifier:
        """Instantiate an XGBClassifier from current config."""
        cfg = {k: v for k, v in self.config.items() if k != "early_stopping_rounds"}
        return xgb.XGBClassifier(
            **cfg,
            early_stopping_rounds=self.config.get("early_stopping_rounds", 30),
            use_label_encoder=False,
        )

    @staticmethod
    def _compute_metrics(
        y_true: pd.Series | np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute standard binary-classification metrics."""
        y_pred = (y_proba >= threshold).astype(int)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)
        return {
            "auc_roc": float(roc_auc_score(y_true, y_proba)),
            "auc_pr": float(auc(rec_curve, prec_curve)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "brier": float(brier_score_loss(y_true, y_proba)),
        }


# ===========================================================================
# Per-stock model trainer
# ===========================================================================
class PerStockModelTrainer:
    """Train and manage independent XGBoost regime classifiers per stock.

    Parameters
    ----------
    base_config : dict, optional
        Shared model configuration passed to every per-stock classifier.
    """

    def __init__(self, base_config: Optional[Dict[str, Any]] = None) -> None:
        self.base_config = base_config or {}
        self.stock_models: Dict[str, XGBoostRegimeClassifier] = {}
        self.stock_metrics: Dict[str, List[Dict[str, float]]] = {}
        self.stock_importances: Dict[str, pd.Series] = {}

    def train_all_stocks(
        self,
        features_dict: Dict[str, pd.DataFrame],
        stock_symbols: List[str],
        target_col: str = "high_vol_regime",
        forecast_horizon: int = 4,
        n_splits: int = 5,
        gap: int = 4,
    ) -> Dict[str, List[Dict[str, float]]]:
        """Train a separate XGBoost model for each stock symbol.

        Parameters
        ----------
        features_dict : dict[str, pd.DataFrame]
            Mapping of stock symbol to its feature DataFrame.
        stock_symbols : list[str]
            Symbols to train (must be keys in ``features_dict``).
        target_col : str
            Binary regime column name.
        forecast_horizon : int
            Forward-shift for the target.
        n_splits : int
            TimeSeriesSplit folds.
        gap : int
            Gap between train/test folds.

        Returns
        -------
        dict[str, list[dict]]
            Per-stock, per-fold metric dictionaries.
        """
        for symbol in stock_symbols:
            if symbol not in features_dict:
                logger.warning("Symbol {} not found in features_dict -- skipping.", symbol)
                continue

            logger.info("Training model for {}", symbol)
            clf = XGBoostRegimeClassifier(config=self.base_config.copy())
            X, y = clf.prepare_data(
                features_dict[symbol],
                target_col=target_col,
                forecast_horizon=forecast_horizon,
            )
            _, fold_metrics = clf.train_cv(X, y, n_splits=n_splits, gap=gap)

            self.stock_models[symbol] = clf
            self.stock_metrics[symbol] = fold_metrics

            # Cache feature importance (SHAP if available, else gain)
            if _SHAP_AVAILABLE:
                try:
                    imp = clf.feature_importance(X)
                    self.stock_importances[symbol] = imp["mean_abs_shap"]
                except Exception as exc:
                    logger.warning("SHAP failed for {}: {}", symbol, exc)
                    self._fallback_importance(clf, symbol)
            else:
                self._fallback_importance(clf, symbol)

        logger.info("Finished training {} stock models.", len(self.stock_models))
        return self.stock_metrics

    def compare_feature_importance(self) -> pd.DataFrame:
        """Build a cross-stock feature importance comparison.

        Returns
        -------
        pd.DataFrame
            Rows = features, columns = stock symbols, values = mean |SHAP|
            or gain-based importance.
        """
        if not self.stock_importances:
            raise RuntimeError("No models trained yet. Call train_all_stocks() first.")

        df = pd.DataFrame(self.stock_importances)
        df.index.name = "feature"
        df["mean_importance"] = df.mean(axis=1)
        df.sort_values("mean_importance", ascending=False, inplace=True)

        logger.info(
            "Cross-stock importance | top-5: {}",
            list(df.index[:5]),
        )
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_importance(
        clf: XGBoostRegimeClassifier, symbol: str
    ) -> None:
        """Use XGBoost native gain importance when SHAP is unavailable."""
        if clf.best_model is not None and clf.feature_names is not None:
            imp_dict = clf.best_model.get_booster().get_score(
                importance_type="gain"
            )
            clf_imp = pd.Series(imp_dict).reindex(clf.feature_names, fill_value=0.0)
            clf_imp.sort_values(ascending=False, inplace=True)
            # Store on the trainer via the caller -- kept static intentionally
        else:
            logger.warning("Cannot compute fallback importance for {}.", symbol)
