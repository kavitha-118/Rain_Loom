"""
Concept Drift Detection for Monsoon-Textile ML System.

Provides online change-detection algorithms (Page-Hinkley, ADWIN) and a
model-health monitor that tracks rolling classification metrics, detects
performance drift, and flags degradation via Kolmogorov-Smirnov tests on
predicted risk-score distributions.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ---------------------------------------------------------------------------
# Page-Hinkley Test
# ---------------------------------------------------------------------------

class PageHinkleyTest:
    """Online change-detection algorithm (Page-Hinkley).

    Monitors the running mean of a stream of values and signals drift when
    the cumulative deviation from the mean exceeds a user-defined threshold.

    Parameters
    ----------
    threshold : float
        Maximum allowed cumulative deviation before drift is signalled.
    alpha : float
        Minimum magnitude of change to consider (tolerance term added at
        each step to avoid false positives from noise).
    """

    def __init__(self, threshold: float = 50.0, alpha: float = 0.005) -> None:
        self.threshold = threshold
        self.alpha = alpha
        self._reset_state()

    # -- public interface ---------------------------------------------------

    def update(self, value: float) -> bool:
        """Ingest a new observation and return *True* if drift is detected."""
        self._n += 1
        # Running mean (numerically stable incremental update).
        self._running_mean += (value - self._running_mean) / self._n

        # Cumulative sum of deviations from the running mean, dampened by
        # the tolerance *alpha*.
        self._cumulative_sum += value - self._running_mean - self.alpha

        # Track the running minimum of the cumulative sum.
        if self._cumulative_sum < self._min_cumulative_sum:
            self._min_cumulative_sum = self._cumulative_sum

        # Drift criterion: the gap between the current cumulative sum and
        # its historical minimum exceeds the threshold.
        if self._cumulative_sum - self._min_cumulative_sum > self.threshold:
            self._drift_detected = True
            self._drift_index = self._n
            logger.info(
                "Page-Hinkley drift detected at observation {}", self._n
            )
            return True

        return False

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self._reset_state()
        logger.debug("Page-Hinkley detector reset")

    # -- properties ---------------------------------------------------------

    @property
    def drift_detected(self) -> bool:
        """Whether drift has been detected since the last reset."""
        return self._drift_detected

    @property
    def drift_index(self) -> Optional[int]:
        """Observation index at which drift was first detected, or *None*."""
        return self._drift_index

    @property
    def cumulative_sum(self) -> float:
        """Current value of the cumulative deviation sum."""
        return self._cumulative_sum

    # -- internals ----------------------------------------------------------

    def _reset_state(self) -> None:
        self._n: int = 0
        self._running_mean: float = 0.0
        self._cumulative_sum: float = 0.0
        self._min_cumulative_sum: float = float("inf")
        self._drift_detected: bool = False
        self._drift_index: Optional[int] = None


# ---------------------------------------------------------------------------
# ADWIN Detector
# ---------------------------------------------------------------------------

class ADWINDetector:
    """Adaptive Windowing (ADWIN) drift detector.

    Maintains a variable-length window of recent observations using a
    bucket-based compression scheme.  When the difference between the means
    of two sub-windows is statistically significant (controlled by *delta*),
    the older portion is dropped and drift is signalled.

    Parameters
    ----------
    delta : float
        Confidence parameter for the cut test.  Smaller values make the
        detector less sensitive (fewer false positives, slower reaction).
    """

    # Bucket growth factor -- each level doubles in size.
    _MAX_BUCKETS_PER_LEVEL: int = 5

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self._reset_state()

    # -- public interface ---------------------------------------------------

    def update(self, value: float) -> bool:
        """Add *value* to the window; return *True* if drift is detected."""
        self._insert(value)
        drift = self._check_drift()
        if drift:
            self._drift_detected = True
            logger.info(
                "ADWIN drift detected – window shrunk to {} observations",
                self._total_n,
            )
        return drift

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self._reset_state()
        logger.debug("ADWIN detector reset")

    # -- properties ---------------------------------------------------------

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def window_size(self) -> int:
        """Number of observations currently in the window."""
        return self._total_n

    @property
    def mean(self) -> float:
        """Mean of the current window."""
        if self._total_n == 0:
            return 0.0
        return self._total_sum / self._total_n

    # -- internals ----------------------------------------------------------

    def _reset_state(self) -> None:
        # Each bucket stores (count, total, variance_times_count).
        # Buckets are grouped into levels; level *k* holds buckets whose
        # capacity is 2^k.  We keep at most _MAX_BUCKETS_PER_LEVEL per level.
        self._buckets: List[List[List[float]]] = []  # [level][bucket_idx] -> [n, sum, var*n]
        self._total_n: int = 0
        self._total_sum: float = 0.0
        self._total_var: float = 0.0  # sum of (var*n) across all buckets
        self._drift_detected: bool = False

    def _insert(self, value: float) -> None:
        """Insert a single observation as a new level-0 bucket."""
        # Level-0 bucket: single observation.
        if len(self._buckets) == 0:
            self._buckets.append([])
        self._buckets[0].append([1.0, value, 0.0])
        self._total_n += 1
        self._total_sum += value

        # Compress: merge oldest buckets when a level overflows.
        self._compress()

    def _compress(self) -> None:
        """Merge buckets that exceed the per-level limit."""
        for level in range(len(self._buckets)):
            if len(self._buckets[level]) <= self._MAX_BUCKETS_PER_LEVEL:
                break
            # Merge the two oldest (last) buckets at this level and push to
            # the next level.
            b1 = self._buckets[level].pop()
            b2 = self._buckets[level].pop()
            merged = self._merge_buckets(b1, b2)

            if level + 1 >= len(self._buckets):
                self._buckets.append([])
            # Insert at the *end* (oldest position) of the next level.
            self._buckets[level + 1].append(merged)

    @staticmethod
    def _merge_buckets(
        b1: List[float], b2: List[float]
    ) -> List[float]:
        n1, s1, v1 = b1
        n2, s2, v2 = b2
        n = n1 + n2
        s = s1 + s2
        # Combined variance * n (parallel / Welford merge).
        if n > 0:
            v = v1 + v2 + (n1 * n2 / n) * ((s1 / max(n1, 1e-15)) - (s2 / max(n2, 1e-15))) ** 2
        else:
            v = 0.0
        return [n, s, v]

    def _check_drift(self) -> bool:
        """Scan sub-window cuts from newest to oldest; drop tail on drift."""
        if self._total_n < 2:
            return False

        # Walk from the newest bucket toward the oldest, accumulating a
        # "recent" sub-window.  Everything not yet accumulated is "old".
        n_recent = 0.0
        sum_recent = 0.0

        for level in range(len(self._buckets)):
            for idx in range(len(self._buckets[level])):
                bn, bs, _ = self._buckets[level][idx]
                n_recent += bn
                sum_recent += bs

                n_old = self._total_n - n_recent
                if n_recent < 1 or n_old < 1:
                    continue

                sum_old = self._total_sum - sum_recent
                mean_recent = sum_recent / n_recent
                mean_old = sum_old / n_old

                # Hoeffding-style bound for the difference of two means.
                n_harmonic = 1.0 / n_recent + 1.0 / n_old
                m = 1.0 / n_harmonic  # harmonic "sample size"
                epsilon = np.sqrt(
                    (1.0 / (2.0 * m)) * np.log(4.0 / self.delta)
                )

                if abs(mean_recent - mean_old) >= epsilon:
                    # Drift found – drop the old portion.
                    self._drop_old(n_recent, sum_recent)
                    return True

        return False

    def _drop_old(self, keep_n: float, keep_sum: float) -> None:
        """Shrink the window by removing all buckets in the old portion.

        For simplicity we rebuild the totals from what we decided to keep.
        This is an approximation consistent with the compressed representation.
        """
        # Rebuild: walk from newest, keep buckets until we cover keep_n.
        new_buckets: List[List[List[float]]] = []
        remaining = keep_n
        for level in range(len(self._buckets)):
            level_kept: List[List[float]] = []
            for idx in range(len(self._buckets[level])):
                bn = self._buckets[level][idx][0]
                if remaining >= bn:
                    level_kept.append(self._buckets[level][idx])
                    remaining -= bn
                if remaining <= 0:
                    break
            if level_kept:
                new_buckets.append(level_kept)
            else:
                new_buckets.append([])
            if remaining <= 0:
                break

        self._buckets = new_buckets
        self._total_n = int(keep_n)
        self._total_sum = keep_sum


# ---------------------------------------------------------------------------
# Model Health Monitor
# ---------------------------------------------------------------------------

class ModelHealthMonitor:
    """Track ML-model classification performance over time and flag drift.

    Maintains rolling windows of ground-truth labels, predictions, and
    (optionally) predicted probabilities.  Periodically computes AUC-ROC,
    F1, and accuracy, then feeds those into Page-Hinkley detectors.  A
    Kolmogorov-Smirnov test compares recent vs. historical predicted-
    probability distributions to detect covariate / concept shift.

    Parameters
    ----------
    window_size : int
        Number of most-recent predictions kept for rolling metric
        calculation and distribution comparison.
    """

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size

        # Prediction buffers.
        self._y_true: deque[int] = deque(maxlen=window_size * 2)
        self._y_pred: deque[int] = deque(maxlen=window_size * 2)
        self._y_prob: deque[Optional[float]] = deque(maxlen=window_size * 2)
        self._timestamps: deque[float] = deque(maxlen=window_size * 2)

        # Rolling metric history (unbounded – summarises entire lifetime).
        self._rolling_auc: List[float] = []
        self._rolling_f1: List[float] = []
        self._rolling_acc: List[float] = []
        self._metric_ts: List[float] = []

        # Drift detectors for key metrics.
        self._auc_detector = PageHinkleyTest(threshold=30, alpha=0.005)
        self._f1_detector = PageHinkleyTest(threshold=30, alpha=0.005)

        self._prediction_count: int = 0
        logger.debug(
            "ModelHealthMonitor initialised (window_size={})", window_size
        )

    # -- recording ----------------------------------------------------------

    def record_prediction(
        self,
        y_true: int,
        y_pred: int,
        y_prob: Optional[float] = None,
    ) -> None:
        """Record a single prediction and update rolling metrics.

        Parameters
        ----------
        y_true : int
            Ground-truth label (0 or 1).
        y_pred : int
            Predicted label (0 or 1).
        y_prob : float, optional
            Predicted probability of the positive class.
        """
        self._y_true.append(int(y_true))
        self._y_pred.append(int(y_pred))
        self._y_prob.append(float(y_prob) if y_prob is not None else None)
        self._timestamps.append(time.time())
        self._prediction_count += 1

        # Recompute rolling metrics every time we have at least a full window.
        if len(self._y_true) >= self.window_size:
            self._update_rolling_metrics()

    # -- metrics ------------------------------------------------------------

    def get_rolling_metrics(self) -> Dict[str, List[Any]]:
        """Return the history of rolling metric values.

        Returns
        -------
        dict
            Keys: ``'auc_roc'``, ``'f1'``, ``'accuracy'``, ``'timestamps'``
            – each a list of floats (timestamps are Unix epoch seconds).
        """
        return {
            "auc_roc": list(self._rolling_auc),
            "f1": list(self._rolling_f1),
            "accuracy": list(self._rolling_acc),
            "timestamps": list(self._metric_ts),
        }

    def check_performance_drift(self) -> Dict[str, Any]:
        """Run drift tests on the most recent metrics and distributions.

        Returns
        -------
        dict
            ``'auc_drift'``, ``'f1_drift'`` – booleans from Page-Hinkley.
            ``'distribution_drift'`` – boolean from a two-sample KS test
            comparing the most recent *window_size* risk scores against
            the preceding *window_size* scores.
            ``'ks_statistic'``, ``'ks_pvalue'`` – KS test outputs.
            ``'status'`` – ``"healthy"``, ``"warning"``, or ``"critical"``.
            ``'status_color'`` – hex colour for dashboard display.
        """
        auc_drift = self._auc_detector.drift_detected
        f1_drift = self._f1_detector.drift_detected

        ks_stat, ks_pvalue, dist_drift = self._ks_distribution_test()

        # Aggregate status.
        n_flags = sum([auc_drift, f1_drift, dist_drift])
        if n_flags >= 2:
            status, colour = "critical", "#ef4444"
        elif n_flags == 1:
            status, colour = "warning", "#f59e0b"
        else:
            status, colour = "healthy", "#10b981"

        result: Dict[str, Any] = {
            "auc_drift": auc_drift,
            "f1_drift": f1_drift,
            "distribution_drift": dist_drift,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "status": status,
            "status_color": colour,
        }

        if status != "healthy":
            logger.warning("Model health status: {} (flags={})", status, n_flags)

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Return a high-level health summary of the monitored model.

        Returns
        -------
        dict
            Contains prediction count, latest metric values, and drift
            status information.
        """
        drift = self.check_performance_drift()
        latest_auc = self._rolling_auc[-1] if self._rolling_auc else None
        latest_f1 = self._rolling_f1[-1] if self._rolling_f1 else None
        latest_acc = self._rolling_acc[-1] if self._rolling_acc else None

        return {
            "prediction_count": self._prediction_count,
            "window_size": self.window_size,
            "latest_auc_roc": latest_auc,
            "latest_f1": latest_f1,
            "latest_accuracy": latest_acc,
            "drift_status": drift["status"],
            "drift_status_color": drift["status_color"],
            "auc_drift": drift["auc_drift"],
            "f1_drift": drift["f1_drift"],
            "distribution_drift": drift["distribution_drift"],
            "ks_statistic": drift["ks_statistic"],
            "ks_pvalue": drift["ks_pvalue"],
        }

    # -- internal helpers ---------------------------------------------------

    def _update_rolling_metrics(self) -> None:
        """Compute metrics over the most recent *window_size* predictions."""
        ws = self.window_size
        yt = list(self._y_true)[-ws:]
        yp = list(self._y_pred)[-ws:]
        yprob = list(self._y_prob)[-ws:]

        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, zero_division=0.0)

        # AUC requires both classes present and probabilities available.
        probs_available = all(p is not None for p in yprob)
        both_classes = len(set(yt)) > 1
        if probs_available and both_classes:
            auc_val = roc_auc_score(yt, yprob)
        else:
            auc_val = float("nan")

        now = time.time()
        self._rolling_auc.append(auc_val)
        self._rolling_f1.append(f1)
        self._rolling_acc.append(acc)
        self._metric_ts.append(now)

        # Feed the detectors (skip NaN to avoid corrupting state).
        if not np.isnan(auc_val):
            # Page-Hinkley reacts to *decreases*, so feed the negative.
            self._auc_detector.update(-auc_val)
        self._f1_detector.update(-f1)

    def _ks_distribution_test(
        self,
    ) -> tuple[float, float, bool]:
        """Two-sample KS test: recent vs. historical risk-score window.

        Returns (statistic, p-value, drift_detected).
        """
        probs = [p for p in self._y_prob if p is not None]
        ws = self.window_size

        if len(probs) < ws * 2:
            # Not enough data for a meaningful comparison.
            return 0.0, 1.0, False

        recent = np.array(probs[-ws:], dtype=np.float64)
        historical = np.array(probs[-2 * ws : -ws], dtype=np.float64)

        ks_stat, ks_pvalue = stats.ks_2samp(recent, historical)
        drift = bool(ks_pvalue < 0.01)

        if drift:
            logger.info(
                "KS distribution drift detected (stat={:.4f}, p={:.6f})",
                ks_stat,
                ks_pvalue,
            )

        return float(ks_stat), float(ks_pvalue), drift
