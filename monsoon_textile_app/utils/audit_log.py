"""
Structured Audit Log — RainLoom Security Layer
===============================================
Appends one JSON-Lines record per significant security/platform event
to logs/audit.jsonl (auto-created).  Thread-safe, zero external deps.

Event categories:
  auth        — API key generated / used / revoked
  subscribe   — email subscribe / unsubscribe / verify
  alert       — alert dispatch / email sent
  admin       — admin panel access / config change
  security    — rate-limit hits, CSRF failures, invalid key attempts
  system      — scheduler start/stop, data refresh

Usage:
    from monsoon_textile_app.utils.audit_log import audit

    audit("auth", "api_key_generated", details={"name": "ICICI"}, ip="1.2.3.4")
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_LOG_DIR  = Path(__file__).resolve().parents[2] / "logs"
_LOG_FILE = _LOG_DIR / "audit.jsonl"
_MAX_BYTES = 5 * 1024 * 1024   # 5 MB — rotate at this size
_LOCK = threading.Lock()

# In-memory ring buffer for the UI (last 500 events)
_ring: list[dict] = []
_RING_MAX = 500


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _rotate_if_needed() -> None:
    """Rename audit.jsonl → audit.jsonl.1 if over size limit."""
    if _LOG_FILE.exists() and _LOG_FILE.stat().st_size >= _MAX_BYTES:
        backup = _LOG_FILE.with_suffix(".jsonl.1")
        _LOG_FILE.rename(backup)


def audit(
    category: str,
    action: str,
    *,
    details: Optional[dict[str, Any]] = None,
    ip: Optional[str] = None,
    user: Optional[str] = None,
    severity: str = "info",
) -> dict:
    """
    Write one audit record to disk and the in-memory ring buffer.

    Parameters
    ----------
    category  : Event category (auth / subscribe / alert / admin / security / system)
    action    : Short action label (e.g. "api_key_generated")
    details   : Arbitrary extra data to log
    ip        : Client IP (optional)
    user      : Actor identifier (email, key_id, …)
    severity  : "info" | "warning" | "critical"

    Returns
    -------
    The record dict (useful for testing).
    """
    record: dict[str, Any] = {
        "ts":       datetime.now(timezone.utc).isoformat(),
        "category": category,
        "action":   action,
        "severity": severity,
        "ip":       ip or "—",
        "user":     user or "—",
        "details":  details or {},
    }

    with _LOCK:
        try:
            _ensure_log_dir()
            _rotate_if_needed()
            with _LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            # Never crash the caller because of logging failure
            record["_log_error"] = str(exc)

        # Ring buffer (most-recent first for UI)
        _ring.insert(0, record)
        if len(_ring) > _RING_MAX:
            _ring.pop()

    return record


# ---------------------------------------------------------------------------
# Read helpers for the Security Audit dashboard page
# ---------------------------------------------------------------------------

def get_recent_events(limit: int = 200) -> list[dict]:
    """Return up to `limit` most-recent events from the in-memory ring."""
    with _LOCK:
        return list(_ring[:limit])


def load_all_events(limit: int = 1000) -> list[dict]:
    """
    Read events from disk (newest-first).
    Falls back to ring buffer if log file is missing.
    """
    _ensure_log_dir()
    if not _LOG_FILE.exists():
        return get_recent_events(limit)

    lines: list[str] = []
    try:
        with _LOG_FILE.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception:
        return get_recent_events(limit)

    records: list[dict] = []
    for line in reversed(lines[-limit:]):
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def event_counts_by_category() -> dict[str, int]:
    """Summary counts by category (from in-memory ring)."""
    counts: dict[str, int] = {}
    with _LOCK:
        for ev in _ring:
            c = ev.get("category", "unknown")
            counts[c] = counts.get(c, 0) + 1
    return counts


def security_summary() -> dict[str, Any]:
    """Quick stats for the security dashboard widget."""
    with _LOCK:
        total      = len(_ring)
        critical   = sum(1 for e in _ring if e.get("severity") == "critical")
        warnings   = sum(1 for e in _ring if e.get("severity") == "warning")
        rate_hits  = sum(1 for e in _ring if e.get("action") == "rate_limit_hit")
        bad_keys   = sum(1 for e in _ring if e.get("action") == "invalid_api_key")
        subs       = sum(1 for e in _ring if e.get("action") in ("subscribed", "verified"))
        dispatched = sum(1 for e in _ring if e.get("action") == "email_dispatched")

    return {
        "total_events":     total,
        "critical_events":  critical,
        "warnings":         warnings,
        "rate_limit_hits":  rate_hits,
        "invalid_key_attempts": bad_keys,
        "new_subscribers":  subs,
        "emails_dispatched": dispatched,
    }
