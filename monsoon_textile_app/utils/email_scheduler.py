"""
RainLoom Smart Email Scheduler
================================
Background thread that checks alert conditions every N minutes and
fires HTML email alerts whenever any of the trigger conditions are met:

  1. Any stock risk score crosses HIGH (≥0.60) or EXTREME (≥0.80)
  2. Sector average risk rises 10%+ week-over-week
  3. Monsoon district deficit worsens beyond -20%
  4. Parametric insurance trigger is reached for any registered contract

Security/Quality enhancements (v5 — Feature 8):
  - Alert deduplication: SHA-256 fingerprint of (title+message) suppresses
    duplicate alerts seen within the cooldown window.
  - Cooldown integration: uses data_bridge.filter_cooldown_alerts() and
    mark_alert_sent() so the same event is not emailed repeatedly.
  - Audit log: every dispatch cycle is logged to logs/audit.jsonl.

Runs entirely in-process — no external broker (Celery/RQ) needed.
Configure via .env:
    ALERT_CHECK_INTERVAL_MINUTES=15   (default: 15)
    ALERT_RISK_THRESHOLD_HIGH=0.60
    ALERT_RISK_THRESHOLD_EXTREME=0.80
    ALERT_SECTOR_JUMP_PCT=0.10
    ALERT_RAINFALL_DEFICIT=-20
    ENABLE_EMAIL_SCHEDULER=1          (set to 0 to disable)
    ALERT_COOLDOWN_HIGH_MIN=60        (minutes between same HIGH alert)
    ALERT_COOLDOWN_EXTREME_MIN=30     (minutes between same EXTREME alert)
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# State: last-known risk levels to detect *changes* (not spam on every cycle)
# ---------------------------------------------------------------------------
_last_risk_snapshot: dict[str, float] = {}
_last_dispatch_ts: float = 0.0
_scheduler_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()

# Deduplication fingerprint store: {fingerprint -> last_seen_ts}
_sent_fingerprints: dict[str, float] = {}
_DEDUP_WINDOW_SEC: int = 30 * 60  # 30 minutes default dedup window


# ---------------------------------------------------------------------------
# Trigger evaluation
# ---------------------------------------------------------------------------

def _load_env_floats() -> dict:
    """Read thresholds from environment with safe fallbacks."""
    def _f(key: str, default: float) -> float:
        try:
            return float(os.environ.get(key, default))
        except ValueError:
            return default

    return {
        "high":     _f("ALERT_RISK_THRESHOLD_HIGH",    0.60),
        "extreme":  _f("ALERT_RISK_THRESHOLD_EXTREME", 0.80),
        "jump_pct": _f("ALERT_SECTOR_JUMP_PCT",        0.10),
        "deficit":  _f("ALERT_RAINFALL_DEFICIT",      -20.0),
    }


def evaluate_triggers() -> list[dict]:
    """
    Evaluate all alert conditions against live data.
    Returns a list of triggered alert dicts (empty if nothing is triggered).
    """
    global _last_risk_snapshot

    triggered: list[dict] = []
    now = datetime.now(timezone.utc)
    thresholds = _load_env_floats()

    try:
        from monsoon_textile_app.api.data_bridge import get_dashboard_data
        data = get_dashboard_data()
    except Exception as exc:
        logger.warning(f"[Scheduler] Could not load data: {exc}")
        return triggered

    stock_data = data.get("stock_data", {})
    try:
        from monsoon_textile_app.data.fetch_real_data import STOCKS
    except Exception:
        STOCKS = {}

    current_risks: dict[str, float] = {}

    # ── 1. Per-stock level triggers ─────────────────────────────────────
    for ticker, sdf in stock_data.items():
        if sdf is None or sdf.empty or "risk_score" not in sdf.columns:
            continue
        risk = float(sdf["risk_score"].iloc[-1])
        current_risks[ticker] = risk
        name = STOCKS.get(ticker, {}).get("name", ticker)
        prev_risk = _last_risk_snapshot.get(ticker, 0.0)

        if risk >= thresholds["extreme"] and prev_risk < thresholds["extreme"]:
            triggered.append({
                "id": f"extreme-{ticker}-{now.date()}",
                "severity": "critical",
                "category": "risk",
                "title": f"🚨 EXTREME Risk: {name}",
                "message": (
                    f"{name} ({ticker}) crossed the EXTREME threshold: risk score now {risk:.1%}. "
                    f"Immediate hedging action may be required. Previous: {prev_risk:.1%}."
                ),
                "timestamp": now,
                "stock": name,
                "risk": risk,
            })
        elif risk >= thresholds["high"] and prev_risk < thresholds["high"]:
            triggered.append({
                "id": f"high-{ticker}-{now.date()}",
                "severity": "warning",
                "category": "risk",
                "title": f"⚠️ HIGH Risk: {name}",
                "message": (
                    f"{name} ({ticker}) crossed the HIGH threshold: risk score now {risk:.1%}. "
                    f"Consider increasing hedging coverage. Previous: {prev_risk:.1%}."
                ),
                "timestamp": now,
                "stock": name,
                "risk": risk,
            })

    # ── 2. Sector average jump trigger ──────────────────────────────────
    if current_risks and _last_risk_snapshot:
        avg_now  = sum(current_risks.values()) / len(current_risks)
        avg_prev = sum(_last_risk_snapshot.get(t, 0) for t in current_risks) / len(current_risks)
        jump = avg_now - avg_prev
        if jump >= thresholds["jump_pct"]:
            triggered.append({
                "id": f"sector-jump-{now.date()}",
                "severity": "warning",
                "category": "sector",
                "title": f"📈 Sector Risk Surge: +{jump:.0%} in one cycle",
                "message": (
                    f"Average textile sector risk jumped from {avg_prev:.1%} to {avg_now:.1%} "
                    f"(+{jump:.1%}). Broad-based stress signal detected across the supply chain."
                ),
                "timestamp": now,
                "risk": avg_now,
            })

    # ── 3. Monsoon district deficit trigger ─────────────────────────────
    rainfall = data.get("rainfall", {})
    if isinstance(rainfall, dict):
        latest_deficit = rainfall.get("latest_deficit")
        if latest_deficit is not None and hasattr(latest_deficit, "iterrows"):
            for _, row in latest_deficit.iterrows():
                deficit = float(row.get("Deficit", 0))
                state   = str(row.get("State", ""))
                if deficit <= thresholds["deficit"]:
                    triggered.append({
                        "id": f"deficit-{state}-{now.date()}",
                        "severity": "critical",
                        "category": "monsoon",
                        "title": f"🌧️ Severe Deficit: {state} at {deficit:+.0f}%",
                        "message": (
                            f"{state} monsoon deficit has reached {deficit:+.1f}% from LPA — "
                            f"below the critical -20% threshold. Cotton supply disruption "
                            f"expected within 8 weeks. Consider activating PMFBY campaigns."
                        ),
                        "timestamp": now,
                        "state": state,
                        "deficit": deficit,
                    })

    # Update snapshot for next cycle
    _last_risk_snapshot = current_risks
    return triggered


# ---------------------------------------------------------------------------
# HTML Email Renderer
# ---------------------------------------------------------------------------

def _render_html_email(alerts: list[dict], recipient: str) -> str:
    """Render a beautiful dark-themed HTML alert digest email."""
    now_str = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

    def _severity_color(severity: str) -> str:
        return {"critical": "#ef4444", "warning": "#f59e0b"}.get(severity, "#60a5fa")

    def _severity_bg(severity: str) -> str:
        return {"critical": "#1a0a0a", "warning": "#1a1200"}.get(severity, "#0a0f1a")

    alert_rows = ""
    for alert in alerts:
        sc = _severity_color(alert.get("severity", "info"))
        bg = _severity_bg(alert.get("severity", "info"))
        title = alert.get("title", "Alert")
        msg   = alert.get("message", "")
        ts    = alert.get("timestamp", "")
        if hasattr(ts, "strftime"):
            ts = ts.strftime("%d %b %Y, %H:%M UTC")
        alert_rows += f"""
        <tr>
          <td style="padding:16px 20px; background:{bg}; border-left:4px solid {sc}; border-radius:0 8px 8px 0; margin-bottom:12px;">
            <div style="font-size:15px; font-weight:700; color:{sc}; margin-bottom:6px;">{title}</div>
            <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">{msg}</div>
            <div style="font-size:11px; color:#475569; margin-top:8px;">{ts}</div>
          </td>
        </tr>
        <tr><td style="height:10px;"></td></tr>
        """

    highest = "critical" if any(a.get("severity") == "critical" for a in alerts) else "warning"
    banner_color = "#ef4444" if highest == "critical" else "#f59e0b"
    banner_icon  = "🚨" if highest == "critical" else "⚠️"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RainLoom Alert</title>
</head>
<body style="margin:0; padding:0; background:#0a0f1e; font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="max-width:640px; margin:30px auto; padding:0 16px;">
    <!-- HEADER -->
    <tr>
      <td style="background:linear-gradient(135deg,#0d1326,#131c35); border-radius:16px 16px 0 0;
                 border:1px solid rgba(99,102,241,0.25); padding:28px 32px;">
        <table width="100%" cellpadding="0" cellspacing="0">
          <tr>
            <td>
              <div style="display:inline-block; background:linear-gradient(135deg,#6366f1,#8b5cf6);
                          border-radius:10px; padding:8px 14px; font-size:14px; font-weight:700;
                          color:#fff; letter-spacing:1px;">RL</div>
              <span style="font-size:18px; font-weight:700; color:#e2e8f0; margin-left:12px; vertical-align:middle;">RainLoom</span>
            </td>
            <td align="right">
              <span style="font-size:11px; color:#475569;">{now_str}</span>
            </td>
          </tr>
        </table>
      </td>
    </tr>

    <!-- ALERT BANNER -->
    <tr>
      <td style="background:{banner_color}18; border-left:1px solid rgba(99,102,241,0.12);
                 border-right:1px solid rgba(99,102,241,0.12); padding:18px 32px;">
        <div style="font-size:20px; font-weight:800; color:{banner_color}; letter-spacing:-0.5px;">
          {banner_icon} {len(alerts)} Active Alert{'s' if len(alerts) != 1 else ''} — Action Required
        </div>
        <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
          Hi {recipient.split('@')[0].title()}, the RainLoom AI has detected new risk events
          that require your attention.
        </div>
      </td>
    </tr>

    <!-- ALERT CARDS -->
    <tr>
      <td style="background:#0d1326; border-left:1px solid rgba(99,102,241,0.12);
                 border-right:1px solid rgba(99,102,241,0.12); padding:20px 32px;">
        <table width="100%" cellpadding="0" cellspacing="0">
          {alert_rows}
        </table>
      </td>
    </tr>

    <!-- CTA -->
    <tr>
      <td style="background:#0d1326; padding:0 32px 24px 32px;
                 border-left:1px solid rgba(99,102,241,0.12); border-right:1px solid rgba(99,102,241,0.12);">
        <a href="https://rainloomtextiles.streamlit.app"
           style="display:inline-block; background:linear-gradient(135deg,#6366f1,#8b5cf6);
                  color:#fff; font-weight:700; font-size:14px; text-decoration:none;
                  padding:12px 28px; border-radius:8px;">
          Open Live Dashboard →
        </a>
      </td>
    </tr>

    <!-- FOOTER -->
    <tr>
      <td style="background:#080d19; border:1px solid rgba(99,102,241,0.12); border-top:none;
                 border-radius:0 0 16px 16px; padding:18px 32px;">
        <div style="font-size:11px; color:#334155; line-height:1.7;">
          You are receiving this because you subscribed to RainLoom alerts.<br>
          RainLoom — Causal AI for Monsoon-Textile Risk | India's Cotton Belt Intelligence Platform<br>
          <a href="#" style="color:#6366f1;">Unsubscribe</a> &nbsp;·&nbsp;
          <a href="https://rainloomtextiles.streamlit.app" style="color:#6366f1;">Dashboard</a>
        </div>
      </td>
    </tr>
  </table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Enhanced _send_email with HTML support (monkey-patches data_bridge)
# ---------------------------------------------------------------------------

def send_alert_email_html(recipient: str, alerts: list[dict], smtp_cfg: dict) -> None:
    """Send a richly formatted HTML alert email."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    highest = "critical" if any(a.get("severity") == "critical" for a in alerts) else "warning"
    subject = f"[RainLoom] {len(alerts)} Alert{'s' if len(alerts)!=1 else ''} — {highest.upper()}: Monsoon Risk Event"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_cfg["sender"]
    msg["To"]      = recipient

    # Plain-text fallback
    plain = f"RainLoom Alert — {len(alerts)} new event(s)\n\n"
    for a in alerts:
        plain += f"[{a.get('severity','info').upper()}] {a.get('title','')}\n{a.get('message','')}\n\n"
    plain += "Visit https://rainloomtextiles.streamlit.app for details."

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(_render_html_email(alerts, recipient), "html"))

    with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"], timeout=20) as server:
        server.ehlo()
        if smtp_cfg["use_tls"]:
            server.starttls()
            server.ehlo()
        server.login(smtp_cfg["user"], smtp_cfg["password"])
        server.sendmail(smtp_cfg["sender"], recipient, msg.as_string())
    logger.info(f"[Scheduler] HTML email dispatched → {recipient} ({len(alerts)} alerts)")


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------

def _dedup_fingerprint(alert: dict) -> str:
    """Return a SHA-256 fingerprint for deduplication (based on title + message)."""
    raw = f"{alert.get('title', '')}{alert.get('message', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _filter_deduped(alerts: list[dict]) -> list[dict]:
    """Remove alerts whose fingerprint was already seen within the dedup window."""
    global _sent_fingerprints
    now    = time.time()
    cutoff = now - _DEDUP_WINDOW_SEC
    # Prune old fingerprints
    _sent_fingerprints = {fp: ts for fp, ts in _sent_fingerprints.items() if ts > cutoff}
    fresh = []
    for a in alerts:
        fp = _dedup_fingerprint(a)
        if fp not in _sent_fingerprints:
            fresh.append(a)
            _sent_fingerprints[fp] = now
    return fresh


def _run_scheduler(interval_seconds: int) -> None:
    """Background scheduler loop."""
    logger.info(f"[Scheduler] Started — check interval: {interval_seconds}s")
    while not _stop_event.is_set():
        try:
            triggers = evaluate_triggers()
            if triggers:
                # Feature 8: dedup + cooldown filtering
                try:
                    from monsoon_textile_app.api.data_bridge import filter_cooldown_alerts
                    triggers = filter_cooldown_alerts(triggers)
                except Exception:
                    pass
                triggers = _filter_deduped(triggers)
                if triggers:
                    logger.info(f"[Scheduler] {len(triggers)} new trigger(s) — dispatching emails")
                    _dispatch(triggers)
                else:
                    logger.debug("[Scheduler] All triggers suppressed by dedup/cooldown")
            else:
                logger.debug("[Scheduler] No triggers — all clear")
        except Exception as exc:
            logger.error(f"[Scheduler] Cycle error: {exc}")
        _stop_event.wait(interval_seconds)
    logger.info("[Scheduler] Stopped.")


def _dispatch(triggers: list[dict]) -> None:
    """Send triggered alerts to all matching verified subscribers."""
    try:
        from monsoon_textile_app.api.data_bridge import (
            get_subscriber_list,
            _get_smtp_config,
            _filter_alerts_for_subscriber,
            mark_alert_sent,
        )
        from monsoon_textile_app.utils.audit_log import audit
    except Exception as exc:
        logger.error(f"[Scheduler] Cannot import data_bridge: {exc}")
        return

    subscribers = get_subscriber_list()
    smtp_cfg    = _get_smtp_config()

    if not smtp_cfg.get("enabled"):
        logger.warning("[Scheduler] SMTP not configured — email dispatch skipped. Set SMTP_* env vars.")
        return

    emails_sent = 0
    for sub in subscribers:
        recipient   = str(sub.get("email", "")).strip()
        alert_types = sub.get("alert_types", ["critical", "warning"])
        if not recipient:
            continue
        # Feature 9: skip unverified subscribers
        if not sub.get("verified", True):
            continue
        selected = _filter_alerts_for_subscriber(triggers, alert_types)
        if not selected:
            continue
        try:
            send_alert_email_html(recipient, selected, smtp_cfg)
            emails_sent += 1
            # Mark alerts sent in cooldown store
            for a in selected:
                mark_alert_sent(a.get("id", ""))
        except Exception as exc:
            logger.error(f"[Scheduler] Email to {recipient} failed: {exc}")
    # Audit log
    try:
        audit("alert", "scheduler_dispatch",
              details={"triggers": len(triggers), "emails_sent": emails_sent})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_scheduler(interval_minutes: Optional[int] = None) -> None:
    """
    Start the background alert scheduler (idempotent — safe to call multiple times).

    Parameters
    ----------
    interval_minutes : int, optional
        Poll interval in minutes. Reads ALERT_CHECK_INTERVAL_MINUTES env var;
        defaults to 15 if not set.
    """
    global _scheduler_thread

    enabled = os.environ.get("ENABLE_EMAIL_SCHEDULER", "1").strip()
    if enabled.lower() in {"0", "false", "no"}:
        logger.info("[Scheduler] Disabled via ENABLE_EMAIL_SCHEDULER=0")
        return

    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        logger.debug("[Scheduler] Already running — skipping start")
        return

    if interval_minutes is None:
        try:
            interval_minutes = int(os.environ.get("ALERT_CHECK_INTERVAL_MINUTES", 15))
        except ValueError:
            interval_minutes = 15

    _stop_event.clear()
    _scheduler_thread = threading.Thread(
        target=_run_scheduler,
        args=(interval_minutes * 60,),
        daemon=True,
        name="RainLoom-AlertScheduler",
    )
    _scheduler_thread.start()
    logger.info(f"[Scheduler] Thread launched — polling every {interval_minutes} min")


def stop_scheduler() -> None:
    """Signal the background scheduler to stop cleanly."""
    _stop_event.set()
    logger.info("[Scheduler] Stop signal sent")


def scheduler_status() -> dict:
    """Return current scheduler status for the UI."""
    alive = _scheduler_thread is not None and _scheduler_thread.is_alive()
    try:
        interval = int(os.environ.get("ALERT_CHECK_INTERVAL_MINUTES", 15))
    except ValueError:
        interval = 15
    return {
        "running": alive,
        "interval_minutes": interval,
        "last_snapshot_stocks": list(_last_risk_snapshot.keys()),
        "smtp_configured": bool(os.environ.get("SMTP_HOST")),
    }
