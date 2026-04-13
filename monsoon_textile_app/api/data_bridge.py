"""
Data Bridge -- connects FastAPI to the existing Streamlit data pipeline.

Security enhancements (v5):
  - python-dotenv used for .env loading (removes custom parser risk)
  - SMTP credentials masked in all log output
  - Subscriber cap enforced (MAX_SUBSCRIBERS env var, default 1000)
  - Email verification flow: subscribers marked verified=False until token clicked
  - HMAC-signed unsubscribe tokens (remove_subscriber / verify_subscriber_token)
  - Alert cooldown store: avoids spamming same alert (Feature 8)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import smtplib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Credentials loaded via python-dotenv (already in requirements.txt) ──────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass  # dotenv optional; env vars may already be set by the OS / hosting platform

_CACHE: dict = {}
_CACHE_TS: float = 0.0
_CACHE_TTL: float = 300  # 5 minutes

# Alert cooldown store: {alert_id -> last_sent_ts}
_COOLDOWN_STORE: dict[str, float] = {}
_COOLDOWN_HIGH_MIN: float = 60.0
_COOLDOWN_EXTREME_MIN: float = 30.0

_MAX_SUBSCRIBERS: int = int(os.environ.get("MAX_SUBSCRIBERS", "1000"))
_APP_BASE_URL: str = os.environ.get(
    "APP_BASE_URL", "https://rainloomtextiles.streamlit.app"
).rstrip("/")

# HMAC secret for signing unsubscribe / verification tokens
_HMAC_SECRET: bytes = (
    os.environ.get("SECRET_KEY", "rainloom-default-hmac-secret-change-in-production")
    .encode("utf-8")
)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def get_dashboard_data(force_refresh: bool = False) -> dict:
    """Get dashboard data, using an in-memory cache with TTL."""
    global _CACHE, _CACHE_TS
    now = time.time()
    if not force_refresh and _CACHE and (now - _CACHE_TS) < _CACHE_TTL:
        return _CACHE
    try:
        from monsoon_textile_app.data.fetch_real_data import load_all_data
        data = load_all_data(use_cache=True)
        _CACHE = data
        _CACHE_TS = now
        return data
    except Exception as e:
        print(f"[API Bridge] Data load failed: {e}")
        return _CACHE or {}


def get_risk_scores() -> list[dict]:
    """Extract current risk scores for all stocks."""
    from monsoon_textile_app.data.fetch_real_data import STOCKS
    data = get_dashboard_data()
    stock_data = data.get("stock_data", {})
    scores = []
    for ticker, info in STOCKS.items():
        sdf = stock_data.get(ticker)
        if sdf is None or sdf.empty:
            continue
        risk  = float(sdf["risk_score"].iloc[-1]) if "risk_score" in sdf.columns else 0.5
        vol   = float(sdf["vol_20d"].iloc[-1])    if "vol_20d"    in sdf.columns else None
        price = float(sdf["price"].iloc[-1])       if "price"      in sdf.columns else None
        label = "LOW" if risk < 0.3 else "MODERATE" if risk < 0.6 else "HIGH" if risk < 0.8 else "EXTREME"
        scores.append({
            "ticker":          ticker,
            "name":            info["name"],
            "sector":          info.get("sector", "Textile"),
            "chain_position":  info.get("chain", "Unknown"),
            "risk_score":      round(risk, 4),
            "risk_label":      label,
            "volatility_20d":  round(vol, 4)   if vol   is not None else None,
            "latest_price":    round(price, 2) if price is not None else None,
        })
    return scores


def get_alerts() -> list[dict]:
    """Generate current alerts based on dashboard data."""
    data = get_dashboard_data()
    alerts = []
    now = datetime.utcnow()
    stock_data = data.get("stock_data", {})
    try:
        from monsoon_textile_app.data.fetch_real_data import STOCKS
    except Exception:
        STOCKS = {}

    for ticker, sdf in stock_data.items():
        if sdf.empty or "risk_score" not in sdf.columns:
            continue
        risk = float(sdf["risk_score"].iloc[-1])
        name = STOCKS.get(ticker, {}).get("name", ticker)
        if risk >= 0.8:
            alerts.append({
                "id":        f"risk-extreme-{ticker}",
                "severity":  "critical",
                "category":  "risk",
                "title":     f"EXTREME risk: {name}",
                "message":   f"{name} ({ticker}) risk score at {risk:.1%}. Consider hedging or reducing exposure.",
                "timestamp": now,
            })
        elif risk >= 0.6:
            alerts.append({
                "id":        f"risk-high-{ticker}",
                "severity":  "warning",
                "category":  "risk",
                "title":     f"HIGH risk: {name}",
                "message":   f"{name} ({ticker}) risk score at {risk:.1%}. Monitor closely.",
                "timestamp": now,
            })

    rainfall = data.get("rainfall", {})
    if isinstance(rainfall, dict):
        annual = rainfall.get("annual_deficit")
        if hasattr(annual, "iloc") and len(annual) > 0:
            latest = annual.iloc[-1]
            vals = [v for v in (list(latest.values) if hasattr(latest, "values") else
                                (list(latest.values()) if isinstance(latest, dict) else []))
                    if isinstance(v, (int, float))]
            avg_def = (sum(vals) / len(vals)) if vals else 0
            if avg_def < -20:
                alerts.append({
                    "id":        "monsoon-severe-deficit",
                    "severity":  "critical",
                    "category":  "monsoon",
                    "title":     "Severe monsoon deficit",
                    "message":   f"Average cotton-belt rainfall deficit at {avg_def:+.1f}% from LPA.",
                    "timestamp": now,
                })
    return alerts


# ── Alert cooldown (Feature 8) ────────────────────────────────────────────────

def is_alert_on_cooldown(alert_id: str, severity: str = "warning") -> bool:
    """Return True if this alert was recently sent and should be suppressed."""
    cooldown_min = (
        _COOLDOWN_EXTREME_MIN if severity == "critical"
        else _COOLDOWN_HIGH_MIN
    )
    last_sent = _COOLDOWN_STORE.get(alert_id, 0.0)
    return (time.time() - last_sent) < (cooldown_min * 60)


def mark_alert_sent(alert_id: str) -> None:
    """Record that an alert was dispatched now."""
    _COOLDOWN_STORE[alert_id] = time.time()


def filter_cooldown_alerts(alerts: list[dict]) -> list[dict]:
    """
    Remove alerts that are within their cooldown window.
    Returns only alerts that should actually be dispatched.
    """
    due = []
    for a in alerts:
        alert_id = a.get("id", "")
        severity = a.get("severity", "warning")
        if not is_alert_on_cooldown(alert_id, severity):
            due.append(a)
    return due


# ── Subscriber store helpers ──────────────────────────────────────────────────

def _sub_file() -> Path:
    return Path(__file__).parent / "subscribers.json"


def get_subscriber_list() -> list[dict]:
    """Load email subscriber list from JSON file."""
    f = _sub_file()
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_subscriber_list(subs: list[dict]) -> None:
    _sub_file().write_text(json.dumps(subs, indent=2, default=str), encoding="utf-8")


# ── HMAC token helpers (Feature 9) ───────────────────────────────────────────

def _make_token(payload: str) -> str:
    """Generate an HMAC-SHA256 token for a given payload string."""
    sig = hmac.new(_HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    import base64
    encoded = base64.urlsafe_b64encode(payload.encode()).decode()
    return f"{encoded}.{sig}"


def _verify_token(token: str) -> Optional[str]:
    """
    Verify an HMAC token and return the payload if valid, else None.
    """
    import base64
    try:
        parts = token.split(".", 1)
        if len(parts) != 2:
            return None
        encoded, sig = parts
        payload = base64.urlsafe_b64decode(encoded.encode()).decode()
        expected_sig = hmac.new(
            _HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        return payload
    except Exception:
        return None


# ── Subscriber operations ─────────────────────────────────────────────────────

def add_subscriber(email: str, alert_types: list[str]) -> dict:
    """
    Add or update an email subscriber.
    Sends a verification email if SMTP is configured (new subscribers only).
    """
    email = email.strip().lower()
    subs  = get_subscriber_list()

    # Cap enforcement
    if len(subs) >= _MAX_SUBSCRIBERS:
        return {
            "status":            "rejected",
            "email":             email,
            "alert_types":       alert_types,
            "verification_sent": False,
            "message":           "Subscriber list is at capacity.",
        }

    # Check existing
    for sub in subs:
        if sub.get("email") == email:
            sub["alert_types"] = alert_types
            _save_subscriber_list(subs)
            return {
                "status":            "updated",
                "email":             email,
                "alert_types":       alert_types,
                "verification_sent": False,
            }

    # New subscriber — add as unverified
    verify_token = _make_token(f"verify:{email}")
    subs.append({
        "email":       email,
        "alert_types": alert_types,
        "verified":    False,
        "subscribed_at": datetime.now(timezone.utc).isoformat(),
        "verify_token": verify_token,
    })
    _save_subscriber_list(subs)

    # Best-effort verification email
    verification_sent = False
    try:
        smtp_cfg = _get_smtp_config()
        if smtp_cfg.get("enabled"):
            _send_verification_email(email, verify_token, smtp_cfg)
            verification_sent = True
    except Exception as exc:
        print(f"[Bridge] Verification email failed for {email}: {exc}")

    return {
        "status":            "subscribed",
        "email":             email,
        "alert_types":       alert_types,
        "verification_sent": verification_sent,
    }


def verify_subscriber_token(token: str) -> dict:
    """Mark a subscriber as verified using their HMAC token."""
    payload = _verify_token(token)
    if not payload or not payload.startswith("verify:"):
        return {"status": "error", "email": "", "message": "Invalid or expired verification token."}

    email = payload[len("verify:"):]
    subs  = get_subscriber_list()
    for sub in subs:
        if sub.get("email") == email:
            if sub.get("verified"):
                return {"status": "already_verified", "email": email,
                        "message": f"{email} is already verified."}
            sub["verified"] = True
            sub.pop("verify_token", None)
            _save_subscriber_list(subs)
            return {"status": "verified", "email": email,
                    "message": f"✅ {email} has been verified and will now receive alerts."}

    return {"status": "error", "email": email,
            "message": "Email address not found in subscriber list."}


def remove_subscriber(email: str, token: Optional[str] = None) -> dict:
    """
    Remove an email subscriber.
    If token is provided, validates it first (for unsubscribe links).
    """
    email = email.strip().lower()
    # Token provided (from unsubscribe link) — validate
    if token:
        payload = _verify_token(token)
        if not payload or payload != f"unsub:{email}":
            return {"status": "error", "email": email,
                    "message": "Invalid or expired unsubscribe token."}

    subs = get_subscriber_list()
    original_len = len(subs)
    subs = [s for s in subs if s.get("email") != email]
    if len(subs) == original_len:
        return {"status": "not_found", "email": email,
                "message": f"{email} was not found in the subscriber list."}
    _save_subscriber_list(subs)
    return {"status": "unsubscribed", "email": email,
            "message": f"✅ {email} has been removed from all RainLoom alerts."}


def get_unsubscribe_link(email: str) -> str:
    """Generate a signed unsubscribe link for use in email footers."""
    token = _make_token(f"unsub:{email}")
    return f"{_APP_BASE_URL}/api/unsubscribe?email={email}&token={token}"


# ── SMTP config ───────────────────────────────────────────────────────────────

def _get_smtp_config() -> dict:
    """Load SMTP configuration from environment variables (loaded via dotenv above)."""
    host     = os.environ.get("SMTP_HOST",   "").strip()
    port_str = os.environ.get("SMTP_PORT",   "587").strip()
    user     = os.environ.get("SMTP_USER",   "").strip()
    password = os.environ.get("SMTP_PASS",   "").strip().replace(" ", "")
    sender   = os.environ.get("SMTP_SENDER", user).strip()
    use_tls  = os.environ.get("SMTP_USE_TLS","1").strip().lower() in {"1", "true", "yes"}

    try:
        port = int(port_str)
    except ValueError:
        port = 587

    enabled = bool(host and port and user and password and sender)

    # Mask password in any debug prints
    masked_pass = (password[:4] + "****") if password else ""
    if enabled:
        print(f"[Bridge] SMTP configured: {user}@{host}:{port} (TLS={use_tls}) pass={masked_pass}")

    return {
        "enabled":  enabled,
        "host":     host,
        "port":     port,
        "user":     user,
        "password": password,
        "sender":   sender,
        "use_tls":  use_tls,
    }


# ── Email senders ─────────────────────────────────────────────────────────────

def _send_verification_email(email: str, token: str, smtp_cfg: dict) -> None:
    """Send an email-verification link to a new subscriber."""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    verify_url = f"{_APP_BASE_URL}/api/verify?token={token}"
    subject    = "✅ Verify your RainLoom alert subscription"
    html = f"""<!DOCTYPE html>
<html><body style="background:#0a0f1e;font-family:Inter,sans-serif;color:#e2e8f0;padding:32px;">
  <div style="max-width:560px;margin:auto;">
    <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);
                border-radius:10px;padding:8px 14px;display:inline-block;
                font-weight:700;color:#fff;margin-bottom:16px;">RL</div>
    <h2 style="color:#e2e8f0;">Confirm your subscription</h2>
    <p style="color:#94a3b8;">Click the button below to verify your email and start receiving
       RainLoom monsoon-risk alerts.</p>
    <a href="{verify_url}"
       style="display:inline-block;background:linear-gradient(135deg,#6366f1,#8b5cf6);
              color:#fff;font-weight:700;text-decoration:none;padding:12px 28px;
              border-radius:8px;margin:16px 0;">Verify Email →</a>
    <p style="color:#475569;font-size:12px;">
      If you did not subscribe, you can safely ignore this email.<br>
      Link: {verify_url}
    </p>
  </div>
</body></html>"""
    plain = f"Verify your RainLoom subscription:\n{verify_url}\n\nIf you did not subscribe, ignore this email."

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_cfg["sender"]
    msg["To"]      = email
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"], timeout=20) as s:
        s.ehlo()
        if smtp_cfg["use_tls"]:
            s.starttls(); s.ehlo()
        s.login(smtp_cfg["user"], smtp_cfg["password"])
        s.sendmail(smtp_cfg["sender"], email, msg.as_string())


def _format_alert_digest(alerts: list[dict]) -> str:
    """Build a plain-text digest for email delivery."""
    lines = ["RainLoom Alert Digest",
             f"Generated at (UTC): {datetime.utcnow().isoformat()}", ""]
    for idx, alert in enumerate(alerts, start=1):
        ts = alert.get("timestamp")
        ts_text = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        lines += [
            f"{idx}. [{str(alert.get('severity','info')).upper()}] {alert.get('title','Alert')}",
            f"   Category: {alert.get('category','unknown')}",
            f"   Message : {alert.get('message','')}",
            f"   Time    : {ts_text}", "",
        ]
    lines.append("You are receiving this because you subscribed to RainLoom alerts.")
    return "\n".join(lines)


def _filter_alerts_for_subscriber(alerts: list[dict], alert_types: list[str]) -> list[dict]:
    """Return only alerts matching subscriber severities."""
    requested = {str(a).strip().lower() for a in (alert_types or []) if str(a).strip()}
    if not requested or "all" in requested:
        return alerts
    return [a for a in alerts if str(a.get("severity", "")).lower() in requested]


def _send_email(recipient: str, alerts: list[dict], smtp_cfg: dict) -> None:
    """Send one plain-text alert email to one recipient."""
    from email.message import EmailMessage
    highest = "critical" if any(a.get("severity") == "critical" for a in alerts) else "warning"
    subject = f"[RainLoom] {len(alerts)} alert(s) - {highest.upper()}"
    body    = _format_alert_digest(alerts)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = smtp_cfg["sender"]
    msg["To"]      = recipient
    msg.set_content(body)
    with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"], timeout=20) as server:
        if smtp_cfg["use_tls"]:
            server.starttls()
        server.login(smtp_cfg["user"], smtp_cfg["password"])
        server.send_message(msg)


# ── Alert dispatch (Feature 8: cooldown-aware, verified-only) ─────────────────

def dispatch_alert_emails(dry_run: bool = False) -> dict:
    """
    Send the current active alerts to all verified subscribers.
    Alerts within their cooldown window are suppressed (Feature 8).
    Only verified subscribers receive emails (Feature 9).
    """
    subscribers = get_subscriber_list()
    all_alerts  = get_alerts()
    smtp_cfg    = _get_smtp_config()

    # Feature 8: filter by cooldown
    alerts = filter_cooldown_alerts(all_alerts)
    suppressed = len(all_alerts) - len(alerts)

    if not dry_run and not smtp_cfg["enabled"]:
        return {
            "status":              "error",
            "timestamp":          datetime.utcnow(),
            "total_subscribers":  len(subscribers),
            "recipients_targeted":0,
            "total_alerts":       len(alerts),
            "emails_sent":        0,
            "deliveries_skipped": len(subscribers),
            "failures":           ["SMTP not configured."],
            "dry_run":            dry_run,
            "suppressed_by_cooldown": suppressed,
            "message":            "Email dispatch not attempted: SMTP not configured.",
        }

    recipients_targeted = 0
    emails_sent         = 0
    deliveries_skipped  = 0
    failures: list[str] = []

    for sub in subscribers:
        recipient   = str(sub.get("email", "")).strip()
        alert_types = sub.get("alert_types", ["critical"])

        if not recipient:
            deliveries_skipped += 1
            continue

        # Feature 9: skip unverified subscribers
        if not sub.get("verified", True):   # legacy subs without 'verified' key pass
            deliveries_skipped += 1
            continue

        selected = _filter_alerts_for_subscriber(alerts, alert_types)
        if not selected:
            deliveries_skipped += 1
            continue

        recipients_targeted += 1
        if dry_run:
            emails_sent += 1
            continue

        try:
            _send_email(recipient=recipient, alerts=selected, smtp_cfg=smtp_cfg)
            emails_sent += 1
            # Mark alerts as sent in cooldown store
            for a in selected:
                mark_alert_sent(a.get("id", ""))
        except Exception as exc:
            failures.append(f"{recipient}: {exc}")

    status = "ok" if not failures else ("partial" if emails_sent > 0 else "error")
    msg = (
        f"Dispatched {emails_sent} email(s) to {recipients_targeted} subscriber(s). "
        f"{suppressed} alert(s) suppressed by cooldown."
        if not dry_run
        else f"Dry run: {emails_sent} would be sent. {suppressed} suppressed by cooldown."
    )

    return {
        "status":              status,
        "timestamp":          datetime.utcnow(),
        "total_subscribers":  len(subscribers),
        "recipients_targeted":recipients_targeted,
        "total_alerts":       len(alerts),
        "emails_sent":        emails_sent,
        "deliveries_skipped": deliveries_skipped,
        "failures":           failures,
        "dry_run":            dry_run,
        "suppressed_by_cooldown": suppressed,
        "message":            msg,
    }
