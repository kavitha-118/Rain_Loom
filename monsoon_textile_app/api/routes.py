"""
FastAPI route definitions for the Monsoon-Textile REST API.

Security enhancements (v5):
  - verify_api_key dependency on all write/sensitive endpoints (Feature 1)
  - Audit log wired to every action (Feature 7)
  - Email verification + unsubscribe endpoints (Feature 9)
  - API key management endpoints (Feature 1)
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from monsoon_textile_app.api.auth import verify_api_key, optional_api_key
from monsoon_textile_app.api.schemas import (
    Alert,
    AlertsResponse,
    ApiKeyResponse,
    DispatchAlertsResponse,
    GenerateApiKeyRequest,
    HealthResponse,
    RevokeKeyRequest,
    RiskScoresResponse,
    StockRisk,
    SubscribeRequest,
    SubscribeResponse,
    UnsubscribeRequest,
    UnsubscribeResponse,
    VerifyEmailRequest,
    VerifyEmailResponse,
)

router = APIRouter(prefix="/api", tags=["Monsoon-Textile API"])

_START_TIME = time.time()


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    return fwd.split(",")[0].strip() if fwd else (
        request.client.host if request.client else "unknown"
    )


# ── Public read-only endpoints ────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint — public, no auth required."""
    from monsoon_textile_app.api.data_bridge import get_dashboard_data
    data = get_dashboard_data()
    n_stocks = len(data.get("stock_data", {}))
    return HealthResponse(
        status="ok" if n_stocks > 0 else "degraded",
        version="5.0.0",
        uptime_seconds=round(time.time() - _START_TIME, 1),
        data_loaded=bool(data),
        n_stocks=n_stocks,
        last_update=datetime.utcnow(),
    )


@router.get("/risk-scores", response_model=RiskScoresResponse)
def get_risk_scores(request: Request, key: Optional[str] = Depends(optional_api_key)):
    """Get current ensemble risk scores. Public endpoint; API key grants higher rate limits."""
    from monsoon_textile_app.api.data_bridge import get_risk_scores as _get_scores
    from monsoon_textile_app.utils.audit_log import audit
    scores = _get_scores()
    if not scores:
        raise HTTPException(status_code=503, detail="Risk data not yet loaded")
    avg_risk = sum(s["risk_score"] for s in scores) / len(scores)
    audit("auth", "risk_scores_read", ip=_client_ip(request), user=key or "anonymous")
    return RiskScoresResponse(
        timestamp=datetime.utcnow(),
        n_stocks=len(scores),
        avg_risk=round(avg_risk, 4),
        scores=[StockRisk(**s) for s in scores],
    )


@router.get("/alerts", response_model=AlertsResponse)
def get_alerts(request: Request):
    """Get current active alerts — public."""
    from monsoon_textile_app.api.data_bridge import get_alerts as _get_alerts
    alerts = _get_alerts()
    return AlertsResponse(
        timestamp=datetime.utcnow(),
        n_alerts=len(alerts),
        alerts=[Alert(**a) for a in alerts],
    )


# ── Write endpoints (require API key) ────────────────────────────────────

@router.post("/subscribe", response_model=SubscribeResponse)
def subscribe(req: SubscribeRequest, request: Request):
    """Subscribe an email address to receive alerts. No API key needed (public subscribe)."""
    from monsoon_textile_app.api.data_bridge import add_subscriber
    from monsoon_textile_app.utils.audit_log import audit
    result = add_subscriber(req.email, req.alert_types)
    audit("subscribe", result["status"], ip=_client_ip(request), user=req.email,
          details={"alert_types": req.alert_types})
    return SubscribeResponse(
        status=result["status"],
        email=result["email"],
        alert_types=result["alert_types"],
        message=f"Successfully {result['status']} {req.email} for {', '.join(req.alert_types)} alerts.",
        verification_sent=result.get("verification_sent", False),
    )


@router.post("/unsubscribe", response_model=UnsubscribeResponse)
def unsubscribe(req: UnsubscribeRequest, request: Request):
    """Remove an email address from the subscriber list."""
    from monsoon_textile_app.api.data_bridge import remove_subscriber
    from monsoon_textile_app.utils.audit_log import audit
    result = remove_subscriber(req.email, req.token)
    audit("subscribe", "unsubscribed", ip=_client_ip(request), user=req.email)
    return UnsubscribeResponse(**result)


@router.post("/verify", response_model=VerifyEmailResponse)
def verify_email(req: VerifyEmailRequest, request: Request):
    """Verify an email address using the token from the confirmation email."""
    from monsoon_textile_app.api.data_bridge import verify_subscriber_token
    from monsoon_textile_app.utils.audit_log import audit
    result = verify_subscriber_token(req.token)
    audit("subscribe", "verified", ip=_client_ip(request), user=result.get("email", "?"))
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return VerifyEmailResponse(**result)


@router.post("/dispatch-alerts", response_model=DispatchAlertsResponse)
def dispatch_alerts(
    request: Request,
    dry_run: bool = False,
    _key: str = Depends(verify_api_key),   # 🔐 requires valid API key
):
    """Send current alerts to all verified subscribers. Requires API key."""
    from monsoon_textile_app.api.data_bridge import dispatch_alert_emails
    from monsoon_textile_app.utils.audit_log import audit
    result = dispatch_alert_emails(dry_run=dry_run)
    audit("alert", "dispatch_triggered", ip=_client_ip(request), user=_key,
          details={"dry_run": dry_run, "sent": result.get("emails_sent", 0)})
    if result["status"] == "error":
        raise HTTPException(status_code=503, detail=result["message"])
    return DispatchAlertsResponse(**result)


# ── API Key management endpoints (require existing valid key to create new ones) ──

@router.post("/generate-key", response_model=ApiKeyResponse)
def generate_key(
    req: GenerateApiKeyRequest,
    request: Request,
    _key: str = Depends(verify_api_key),   # 🔐 only existing key holders can generate new ones
):
    """Generate a new hashed API key. Raw key returned ONCE — store it securely."""
    from monsoon_textile_app.api.auth import generate_api_key
    from monsoon_textile_app.utils.audit_log import audit
    result = generate_api_key(name=req.name, created_by=_key[:12] + "...")
    audit("auth", "api_key_generated", ip=_client_ip(request), user=_key,
          details={"name": req.name, "key_id": result["key_id"]})
    return ApiKeyResponse(
        **result,
        message="⚠️ Save this key now — it will NOT be shown again.",
    )


@router.get("/keys", response_model=list[ApiKeyResponse])
def list_keys(_key: str = Depends(verify_api_key)):
    """List all API keys (no raw keys — hashes are never returned)."""
    from monsoon_textile_app.api.auth import list_api_keys
    records = list_api_keys()
    return [ApiKeyResponse(**r, message="") for r in records]


@router.post("/revoke-key")
def revoke_key(
    req: RevokeKeyRequest,
    request: Request,
    _key: str = Depends(verify_api_key),
):
    """Deactivate an API key by its key_id."""
    from monsoon_textile_app.api.auth import revoke_api_key
    from monsoon_textile_app.utils.audit_log import audit
    success = revoke_api_key(req.key_id)
    audit("auth", "api_key_revoked", ip=_client_ip(request), user=_key,
          details={"revoked_key_id": req.key_id}, severity="warning")
    if not success:
        raise HTTPException(status_code=404, detail=f"Key '{req.key_id}' not found.")
    return {"status": "revoked", "key_id": req.key_id}

