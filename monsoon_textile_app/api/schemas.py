"""
Pydantic schemas for the REST API request/response models.

Security enhancements (v5):
  - SubscribeRequest validates email with a simple regex + strip sanitizer
  - New UnsubscribeRequest / VerifyEmailRequest schemas (Feature 9)
  - New ApiKeyResponse / RevokeKeyRequest schemas (Feature 1)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# RFC-5322 simplified email pattern
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+$")
_ALLOWED_ALERT_TYPES = {"critical", "warning", "all"}


def _validate_email(v: str) -> str:
    v = v.strip().lower()
    if len(v) > 254:
        raise ValueError("Email address too long (max 254 chars).")
    if not _EMAIL_RE.match(v):
        raise ValueError(f"'{v}' is not a valid email address.")
    return v


# ── Response Models ──────────────────────────────────────────────────────

class StockRisk(BaseModel):
    ticker: str
    name: str
    sector: str
    chain_position: str
    risk_score: float = Field(ge=0, le=1)
    risk_label: str
    volatility_20d: Optional[float] = None
    latest_price: Optional[float] = None


class RiskScoresResponse(BaseModel):
    timestamp: datetime
    n_stocks: int
    avg_risk: float
    scores: list[StockRisk]


class Alert(BaseModel):
    id: str
    severity: str  # "info", "warning", "critical"
    category: str  # "risk", "monsoon", "cotton", "model"
    title: str
    message: str
    timestamp: datetime


class AlertsResponse(BaseModel):
    timestamp: datetime
    n_alerts: int
    alerts: list[Alert]


class SubscribeRequest(BaseModel):
    email: str
    alert_types: list[str] = Field(
        default=["critical"],
        description="Alert severities to subscribe to: 'critical', 'warning', or 'all'",
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return _validate_email(v)

    @field_validator("alert_types")
    @classmethod
    def validate_alert_types(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip().lower() for t in v if t.strip()]
        invalid = [t for t in cleaned if t not in _ALLOWED_ALERT_TYPES]
        if invalid:
            raise ValueError(
                f"Invalid alert types: {invalid}. Allowed: {_ALLOWED_ALERT_TYPES}"
            )
        return cleaned or ["critical"]


class SubscribeResponse(BaseModel):
    status: str
    email: str
    alert_types: list[str]
    message: str
    verification_sent: bool = False


class UnsubscribeRequest(BaseModel):
    email: str
    token: Optional[str] = None   # HMAC-signed unsubscribe token

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return _validate_email(v)


class UnsubscribeResponse(BaseModel):
    status: str
    email: str
    message: str


class VerifyEmailRequest(BaseModel):
    token: str = Field(min_length=32, max_length=256)


class VerifyEmailResponse(BaseModel):
    status: str
    email: str
    message: str


class DispatchAlertsResponse(BaseModel):
    status: str
    timestamp: datetime
    total_subscribers: int
    recipients_targeted: int
    total_alerts: int
    emails_sent: int
    deliveries_skipped: int
    failures: list[str]
    dry_run: bool
    message: str


class HealthResponse(BaseModel):
    status: str  # "ok", "degraded", "error"
    version: str
    uptime_seconds: float
    data_loaded: bool
    n_stocks: int
    last_update: Optional[datetime] = None


class MonsoonStatus(BaseModel):
    avg_deficit_pct: float
    stressed_states: list[str]
    spatial_breadth_pct: float


class CottonStatus(BaseModel):
    price_change_30d_pct: float
    regime: str
    latest_price: Optional[float] = None


class DashboardSummaryResponse(BaseModel):
    timestamp: datetime
    risk: RiskScoresResponse
    monsoon: MonsoonStatus
    cotton: CottonStatus
    vix: Optional[float] = None


# ── API Key Management Schemas (Feature 1) ────────────────────────────────

class GenerateApiKeyRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80, description="Descriptive name for this key")

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        # Strip any HTML / control characters
        safe = re.sub(r"[<>\"'&]", "", v).strip()
        if not safe:
            raise ValueError("Key name must contain printable characters.")
        return safe


class ApiKeyResponse(BaseModel):
    key_id: str
    name: str
    raw_key: Optional[str] = None   # Only set on generation — never returned again
    created_at: str
    last_used: Optional[str] = None
    call_count: int = 0
    active: bool = True
    message: str = ""


class RevokeKeyRequest(BaseModel):
    key_id: str = Field(min_length=4, max_length=50)

