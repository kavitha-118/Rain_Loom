"""
FastAPI Application -- Monsoon-Textile REST API Gateway
=======================================================

Security enhancements (v5):
  - CORS restricted to env-driven allowlist (CORS_ORIGINS)
  - RateLimitMiddleware enforces per-IP sliding-window limits
  - Security HTTP headers injected on every response
  - Audit log wired to startup / shutdown events

Run standalone:
    uvicorn monsoon_textile_app.api.app:app --host 0.0.0.0 --port 8000

Or enable alongside Streamlit by setting ENABLE_API=1 in environment.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from monsoon_textile_app.api.rate_limiter import RateLimitMiddleware
from monsoon_textile_app.api.routes import router

# ---------------------------------------------------------------------------
# CORS allowlist — read from env; fall back to dev defaults
# ---------------------------------------------------------------------------
def _cors_origins() -> list[str]:
    raw = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:8501,https://rainloomtextiles.streamlit.app,https://rainloom.vercel.app",
    ).strip()
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="RainLoom Volatility API",
    description=(
        "REST API for the RainLoom Risk System. "
        "Provides programmatic access to ensemble risk scores, "
        "alerts, and monsoon-cotton-textile transmission chain data."
    ),
    version="5.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── 1. Rate limiting (before CORS so limits apply regardless of origin) ──
app.add_middleware(RateLimitMiddleware)

# ── 2. CORS — restricted allowlist, credentials disabled for public routes ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,          # credentials=True + wildcard is a CORS misconfiguration
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Api-Key", "Authorization"],
    expose_headers=["Retry-After"],
)


# ── 3. Security response headers middleware ─────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next: Callable) -> Response:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    # Only add HSTS on production (skip localhost)
    host = request.headers.get("host", "")
    if "localhost" not in host and "127.0.0.1" not in host:
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    return response


app.include_router(router)


# ── 4. Startup / shutdown audit events ─────────────────────────────────
@app.on_event("startup")
async def _on_startup():
    try:
        from monsoon_textile_app.utils.audit_log import audit
        audit("system", "api_startup", details={"version": "5.0.0"})
    except Exception:
        pass


@app.on_event("shutdown")
async def _on_shutdown():
    try:
        from monsoon_textile_app.utils.audit_log import audit
        audit("system", "api_shutdown", details={"version": "5.0.0"})
    except Exception:
        pass


@app.get("/")
def root():
    return {
        "name": "Monsoon-Textile Volatility API",
        "version": "5.0.0",
        "docs": "/api/docs",
        "security": "X-Api-Key header required on write endpoints",
        "endpoints": [
            "/api/health",
            "/api/risk-scores",
            "/api/alerts",
            "/api/subscribe",
            "/api/unsubscribe",
            "/api/verify",
            "/api/dispatch-alerts",
            "/api/generate-key",
            "/api/keys",
            "/api/revoke-key",
        ],
    }


def start_api_background(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server in a background thread (for Streamlit co-hosting)."""
    import threading
    import uvicorn

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True, name="fastapi-bg")
    thread.start()
    print(f"[API] FastAPI server started on http://{host}:{port}")
    return thread


def start_email_dispatch_scheduler() -> "threading.Thread":
    """
    Start periodic email dispatch in a background thread.

    Env vars:
    - ENABLE_EMAIL_SCHEDULER: 1/true/yes to enable
    - EMAIL_DISPATCH_INTERVAL_SECONDS: default 900 (15 minutes)
    - EMAIL_DISPATCH_DRY_RUN: 1/true/yes to simulate only
    """
    import threading
    from monsoon_textile_app.api.data_bridge import dispatch_alert_emails

    interval_raw = os.environ.get("EMAIL_DISPATCH_INTERVAL_SECONDS", "900").strip()
    try:
        interval_seconds = max(30, int(interval_raw))
    except ValueError:
        interval_seconds = 900

    dry_run = os.environ.get("EMAIL_DISPATCH_DRY_RUN", "0").strip().lower() in ("1", "true", "yes")

    def _run():
        print(
            f"[API] Email scheduler started | interval={interval_seconds}s | dry_run={dry_run}"
        )
        while True:
            try:
                result = dispatch_alert_emails(dry_run=dry_run)
                print(
                    "[API] Email dispatch | status={status} sent={sent} targeted={targeted} alerts={alerts}".format(
                        status=result.get("status"),
                        sent=result.get("emails_sent"),
                        targeted=result.get("recipients_targeted"),
                        alerts=result.get("total_alerts"),
                    )
                )
                if result.get("failures"):
                    print(f"[API] Email dispatch failures: {result.get('failures')}")
            except Exception as exc:
                print(f"[API] Email scheduler error: {exc}")

            time.sleep(interval_seconds)

    thread = threading.Thread(target=_run, daemon=True, name="email-dispatch-scheduler")
    thread.start()
    return thread


# Auto-start when ENABLE_API=1
if os.environ.get("ENABLE_API", "").strip() in ("1", "true", "yes"):
    start_api_background()


# Optional periodic alert dispatch
if os.environ.get("ENABLE_EMAIL_SCHEDULER", "").strip().lower() in ("1", "true", "yes"):
    start_email_dispatch_scheduler()
