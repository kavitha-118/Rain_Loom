"""
In-Memory Sliding Window Rate Limiter — RainLoom Security Layer
===============================================================
No Redis needed — uses per-IP sliding window counters stored in memory.

Limits:
  - Read endpoints  : 60 requests / 60 s per IP
  - Write endpoints : 10 requests / 60 s per IP
  - Subscribe       :  5 requests / 60 s per IP (anti-spam)

Returns HTTP 429 with Retry-After header when exceeded.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Per-route limit configuration  (path_prefix → (max_calls, window_secs))
# ---------------------------------------------------------------------------
_ROUTE_LIMITS: dict[str, tuple[int, int]] = {
    "/api/subscribe":       (5,  60),   # anti-spam subscribe
    "/api/dispatch-alerts": (5,  60),   # prevent dispatch flood
    "/api/verify":          (10, 60),   # verification attempts
    "/api/unsubscribe":     (10, 60),
    "/api/generate-key":    (10, 60),   # key generation
}
_DEFAULT_READ_LIMIT:  tuple[int, int] = (60, 60)   # 60 req / 60 s
_DEFAULT_WRITE_LIMIT: tuple[int, int] = (20, 60)   # 20 req / 60 s
_GLOBAL_LIMIT:        tuple[int, int] = (300, 60)  # 300 req / 60 s total

# ---------------------------------------------------------------------------
# Thread-safe storage
# ---------------------------------------------------------------------------
# {ip: deque of timestamps}
_per_ip: dict[str, deque] = defaultdict(deque)
_global_ts: deque = deque()
_lock = Lock()


def _is_rate_limited(ip: str, path: str, method: str) -> tuple[bool, int]:
    """
    Check whether the request should be rate-limited.

    Returns
    -------
    (limited: bool, retry_after_secs: int)
    """
    now = time.time()

    # Find the applicable limit for this route
    route_limit = None
    for prefix, limit in _ROUTE_LIMITS.items():
        if path.startswith(prefix):
            route_limit = limit
            break

    if route_limit is None:
        route_limit = (
            _DEFAULT_WRITE_LIMIT if method in ("POST", "PUT", "DELETE", "PATCH")
            else _DEFAULT_READ_LIMIT
        )

    max_calls, window = route_limit
    cutoff = now - window

    with _lock:
        # Prune old timestamps
        q = _per_ip[ip]
        while q and q[0] < cutoff:
            q.popleft()

        # Prune global
        global_cutoff = now - _GLOBAL_LIMIT[1]
        while _global_ts and _global_ts[0] < global_cutoff:
            _global_ts.popleft()

        # Check global limit first
        if len(_global_ts) >= _GLOBAL_LIMIT[0]:
            retry_after = int(_global_ts[0] + _GLOBAL_LIMIT[1] - now) + 1
            return True, max(retry_after, 1)

        # Check per-IP limit
        if len(q) >= max_calls:
            retry_after = int(q[0] + window - now) + 1
            return True, max(retry_after, 1)

        # Allow — record this request
        q.append(now)
        _global_ts.append(now)

    return False, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Starlette/FastAPI middleware that enforces sliding-window rate limits.
    Skips the health endpoint and the Streamlit UI entirely.
    """

    SKIP_PATHS = {"/api/health", "/", "/api/docs", "/api/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next: Callable):
        path   = request.url.path
        method = request.method

        # Skip non-API paths and health checks
        if not path.startswith("/api") or path in self.SKIP_PATHS:
            return await call_next(request)

        # Resolve client IP (handles X-Forwarded-For from proxies)
        x_forwarded = request.headers.get("X-Forwarded-For")
        ip = (
            x_forwarded.split(",")[0].strip()
            if x_forwarded
            else (request.client.host if request.client else "unknown")
        )

        limited, retry_after = _is_rate_limited(ip, path, method)
        if limited:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": (
                        f"Too many requests from {ip}. "
                        f"Please wait {retry_after} seconds before retrying."
                    ),
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
