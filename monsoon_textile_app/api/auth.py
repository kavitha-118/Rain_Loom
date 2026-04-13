"""
API Key Authentication — RainLoom Security Layer
=================================================
Provides hashed key storage and FastAPI Security dependency for
protecting write/sensitive endpoints.

Key design decisions:
  - Keys are displayed to the user exactly once (at generation time)
  - Only sha256(key) is persisted on disk — never the raw key
  - Key format: rl_live_<32 hex chars>
  - Invalid / missing key → HTTP 403 Forbidden
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_KEY_FILE = Path(__file__).parent / "api_keys.json"
_KEY_HEADER = APIKeyHeader(name="X-Api-Key", auto_error=False)
_KEY_PREFIX = "rl_live_"

# In-memory lookup: hash -> metadata (populated on first request)
_key_hash_cache: dict[str, dict] = {}
_cache_loaded_at: float = 0.0
_CACHE_TTL = 60  # reload from disk every 60 s


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _hash_key(raw_key: str) -> str:
    """Return sha256 hex digest of raw_key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _load_key_store() -> list[dict]:
    """Load api_keys.json from disk (creates empty file if absent)."""
    if not _KEY_FILE.exists():
        _KEY_FILE.write_text("[]", encoding="utf-8")
    try:
        return json.loads(_KEY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_key_store(records: list[dict]) -> None:
    _KEY_FILE.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


def _refresh_cache() -> None:
    global _key_hash_cache, _cache_loaded_at
    records = _load_key_store()
    _key_hash_cache = {r["key_hash"]: r for r in records if "key_hash" in r}
    _cache_loaded_at = time.time()


def _ensure_cache() -> None:
    if time.time() - _cache_loaded_at > _CACHE_TTL:
        _refresh_cache()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_api_key(name: str, created_by: str = "dashboard") -> dict:
    """
    Generate a new API key, store only its hash, return the raw key once.

    Returns
    -------
    dict with keys: raw_key (show once!), key_id, name, created_at
    """
    raw_key = _KEY_PREFIX + secrets.token_hex(16)
    key_hash = _hash_key(raw_key)
    key_id = "kid_" + secrets.token_hex(6)
    created_at = datetime.now(timezone.utc).isoformat()

    records = _load_key_store()
    records.append(
        {
            "key_id": key_id,
            "name": name,
            "key_hash": key_hash,
            "created_at": created_at,
            "created_by": created_by,
            "last_used": None,
            "call_count": 0,
            "active": True,
        }
    )
    _save_key_store(records)
    _refresh_cache()

    return {
        "raw_key": raw_key,        # ← show to user ONCE, never stored
        "key_id": key_id,
        "name": name,
        "created_at": created_at,
    }


def list_api_keys() -> list[dict]:
    """Return all key records (without raw keys — they are never stored)."""
    _ensure_cache()
    records = _load_key_store()
    # Strip key_hash before returning to UI
    return [
        {k: v for k, v in r.items() if k != "key_hash"}
        for r in records
    ]


def revoke_api_key(key_id: str) -> bool:
    """Mark a key as inactive. Returns True if found."""
    records = _load_key_store()
    for r in records:
        if r.get("key_id") == key_id:
            r["active"] = False
            _save_key_store(records)
            _refresh_cache()
            return True
    return False


def _is_valid_key(raw_key: str) -> bool:
    """Return True if raw_key matches an active hashed key."""
    _ensure_cache()
    if not raw_key or not raw_key.startswith(_KEY_PREFIX):
        return False
    h = _hash_key(raw_key)
    rec = _key_hash_cache.get(h)
    if rec and rec.get("active", True):
        # Update last_used + call_count asynchronously (best-effort)
        try:
            records = _load_key_store()
            for r in records:
                if r.get("key_hash") == h:
                    r["last_used"] = datetime.now(timezone.utc).isoformat()
                    r["call_count"] = r.get("call_count", 0) + 1
                    break
            _save_key_store(records)
        except Exception:
            pass
        return True
    return False


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def verify_api_key(key: Optional[str] = Security(_KEY_HEADER)) -> str:
    """
    FastAPI Security dependency — raises 403 if key is missing or invalid.
    Use as: `Depends(verify_api_key)` on any protected route.
    """
    if not key or not _is_valid_key(key):
        raise HTTPException(
            status_code=403,
            detail=(
                "Invalid or missing API key. "
                "Pass your key in the 'X-Api-Key' header. "
                "Generate one at the RainLoom Institutional API page."
            ),
        )
    return key


def optional_api_key(key: Optional[str] = Security(_KEY_HEADER)) -> Optional[str]:
    """
    Like verify_api_key but returns None instead of raising 403.
    Useful for endpoints that are public but enhanced with auth.
    """
    if key and _is_valid_key(key):
        return key
    return None
