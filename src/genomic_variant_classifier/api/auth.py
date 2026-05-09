"""
src/api/auth.py
===============
API key authentication for the variant pathogenicity API.

Authentication scheme
---------------------
Header: ``X-API-Key: <key>``

When the ``API_KEYS`` environment variable is empty or unset, all requests
are allowed (development / testing mode).  Set it to a comma-separated
list of valid keys to require authentication:

    API_KEYS=secret-key-1,secret-key-2

/health is intentionally exempt — do not apply this dependency there.
Load-balancer probes and Docker HEALTHCHECK must reach /health unauthenticated.
"""

from __future__ import annotations

import os

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_VALID_KEYS: frozenset[str] = frozenset(
    k.strip()
    for k in os.environ.get("API_KEYS", "").split(",")
    if k.strip()
)


async def require_api_key(key: str | None = Depends(_API_KEY_HEADER)) -> str:
    """
    FastAPI dependency.  Returns the accepted key (or "dev" when auth is disabled).

    Usage::

        @app.post("/predict")
        async def predict(
            body: VariantRequest,
            _key: str = Depends(require_api_key),
        ) -> PredictResponse:
            ...
    """
    if not _VALID_KEYS:
        return "dev"   # auth disabled — all requests allowed
    if not key or key not in _VALID_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
    return key
