"""API key authentication middleware for the VeraLLM miner server."""

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require a valid API key on all requests except /health.

    Reads the expected key from ``VERATHOS_API_KEY`` env var (or the
    *api_key* constructor argument).  When no key is configured, all
    requests are allowed through (dev mode).

    Clients can authenticate via either header:
        Authorization: Bearer <key>
        X-API-Key: <key>
    """

    def __init__(self, app, api_key: str | None = None):
        super().__init__(app)
        self.api_key = api_key or os.environ.get("VERATHOS_API_KEY")

    async def dispatch(self, request: Request, call_next):
        if not self.api_key:
            return await call_next(request)
        if request.url.path == "/health":
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        x_key = request.headers.get("x-api-key", "")
        token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else x_key

        if token != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key"},
            )
        return await call_next(request)
