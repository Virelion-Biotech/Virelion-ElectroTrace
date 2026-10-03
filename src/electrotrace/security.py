"""Small deployment-safety helpers for the ElectroTrace API."""
from __future__ import annotations

import hmac
import ipaddress
import os
from typing import Callable

from flask import Response, request


class ApiKeyMiddleware:
    """Require a configured API key for non-local API requests.

    The UI/static assets remain public. API protection is enabled whenever
    ELECTROTRACE_API_KEY is configured.
    """

    def __init__(self, app, api_key: str):
        if not api_key:
            raise ValueError("api_key must not be empty")
        self.app = app
        self.api_key = api_key.encode("utf-8")

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if not path.startswith("/api/"):
            return self.app(environ, start_response)
        supplied = environ.get("HTTP_AUTHORIZATION", "")
        prefix = "Bearer "
        token = supplied[len(prefix):] if supplied.startswith(prefix) else ""
        if not token or not hmac.compare_digest(token.encode("utf-8"), self.api_key):
            response = Response('{"error":"API authentication required"}\n', status=401, content_type="application/json")
            return response(environ, start_response)
        return self.app(environ, start_response)


def configured_api_key() -> str | None:
    value = os.getenv("ELECTROTRACE_API_KEY", "").strip()
    return value or None


def validate_bind(host: str, api_key: str | None) -> None:
    """Refuse an externally reachable bind unless an API key is configured."""
    host = str(host).strip()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return
    try:
        ip = ipaddress.ip_address(host)
        external = not ip.is_loopback
    except ValueError:
        external = True
    if external and not api_key:
        raise RuntimeError(
            "Refusing non-localhost bind without ELECTROTRACE_API_KEY; "
            "set a strong API key before exposing ElectroTrace beyond localhost"
        )
