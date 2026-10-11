"""A12: the UI's only door into DocuMind -- HTTP calls to the API of A11.

The UI never imports app/: no database connection, no models, no Gemini key live on
this side. It sends the question to POST /ask and draws what comes back, exactly like
a mobile app or a website widget would. Two consequences:
  - the API's rules (audience filter, admin key, input checks) protect the UI too,
    nothing here can skip them
  - the UI can run on another machine or container (A13) and stays small
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_URL = os.environ.get("DOCUMIND_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 90.0  # seconds. A11 measured a retried answer at ~60 s on a slow free-tier day.


class ApiError(Exception):
    """The API said no, or could not be reached. status=None means no answer at all."""

    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class ApiClient:
    base_url: str = DEFAULT_URL
    api_key: str | None = None  # the support agent's X-API-Key; None for customers
    transport: httpx.BaseTransport | None = None  # tests plug a fake API in here

    def ask(self, question: str, audience: str = "public", k: int = 5) -> dict[str, Any]:
        return self._request("POST", "/ask", json={"question": question, "audience": audience, "k": k})

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", timeout=5.0)

    def _request(self, method: str, path: str, timeout: float = TIMEOUT, **kwargs: Any) -> dict[str, Any]:
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            with httpx.Client(base_url=self.base_url, timeout=timeout, transport=self.transport) as http:
                response = http.request(method, path, headers=headers, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiError(None, f"The API did not answer within {timeout:.0f} s.") from exc
        except httpx.TransportError as exc:
            raise ApiError(
                None, f"Cannot reach the API at {self.base_url}. Is it running? (uv run uvicorn app.api:app)"
            ) from exc
        if response.is_error:
            raise ApiError(response.status_code, error_detail(response))
        return response.json()


def error_detail(response: httpx.Response) -> str:
    """FastAPI's error text: {"detail": "..."} or, for a rejected request, a list of problems."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(detail, list):  # 422 from pydantic: [{"loc": [...], "msg": "..."}, ...]
        return "; ".join(f"{item['loc'][-1]}: {item['msg']}" for item in detail)
    return str(detail)
