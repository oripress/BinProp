"""OpenAI-compatible chat client, defaulting to Nebius Token Factory."""

from __future__ import annotations

import http.client
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_API_KEY_ENV = "NEBIUS_API_KEY"


class ProviderError(RuntimeError):
    """An API failure with a retry decision independent of its message."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def _open(request, *, timeout):
    # A provider redirect must never forward the API key to another destination.
    return urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout)


@dataclass
class ChatClient:
    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = DEFAULT_API_KEY_ENV
    max_tokens: int = 4096
    timeout: float = 120.0
    attempts: int = 2
    extra_body: dict = field(default_factory=dict)

    def __post_init__(self):
        url = urllib.parse.urlsplit(self.base_url)
        if (
            url.scheme not in ("https", "http")
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) API base URL without credentials or query parameters"
            )
        if not self.model.strip():
            raise ValueError("A model is required")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if type(self.attempts) is not int or not 1 <= self.attempts <= 5:
            raise ValueError("attempts must be between 1 and 5")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if not isinstance(self.extra_body, dict):
            raise ValueError("extra_body must be a JSON object")
        if set(self.extra_body) & {"model", "messages", "stream", "max_tokens"}:
            raise ValueError(
                "extra_body cannot override model, messages, stream, or max_tokens"
            )
        self.base_url = self.base_url.rstrip("/")

    def complete(self, messages: list[dict[str, str]]) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(
                f"Set the {self.api_key_env} environment variable", retryable=False
            )
        body = json.dumps(
            {
                **self.extra_body,
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "stream": False,
            },
            allow_nan=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
            },
        )
        for attempt in range(self.attempts):
            try:
                with _open(request, timeout=self.timeout) as response:
                    value = json.load(response)
                text = value["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ProviderError("The model returned no text")
                if value["choices"][0].get("finish_reason") == "length":
                    raise ProviderError(
                        "The model output was truncated; increase max_tokens"
                    )
                return text
            except urllib.error.HTTPError as exc:
                retryable = exc.code in (429, 500, 502, 503, 504)
                status = exc.code
                exc.close()
                if not retryable or attempt + 1 == self.attempts:
                    # Never echo response bodies, request headers, or credentials.
                    raise ProviderError(
                        f"Model endpoint returned HTTP {status}",
                        retryable=retryable,
                        status_code=status,
                    ) from None
            except (OSError, http.client.HTTPException):
                # A connection can fail while reading an already-open response.
                if attempt + 1 == self.attempts:
                    raise ProviderError(
                        "Model endpoint could not be reached within the request timeout"
                    ) from None
            except (ValueError, KeyError, IndexError, TypeError):
                raise ProviderError(
                    "Model endpoint returned an invalid chat-completion response"
                ) from None
            time.sleep(min(2**attempt, 8))
        raise ProviderError("Model request failed")
