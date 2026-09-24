"""Thin, swappable LLM boundary (Gemini REST API).

Only this module knows about the provider. Responses are cached on disk keyed
by (model, prompt, schema) so that re-running the pipeline on the same inputs
is reproducible and does not re-bill the API.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import requests

DEFAULT_MODEL = "gemini-2.5-flash-lite"
_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    model: str

    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]: ...


class JsonFileCache:
    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, Any] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._data = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self._data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")


def cache_key(model: str, prompt: str, schema: Dict[str, Any]) -> str:
    blob = json.dumps({"model": model, "prompt": prompt, "schema": schema}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ReplayClient:
    """Answers only from the response cache - never calls the network.

    Used to reproduce an LLM-mode run exactly (e.g. by validate.py) without an API key.
    A cache miss raises LLMError, which the pipeline handles like any model failure.
    """

    def __init__(self, model: str, cache: JsonFileCache):
        self.model = model
        self.cache = cache

    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        hit = self.cache.get(cache_key(self.model, prompt, schema))
        if hit is None:
            raise LLMError("no cached response (replay mode)")
        return hit


class GeminiClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, cache: Optional[JsonFileCache] = None,
                 timeout: float = 60.0, max_retries: int = 4):
        self.api_key = api_key
        self.max_retries = max_retries
        self.model = model
        self.cache = cache
        self.timeout = timeout

    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        key = cache_key(self.model, prompt, schema)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        generation_config: Dict[str, Any] = {
            "temperature": 0.0,
            "topP": 1.0,
            "candidateCount": 1,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        }
        if self.model.startswith("gemini-2.5-flash"):
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": generation_config}
        resp = self._post(body)
        try:
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            data = json.loads(text)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise LLMError(f"unparseable response: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMError("response is not a JSON object")
        if self.cache is not None:
            self.cache.set(key, data)
        return data

    def _post(self, body: Dict[str, Any]) -> requests.Response:
        """POST with bounded retries on rate limits (429) and transient 5xx errors."""
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    _API_URL.format(model=self.model),
                    headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                    json=body,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"request failed: {exc.__class__.__name__}") from exc
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429 and "PerDay" in resp.text:
                # Daily quota exhausted: waiting won't help, fail fast so the pipeline falls back.
                raise LLMError(f"HTTP 429 (daily quota exhausted): {_error_message(resp)}")
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                time.sleep(_retry_delay(resp, default=2 ** (attempt + 1)))
                continue
            raise LLMError(f"HTTP {resp.status_code}: {_error_message(resp)}")
        raise LLMError("retries exhausted")


def _error_message(resp: requests.Response) -> str:
    try:
        return str(resp.json()["error"]["message"]).splitlines()[0][:120]
    except (ValueError, KeyError, TypeError):
        return resp.text[:120]


def _retry_delay(resp: requests.Response, default: float) -> float:
    """Honour the server-suggested retry delay (e.g. "retryDelay": "17s"), capped at 30s."""
    match = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', resp.text)
    delay = float(match.group(1)) + 1 if match else default
    return min(delay, 30.0)


def client_from_env(cache_path: Path) -> Optional[GeminiClient]:
    """Return a Gemini client if GEMINI_API_KEY is set, else None (offline mode)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    return GeminiClient(key, model=model, cache=JsonFileCache(cache_path))
