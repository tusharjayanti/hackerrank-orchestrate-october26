"""The only place that talks to the model.

Every call returns schema-constrained JSON, is cached on disk by
sha256(model + prompt_version + system + content + schema), and is logged to
runs/llm_calls.jsonl with token usage. A cached answer costs nothing and is
logged as such, so reruns are deterministic and free.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

from . import CODE_ROOT, REPO_ROOT
from .budget import Budget, BudgetExceeded
from .obs import get_logger

log = get_logger(__name__)

MODEL = "claude-opus-5-5"
SONNET = "claude-sonnet-5-5"
# Per-stage models, overridable from the environment or .env. Vision and
# intent are transcription/classification with an agreement check behind
# them; the agent and the vision tiebreak are judgment calls with no second
# check, so they default to the stronger model.
STAGE_DEFAULTS = {"VISION": SONNET, "INTENT": SONNET, "AGENT": MODEL}


def model_for(stage: str) -> str:
    _load_dotenv()
    return os.environ.get(f"MODEL_{stage}") or STAGE_DEFAULTS[stage]
CACHE_DIR = CODE_ROOT / ".cache" / "llm"
CALL_LOG = CODE_ROOT / "runs" / "llm_calls.jsonl"


class LLMUnavailable(RuntimeError):
    pass


def _load_dotenv() -> None:
    """Read ANTHROPIC_* keys from an uncommitted .env at the repo root, if
    present. Values are never logged."""
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if (k.startswith("ANTHROPIC_") or k.startswith("MODEL_")) and k not in os.environ:
            os.environ[k] = v.strip().strip('"').strip("'")


def credentials_present() -> bool:
    if os.environ.get("GHOSTAUDIT_OFFLINE") == "1":
        # Tests and engine-only runs never reach the network.
        return False
    _load_dotenv()
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def replay_cache_present() -> bool:
    """A populated answer cache can be replayed without credentials."""
    return CACHE_DIR.is_dir() and any(CACHE_DIR.glob("*.json"))


def model_access() -> str:
    """live: credentials present (cache hits still replay, misses call the
    API); replay: no credentials but a cache, so only cached answers are used
    and a miss degrades with a flag; none: offline or nothing to replay."""
    if os.environ.get("GHOSTAUDIT_OFFLINE") == "1":
        return "none"
    if credentials_present():
        return "live"
    return "replay" if replay_cache_present() else "none"


def image_block(path: Path, media_type: str) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    kind = "document" if media_type == "application/pdf" else "image"
    return {"type": kind, "source": {"type": "base64", "media_type": media_type, "data": data}}


class LLMExtractionFailed(RuntimeError):
    """The model could not produce a usable answer (retries exhausted,
    refusal, truncation, malformed JSON twice, or budget reached). Callers
    proceed without that extraction and flag the affected rows."""


SDK_RETRIES = 4  # exponential backoff on 408/409/429/5xx and connection errors
JSON_ATTEMPTS = 2  # one retry when the response is not valid JSON


class LLM:
    def __init__(self, budget: Budget | None = None, model: str = MODEL, run_id: str = "", allow_live: bool = True):
        self.budget = budget or Budget()
        self.model = model
        self.run_id = run_id
        # False: answers come from the cache only, even with credentials
        # (measure runs without --allow-live); a miss is LLMUnavailable.
        self.allow_live = allow_live
        self._client = None

    def _client_or_raise(self):
        if not self.allow_live:
            raise LLMUnavailable("live model calls are refused in this run (measure mode without --allow-live)")
        if self._client is None:
            if not credentials_present():
                raise LLMUnavailable("no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN in environment or .env")
            import anthropic

            self._client = anthropic.Anthropic(max_retries=SDK_RETRIES)
        return self._client

    def cache_key(self, model: str, prompt_version: str, system: str, content: list[dict], schema: dict) -> str:
        h = hashlib.sha256()
        for part in (model, prompt_version, system, json.dumps(content, sort_keys=True),
                     json.dumps(schema, sort_keys=True)):
            h.update(part.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    def json_call(self, *, purpose: str, prompt_version: str, system: str, content: list[dict], schema: dict,
                  max_tokens: int = 8000, effort: str = "medium", model: str | None = None) -> dict:
        """Schema-constrained JSON from the model, served from cache when the
        exact request was answered before. Raises LLMExtractionFailed."""
        model = model or self.model
        key = self.cache_key(model, prompt_version, system, content, schema)
        cached = CACHE_DIR / f"{key}.json"
        base = {"purpose": purpose, "prompt_version": prompt_version, "cache_key": key}
        if cached.is_file():
            try:
                out = json.loads(cached.read_text(encoding="utf-8"))
                result = out["result"]
            except (OSError, ValueError, KeyError, TypeError):
                # A damaged cache entry is a miss, never a crash.
                log.warning("llm_cache_unreadable", purpose=purpose, cache_key=key)
            else:
                self._log({**base, "model": out.get("model", model), "cached": True, "input_tokens": 0,
                           "output_tokens": 0, "usd": 0.0})
                return result
        for attempt in range(1, JSON_ATTEMPTS + 1):
            result, served_by = self._request(base, model, system, content, schema, max_tokens, effort, attempt)
            if result is not None:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps({"model": served_by, "prompt_version": prompt_version,
                                              "result": result}, sort_keys=True, indent=1), encoding="utf-8")
                return result
        raise LLMExtractionFailed(f"{purpose}: malformed JSON after {JSON_ATTEMPTS} attempts")

    def _request(self, base: dict, model: str, system: str, content: list[dict], schema: dict, max_tokens: int,
                 effort: str, attempt: int) -> tuple[dict | None, str]:
        """One API call. Returns (parsed JSON or None if malformed, model)."""
        import anthropic

        try:
            self.budget.check()
        except BudgetExceeded as e:
            raise LLMExtractionFailed(str(e)) from e
        client = self._client_or_raise()
        start = time.monotonic()
        try:
            # No server-side fallback: a refusal is surfaced and logged with the
            # model that refused, so every usage row names the model that read.
            resp = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.APIError as e:  # raised only after the SDK's own retries
            log.error("llm_api_error", purpose=base["purpose"], error=type(e).__name__,
                      status=getattr(e, "status_code", None), request_id=getattr(e, "request_id", None))
            raise LLMExtractionFailed(f"{base['purpose']}: {type(e).__name__}") from e
        usage = resp.usage
        row = {**base, "model": resp.model, "cached": False, "attempt": attempt, "stop_reason": resp.stop_reason,
               "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
               "usd": round(self.budget.record(resp.model, usage.input_tokens, usage.output_tokens), 6),
               "latency_s": round(time.monotonic() - start, 2), "request_id": resp._request_id}
        if resp.stop_reason in ("refusal", "max_tokens"):
            row["refusal_category"] = getattr(getattr(resp, "stop_details", None), "category", None)
            self._log(row)
            log.error("llm_unusable_stop", purpose=base["purpose"], model=resp.model, stop_reason=resp.stop_reason,
                      category=row["refusal_category"], request_id=resp._request_id)
            raise LLMExtractionFailed(f"{base['purpose']}: model stopped with {resp.stop_reason}")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            row["malformed_json"] = True
            self._log(row)
            log.warning("llm_malformed_json", purpose=base["purpose"], attempt=attempt, request_id=resp._request_id)
            return None, resp.model
        self._log(row)
        return result, resp.model

    def _log(self, row: dict) -> None:
        row = {**row, "run_id": self.run_id}
        CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CALL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


__all__ = ["LLM", "LLMUnavailable", "LLMExtractionFailed", "BudgetExceeded", "credentials_present", "image_block",
           "MODEL", "SONNET", "model_for", "model_access"]
