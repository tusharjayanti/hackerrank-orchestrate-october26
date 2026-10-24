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
        if k.startswith("ANTHROPIC_") and k not in os.environ:
            os.environ[k] = v.strip().strip('"').strip("'")


def credentials_present() -> bool:
    if os.environ.get("GHOSTAUDIT_OFFLINE") == "1":
        # Tests and engine-only runs never reach the network.
        return False
    _load_dotenv()
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def image_block(path: Path, media_type: str) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    kind = "document" if media_type == "application/pdf" else "image"
    return {"type": kind, "source": {"type": "base64", "media_type": media_type, "data": data}}


class LLM:
    def __init__(self, budget: Budget | None = None, model: str = MODEL):
        self.budget = budget or Budget()
        self.model = model
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            if not credentials_present():
                raise LLMUnavailable("no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN in environment or .env")
            import anthropic

            self._client = anthropic.Anthropic()
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
        model = model or self.model
        key = self.cache_key(model, prompt_version, system, content, schema)
        cached = CACHE_DIR / f"{key}.json"
        if cached.is_file():
            out = json.loads(cached.read_text(encoding="utf-8"))
            self._log({"purpose": purpose, "prompt_version": prompt_version, "model": out["model"], "cache_key": key,
                       "cached": True, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
            return out["result"]
        self.budget.check()
        client = self._client_or_raise()
        import anthropic

        start = time.monotonic()
        try:
            # No server-side fallback: a refusal is surfaced and logged with the
            # model that refused, so every usage row names the model that read.
            resp = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.RateLimitError:
            raise
        except anthropic.APIStatusError as e:
            log.error("llm_api_error", purpose=purpose, status=e.status_code, request_id=getattr(e, "request_id", None))
            raise
        if resp.stop_reason in ("refusal", "max_tokens"):
            category = getattr(getattr(resp, "stop_details", None), "category", None)
            log.error("llm_unusable_stop", purpose=purpose, model=resp.model, stop_reason=resp.stop_reason,
                      category=category, request_id=resp._request_id)
            self._log({"purpose": purpose, "prompt_version": prompt_version, "model": resp.model, "cache_key": key,
                       "cached": False, "stop_reason": resp.stop_reason, "refusal_category": category,
                       "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                       "usd": round(self.budget.record(resp.model, resp.usage.input_tokens,
                                                       resp.usage.output_tokens), 6),
                       "request_id": resp._request_id})
            raise RuntimeError(f"model stopped with {resp.stop_reason}")
        text = next(b.text for b in resp.content if b.type == "text")
        result = json.loads(text)
        usage = resp.usage
        cost = self.budget.record(model, usage.input_tokens, usage.output_tokens)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"model": resp.model, "prompt_version": prompt_version, "result": result},
                                     sort_keys=True, indent=1), encoding="utf-8")
        self._log({"purpose": purpose, "prompt_version": prompt_version, "model": resp.model, "cache_key": key,
                   "cached": False, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                   "usd": round(cost, 6), "latency_s": round(time.monotonic() - start, 2),
                   "request_id": resp._request_id})
        return result

    def _log(self, row: dict) -> None:
        CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CALL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


__all__ = ["LLM", "LLMUnavailable", "BudgetExceeded", "credentials_present", "image_block", "MODEL", "SONNET"]
