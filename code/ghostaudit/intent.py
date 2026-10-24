"""Request intent: does the request authorize deletion?

Unique texts are classified in batches of 25 with Sonnet; the output is an
enum per text and nothing else, so an instruction embedded in a request can
change at most its label, never the steps. The engine never reads the text.

Fallbacks: a text the model does not return, or labels with low confidence,
is treated as DELETION (the outcome safer for the subject) and flagged.
Without credentials every text is DELETION and flagged INTENT_NOT_CLASSIFIED.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import intent_prompts as ip
from .llm import LLM, SONNET, LLMUnavailable
from .obs import get_logger
from .schemas import Intent, IntentLabel

log = get_logger(__name__)

BATCH_SIZE = 25


@dataclass(frozen=True)
class Classified:
    intent: Intent  # effective label, after fallbacks
    language: str
    flags: tuple[str, ...]
    raw_label: str = ""  # what the model said


def _batch_content(texts: list[str]) -> list[dict]:
    body = "\n----\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    return [{"type": "text", "text": f"{ip.FEW_SHOT}\n\n{ip.INSTRUCTION}\n\n{body}"}]


def classify(texts: list[str], llm: LLM | None) -> dict[str, Classified]:
    unique = sorted(set(texts))
    out: dict[str, Classified] = {}
    if llm is None:
        for t in unique:
            out[t] = Classified(Intent(label=IntentLabel.DELETION, confidence="low"), "", ("INTENT_NOT_CLASSIFIED",))
        return out
    for start in range(0, len(unique), BATCH_SIZE):
        batch = unique[start:start + BATCH_SIZE]
        try:
            res = llm.json_call(purpose=f"intent:batch{start // BATCH_SIZE}", prompt_version=ip.PROMPT_VERSION,
                                system=ip.SYSTEM, content=_batch_content(batch), schema=ip.SCHEMA,
                                max_tokens=8000, effort="medium", model=SONNET)
            items = {it["index"]: it for it in res.get("items", [])}
        except LLMUnavailable:
            items = {}
        for i, text in enumerate(batch):
            it = items.get(i)
            if it is None:
                log.warning("intent_missing", batch=start // BATCH_SIZE, index=i)
                out[text] = Classified(Intent(label=IntentLabel.DELETION, confidence="low"), "",
                                       ("INTENT_MISSING",))
                continue
            label = IntentLabel(it["label"])
            flags = []
            if it["confidence"] == "low" and label is not IntentLabel.DELETION:
                # Fallback: an uncertain non-deletion label would close the
                # case with no audit; auditing a non-deletion costs less.
                flags.append(f"INTENT_LOW_CONFIDENCE:{label.value}")
                label = IntentLabel.DELETION
            if it["injection_suspected"]:
                flags.append("INJECTION_SUSPECTED")
            out[text] = Classified(
                Intent(label=label, confidence=it["confidence"], injection_suspected=it["injection_suspected"]),
                it["language"], tuple(flags), it["label"])
    return out
