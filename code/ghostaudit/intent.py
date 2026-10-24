"""Request intent: does the request authorize deletion?

Unique texts are classified in batches of 25 with Sonnet; the output is an
enum per text and nothing else, so an instruction embedded in a request can
change at most its label, never the steps. The engine never reads the text.

Fallbacks: a text the model does not return is treated as DELETION and
flagged; so is a non-deletion label on a text flagged as carrying processing
instructions. Low confidence becomes DELETION only when DELETION is the top or the
second label; low confidence between two non-deletion labels stays
NOT_APPLICABLE and is flagged for review.
Without credentials every text is DELETION and flagged INTENT_UNAVAILABLE.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import intent_prompts as ip
from .llm import LLM, LLMExtractionFailed, LLMUnavailable, model_for
from .obs import get_logger
from .schemas import Flag, FlagHit, Intent, IntentLabel

log = get_logger(__name__)

BATCH_SIZE = 25


@dataclass(frozen=True)
class Classified:
    intent: Intent  # effective label, after fallbacks
    language: str
    flags: tuple[FlagHit, ...]
    raw_label: str = ""  # what the model said
    second_label: str = ""


def _batch_content(texts: list[str]) -> list[dict]:
    body = "\n----\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    return [{"type": "text", "text": f"{ip.FEW_SHOT}\n\n{ip.INSTRUCTION}\n\n{body}"}]


def classify(texts: list[str], llm: LLM | None) -> dict[str, Classified]:
    unique = sorted(set(texts))
    out: dict[str, Classified] = {}
    if llm is None:
        for t in unique:
            out[t] = Classified(Intent(label=IntentLabel.DELETION, confidence="low"), "",
                                (FlagHit(flag=Flag.INTENT_UNAVAILABLE, detail="no model credentials"),))
        return out
    for start in range(0, len(unique), BATCH_SIZE):
        batch = unique[start:start + BATCH_SIZE]
        try:
            res = llm.json_call(purpose=f"intent:batch{start // BATCH_SIZE}", prompt_version=ip.PROMPT_VERSION,
                                system=ip.SYSTEM, content=_batch_content(batch), schema=ip.SCHEMA,
                                max_tokens=8000, effort="medium", model=model_for("INTENT"))
            items = {it["index"]: it for it in res.get("items", [])}
        except (LLMUnavailable, LLMExtractionFailed) as e:
            # Fallback: the batch has no answer; every text in it is audited as
            # a deletion and flagged INTENT_UNAVAILABLE below.
            log.error("intent_batch_failed", batch=start // BATCH_SIZE, error=str(e))
            items = {}
        for i, text in enumerate(batch):
            it = items.get(i)
            if it is None:
                log.warning("intent_missing", batch=start // BATCH_SIZE, index=i)
                out[text] = Classified(Intent(label=IntentLabel.DELETION, confidence="low"), "",
                                       (FlagHit(flag=Flag.INTENT_UNAVAILABLE, detail="no answer for this text"),))
                continue
            label = IntentLabel(it["label"])
            flags = []
            if it["confidence"] == "low":
                if IntentLabel.DELETION.value in (it["label"], it["second_label"]):
                    # Fallback: hesitation between DELETION and anything else
                    # is audited as a deletion; a missed deletion leaves data
                    # behind, an extra audit does not.
                    flags.append(FlagHit(flag=Flag.INTENT_REVIEW,
                                         detail=f"{it['label']}/{it['second_label']}, audited as DELETION"))
                    label = IntentLabel.DELETION
                else:
                    # Two non-deletion readings: still not a deletion request.
                    flags.append(FlagHit(flag=Flag.INTENT_REVIEW,
                                         detail=f"{it['label']}/{it['second_label']}, not a deletion"))
            if it["injection_suspected"]:
                flags.append(FlagHit(flag=Flag.INJECTION_SUSPECTED))
                if label is not IntentLabel.DELETION:
                    # A request carrying processing instructions is never
                    # closed as NOT_APPLICABLE on the model's word: the
                    # instruction may be what moved the label. Audit it.
                    flags.append(FlagHit(flag=Flag.INTENT_REVIEW,
                                         detail=f"{it['label']} with suspected injection, audited as DELETION"))
                    label = IntentLabel.DELETION
            out[text] = Classified(
                Intent(label=label, confidence=it["confidence"], injection_suspected=it["injection_suspected"]),
                it["language"], tuple(flags), it["label"], it["second_label"])
    return out
