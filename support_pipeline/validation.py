"""Output validation: grounding, unsupported-claim detection, refusal and schema checks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import ValidationError

from .retrieval import RetrievedArticle
from .schemas import ALLOWED_INTENTS, ALLOWED_SAFETY_FLAGS, RESULT_JSON_SCHEMA, Intent, TicketResult
from .text import split_sentences, tokenize

# Conversational words that carry no factual claim.
_CONVERSATIONAL = set(tokenize(
    "thanks thank sorry hear reach reaching contact contacting help happy glad understand "
    "frustrating hello hi dear assist question issue look looking reply team support member "
    "please pass passing passed flag review securely secure channel message trouble start "
    "customer good place like learn want able get getting know let sure step steps try "
    "information info additional also recommend suggest having doing sorry"
))

SUPPORT_THRESHOLD = 0.5  # majority of content tokens must appear in the retrieved context


@dataclass
class GroundingReport:
    ratio: float                      # share of factual sentences that are supported
    grounded: bool
    sentences: List[Dict[str, Any]] = field(default_factory=list)


def check_grounding(reply: str, ticket_message: str, context: Sequence[RetrievedArticle]) -> GroundingReport:
    """Lexical grounding: each factual sentence must be mostly made of words that appear
    in the retrieved articles (or restate the customer's own message)."""
    article_vocab = {c.article.article_id: set(tokenize(c.article.text)) for c in context}
    ticket_vocab = set(tokenize(ticket_message))
    all_vocab = set().union(*article_vocab.values()) if article_vocab else set()

    rows: List[Dict[str, Any]] = []
    factual = supported = 0
    for sent in split_sentences(reply):
        content = [t for t in tokenize(sent) if t not in _CONVERSATIONAL]
        if len(content) <= 1:
            rows.append({"sentence": sent, "source": None, "support": None, "kind": "boilerplate"})
            continue
        factual += 1
        covered = [t for t in content if t in all_vocab or t in ticket_vocab]
        support = len(covered) / len(content)
        # best single source for traceability
        best_id, best_overlap = None, 0
        for aid, vocab in sorted(article_vocab.items()):
            overlap = sum(1 for t in content if t in vocab)
            if overlap > best_overlap:
                best_id, best_overlap = aid, overlap
        ok = support >= SUPPORT_THRESHOLD and best_id is not None
        supported += int(ok)
        rows.append({"sentence": sent, "source": best_id, "support": round(support, 3),
                     "kind": "supported" if ok else "unsupported"})

    ratio = supported / factual if factual else 0.0
    return GroundingReport(round(ratio, 3), factual > 0 and supported == factual, rows)


# ---------------------------------------------------------------- claim / safety checks
_NEGATION = re.compile(r"\b(can'?t|cannot|can not|not|unable|won'?t|don'?t|never|no)\b", re.I)

_PROMISE_PATTERNS = [
    r"\b(will|shall|is going to) (definitely |certainly )?(be )?(approved|processed|credited|completed|released|unlocked|reflected|resolved)\b",
    r"\bwill (arrive|appear|show up|go through|reflect)\b",
    r"\bguarantee[ds]?\b",
    r"\bwithin \d+\s*(minutes?|hours?|days?|business days?)\b",
    r"\b(definitely|certainly|for sure)\b",
]
_ACCOUNT_FACT_PATTERNS = [
    r"\b(i|we) (have |'ve )?(checked|reviewed|looked at|verified|confirmed|can see) your\b",
    r"\byour (account|balance|withdrawal|deposit|payment|transaction|verification)s? "
    r"(was|were|has been|have been|is) (approved|declined|rejected|credited|verified|flagged|restricted|suspended|unlocked|received|complete)",
    r"\b(was|has been) (declined|rejected) (because|due to|as)\b",
]
_ADVICE_PATTERNS = [
    r"\b(you should|i recommend|we recommend|i suggest|consider) (buy|sell|invest|trad|go long|go short)",
    r"\bwill (go up|rise|increase|climb|fall|drop|go down)\b",
    r"\b(buy|sell) (now|today)\b",
    r"\b(good|best) (asset|stock|coin|investment) (to|for)\b",
]


def find_unsupported_claims(reply: str, ticket_message: str, source_texts: Sequence[str]) -> List[str]:
    """Return human-readable descriptions of claims the retrieved sources cannot support."""
    issues: List[str] = []
    source_text = " ".join(source_texts) + " " + ticket_message
    for sent in split_sentences(reply):
        low = sent.lower()
        negated = bool(_NEGATION.search(low))
        for p in _PROMISE_PATTERNS:
            if re.search(p, low) and not negated:
                issues.append(f"outcome/timing promise: {sent!r}")
        for p in _ACCOUNT_FACT_PATTERNS:
            if re.search(p, low) and not negated:
                issues.append(f"account-specific fact: {sent!r}")
        for p in _ADVICE_PATTERNS:
            if re.search(p, low) and not negated:
                issues.append(f"trading advice: {sent!r}")
    # numbers (amounts, durations, percentages) that appear nowhere in the sources
    for num in re.findall(r"\d+(?:[.,]\d+)?", reply):
        if num not in source_text:
            issues.append(f"number not in sources: {num}")
    return sorted(set(issues))


_REFUSAL_RE = re.compile(
    r"\b(can'?t|cannot|can not|unable to|not able to|won'?t|do not|don'?t|are not permitted to)\b"
    r"[^.]*\b(provide|give|offer|share|make|tell|recommend|predict)", re.I)
_REFUSAL_TOPIC_RE = re.compile(r"\b(predict|recommend|advice|profit|asset|market|invest|trad)", re.I)
_POLITE_RE = re.compile(r"\b(sorry|thanks?|thank you|unfortunately|apologi[sz]e|appreciate)\b", re.I)


def is_refusal(reply: str) -> bool:
    """A polite refusal: explicitly declines, uses a courtesy marker, and contains no advice."""
    if not (_REFUSAL_RE.search(reply) and _REFUSAL_TOPIC_RE.search(reply) and _POLITE_RE.search(reply)):
        return False
    return not any(re.search(p, s.lower()) and not _NEGATION.search(s.lower())
                   for s in split_sentences(reply) for p in _ADVICE_PATTERNS)


# ---------------------------------------------------------------- schema / result validation
def _check_json_schema(obj: Dict[str, Any]) -> List[str]:
    """Minimal JSON-schema check against RESULT_JSON_SCHEMA (no external dependency)."""
    errs: List[str] = []
    props = RESULT_JSON_SCHEMA["properties"]
    for key in RESULT_JSON_SCHEMA["required"]:
        if key not in obj:
            errs.append(f"missing field {key!r}")
    for key in obj:
        if key not in props:
            errs.append(f"unexpected field {key!r}")
    types = {"string": str, "boolean": bool, "array": list}
    for key, spec in props.items():
        if key not in obj:
            continue
        val = obj[key]
        t = spec["type"]
        if t == "number":
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                errs.append(f"{key} must be a number")
        elif not isinstance(val, types[t]):
            errs.append(f"{key} must be {t}")
    return errs


def validate_result(obj: Dict[str, Any], known_article_ids: Iterable[str],
                    article_texts: Optional[Dict[str, str]] = None, ticket_message: str = "") -> List[str]:
    """Structural + policy checks on a single serialised result. Returns a list of errors.

    If `article_texts` (article_id -> text) is given, the reply is also scanned for
    unsupported claims, which must be covered by a safety flag or escalation.
    """
    errs = _check_json_schema(obj)
    if errs:
        return errs
    try:
        TicketResult.model_validate(obj)
    except ValidationError as exc:
        errs.extend(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    known = set(known_article_ids)
    if obj["intent"] not in ALLOWED_INTENTS:
        errs.append(f"intent {obj['intent']!r} not allowed")
    if not 1 <= len(obj["retrieved_articles"]) <= 3:
        errs.append("retrieved_articles must contain 1-3 ids")
    unknown = [a for a in obj["retrieved_articles"] if a not in known]
    if unknown:
        errs.append(f"unknown article ids: {unknown}")
    if not 0.0 <= float(obj["confidence"]) <= 1.0:
        errs.append("confidence out of [0, 1]")
    bad_flags = [f for f in obj["safety_flags"] if f not in ALLOWED_SAFETY_FLAGS]
    if bad_flags:
        errs.append(f"unknown safety flags: {bad_flags}")
    if obj["intent"] == Intent.TRADING_ADVICE_REQUEST.value:
        if not is_refusal(obj["reply_draft"]):
            errs.append("trading advice request not refused")
        if "advice_request" not in obj["safety_flags"]:
            errs.append("trading advice request missing advice_request flag")
    if not obj["grounded"] and not (obj["needs_human_escalation"] or "insufficient_grounding" in obj["safety_flags"]):
        errs.append("ungrounded reply without flag or escalation")
    if article_texts is not None:
        sources = [article_texts[a] for a in obj["retrieved_articles"] if a in article_texts]
        claims = find_unsupported_claims(obj["reply_draft"], ticket_message, sources)
        if claims and not (obj["needs_human_escalation"] or "unsupported_claim" in obj["safety_flags"]):
            errs.append(f"unsupported claims without flag or escalation: {claims}")
    return errs
