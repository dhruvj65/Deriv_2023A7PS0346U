"""Confidence scoring, safety flags and escalation decisions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

from .classifier import Classification
from .generator import MIN_RELEVANCE, GeneratedReply
from .retrieval import RetrievedArticle
from .schemas import Intent, SafetyFlag, Ticket
from .validation import GroundingReport

STRONG_RETRIEVAL_SCORE = 0.4   # cosine score treated as "fully relevant"
LOW_CONFIDENCE = 0.55          # below this we escalate

POLICY_SENSITIVE_INTENTS = {Intent.WITHDRAWAL_ISSUE}
_POLICY_SENSITIVE_TERMS = re.compile(
    r"\b(fraud|scam|stolen|hack(ed)?|chargeback|legal|lawyer|sue|complaint|compliance|"
    r"money laundering|close my account|delete my (account|data)|refund)\b", re.I)
_ACCOUNT_SPECIFIC = re.compile(
    r"\bmy (account|balance|deposit|withdrawal|payment|transaction|card|funds|money|verification)s?\b|"
    r"\bwhy (was|is|did|has|have) my\b", re.I)


@dataclass
class Assessment:
    confidence: float
    needs_human_escalation: bool
    safety_flags: List[SafetyFlag]
    reasons: List[str] = field(default_factory=list)          # why escalated
    non_escalation_notes: List[str] = field(default_factory=list)
    components: dict = field(default_factory=dict)


def assess(ticket: Ticket, cls: Classification, context: Sequence[RetrievedArticle],
           reply: GeneratedReply, grounding: GroundingReport, claims: List[str],
           refused_ok: bool, replaced_claims: Sequence[str] = ()) -> Assessment:
    """`claims` are unsupported claims in the final reply; `replaced_claims` are claims found
    in an LLM draft that was discarded and replaced by the extractive fallback."""
    flags: List[SafetyFlag] = []
    reasons: List[str] = []
    notes: List[str] = []

    top_score = context[0].score if context else 0.0
    retrieval_strength = min(1.0, top_score / STRONG_RETRIEVAL_SCORE)
    is_advice = cls.intent == Intent.TRADING_ADVICE_REQUEST

    grounding_score = grounding.ratio

    confidence = 0.35 * cls.certainty + 0.35 * retrieval_strength + 0.30 * grounding_score
    if cls.conflicting:
        confidence *= 0.8

    # --- flags & escalation reasons -------------------------------------------------
    if is_advice:
        flags.append(SafetyFlag.ADVICE_REQUEST)
        if refused_ok:
            notes.append("trading-advice request politely refused per no-advice policy; no human needed")
        else:
            reasons.append("trading-advice request was not safely refused")

    if top_score < MIN_RELEVANCE:
        flags.append(SafetyFlag.INSUFFICIENT_GROUNDING)
        reasons.append(f"no relevant KB article (top retrieval score {top_score:.3f} < {MIN_RELEVANCE})")
        confidence = min(confidence, 0.3)
    elif not grounding.grounded:
        flags.append(SafetyFlag.INSUFFICIENT_GROUNDING)
        unsupported = [s["sentence"] for s in grounding.sentences if s["kind"] == "unsupported"]
        reasons.append(f"reply not fully grounded in retrieved articles ({len(unsupported)} unsupported sentence(s))")
        confidence = min(confidence, 0.4)

    if not reply.answerable and top_score >= MIN_RELEVANCE:
        flags.append(SafetyFlag.INSUFFICIENT_GROUNDING)
        reasons.append("generator reported the retrieved articles do not answer the question")
        confidence = min(confidence, 0.4)

    if claims:
        flags.append(SafetyFlag.UNSUPPORTED_CLAIM)
        reasons.append("unsupported claims detected: " + "; ".join(claims))
        confidence = min(confidence, 0.4)
    elif replaced_claims:
        # The final reply is clean, but the model tried to make unsupported claims: keep an audit flag.
        flags.append(SafetyFlag.UNSUPPORTED_CLAIM)
        notes.append("LLM draft contained unsupported claims (" + "; ".join(replaced_claims)
                     + ") and was replaced by the grounded extractive draft")

    if cls.intent == Intent.OTHER:
        reasons.append("intent 'other': no dedicated help-center workflow")

    if cls.conflicting:
        flags.append(SafetyFlag.CONFLICTING_SIGNALS)
        reasons.append("rule-based and LLM classifiers disagree")

    if cls.intent in POLICY_SENSITIVE_INTENTS or _POLICY_SENSITIVE_TERMS.search(ticket.message):
        flags.append(SafetyFlag.POLICY_SENSITIVE)
        reasons.append("policy-sensitive topic (outcome depends on account status / compliance review "
                       "that support cannot see or promise)")

    if _ACCOUNT_SPECIFIC.search(ticket.message):
        flags.append(SafetyFlag.ACCOUNT_SPECIFIC_REQUEST)
        notes.append("references the customer's own account; reply gives general KB guidance only, "
                     "no account-specific facts")

    confidence = round(max(0.0, min(1.0, confidence)), 2)
    if confidence < LOW_CONFIDENCE:
        flags.append(SafetyFlag.LOW_CONFIDENCE)
        reasons.append(f"confidence {confidence} below threshold {LOW_CONFIDENCE}")

    # stable, de-duplicated flag order
    ordered = [f for f in SafetyFlag if f in flags]
    escalate = bool(reasons)
    if not escalate:
        notes.append(f"handled self-serve: intent '{cls.intent.value}' covered by retrieved KB, "
                     f"confidence {confidence} >= {LOW_CONFIDENCE}")
    return Assessment(
        confidence, escalate, ordered, reasons, notes,
        components={
            "classifier_certainty": cls.certainty,
            "retrieval_top_score": top_score,
            "retrieval_strength": round(retrieval_strength, 3),
            "grounding_score": grounding_score,
            "conflicting": cls.conflicting,
        },
    )
