"""Data contracts shared by every pipeline stage."""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    DEPOSIT_ISSUE = "deposit_issue"
    PASSWORD_RESET = "password_reset"
    WITHDRAWAL_ISSUE = "withdrawal_issue"
    ACCOUNT_LOCK = "account_lock"
    TRADING_ADVICE_REQUEST = "trading_advice_request"
    OTHER = "other"


ALLOWED_INTENTS = [i.value for i in Intent]


class SafetyFlag(str, Enum):
    ADVICE_REQUEST = "advice_request"
    INSUFFICIENT_GROUNDING = "insufficient_grounding"
    POLICY_SENSITIVE = "policy_sensitive"
    LOW_CONFIDENCE = "low_confidence"
    ACCOUNT_SPECIFIC_REQUEST = "account_specific_request"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONFLICTING_SIGNALS = "conflicting_signals"


ALLOWED_SAFETY_FLAGS = [f.value for f in SafetyFlag]


class Ticket(BaseModel):
    id: str
    message: str
    language: str = "en"


class Article(BaseModel):
    article_id: str
    title: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.title}. {self.body}"


class TicketResult(BaseModel):
    """Required output schema per ticket (written to results.json)."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    intent: Intent
    retrieved_articles: List[str] = Field(min_length=1, max_length=3)
    reply_draft: str = Field(min_length=1)
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_escalation: bool
    safety_flags: List[SafetyFlag]


# JSON Schema used by validate.py (kept independent of pydantic on purpose).
RESULT_JSON_SCHEMA = {
    "type": "object",
    "required": [
        "ticket_id", "intent", "retrieved_articles", "reply_draft",
        "grounded", "confidence", "needs_human_escalation", "safety_flags",
    ],
    "additionalProperties": False,
    "properties": {
        "ticket_id": {"type": "string"},
        "intent": {"type": "string", "enum": ALLOWED_INTENTS},
        "retrieved_articles": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "reply_draft": {"type": "string", "minLength": 1},
        "grounded": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "needs_human_escalation": {"type": "boolean"},
        "safety_flags": {"type": "array", "items": {"type": "string", "enum": ALLOWED_SAFETY_FLAGS}},
    },
}
