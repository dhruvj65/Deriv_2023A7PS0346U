"""Intent classification stage.

* `RuleBasedClassifier`  - offline, deterministic keyword/regex scoring.
* `LLMClassifier`        - Gemini with an enum-constrained JSON schema.
* `HybridClassifier`     - runs both, reconciles them, reports conflicts.

All classifiers return a `Classification` whose label is always one of the
allowed intents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from .llm import LLMClient, LLMError
from .schemas import ALLOWED_INTENTS, Intent, Ticket


@dataclass
class Classification:
    intent: Intent
    certainty: float  # 0..1
    method: str
    evidence: List[str] = field(default_factory=list)
    conflicting: bool = False
    notes: List[str] = field(default_factory=list)


class IntentClassifier(Protocol):
    def classify(self, ticket: Ticket) -> Classification: ...


# (pattern, weight). Patterns are matched case-insensitively on the raw message.
_RULES: Dict[Intent, List[Tuple[str, float]]] = {
    Intent.DEPOSIT_ISSUE: [
        (r"\bdeposit(ed|s|ing)?\b", 3.0),
        (r"\btop[- ]?up\b|\bfund(ed|ing)? (my )?account\b", 2.0),
        (r"\b(bank |credit |debit )?card\b", 1.0),
        (r"\bbalance\b", 1.0),
        (r"\b(not|never) (credited|received|showing|arrived)\b", 1.0),
    ],
    Intent.PASSWORD_RESET: [
        (r"\bpassword\b", 3.0),
        (r"\breset\b", 2.0),
        (r"\bforg(o|e)t\b", 1.5),
        (r"\b(reset|recovery) (email|link)\b", 1.5),
    ],
    Intent.WITHDRAWAL_ISSUE: [
        (r"\bwithdraw(al|als|n|ing)?\b", 3.0),
        (r"\bcash[- ]?out\b|\bpayout\b", 2.5),
        (r"\b(declined|rejected|refused|pending)\b", 1.0),
    ],
    Intent.ACCOUNT_LOCK: [
        (r"\b(locked|lock(ed)? out|lockout|blocked|suspended)\b", 3.0),
        (r"\btoo many (login|sign[- ]?in)? ?attempts\b|\bfailed (login|sign[- ]?in)", 2.0),
        (r"\b(log ?in|sign[- ]?in)\b", 1.0),
    ],
    Intent.TRADING_ADVICE_REQUEST: [
        (r"\b(which|what) (asset|stock|coin|crypto|pair|market|trade)s?\b", 3.0),
        (r"\b(go(ing)? up|go(ing)? down|rise|fall|pump|moon)\b", 2.0),
        (r"\b(make|guarantee[ds]?|earn) (a )?(profit|money)\b|\bprofit(able)?\b", 2.5),
        (r"\b(should i|recommend|advice|tip|signal|predict(ion)?|forecast)\b.*\b(buy|sell|trade|invest|asset|market)", 3.0),
        (r"\b(buy|sell|invest) (in|now|today)\b", 2.0),
    ],
}

_MIN_SCORE = 2.5  # below this the ticket is "other"


class RuleBasedClassifier:
    def classify(self, ticket: Ticket) -> Classification:
        text = ticket.message.lower()
        scores: Dict[Intent, float] = {}
        evidence: Dict[Intent, List[str]] = {}
        for intent, rules in _RULES.items():
            for pattern, weight in rules:
                m = re.search(pattern, text)
                if m:
                    scores[intent] = scores.get(intent, 0.0) + weight
                    evidence.setdefault(intent, []).append(m.group(0))

        if not scores:
            return Classification(Intent.OTHER, 0.5, "rules", notes=["no rule matched"])

        # Deterministic ranking: score desc, then label order.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], ALLOWED_INTENTS.index(kv[0].value)))
        best, best_score = ranked[0]
        if best_score < _MIN_SCORE:
            return Classification(Intent.OTHER, 0.4, "rules", evidence=evidence.get(best, []),
                                  notes=[f"weak rule signal for {best.value} ({best_score})"])

        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = (best_score - runner_up) / best_score
        coverage = min(1.0, best_score / 6.0)
        certainty = round(0.5 * margin + 0.5 * coverage, 3)
        notes = []
        if len(ranked) > 1:
            notes.append(f"runner-up {ranked[1][0].value} ({runner_up})")
        return Classification(best, certainty, "rules", evidence=evidence[best], notes=notes)


_LLM_CLASSIFY_PROMPT = """You are an intent classifier for a customer-support desk of a digital platform
(accounts, payments, trading). Classify the ticket into exactly ONE label:

- deposit_issue: money deposited/added but not showing, deposit failures or delays
- password_reset: forgotten password, reset email/link problems
- withdrawal_issue: withdrawals declined, delayed or pending
- account_lock: account locked/blocked after sign-in attempts or security locks
- trading_advice_request: asks for market predictions, which asset to buy/sell, profit tips
- other: anything else

Return JSON with "intent" (one label) and "certainty" (0 to 1).

Ticket:
\"\"\"{message}\"\"\"
"""

_LLM_CLASSIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {"type": "STRING", "enum": ALLOWED_INTENTS},
        "certainty": {"type": "NUMBER"},
    },
    "required": ["intent", "certainty"],
}


class LLMClassifier:
    def __init__(self, client: LLMClient):
        self.client = client

    def classify(self, ticket: Ticket) -> Classification:
        data = self.client.generate_json(_LLM_CLASSIFY_PROMPT.format(message=ticket.message), _LLM_CLASSIFY_SCHEMA)
        label = data.get("intent")
        if label not in ALLOWED_INTENTS:  # never trust the model blindly
            raise LLMError(f"LLM returned invalid intent {label!r}")
        try:
            certainty = float(data.get("certainty", 0.5))
        except (TypeError, ValueError):
            certainty = 0.5
        return Classification(Intent(label), max(0.0, min(1.0, certainty)), "llm")


class HybridClassifier:
    """Rules always run; the LLM (if configured) refines the label.

    Reconciliation policy:
      * agreement             -> boosted certainty
      * either side says trading_advice_request -> trading_advice_request (safety first)
      * other disagreement    -> LLM label, reduced certainty, flagged as conflicting
      * LLM failure           -> rules result
    """

    def __init__(self, rules: RuleBasedClassifier, llm: Optional[LLMClassifier] = None):
        self.rules = rules
        self.llm = llm

    def classify(self, ticket: Ticket) -> Classification:
        rule_res = self.rules.classify(ticket)
        if self.llm is None:
            return rule_res
        try:
            llm_res = self.llm.classify(ticket)
        except LLMError as exc:
            rule_res.notes.append(f"LLM classifier unavailable, used rules ({exc})")
            return rule_res

        notes = rule_res.notes + [f"rules={rule_res.intent.value}, llm={llm_res.intent.value}"]
        if llm_res.intent == rule_res.intent:
            certainty = round(min(1.0, 0.5 * (rule_res.certainty + llm_res.certainty) + 0.1), 3)
            return Classification(llm_res.intent, certainty, "hybrid", rule_res.evidence, notes=notes)

        advice = Intent.TRADING_ADVICE_REQUEST
        if advice in (rule_res.intent, llm_res.intent):
            notes.append("disagreement involving trading advice: resolved conservatively")
            return Classification(advice, 0.5, "hybrid", rule_res.evidence, conflicting=True, notes=notes)

        certainty = round(min(rule_res.certainty, llm_res.certainty) * 0.6, 3)
        notes.append("rules and LLM disagree")
        return Classification(llm_res.intent, certainty, "hybrid", rule_res.evidence, conflicting=True, notes=notes)
