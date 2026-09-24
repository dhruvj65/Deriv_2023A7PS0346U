"""Reply generation stage.

Generators only see the ticket, the predicted intent and the *already
retrieved* articles - they never search the KB themselves.

* `TemplateGenerator` - offline, extractive: rewrites retrieved KB sentences
  into client-facing voice. Every sentence is traceable to an article.
* `LLMGenerator`      - Gemini, prompted with the retrieved snippets only and
  instructed not to go beyond them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Protocol, Sequence, Tuple

from .llm import LLMClient, LLMError
from .retrieval import RetrievedArticle
from .schemas import Intent, Ticket
from .text import split_sentences

# Below this retrieval score an article is not considered relevant enough to answer from.
MIN_RELEVANCE = 0.1

# Used only if no retrieved article states the no-advice policy (the reply is then
# marked ungrounded and escalated).
DEFAULT_REFUSAL_ITEMS = "market predictions, trading advice, or profit guarantees"

NO_CONTEXT_TEXT = (
    "Thanks for contacting us. I couldn't find help-center guidance that covers this "
    "request, so I'm passing it to a member of our support team who can look into it."
)

HANDOFF_TEXT = (
    "I've also passed this to our support team so they can review it for you through a secure channel."
)


@dataclass
class GeneratedReply:
    reply: str
    cited_article_ids: List[str]
    method: str
    refused: bool = False
    answerable: bool = True  # False when the generator could not answer from the context
    notes: List[str] = field(default_factory=list)


class ReplyGenerator(Protocol):
    def generate(self, ticket: Ticket, intent: Intent, context: Sequence[RetrievedArticle]) -> GeneratedReply: ...


def relevant(context: Sequence[RetrievedArticle]) -> List[RetrievedArticle]:
    return [c for c in context if c.score >= MIN_RELEVANCE]


# ---------------------------------------------------------------- offline generator
_INTRO = {
    Intent.DEPOSIT_ISSUE: "Thanks for reaching out about your deposit.",
    Intent.PASSWORD_RESET: "Sorry you're having trouble resetting your password.",
    Intent.WITHDRAWAL_ISSUE: "Thanks for reaching out about your withdrawal.",
    Intent.ACCOUNT_LOCK: "Sorry to hear you can't get into your account.",
    Intent.OTHER: "Thanks for contacting us.",
}

# KB sentences addressed to agents only (policy instructions) - obeyed, not quoted.
_INTERNAL_SENTENCE = re.compile(r"^(support( agents?)?|agents?|such requests)\b.*\b(should|must)\b", re.I)

# Agent-voice -> client-voice rewrites, applied in order.
_REWRITES = [
    (re.compile(r"^(Ask|Advise|Tell) the client to\s+", re.I), "Please "),
    (re.compile(r"^If the client does not\b", re.I), "If you do not"),
    (re.compile(r"^If the client\b", re.I), "If you"),
    (re.compile(r"\bthe client's\b", re.I), "your"),
    (re.compile(r"\bthe client\b", re.I), "you"),
    (re.compile(r"\block the account\b", re.I), "lock your account"),
    (re.compile(r"\bbefore escalating\b", re.I), "before contacting us again"),
    (re.compile(r"^Avoid\b"), "Please avoid"),
]

_REFUSAL_ARTICLE = re.compile(r"\b(investment|trading) advice\b|\bmarket predictions?\b", re.I)
# "... must not provide market predictions, profit guarantees, or asset recommendations."
_PROHIBITED_ITEMS = re.compile(r"\b(?:must not|should not|cannot|may not|do not) (?:provide|give|offer|make) ([^.]+)\.", re.I)
# "... redirected to general educational resources if available."
_REDIRECT = re.compile(r"\bredirect(?:ed)? to ([^.]+?)(?: if available)?\.", re.I)


def build_refusal(context: Sequence[RetrievedArticle]) -> Tuple[str, List[str]]:
    """Polite refusal assembled from the retrieved no-advice policy article(s)."""
    policy = [c for c in context if _REFUSAL_ARTICLE.search(c.article.text)]
    items = redirect = None
    for c in policy:
        if items is None and (m := _PROHIBITED_ITEMS.search(c.article.body)):
            items = m.group(1)
        if redirect is None and (m := _REDIRECT.search(c.article.body)):
            redirect = m.group(1)
    parts = [f"Thanks for your question. I'm sorry, but we're not able to provide {items or DEFAULT_REFUSAL_ITEMS}."]
    if redirect:
        parts.append(f"If you'd like to learn more, you may find our {redirect} helpful, where available.")
    return " ".join(parts), [c.article.article_id for c in policy]


def _client_voice(sentence: str) -> str:
    for pattern, repl in _REWRITES:
        sentence = pattern.sub(repl, sentence)
    # "If you do not receive ..., confirm ..." -> "..., please confirm ..."
    sentence = re.sub(r"^(If you [^,]+),\s+(?!please)", r"\1, please ", sentence)
    return sentence[0].upper() + sentence[1:] if sentence else sentence


class TemplateGenerator:
    """Deterministic extractive generator (the offline fallback)."""

    def __init__(self, max_sentences: int = 4):
        self.max_sentences = max_sentences

    def generate(self, ticket: Ticket, intent: Intent, context: Sequence[RetrievedArticle]) -> GeneratedReply:
        ctx = relevant(context)

        if intent == Intent.TRADING_ADVICE_REQUEST:
            text, policy = build_refusal(context)
            return GeneratedReply(
                text, policy, "template", refused=True,
                notes=[] if policy else ["no-advice policy article not retrieved; refused by built-in policy"],
            )

        if not ctx:
            return GeneratedReply(NO_CONTEXT_TEXT, [], "template", answerable=False,
                                  notes=["no retrieved article above relevance threshold"])

        intro = _INTRO.get(intent, _INTRO[Intent.OTHER])
        body: List[str] = []
        cited: List[str] = []
        for item in ctx:
            for sent in split_sentences(item.article.body):
                if _INTERNAL_SENTENCE.search(sent):
                    continue
                if len(body) >= self.max_sentences:
                    break
                rewritten = _client_voice(sent)
                body.append(rewritten)
                if item.article.article_id not in cited:
                    cited.append(item.article.article_id)

        reply = " ".join([intro] + body)
        return GeneratedReply(reply, cited, "template")


# ---------------------------------------------------------------- LLM generator
_GEN_PROMPT = """You are drafting a reply for a customer-support agent of a digital platform
(accounts, payments, trading). Write the reply to the customer.

STRICT RULES
1. Use ONLY the information in the HELP-CENTER SNIPPETS below. Do not add facts, steps,
   timeframes, numbers, fees, links or policies that are not in the snippets.
   Stay close to the snippets' own wording so every sentence can be traced back to them.
2. Never state or guess anything about this customer's specific account, balance,
   transaction or verification status. You cannot see their account.
3. Never promise outcomes or timing for payments, deposits or withdrawals
   (no "will be approved", "will arrive", "within X hours").
4. If the customer asks for market predictions, which asset to buy/sell, or how to make
   profit, thank them and politely refuse: do not give any prediction or recommendation.
   If the snippets mention general educational resources, suggest them. Set "refused" to true.
5. Some snippets are internal guidance for agents (e.g. "Support should not ...").
   Follow that guidance but do not quote it to the customer.
6. If the snippets do not answer the question, say you are passing it to the support team
   and set "answerable" to false.
7. Answer the customer's question directly, in 2-5 short sentences, friendly and plain.
8. List in "cited_article_ids" only the snippet IDs you actually used.

PREDICTED INTENT: {intent}

HELP-CENTER SNIPPETS
{snippets}

CUSTOMER MESSAGE
\"\"\"{message}\"\"\"
"""

_GEN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reply_draft": {"type": "STRING"},
        "cited_article_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "answerable": {"type": "BOOLEAN"},
        "refused": {"type": "BOOLEAN"},
    },
    "required": ["reply_draft", "cited_article_ids", "answerable", "refused"],
}


def build_prompt(ticket: Ticket, intent: Intent, context: Sequence[RetrievedArticle]) -> str:
    snippets = "\n".join(
        f"[{c.article.article_id}] {c.article.title}: {c.article.body}" for c in context
    ) or "(none)"
    return _GEN_PROMPT.format(intent=intent.value, snippets=snippets, message=ticket.message)


class LLMGenerator:
    def __init__(self, client: LLMClient):
        self.client = client

    def generate(self, ticket: Ticket, intent: Intent, context: Sequence[RetrievedArticle]) -> GeneratedReply:
        ctx = relevant(context)
        if intent == Intent.TRADING_ADVICE_REQUEST:
            ctx = list(context)  # always show the model the policy snippets it must follow
        elif not ctx:
            # Nothing to ground on: don't let the model improvise.
            return GeneratedReply(NO_CONTEXT_TEXT, [], "llm-skipped", answerable=False,
                                  notes=["no relevant context; LLM not called"])
        data = self.client.generate_json(build_prompt(ticket, intent, ctx), _GEN_SCHEMA)
        reply = str(data.get("reply_draft", "")).strip()
        if not reply:
            raise LLMError("empty reply_draft")
        cited = [str(x) for x in data.get("cited_article_ids", []) if isinstance(x, (str, int))]
        answerable = bool(data.get("answerable", True))
        notes = [] if answerable else ["model reported question not answerable from snippets"]
        return GeneratedReply(
            reply, cited, f"llm:{self.client.model}", refused=bool(data.get("refused", False)),
            answerable=answerable, notes=notes,
        )
