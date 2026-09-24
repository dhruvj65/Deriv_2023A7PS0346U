"""Pipeline orchestration.

LOAD_DATA -> INDEX_KB -> CLASSIFY_INTENT -> RETRIEVE_CONTEXT -> GENERATE_REPLY -> VALIDATE_OUTPUT -> WRITE_RESULTS

Each stage is a separate method; components are injected so that the
classifier, retriever and generator can be swapped independently.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .classifier import Classification, HybridClassifier, IntentClassifier, LLMClassifier, RuleBasedClassifier
from .generator import (HANDOFF_TEXT, NO_CONTEXT_TEXT, GeneratedReply, LLMGenerator, ReplyGenerator,
                        TemplateGenerator)
from .llm import JsonFileCache, LLMError, ReplayClient, client_from_env
from .policy import Assessment, assess
from .retrieval import RetrievedArticle, Retriever, TfidfRetriever
from .schemas import Article, Intent, Ticket, TicketResult
from .validation import GroundingReport, check_grounding, find_unsupported_claims, is_refusal, validate_result

log = logging.getLogger("support_pipeline")

_NO_ADVICE_POLICY = re.compile(r"\b(investment|trading) advice\b|\bmarket predictions?\b", re.I)


class PipelineError(RuntimeError):
    pass


@dataclass
class PipelineConfig:
    tickets_path: Path = Path("tickets.json")
    kb_path: Path = Path("kb_articles.json")
    results_path: Path = Path("results.json")
    debug_json_path: Path = Path("debug_report.json")
    debug_md_path: Path = Path("debug_report.md")
    cache_path: Path = Path(".cache/llm_cache.json")
    offline: bool = False
    # If set, answer LLM calls for this model only from the response cache (no network).
    replay_model: Optional[str] = None


@dataclass
class TicketRun:
    result: TicketResult
    debug: Dict[str, Any] = field(default_factory=dict)


class SupportPipeline:
    def __init__(self, classifier: IntentClassifier, retriever: Retriever, generator: ReplyGenerator,
                 fallback_generator: Optional[ReplyGenerator] = None, mode: str = "offline"):
        self.classifier = classifier
        self.retriever = retriever
        self.generator = generator
        self.fallback = fallback_generator or TemplateGenerator()
        self.mode = mode
        self.articles: Dict[str, Article] = {}

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_config(cls, config: PipelineConfig) -> "SupportPipeline":
        if config.offline:
            client = None
        elif config.replay_model:
            client = ReplayClient(config.replay_model, JsonFileCache(config.cache_path))
        else:
            client = client_from_env(config.cache_path)
        rules = RuleBasedClassifier()
        if client is None:
            return cls(rules, TfidfRetriever(), TemplateGenerator(), mode="offline")
        return cls(HybridClassifier(rules, LLMClassifier(client)), TfidfRetriever(), LLMGenerator(client),
                   fallback_generator=TemplateGenerator(), mode=f"llm:{client.model}")

    # ------------------------------------------------------------------ stages
    @staticmethod
    def load_data(tickets_path: Path, kb_path: Path) -> Tuple[List[Ticket], List[Article]]:
        """LOAD_DATA"""
        log.info("[LOAD_DATA] %s, %s", tickets_path, kb_path)
        try:
            raw_tickets = json.loads(Path(tickets_path).read_text(encoding="utf-8"))
            raw_kb = json.loads(Path(kb_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineError(f"could not load input files: {exc}") from exc
        tickets = [Ticket.model_validate(t) for t in raw_tickets]
        articles = [Article.model_validate(a) for a in raw_kb]
        if len({t.id for t in tickets}) != len(tickets):
            raise PipelineError("duplicate ticket ids")
        if len({a.article_id for a in articles}) != len(articles):
            raise PipelineError("duplicate article ids")
        if not articles:
            raise PipelineError("knowledge base is empty")
        return tickets, articles

    def index_kb(self, articles: Sequence[Article]) -> None:
        """INDEX_KB"""
        self.articles = {a.article_id: a for a in articles}
        self.retriever.index(articles)

    def classify_intent(self, ticket: Ticket) -> Classification:
        """CLASSIFY_INTENT"""
        return self.classifier.classify(ticket)

    def retrieve_context(self, ticket: Ticket) -> List[RetrievedArticle]:
        """RETRIEVE_CONTEXT - the ticket text is the query; intent is NOT used."""
        return self.retriever.retrieve(ticket.message)

    def generate_reply(self, ticket: Ticket, intent: Intent, context: Sequence[RetrievedArticle]) -> GeneratedReply:
        """GENERATE_REPLY - primary generator with offline fallback on model errors."""
        try:
            return self.generator.generate(ticket, intent, context)
        except LLMError as exc:
            log.warning("generator failed for %s (%s); using fallback", ticket.id, exc)
            reply = self.fallback.generate(ticket, intent, context)
            reply.notes.append(f"primary generator failed ({exc}); used offline fallback")
            return reply

    def _check_reply(self, ticket: Ticket, intent: Intent, context: Sequence[RetrievedArticle],
                     reply: GeneratedReply) -> Tuple[GroundingReport, List[str], bool, List[str]]:
        grounding = check_grounding(reply.reply, ticket.message, context)
        claims = find_unsupported_claims(reply.reply, ticket.message, [c.article.text for c in context])
        refused_ok = is_refusal(reply.reply) if intent == Intent.TRADING_ADVICE_REQUEST else True
        problems = list(claims)
        retrieved_ids = {c.article.article_id for c in context}
        bad_cites = [a for a in reply.cited_article_ids if a not in retrieved_ids]
        if bad_cites:
            problems.append(f"cited non-retrieved articles {bad_cites}")
        if not refused_ok:
            problems.append("trading advice request not politely refused")
        if intent == Intent.TRADING_ADVICE_REQUEST:
            # A refusal is grounded when the no-advice policy article was retrieved.
            policy_ids = [c.article.article_id for c in context if _NO_ADVICE_POLICY.search(c.article.text)]
            ok = refused_ok and bool(policy_ids)
            grounding = GroundingReport(1.0 if ok else 0.0, ok, [
                {"sentence": s["sentence"], "source": ",".join(policy_ids) or None,
                 "support": 1.0 if ok else 0.0, "kind": "policy_refusal"} for s in grounding.sentences])
        elif not grounding.grounded and reply.reply != NO_CONTEXT_TEXT:
            problems.append("reply contains sentences not supported by retrieved articles")
        return grounding, claims, refused_ok, problems

    def validate_output(self, ticket: Ticket, cls: Classification, context: Sequence[RetrievedArticle],
                        reply: GeneratedReply) -> TicketRun:
        """VALIDATE_OUTPUT - grounding/claim/refusal checks, safe fallback, confidence,
        escalation, and schema validation."""
        grounding, claims, refused_ok, problems = self._check_reply(ticket, cls.intent, context, reply)
        replaced: Optional[Dict[str, Any]] = None
        replaced_claims: List[str] = []
        if problems and not reply.method.startswith("template"):
            # Model output failed a check: replace it with the deterministic extractive draft.
            replaced = {"original_reply": reply.reply, "original_method": reply.method, "problems": problems}
            replaced_claims = claims
            reply = self.fallback.generate(ticket, cls.intent, context)
            reply.notes.append("LLM draft failed validation; replaced by offline extractive draft")
            grounding, claims, refused_ok, problems = self._check_reply(ticket, cls.intent, context, reply)

        assessment = assess(ticket, cls, context, reply, grounding, claims, refused_ok, replaced_claims)

        text = reply.reply
        if assessment.needs_human_escalation and text != NO_CONTEXT_TEXT and HANDOFF_TEXT not in text:
            text = f"{text} {HANDOFF_TEXT}"

        result_obj = {
            "ticket_id": ticket.id,
            "intent": cls.intent.value,
            "retrieved_articles": [c.article.article_id for c in context],
            "reply_draft": text,
            "grounded": grounding.grounded,
            "confidence": assessment.confidence,
            "needs_human_escalation": assessment.needs_human_escalation,
            "safety_flags": [f.value for f in assessment.safety_flags],
        }
        errors = validate_result(result_obj, self.articles.keys(),
                                 {aid: a.text for aid, a in self.articles.items()}, ticket.message)
        if errors:
            raise PipelineError(f"ticket {ticket.id} failed output validation: {errors}")
        result = TicketResult.model_validate(result_obj)
        return TicketRun(result, self._debug_entry(ticket, cls, context, reply, grounding, claims,
                                                   assessment, replaced))

    def _debug_entry(self, ticket: Ticket, cls: Classification, context: Sequence[RetrievedArticle],
                     reply: GeneratedReply, grounding: GroundingReport, claims: List[str],
                     a: Assessment, replaced: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "ticket_id": ticket.id,
            "message": ticket.message,
            "predicted_intent": cls.intent.value,
            "classifier": {"method": cls.method, "certainty": cls.certainty,
                           "evidence": cls.evidence, "notes": cls.notes},
            "retrieved": [{"article_id": c.article.article_id, "title": c.article.title,
                           "score": c.score, "rank": c.rank} for c in context],
            "generator": {"method": reply.method, "cited_article_ids": reply.cited_article_ids,
                          "notes": reply.notes, "replaced_llm_draft": replaced},
            "grounding": {"grounded": grounding.grounded, "ratio": grounding.ratio,
                          "sentence_trace": grounding.sentences},
            "unsupported_claims": claims,
            "confidence": a.confidence,
            "confidence_components": a.components,
            "needs_human_escalation": a.needs_human_escalation,
            "escalation_reasons": a.reasons,
            "notes": a.non_escalation_notes,
            "safety_flags": [f.value for f in a.safety_flags],
        }

    # ------------------------------------------------------------------ run
    def process(self, tickets: Sequence[Ticket], articles: Sequence[Article]) -> List[TicketRun]:
        log.info("[INDEX_KB] %d articles", len(articles))
        self.index_kb(articles)
        runs: List[TicketRun] = []
        for ticket in tickets:
            cls = self.classify_intent(ticket)
            log.debug("[CLASSIFY_INTENT] %s -> %s", ticket.id, cls.intent.value)
            context = self.retrieve_context(ticket)
            log.debug("[RETRIEVE_CONTEXT] %s -> %s", ticket.id, [c.article.article_id for c in context])
            reply = self.generate_reply(ticket, cls.intent, context)
            log.debug("[GENERATE_REPLY] %s via %s", ticket.id, reply.method)
            runs.append(self.validate_output(ticket, cls, context, reply))
            log.debug("[VALIDATE_OUTPUT] %s ok", ticket.id)
        log.info("[CLASSIFY_INTENT -> RETRIEVE_CONTEXT -> GENERATE_REPLY -> VALIDATE_OUTPUT] %d tickets (mode %s)",
                 len(runs), self.mode)
        return runs

    def write_results(self, runs: Sequence[TicketRun], config: PipelineConfig) -> None:
        """WRITE_RESULTS"""
        log.info("[WRITE_RESULTS] %s, %s, %s", config.results_path, config.debug_json_path, config.debug_md_path)
        results = [r.result.model_dump(mode="json") for r in runs]
        config.results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        debug = {"mode": self.mode, "tickets": [r.debug for r in runs]}
        config.debug_json_path.write_text(json.dumps(debug, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        config.debug_md_path.write_text(render_markdown(debug), encoding="utf-8")


def run(config: PipelineConfig) -> List[TicketRun]:
    pipeline = SupportPipeline.from_config(config)
    tickets, articles = pipeline.load_data(config.tickets_path, config.kb_path)
    runs = pipeline.process(tickets, articles)
    pipeline.write_results(runs, config)
    return runs


def render_markdown(debug: Dict[str, Any]) -> str:
    lines = ["# Debug report", "", f"Mode: `{debug['mode']}`", "",
             "| Ticket | Intent | Retrieved articles | Confidence | Escalated | Flags |",
             "|---|---|---|---|---|---|"]
    for t in debug["tickets"]:
        arts = "; ".join(f"{r['article_id']} {r['title']} ({r['score']:.2f})" for r in t["retrieved"])
        lines.append(f"| {t['ticket_id']} | {t['predicted_intent']} | {arts} | {t['confidence']} | "
                     f"{'yes' if t['needs_human_escalation'] else 'no'} | {', '.join(t['safety_flags']) or '-'} |")
    for t in debug["tickets"]:
        lines += ["", f"## {t['ticket_id']}", "", f"> {t['message']}", "",
                  f"- **Predicted intent:** `{t['predicted_intent']}` "
                  f"(method: {t['classifier']['method']}, certainty {t['classifier']['certainty']})",
                  f"- **Classifier evidence:** {', '.join(t['classifier']['evidence']) or '-'}"]
        if t["classifier"]["notes"]:
            lines.append(f"- **Classifier notes:** {'; '.join(t['classifier']['notes'])}")
        lines.append("- **Retrieved articles:** " + "; ".join(
            f"{r['article_id']} \"{r['title']}\" (score {r['score']:.3f})" for r in t["retrieved"]))
        lines.append(f"- **Generator:** {t['generator']['method']}")
        if t["generator"]["notes"]:
            lines.append(f"- **Generator notes:** {'; '.join(t['generator']['notes'])}")
        lines.append(f"- **Confidence:** {t['confidence']} (components: {t['confidence_components']})")
        if t["needs_human_escalation"]:
            lines.append("- **Escalated: yes** - " + "; ".join(t["escalation_reasons"]))
        else:
            lines.append("- **Escalated: no** - " + "; ".join(t["notes"]))
        if t["needs_human_escalation"] and t["notes"]:
            lines.append("- **Other notes:** " + "; ".join(t["notes"]))
        lines += ["- **Reply trace (sentence -> source):**"]
        for s in t["grounding"]["sentence_trace"]:
            src = s["source"] or "-"
            lines.append(f"  - [{s['kind']}, {src}] {s['sentence']}")
    return "\n".join(lines) + "\n"
