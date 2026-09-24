"""Pipeline tests. All run offline (no API key needed); LLM behaviour is mocked."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from support_pipeline.classifier import HybridClassifier, LLMClassifier, RuleBasedClassifier
from support_pipeline.generator import LLMGenerator, TemplateGenerator
from support_pipeline.llm import LLMError
from support_pipeline.pipeline import PipelineConfig, SupportPipeline, run
from support_pipeline.retrieval import TfidfRetriever
from support_pipeline.schemas import RESULT_JSON_SCHEMA, Ticket
from support_pipeline.validation import find_unsupported_claims, is_refusal, validate_result

ROOT = Path(__file__).resolve().parents[1]


def offline_pipeline() -> SupportPipeline:
    return SupportPipeline(RuleBasedClassifier(), TfidfRetriever(), TemplateGenerator(), mode="offline")


def process(pipe: SupportPipeline, tickets: List[Ticket] | None = None):
    loaded, articles = pipe.load_data(ROOT / "tickets.json", ROOT / "kb_articles.json")
    return {r.result.ticket_id: r for r in pipe.process(tickets or loaded, articles)}


@pytest.fixture(scope="module")
def results():
    return {k: v.result.model_dump(mode="json") for k, v in process(offline_pipeline()).items()}


@pytest.fixture(scope="module")
def known_ids():
    return {a["article_id"] for a in json.loads((ROOT / "kb_articles.json").read_text())}


# ---------------------------------------------------------------- required checks
def test_t4_is_refused_as_trading_advice(results):
    t4 = results["T4"]
    assert t4["intent"] == "trading_advice_request"
    assert "advice_request" in t4["safety_flags"]
    assert is_refusal(t4["reply_draft"])
    assert "A5" in t4["retrieved_articles"]


def test_t1_retrieves_deposit_article(results):
    assert results["T1"]["intent"] == "deposit_issue"
    assert results["T1"]["retrieved_articles"][0] == "A1"


def test_output_matches_required_schema(results, known_ids):
    for r in results.values():
        assert set(r) == set(RESULT_JSON_SCHEMA["required"])
        assert validate_result(r, known_ids) == []


# ---------------------------------------------------------------- behaviour
@pytest.mark.parametrize("tid,intent,article", [
    ("T1", "deposit_issue", "A1"), ("T2", "password_reset", "A2"), ("T3", "withdrawal_issue", "A3"),
    ("T4", "trading_advice_request", "A5"), ("T5", "account_lock", "A4"),
])
def test_intents_and_top_article(results, tid, intent, article):
    assert results[tid]["intent"] == intent
    assert results[tid]["retrieved_articles"][0] == article


def test_withdrawal_is_policy_sensitive_and_escalated(results):
    t3 = results["T3"]
    assert t3["needs_human_escalation"] is True
    assert "policy_sensitive" in t3["safety_flags"]
    assert "approved" not in t3["reply_draft"].lower()


def test_pipeline_is_deterministic():
    a = {k: v.result for k, v in process(offline_pipeline()).items()}
    b = {k: v.result for k, v in process(offline_pipeline()).items()}
    assert a == b


def test_logic_does_not_depend_on_ticket_ids():
    tickets = [Ticket(id=f"X{i}", message=m) for i, m in enumerate([
        "Which crypto should I buy now to make money fast?",
        "My deposit via debit card has not arrived in my balance.",
        "What are your office opening hours?",
    ])]
    out = {k: v.result for k, v in process(offline_pipeline(), tickets).items()}
    assert out["X0"].intent.value == "trading_advice_request" and is_refusal(out["X0"].reply_draft)
    assert out["X1"].intent.value == "deposit_issue" and out["X1"].retrieved_articles[0] == "A1"
    assert out["X2"].intent.value == "other" and out["X2"].needs_human_escalation


def test_irrelevant_ticket_is_flagged_and_escalated():
    out = process(offline_pipeline(), [Ticket(id="Z", message="Do you have a dress code for the gala?")])["Z"]
    assert out.result.needs_human_escalation
    assert "insufficient_grounding" in [f.value for f in out.result.safety_flags]
    assert out.result.grounded is False


def test_retrieval_returns_one_to_three_known_ids(known_ids):
    r = TfidfRetriever()
    pipe = offline_pipeline()
    _, articles = pipe.load_data(ROOT / "tickets.json", ROOT / "kb_articles.json")
    r.index(articles)
    for q in ["card deposit", "password", "hello", "withdrawal declined and account locked"]:
        hits = r.retrieve(q)
        assert 1 <= len(hits) <= 3
        assert {h.article.article_id for h in hits} <= known_ids


# ---------------------------------------------------------------- validation units
def test_unsupported_claims_are_detected():
    claims = find_unsupported_claims("Your withdrawal will be approved within 24 hours.", "", [])
    assert any("promise" in c for c in claims) and any("number" in c for c in claims)
    assert find_unsupported_claims("I have checked your account and it is fine.", "", [])
    assert find_unsupported_claims("You should buy gold today, it will go up.", "", [])


def test_validator_rejects_bad_results(known_ids):
    good = {"ticket_id": "T", "intent": "other", "retrieved_articles": ["A1"], "reply_draft": "x",
            "grounded": True, "confidence": 0.5, "needs_human_escalation": True, "safety_flags": []}
    assert validate_result(good, known_ids) == []
    assert validate_result({**good, "intent": "refund"}, known_ids)
    assert validate_result({**good, "retrieved_articles": ["A99"]}, known_ids)
    assert validate_result({**good, "confidence": 1.5}, known_ids)
    assert validate_result({k: v for k, v in good.items() if k != "grounded"}, known_ids)
    advice = {**good, "intent": "trading_advice_request", "safety_flags": ["advice_request"],
              "reply_draft": "You should buy gold today, it will go up."}
    assert "trading advice request not refused" in validate_result(advice, known_ids)
    ungrounded = {**good, "grounded": False, "needs_human_escalation": False}
    assert validate_result(ungrounded, known_ids)


# ---------------------------------------------------------------- LLM boundary (mocked)
class FakeClient:
    model = "fake"

    def __init__(self, intent: str | None = None, reply: str | None = None, fail: bool = False):
        self.intent, self.reply, self.fail = intent, reply, fail

    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if self.fail:
            raise LLMError("simulated outage")
        if "intent" in schema["properties"]:
            return {"intent": self.intent, "certainty": 0.9}
        cited = re.findall(r"^\[(\w+)\]", prompt, re.M)[:1]  # cite the top snippet shown to the model
        return {"reply_draft": self.reply, "cited_article_ids": cited, "answerable": True, "refused": False}


def llm_pipeline(client: FakeClient) -> SupportPipeline:
    return SupportPipeline(HybridClassifier(RuleBasedClassifier(), LLMClassifier(client)), TfidfRetriever(),
                           LLMGenerator(client), fallback_generator=TemplateGenerator(), mode="llm:fake")


def test_llm_outage_falls_back_to_offline():
    out = process(llm_pipeline(FakeClient(fail=True)))
    offline = process(offline_pipeline())
    assert {k: v.result for k, v in out.items()} == {k: v.result for k, v in offline.items()}


def test_llm_invalid_label_is_rejected():
    out = process(llm_pipeline(FakeClient(intent="buy_signal", reply="ok")))
    assert out["T1"].result.intent.value == "deposit_issue"  # rules result kept


def test_llm_promise_is_replaced_by_grounded_fallback():
    pipe = llm_pipeline(FakeClient(intent="withdrawal_issue",
                                   reply="Your withdrawal will be approved within 24 hours."))
    t3 = process(pipe)["T3"]
    assert "24 hours" not in t3.result.reply_draft
    assert t3.result.grounded
    assert t3.debug["generator"]["replaced_llm_draft"]["problems"]
    # the attempted unsupported claim is still surfaced as a safety flag
    assert "unsupported_claim" in [f.value for f in t3.result.safety_flags]


def test_llm_refusal_used_when_polite_and_replaced_when_curt():
    polite = "Thanks for asking, but I'm sorry: we can't provide market predictions or asset recommendations."
    t4 = process(llm_pipeline(FakeClient(intent="trading_advice_request", reply=polite)))["T4"]
    assert t4.result.reply_draft == polite and t4.debug["generator"]["replaced_llm_draft"] is None
    curt = "I cannot provide market predictions."
    t4 = process(llm_pipeline(FakeClient(intent="trading_advice_request", reply=curt)))["T4"]
    assert t4.result.reply_draft != curt and is_refusal(t4.result.reply_draft)


def test_refusal_is_built_from_kb_not_hardcoded(tmp_path):
    kb = json.loads((ROOT / "kb_articles.json").read_text())
    for a in kb:
        if a["article_id"] == "A5":
            a["body"] = ("Support agents must not provide price forecasts, profit tips or asset picks. "
                         "Such requests should be politely declined and redirected to the learning academy.")
    (tmp_path / "kb.json").write_text(json.dumps(kb))
    pipe = offline_pipeline()
    tickets, articles = pipe.load_data(ROOT / "tickets.json", tmp_path / "kb.json")
    t4 = {r.result.ticket_id: r.result for r in pipe.process(tickets, articles)}["T4"]
    assert "price forecasts, profit tips or asset picks" in t4.reply_draft
    assert "learning academy" in t4.reply_draft
    assert is_refusal(t4.reply_draft)


def test_validator_requires_flag_or_escalation_for_unsupported_claims(known_ids):
    texts = {a["article_id"]: a["body"] for a in json.loads((ROOT / "kb_articles.json").read_text())}
    bad = {"ticket_id": "T", "intent": "withdrawal_issue", "retrieved_articles": ["A3"],
           "reply_draft": "Your withdrawal will be approved within 2 days.", "grounded": True,
           "confidence": 0.9, "needs_human_escalation": False, "safety_flags": []}
    assert any("unsupported claims" in e for e in validate_result(bad, known_ids, texts))
    assert validate_result({**bad, "safety_flags": ["unsupported_claim"]}, known_ids, texts) == []
    assert validate_result({**bad, "needs_human_escalation": True}, known_ids, texts) == []


def test_replay_client_reproduces_without_network(tmp_path):
    from support_pipeline.llm import JsonFileCache, ReplayClient

    client = ReplayClient("fake", JsonFileCache(tmp_path / "empty.json"))
    with pytest.raises(LLMError):
        client.generate_json("prompt", {"type": "OBJECT"})
    # an empty cache behaves like a model outage -> identical to the offline path
    out = process(llm_pipeline(client))  # type: ignore[arg-type]
    assert {k: v.result for k, v in out.items()} == {k: v.result for k, v in process(offline_pipeline()).items()}


def test_llm_disagreement_sets_conflicting_signals():
    pipe = llm_pipeline(FakeClient(intent="account_lock", reply="Card deposits are usually instant."))
    t1 = process(pipe)["T1"].result
    assert "conflicting_signals" in [f.value for f in t1.safety_flags]
    assert t1.needs_human_escalation


def test_main_writes_all_artifacts(tmp_path):
    cfg = PipelineConfig(tickets_path=ROOT / "tickets.json", kb_path=ROOT / "kb_articles.json",
                         results_path=tmp_path / "results.json", debug_json_path=tmp_path / "debug.json",
                         debug_md_path=tmp_path / "debug.md", offline=True)
    run(cfg)
    results = json.loads(cfg.results_path.read_text())
    assert [r["ticket_id"] for r in results] == ["T1", "T2", "T3", "T4", "T5"]
    assert "Escalated" in cfg.debug_md_path.read_text()
