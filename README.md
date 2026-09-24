# Support Ticket AI Service

A small, runnable pipeline that takes customer support tickets and, for each one:

1. classifies the intent,
2. retrieves the most relevant help-center articles, and
3. writes a grounded reply draft with a confidence score, an escalation signal and safety flags.

It works in two modes:

| Mode | When | Classifier | Generator |
|---|---|---|---|
| **LLM** | `GEMINI_API_KEY` is set in `.env` | Rules + Gemini (hybrid, enum-constrained) | Gemini, prompted only with retrieved snippets |
| **Offline** | no key, or `--offline` | Rules | Deterministic extractive template |

Retrieval (TF-IDF) and validation are the same in both modes. If a Gemini call fails, or an
LLM draft fails validation, that ticket falls back to the offline path automatically.

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

```bash
python validate.py
```

```bash
python -m pytest -q
```

* `python main.py` reads `tickets.json` and `kb_articles.json`, then writes `results.json`,
  `debug_report.json` and `debug_report.md`.
* `python main.py --offline` never calls a model, so it needs no secrets.
* Optional: copy `.env.example` to `.env` and set `GEMINI_API_KEY` (and optionally
  `GEMINI_MODEL`; the default is `gemini-3.5-flash-lite`).
* `python validate.py` checks the artifacts and exits non-zero if any check fails.
* The pytest suite runs fully offline; the LLM is mocked.

---

## Pipeline

```text
LOAD_DATA -> INDEX_KB -> CLASSIFY_INTENT -> RETRIEVE_CONTEXT -> GENERATE_REPLY -> VALIDATE_OUTPUT -> WRITE_RESULTS
```

Each stage is its own method on `SupportPipeline` in `support_pipeline/pipeline.py`:

| Stage | What it does |
|---|---|
| `load_data` | Loads both JSON files and validates them with pydantic. Rejects duplicate IDs and an empty KB. |
| `index_kb` | Fits the retriever on the KB, sorted by `article_id` so file order doesn't matter. |
| `classify_intent` | Returns one of the six allowed labels, plus a certainty score and the evidence behind it. |
| `retrieve_context` | Uses **only the ticket text** as the query and returns the top 1–3 articles. It never looks at the intent. |
| `generate_reply` | Receives only the retrieved articles. Falls back to the template generator if the model errors. |
| `validate_output` | Runs grounding, unsupported-claim and refusal checks, replaces a failed LLM draft with the safe version, computes confidence, escalation and flags, and validates the schema. |
| `write_results` | Writes `results.json`, `debug_report.json` and `debug_report.md`. |

### Swappable components

Components are injected and typed with `Protocol`s, so each one can be replaced without touching the others:

| Interface | Current implementations | Could be swapped for |
|---|---|---|
| `IntentClassifier` | `RuleBasedClassifier`, `LLMClassifier`, `HybridClassifier` | a fine-tuned model |
| `Retriever` | `TfidfRetriever` | an embeddings or BM25 retriever |
| `ReplyGenerator` | `TemplateGenerator` (offline), `LLMGenerator` | another hosted model |
| `LLMClient` | `GeminiClient` (REST, with a disk cache) | any provider with a `generate_json(prompt, schema)` method |

`llm.py` is the only module that knows which provider is in use.

---

## How each part works

### Intent classification (`classifier.py`)

* **Rules.** Weighted regex patterns for each intent. Certainty comes from the margin over the
  runner-up intent and from how many rules matched. If the best score is too weak, the label is `other`.
* **LLM.** Gemini with a `responseSchema` whose `intent` field is an **enum of the six allowed
  labels**. The code re-checks the returned label and rejects anything outside the set.
* **Hybrid reconciliation:**
  * If rules and the LLM agree, certainty goes up.
  * If either one says `trading_advice_request`, that label wins (the safe choice).
  * Any other disagreement uses the LLM label, lowers certainty and adds `conflicting_signals`.

### Retrieval (`retrieval.py`, `text.py`)

* TF-IDF cosine similarity over title + body (the title is counted twice).
* A custom deterministic analyzer handles stop words, light stemming and a few domain synonyms.
  For example, `login` and `sign-in` both become `signin`, and `locked` becomes `lock`.
* **Always returns 1–3 articles.** The top article is always included. Extra articles must score
  at least `0.08` and at least half the top score.
* Ties are broken by score, then `article_id`, so the same input always gives the same output.

### Grounded generation (`generator.py`)

* **LLM prompt.** Contains only the retrieved snippets and explicitly tells the model:
  * use only the snippets;
  * don't state account-specific facts;
  * don't promise payment or withdrawal outcomes or timing;
  * don't quote internal agent guidance;
  * refuse trading advice.

  The model returns JSON: `reply_draft`, `cited_article_ids`, `answerable`, `refused`.
* **Offline template.** Pulls sentences from the retrieved articles and rewrites them from agent
  voice into customer voice (for example, "Ask the client to…" becomes "Please…"). It skips
  internal-only sentences such as "Support should not promise…". Replies are assembled from the
  KB, not hardcoded per ticket.
* **Trading advice.** In LLM mode the model drafts the refusal from the no-advice snippet. The
  offline refusal is **assembled from the retrieved policy article**: the list of what support
  must not provide, and where to redirect the customer, are taken from A5's text. Only short
  courtesy phrases are fixed. If no policy article is retrieved, a built-in minimal refusal is
  used, and the reply is marked ungrounded and escalated.
* **No relevant article.** Neither generator improvises. The reply says the ticket is being
  passed to the team, and the ticket is escalated.

### Validation (`validation.py`)

Every result goes through these checks before it is written:

* **Structure:** required fields are present with the right types, and no extra fields.
* **Allowed values:** `intent` is an allowed label, `retrieved_articles` has 1–3 IDs that all
  exist in the KB, `confidence` is in [0, 1], and every safety flag is a known flag.
* **Trading advice:** these requests must be *politely* refused (explicit decline, a courtesy
  marker such as "sorry" or "thanks", and no advice patterns) and must carry the `advice_request` flag.
* **Grounding:** each factual sentence is checked for word overlap with the retrieved articles.
  A sentence counts as grounded if at least 50% of its content words appear in them (restating
  the customer's own words is allowed). Every sentence is mapped to its best source article in the debug report.
* **Unsupported claims** are caught by pattern:
  * outcome or timing promises ("will be approved", "within 24 hours", "guaranteed");
  * invented account facts ("I checked your account", "your withdrawal was declined because");
  * trading advice ("you should buy", "will go up");
  * numbers that don't appear in the sources.
* **Failure handling:** unsupported claims, an ungrounded sentence or a bad citation in an LLM
  draft cause the draft to be **replaced** by the offline extractive draft. The original draft
  is kept in the debug report.
  * If the discarded draft contained unsupported claims, the result still carries the
    `unsupported_claim` flag for auditing.
  * If the final reply itself still has problems, it gets `unsupported_claim` or
    `insufficient_grounding` and is escalated.
  * `validate_result` re-scans the final reply. Any unsupported claim without a flag or
    escalation is a validation error, and the result is not written.

### Confidence and escalation (`policy.py`)

```text
confidence = 0.35 * classifier_certainty + 0.35 * retrieval_strength + 0.30 * grounding_ratio
```

* `retrieval_strength` is `min(1, top_score / 0.4)`.
* Confidence is multiplied by 0.8 when the classifiers conflict.
* Confidence is capped at 0.3–0.4 when grounding is missing or a claim is unsupported.

A ticket is sent to a human (`needs_human_escalation = true`) if any of these apply:

| Flag / reason | Trigger |
|---|---|
| `low_confidence` | confidence < 0.55 |
| `insufficient_grounding` | top retrieval score < 0.1, an ungrounded reply, or the model reports the question can't be answered from the snippets |
| `unsupported_claim` | the final reply fails the claim checks. If only a discarded LLM draft failed, the flag is set without escalating. |
| `policy_sensitive` | withdrawal issues, or terms like fraud, chargeback or compliance (A3: support must not promise outcomes without checking account status) |
| `conflicting_signals` | the rule and LLM classifiers disagree |
| intent `other` | no help-center workflow covers it |

Two flags are recorded but don't escalate on their own:

* `account_specific_request`: the customer refers to their own account. The reply sticks to general KB guidance.
* `advice_request`: escalates only if the refusal failed.

When a ticket escalates, the reply ends with a short hand-off sentence.

---

## Outputs

* **`results.json`:** one entry per ticket, in exactly the required schema.
* **`debug_report.json` / `debug_report.md`:** for each ticket:
  * the predicted intent, with classifier evidence and notes;
  * retrieved article titles and scores;
  * the generator used, and any replaced LLM draft;
  * the sentence-to-article trace;
  * confidence components;
  * **why it was or wasn't escalated.**

## Reproducibility

* **Offline mode** is fully deterministic: no randomness, and ties are broken in a stable order.
* **LLM mode** uses `temperature=0`. Responses are cached in `.cache/llm_cache.json`, keyed by
  a hash of the model, prompt and schema, so re-running on the same inputs gives identical
  output without calling the API again.
* Gemini calls retry on 429 and 5xx errors with bounded backoff.
* `validate.py` re-runs the pipeline in the mode that produced `results.json` and checks the
  output is identical. It also runs offline mode twice and compares the two runs.
  * For an LLM run it uses **replay mode**: model responses come only from the cache, with no
    API key and no network calls. That is why `.cache/llm_cache.json` is committed; it holds
    model outputs only, no secrets.
  * A cache miss behaves like a model outage, so the ticket takes the offline fallback path.
* `validate.py`'s ticket checks are **content-based, not ID-based**, so they still work if
  `tickets.json` is replaced:
  * every ticket that looks like an advice request (using a keyword check separate from the
    pipeline's classifier), or is classified as one, must be politely refused and flagged;
  * every ticket that mentions a deposit must retrieve a deposit article.

## Project layout

```text
main.py                  # single run command
validate.py              # single validation command
tickets.json             # sample input
kb_articles.json         # sample KB
results.json             # output (generated)
debug_report.json/.md    # explainability artifacts (generated)
support_pipeline/
  schemas.py             # pydantic models, allowed labels and flags, JSON schema
  text.py                # deterministic tokenizer, stemmer, synonyms
  retrieval.py           # Retriever protocol + TF-IDF retriever
  classifier.py          # rule, LLM and hybrid classifiers
  generator.py           # template + LLM generators, prompt
  llm.py                 # Gemini REST client, cache, retries (provider boundary)
  validation.py          # grounding, claim and refusal checks, result validation
  policy.py              # confidence, safety flags, escalation
  pipeline.py            # stage orchestration + report writing
tests/test_pipeline.py   # pytest suite (offline, LLM mocked)
```

## Limitations

* The grounding check measures word overlap, not meaning. A paraphrase that uses different
  words can be marked unsupported, which is the safe direction: the draft falls back to the
  extractive version or gets escalated.
* Rules and synonyms are tuned for English and this domain. The `language` field is carried
  through but not used.
* The KB has no concrete processing windows, so replies never give specific timeframes.
