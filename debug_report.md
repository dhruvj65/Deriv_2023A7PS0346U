# Debug report

Mode: `llm:gemini-2.5-flash-lite`

| Ticket | Intent | Retrieved articles | Confidence | Escalated | Flags |
|---|---|---|---|---|---|
| T1 | deposit_issue | A1 Card deposit processing times (0.55) | 0.97 | no | account_specific_request |
| T2 | password_reset | A2 Password reset troubleshooting (0.59) | 1.0 | no | - |
| T3 | withdrawal_issue | A3 Withdrawal review and declines (0.46) | 0.94 | yes | policy_sensitive, account_specific_request |
| T4 | trading_advice_request | A5 No investment or trading advice (0.28) | 0.9 | no | advice_request |
| T5 | account_lock | A4 Account lock after failed sign-in attempts (0.66) | 1.0 | no | account_specific_request |

## T1

> I deposited by bank card two hours ago but my balance still shows zero. What should I check?

- **Predicted intent:** `deposit_issue` (method: rules, certainty 0.917)
- **Classifier evidence:** deposited, bank card, balance
- **Classifier notes:** LLM classifier unavailable, used rules (HTTP 404: This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemin)
- **Retrieved articles:** A1 "Card deposit processing times" (score 0.550)
- **Generator:** template
- **Generator notes:** primary generator failed (HTTP 404: This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemin); used offline fallback
- **Confidence:** 0.97 (components: {'classifier_certainty': 0.917, 'retrieval_top_score': 0.549634, 'retrieval_strength': 1.0, 'grounding_score': 1.0, 'conflicting': False})
- **Escalated: no** - references the customer's own account; reply gives general KB guidance only, no account-specific facts; handled self-serve: intent 'deposit_issue' covered by retrieved KB, confidence 0.97 >= 0.55
- **Reply trace (sentence -> source):**
  - [boilerplate, -] Thanks for reaching out about your deposit.
  - [supported, A1] Card deposits are usually instant, but may be delayed by issuer checks, network issues, or pending review.
  - [supported, A1] Please confirm successful payment, check transaction history, and wait up to the documented processing window before contacting us again.

## T2

> I forgot my password and I am not receiving the reset email.

- **Predicted intent:** `password_reset` (method: rules, certainty 1.0)
- **Classifier evidence:** password, reset, forgot, reset email
- **Classifier notes:** LLM classifier unavailable, used rules (HTTP 404: This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemin)
- **Retrieved articles:** A2 "Password reset troubleshooting" (score 0.592)
- **Generator:** template
- **Generator notes:** primary generator failed (HTTP 404: This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemin); used offline fallback
- **Confidence:** 1.0 (components: {'classifier_certainty': 1.0, 'retrieval_top_score': 0.592095, 'retrieval_strength': 1.0, 'grounding_score': 1.0, 'conflicting': False})
- **Escalated: no** - handled self-serve: intent 'password_reset' covered by retrieved KB, confidence 1.0 >= 0.55
- **Reply trace (sentence -> source):**
  - [supported, A2] Sorry you're having trouble resetting your password.
  - [supported, A2] If you do not receive a reset email, please confirm the registered email address, check spam folders, wait a few minutes, and retry.
  - [supported, A2] Repeated failed attempts may trigger temporary rate limits.

## T3

> Why was my withdrawal declined after verification?

- **Predicted intent:** `withdrawal_issue` (method: rules, certainty 0.833)
- **Classifier evidence:** withdrawal, declined
- **Classifier notes:** LLM classifier unavailable, used rules (HTTP 404: This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemin)
- **Retrieved articles:** A3 "Withdrawal review and declines" (score 0.463)
- **Generator:** template
- **Generator notes:** primary generator failed (HTTP 404: This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemin); used offline fallback
- **Confidence:** 0.94 (components: {'classifier_certainty': 0.833, 'retrieval_top_score': 0.463309, 'retrieval_strength': 1.0, 'grounding_score': 1.0, 'conflicting': False})
- **Escalated: yes** - policy-sensitive topic (outcome depends on account status / compliance review that support cannot see or promise)
- **Other notes:** references the customer's own account; reply gives general KB guidance only, no account-specific facts
- **Reply trace (sentence -> source):**
  - [boilerplate, -] Thanks for reaching out about your withdrawal.
  - [supported, A3] Withdrawals may be declined due to incomplete verification, mismatched payment method details, compliance review, or account restrictions.

## T4

> Can you tell me which asset will go up today so I can make profit?

- **Predicted intent:** `trading_advice_request` (method: rules, certainty 1.0)
- **Classifier evidence:** which asset, go up, make profit
- **Classifier notes:** LLM classifier unavailable, used rules (HTTP 404: This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemin)
- **Retrieved articles:** A5 "No investment or trading advice" (score 0.283)
- **Generator:** template
- **Generator notes:** primary generator failed (HTTP 404: This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemin); used offline fallback
- **Confidence:** 0.9 (components: {'classifier_certainty': 1.0, 'retrieval_top_score': 0.283398, 'retrieval_strength': 0.708, 'grounding_score': 1.0, 'conflicting': False})
- **Escalated: no** - trading-advice request politely refused per no-advice policy; no human needed; handled self-serve: intent 'trading_advice_request' covered by retrieved KB, confidence 0.9 >= 0.55
- **Reply trace (sentence -> source):**
  - [policy_refusal, A5] Thanks for your question.
  - [policy_refusal, A5] I'm sorry, but we're not able to provide market predictions, profit guarantees, or asset recommendations.
  - [policy_refusal, A5] If you'd like to learn more, you may find our general educational resources helpful, where available.

## T5

> My account was locked after too many login attempts. What do I do now?

- **Predicted intent:** `account_lock` (method: rules, certainty 1.0)
- **Classifier evidence:** locked, too many login attempts, login
- **Classifier notes:** LLM classifier unavailable, used rules (HTTP 404: This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemin)
- **Retrieved articles:** A4 "Account lock after failed sign-in attempts" (score 0.661)
- **Generator:** template
- **Generator notes:** primary generator failed (HTTP 404: This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemin); used offline fallback
- **Confidence:** 1.0 (components: {'classifier_certainty': 1.0, 'retrieval_top_score': 0.661071, 'retrieval_strength': 1.0, 'grounding_score': 1.0, 'conflicting': False})
- **Escalated: no** - references the customer's own account; reply gives general KB guidance only, no account-specific facts; handled self-serve: intent 'account_lock' covered by retrieved KB, confidence 1.0 >= 0.55
- **Reply trace (sentence -> source):**
  - [boilerplate, -] Sorry to hear you can't get into your account.
  - [supported, A4] For security reasons, repeated failed sign-in attempts can temporarily lock your account.
  - [supported, A4] Please wait for the cooldown period or use the secure recovery flow.
  - [supported, A4] Please avoid sharing account-specific details in unsecured channels.
