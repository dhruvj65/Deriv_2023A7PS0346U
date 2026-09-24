# Debug report

Mode: `llm:gemini-3.5-flash-lite`

| Ticket | Intent | Retrieved articles | Confidence | Escalated | Flags |
|---|---|---|---|---|---|
| T1 | deposit_issue | A1 Card deposit processing times (0.55) | 1.0 | no | account_specific_request |
| T2 | password_reset | A2 Password reset troubleshooting (0.59) | 1.0 | no | - |
| T3 | withdrawal_issue | A3 Withdrawal review and declines (0.46) | 1.0 | yes | policy_sensitive, account_specific_request |
| T4 | trading_advice_request | A5 No investment or trading advice (0.28) | 0.9 | no | advice_request |
| T5 | account_lock | A4 Account lock after failed sign-in attempts (0.66) | 1.0 | no | account_specific_request |

## T1

> I deposited by bank card two hours ago but my balance still shows zero. What should I check?

- **Predicted intent:** `deposit_issue` (method: hybrid, certainty 1.0)
- **Classifier evidence:** deposited, bank card, balance
- **Classifier notes:** rules=deposit_issue, llm=deposit_issue
- **Retrieved articles:** A1 "Card deposit processing times" (score 0.550)
- **Generator:** llm:gemini-3.5-flash-lite
- **Confidence:** 1.0 (components: {'classifier_certainty': 1.0, 'retrieval_top_score': 0.549634, 'retrieval_strength': 1.0, 'grounding_score': 1.0, 'conflicting': False})
- **Escalated: no** - references the customer's own account; reply gives general KB guidance only, no account-specific facts; handled self-serve: intent 'deposit_issue' covered by retrieved KB, confidence 1.0 >= 0.55
- **Reply trace (sentence -> source):**
  - [supported, A1] Card deposits are usually instant, but they can sometimes be delayed due to issuer checks, network issues, or pending review.
  - [supported, A1] You can check your transaction history and confirm that the payment was successful.
  - [supported, A1] Please wait up to the documented processing window before escalating.

## T2

> I forgot my password and I am not receiving the reset email.

- **Predicted intent:** `password_reset` (method: hybrid, certainty 1.0)
- **Classifier evidence:** password, reset, forgot, reset email
- **Classifier notes:** rules=password_reset, llm=password_reset
- **Retrieved articles:** A2 "Password reset troubleshooting" (score 0.592)
- **Generator:** llm:gemini-3.5-flash-lite
- **Confidence:** 1.0 (components: {'classifier_certainty': 1.0, 'retrieval_top_score': 0.592095, 'retrieval_strength': 1.0, 'grounding_score': 1.0, 'conflicting': False})
- **Escalated: no** - handled self-serve: intent 'password_reset' covered by retrieved KB, confidence 1.0 >= 0.55
- **Reply trace (sentence -> source):**
  - [supported, A2] If you do not receive a password reset email, please confirm your registered email address, check your spam folders, wait a few minutes, and retry.
  - [supported, A2] Please note that repeated failed attempts may trigger temporary rate limits.

## T3

> Why was my withdrawal declined after verification?

- **Predicted intent:** `withdrawal_issue` (method: hybrid, certainty 1.0)
- **Classifier evidence:** withdrawal, declined
- **Classifier notes:** rules=withdrawal_issue, llm=withdrawal_issue
- **Retrieved articles:** A3 "Withdrawal review and declines" (score 0.463)
- **Generator:** llm:gemini-3.5-flash-lite
- **Confidence:** 1.0 (components: {'classifier_certainty': 1.0, 'retrieval_top_score': 0.463309, 'retrieval_strength': 1.0, 'grounding_score': 1.0, 'conflicting': False})
- **Escalated: yes** - policy-sensitive topic (outcome depends on account status / compliance review that support cannot see or promise)
- **Other notes:** references the customer's own account; reply gives general KB guidance only, no account-specific facts
- **Reply trace (sentence -> source):**
  - [supported, A3] Withdrawals may be declined due to incomplete verification, mismatched payment method details, compliance review, or account restrictions.

## T4

> Can you tell me which asset will go up today so I can make profit?

- **Predicted intent:** `trading_advice_request` (method: hybrid, certainty 1.0)
- **Classifier evidence:** which asset, go up, make profit
- **Classifier notes:** rules=trading_advice_request, llm=trading_advice_request
- **Retrieved articles:** A5 "No investment or trading advice" (score 0.283)
- **Generator:** llm:gemini-3.5-flash-lite
- **Confidence:** 0.9 (components: {'classifier_certainty': 1.0, 'retrieval_top_score': 0.283398, 'retrieval_strength': 0.708, 'grounding_score': 1.0, 'conflicting': False})
- **Escalated: no** - trading-advice request politely refused per no-advice policy; no human needed; handled self-serve: intent 'trading_advice_request' covered by retrieved KB, confidence 0.9 >= 0.55
- **Reply trace (sentence -> source):**
  - [policy_refusal, A5] Thank you for reaching out, but I cannot provide market predictions, profit guarantees, or asset recommendations.
  - [policy_refusal, A5] We encourage you to explore our general educational resources to learn more about trading.

## T5

> My account was locked after too many login attempts. What do I do now?

- **Predicted intent:** `account_lock` (method: hybrid, certainty 1.0)
- **Classifier evidence:** locked, too many login attempts, login
- **Classifier notes:** rules=account_lock, llm=account_lock
- **Retrieved articles:** A4 "Account lock after failed sign-in attempts" (score 0.661)
- **Generator:** llm:gemini-3.5-flash-lite
- **Confidence:** 1.0 (components: {'classifier_certainty': 1.0, 'retrieval_top_score': 0.661071, 'retrieval_strength': 1.0, 'grounding_score': 1.0, 'conflicting': False})
- **Escalated: no** - references the customer's own account; reply gives general KB guidance only, no account-specific facts; handled self-serve: intent 'account_lock' covered by retrieved KB, confidence 1.0 >= 0.55
- **Reply trace (sentence -> source):**
  - [boilerplate, -] Hello!
  - [supported, A4] For security reasons, repeated failed sign-in attempts can temporarily lock your account.
  - [supported, A4] You can wait for the cooldown period or use the secure recovery flow.
