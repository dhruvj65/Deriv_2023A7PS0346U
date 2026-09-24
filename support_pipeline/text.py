"""Tiny, dependency-free text normalisation shared by retrieval and grounding checks.

Deterministic by construction: pure functions, no randomness, no external models.
"""
from __future__ import annotations

import re
from typing import List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Domain synonyms folded onto one canonical token so that e.g. "login" matches "sign-in".
SYNONYMS = {
    "login": "signin",
    "logon": "signin",
    "log": "signin",
    "sign": "signin",
    "signin": "signin",
    "passwords": "password",
    "pwd": "password",
    "locked": "lock",
    "lockout": "lock",
    "blocked": "lock",
    "mail": "email",
    "e-mail": "email",
    "cashout": "withdrawal",
    "payout": "withdrawal",
    "withdraw": "withdrawal",
    "funded": "deposit",
    "topup": "deposit",
    "stock": "asset",
    "stocks": "asset",
    "crypto": "asset",
    "shares": "asset",
    "profits": "profit",
    "invest": "investment",
    "investing": "investment",
    "declined": "decline",
    "rejected": "decline",
    "refused": "decline",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SUFFIXES = ("ations", "ation", "ings", "ing", "ily", "ies", "ly", "ed", "es", "s")

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)


def stem(token: str) -> str:
    """Very small suffix stripper (deliberately conservative)."""
    if token in SYNONYMS:
        return SYNONYMS[token]
    for suf in _SUFFIXES:
        if len(token) > len(suf) + 3 and token.endswith(suf):
            if suf in ("ies", "ily"):
                token = token[: -len(suf)] + "y"
            else:
                token = token[: -len(suf)]
                # "resetting" -> "resett" -> "reset"
                if suf in ("ing", "ed") and len(token) > 3 and token[-1] == token[-2] and token[-1] not in "lsz":
                    token = token[:-1]
            break
    return SYNONYMS.get(token, token)


def tokenize(text: str) -> List[str]:
    """Lowercase, split, drop stop words, stem, fold synonyms."""
    out: List[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw == "sign-in":
            out.append("signin")
            continue
        for part in raw.split("-"):
            if part in STOP_WORDS or len(part) < 2:
                continue
            out.append(stem(part))
    return out


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
