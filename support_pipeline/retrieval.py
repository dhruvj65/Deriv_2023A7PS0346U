"""Retrieval stage: index the KB and return the top-k articles for a ticket.

Retrieval is fully independent of generation. Any class implementing the
`Retriever` protocol (e.g. an embeddings retriever) can be dropped in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .schemas import Article
from .text import tokenize


@dataclass(frozen=True)
class RetrievedArticle:
    article: Article
    score: float
    rank: int


class Retriever(Protocol):
    def index(self, articles: Sequence[Article]) -> None: ...

    def retrieve(self, query: str) -> List[RetrievedArticle]: ...


class TfidfRetriever:
    """Deterministic TF-IDF cosine retriever over title + body.

    Returns 1..top_k articles. The best article is always returned (so the
    downstream stages can decide whether it is good enough); additional
    articles are only kept if they clear both an absolute score floor and a
    fraction of the top score.
    """

    def __init__(self, top_k: int = 3, min_score: float = 0.08, relative_cutoff: float = 0.5):
        if not 1 <= top_k <= 3:
            raise ValueError("top_k must be between 1 and 3")
        self.top_k = top_k
        self.min_score = min_score
        self.relative_cutoff = relative_cutoff
        self._articles: List[Article] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def index(self, articles: Sequence[Article]) -> None:
        # Sort by id so the index (and tie-breaking) never depends on file order.
        self._articles = sorted(articles, key=lambda a: a.article_id)
        self._vectorizer = TfidfVectorizer(
            analyzer=tokenize,  # custom deterministic analyzer (stemming + synonyms)
            sublinear_tf=True,
            norm="l2",
        )
        # Title is repeated once to give it slightly more weight than the body.
        docs = [f"{a.title}. {a.title}. {a.body}" for a in self._articles]
        self._matrix = self._vectorizer.fit_transform(docs)

    def retrieve(self, query: str) -> List[RetrievedArticle]:
        if self._vectorizer is None:
            raise RuntimeError("index() must be called before retrieve()")
        q = self._vectorizer.transform([query])
        scores = (self._matrix @ q.T).toarray().ravel()
        # Stable ordering: score desc, then article_id asc.
        order = sorted(range(len(scores)), key=lambda i: (-round(float(scores[i]), 10), self._articles[i].article_id))

        results: List[RetrievedArticle] = []
        top_score = float(scores[order[0]]) if order else 0.0
        for rank, i in enumerate(order[: self.top_k]):
            s = float(np.round(scores[i], 6))
            if rank > 0 and (s < self.min_score or s < self.relative_cutoff * top_score):
                break
            results.append(RetrievedArticle(article=self._articles[i], score=s, rank=rank + 1))
        return results
