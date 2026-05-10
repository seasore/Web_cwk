"""
Query processing, ranking, and vocabulary suggestions ("query suggestions").

For conjunctive AND queries the dominant cost is posting-list intersection::

    O( sum_i |postings(term_i)| )

Reranking adds O(|candidates| · |Q_unique|).

**Ranking modes** (see :func:`rank_with_mode`):

- *tfidf* — logarithmic TF with smoothed IDF (assignment baseline).
- *bm25* — Okapi BM25 (Robertson & Walker, probability ranking framework);
  strong baseline for short heterogeneous texts such as quotations.
- *hybrid* — min–max normalised average of TF-IDF and BM25 per query.
- *bm25_proximity* — BM25 plus a small bonus when query terms appear close
  within the same document (minimum positional window covering all terms),
  related to *proximity / phrasal* heuristics in web search.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Iterable, Literal, Sequence

from .indexer import InvertedIndex

logger = logging.getLogger(__name__)

RankMode = Literal["tfidf", "bm25", "hybrid", "bm25_proximity"]

BM25_K1 = 1.2
BM25_B = 0.75
PROXIMITY_LAMBDA = 0.08

@dataclass(slots=True)
class RankedHit:
    """One ranked retrieval result."""

    doc_id: int
    url: str
    score: float
    term_weights: dict[str, float]


def posting_map_for_term(index: InvertedIndex, term: str) -> dict[str, dict[str, Any]]:
    """Return the inner posting map ``doc_id(str) -> {tf, positions}``."""
    return index.postings.get(term, {})


def conjunctive_doc_ids(index: InvertedIndex, terms: Sequence[str]) -> set[int]:
    """
    Documents containing *all* case-folded ``terms``.

    Duplicate terms do not raise the required occurrence count—they are treated
    as one constraint (standard bag-of-words AND). Empty ``terms`` → empty set.
    """
    uniq: list[str] = []
    seen: set[str] = set()
    for t in terms:
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    if not uniq:
        return set()

    sets_list: list[set[int]] = []
    for t in uniq:
        m = posting_map_for_term(index, t)
        sets_list.append({int(k) for k in m.keys()})

    acc = sets_list[0]
    for s in sets_list[1:]:
        acc = acc.intersection(s)
    return acc


def tf_idf_score_for_document(
    index: InvertedIndex,
    doc_id: int,
    terms: Sequence[str],
) -> tuple[float, dict[str, float]]:
    """Sum TF-IDF weights for ``terms`` in ``doc_id`` (vector-space lite)."""
    uniq: list[str] = []
    seen: set[str] = set()
    for t in terms:
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(t)

    weights: dict[str, float] = {}
    total = 0.0
    ds = str(doc_id)
    for t in uniq:
        rec = posting_map_for_term(index, t).get(ds)
        if not rec:
            continue
        tf = int(rec["tf"])
        w = index.tf_idf_weight(t, tf)
        weights[t] = w
        total += w
    return total, weights


def rank_documents(
    index: InvertedIndex,
    doc_ids: Iterable[int],
    terms: Sequence[str],
    *,
    top_k: int | None = None,
) -> list[RankedHit]:
    """Score candidate documents and sort descending by summed TF-IDF weights."""
    hits: list[RankedHit] = []
    for did in doc_ids:
        score, tw = tf_idf_score_for_document(index, did, terms)
        if score <= 0 and terms:
            continue
        url = index.documents[did].url
        hits.append(RankedHit(doc_id=did, url=url, score=score, term_weights=tw))

    hits.sort(key=lambda h: h.score, reverse=True)
    if top_k is not None and top_k >= 0:
        hits = hits[:top_k]
    return hits


def _unique_query_terms(terms: Sequence[str]) -> list[str]:
    uniq: list[str] = []
    seen: set[str] = set()
    for t in terms:
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


def average_document_length(index: InvertedIndex) -> float:
    """Mean token count per indexed document (BM25 length normalisation)."""
    n = index.num_documents
    if n == 0:
        return 1.0
    return max(1.0, sum(d.token_count for d in index.documents) / n)


def bm25_idf_rsj(N: int, df: int) -> float:
    """
    Robertson-Spärck Jones IDF variant used inside Okapi BM25:

    log((N - df + 0.5) / (df + 0.5) + 1)
    """
    if N <= 0:
        return 0.0
    df = max(0, min(df, N))
    return math.log((N - df + 0.5) / (df + 0.5) + 1.0)


def bm25_score_for_document(
    index: InvertedIndex,
    doc_id: int,
    terms: Sequence[str],
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> tuple[float, dict[str, float]]:
    """Sum BM25 weights for independent query terms (bag-of-words query)."""
    uniq = _unique_query_terms(terms)
    if not uniq:
        return 0.0, {}
    N = index.num_documents
    avgdl = average_document_length(index)
    dl = max(1, index.documents[doc_id].token_count)
    ds = str(doc_id)
    weights: dict[str, float] = {}
    total = 0.0
    for t in uniq:
        rec = posting_map_for_term(index, t).get(ds)
        if not rec:
            continue
        tf = int(rec["tf"])
        idf = bm25_idf_rsj(N, index.doc_freq.get(t, 0))
        denom = tf + k1 * (1.0 - b + b * dl / avgdl)
        w = idf * (tf * (k1 + 1.0)) / denom
        weights[t] = w
        total += w
    return total, weights


def rank_documents_bm25(
    index: InvertedIndex,
    doc_ids: Iterable[int],
    terms: Sequence[str],
    *,
    top_k: int | None = None,
) -> list[RankedHit]:
    """Rank by summed Okapi BM25 scores."""
    hits: list[RankedHit] = []
    for did in doc_ids:
        score, tw = bm25_score_for_document(index, did, terms)
        if score <= 0.0 and terms:
            continue
        hits.append(RankedHit(doc_id=did, url=index.documents[did].url, score=score, term_weights=tw))
    hits.sort(key=lambda h: h.score, reverse=True)
    if top_k is not None and top_k >= 0:
        hits = hits[:top_k]
    return hits


def min_cooccurrence_position_span(
    index: InvertedIndex,
    doc_id: int,
    terms: Sequence[str],
) -> int | None:
    """
    Minimum (max_pos - min_pos) of a window that contains at least one
    occurrence of every distinct query term in ``doc_id``.
    """
    uniq = _unique_query_terms(terms)
    if len(uniq) < 2:
        return 0
    ds = str(doc_id)
    events: list[tuple[int, int]] = []
    for ti, t in enumerate(uniq):
        rec = posting_map_for_term(index, t).get(ds)
        if not rec:
            return None
        for pos in rec["positions"]:
            events.append((pos, ti))
    if len(events) < len(uniq):
        return None
    events.sort(key=lambda x: x[0])
    needed = len(uniq)
    counts: Counter[int] = Counter()
    cover = 0
    left = 0
    best: int | None = None
    for right in range(len(events)):
        pr, ti = events[right]
        if counts[ti] == 0:
            cover += 1
        counts[ti] += 1
        while cover == needed:
            span = pr - events[left][0]
            if best is None or span < best:
                best = span
            _pleft, tileft = events[left]
            counts[tileft] -= 1
            if counts[tileft] == 0:
                cover -= 1
            left += 1
    return best


def rank_documents_bm25_proximity(
    index: InvertedIndex,
    doc_ids: Iterable[int],
    terms: Sequence[str],
    *,
    top_k: int | None = None,
    proximity_weight: float = PROXIMITY_LAMBDA,
) -> list[RankedHit]:
    """BM25 plus a proximity bonus lambda/(1+span) from minimal co-occurrence window."""
    hits: list[RankedHit] = []
    for did in doc_ids:
        base, tw = bm25_score_for_document(index, did, terms)
        span = min_cooccurrence_position_span(index, did, terms)
        if span is None:
            continue
        bonus = 0.0
        if len(_unique_query_terms(terms)) >= 2:
            bonus = proximity_weight / (1.0 + float(span))
        score = base + bonus
        if score <= 0.0 and terms:
            continue
        hits.append(RankedHit(doc_id=did, url=index.documents[did].url, score=score, term_weights=tw))
    hits.sort(key=lambda h: h.score, reverse=True)
    if top_k is not None and top_k >= 0:
        hits = hits[:top_k]
    return hits


def _minmax_normalise(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def rank_documents_hybrid(
    index: InvertedIndex,
    doc_ids: Iterable[int],
    terms: Sequence[str],
    *,
    top_k: int | None = None,
) -> list[RankedHit]:
    """Average min-max normalised TF-IDF and BM25 (per-query scale fusion)."""
    doc_list = list(doc_ids)
    uniq = _unique_query_terms(terms)
    tf_scores: dict[int, float] = {}
    bm_scores: dict[int, float] = {}
    tw_tf: dict[int, dict[str, float]] = {}
    tw_bm: dict[int, dict[str, float]] = {}
    for did in doc_list:
        ts, ttw = tf_idf_score_for_document(index, did, terms)
        bs, tbw = bm25_score_for_document(index, did, terms)
        tf_scores[did] = ts
        bm_scores[did] = bs
        tw_tf[did] = ttw
        tw_bm[did] = tbw
    n_tf = _minmax_normalise(tf_scores)
    n_bm = _minmax_normalise(bm_scores)
    hits: list[RankedHit] = []
    for did in doc_list:
        if tf_scores[did] <= 0.0 and bm_scores[did] <= 0.0 and terms:
            continue
        comb = 0.5 * n_tf[did] + 0.5 * n_bm[did]
        merged = {t: 0.5 * tw_tf[did].get(t, 0.0) + 0.5 * tw_bm[did].get(t, 0.0) for t in uniq}
        hits.append(RankedHit(doc_id=did, url=index.documents[did].url, score=comb, term_weights=merged))
    hits.sort(key=lambda h: h.score, reverse=True)
    if top_k is not None and top_k >= 0:
        hits = hits[:top_k]
    return hits


def normalise_rank_mode(raw: str | None) -> RankMode:
    """Map env string to ranker; unknown values warn and fall back to tfidf."""
    if not raw:
        return "tfidf"
    key = raw.strip().lower().replace("-", "_")
    mapping: dict[str, RankMode] = {
        "tfidf": "tfidf",
        "tf-idf": "tfidf",
        "tf_idf": "tfidf",
        "bm25": "bm25",
        "okapi": "bm25",
        "hybrid": "hybrid",
        "bm25_proximity": "bm25_proximity",
        "proximity": "bm25_proximity",
    }
    mode = mapping.get(key)
    if mode is None:
        logger.warning("Unknown SEARCH_ENGINE_RANKER=%r — using tfidf", raw)
        return "tfidf"
    return mode


def rank_with_mode(
    index: InvertedIndex,
    doc_ids: Iterable[int],
    terms: Sequence[str],
    mode: RankMode,
    *,
    top_k: int | None = None,
) -> list[RankedHit]:
    """Dispatch ranking by strategy name."""
    if mode == "bm25":
        return rank_documents_bm25(index, doc_ids, terms, top_k=top_k)
    if mode == "hybrid":
        return rank_documents_hybrid(index, doc_ids, terms, top_k=top_k)
    if mode == "bm25_proximity":
        return rank_documents_bm25_proximity(index, doc_ids, terms, top_k=top_k)
    return rank_documents(index, doc_ids, terms, top_k=top_k)


def print_term_postings(index: InvertedIndex, term: str) -> dict[str, Any]:
    """Materialise the inverted list for ``term`` with DF / IDF metadata."""
    m = posting_map_for_term(index, term)
    postings: list[dict[str, Any]] = []
    for ds in sorted(m.keys(), key=int):
        rec = m[ds]
        did = int(ds)
        postings.append(
            {
                "doc_id": did,
                "url": index.documents[did].url,
                "tf": int(rec["tf"]),
                "positions": list(rec["positions"]),
            },
        )
    return {
        "term": term,
        "doc_freq": index.doc_freq.get(term, 0),
        "idf": index.idf(term),
        "postings": postings,
    }


def suggest_terms(index: InvertedIndex, word: str, *, n: int = 5, cutoff: float = 0.6) -> list[str]:
    """
    Lightweight "did you mean" suggestions via :mod:`difflib`.

    Returns an empty list when ``word`` already exists or is blank.
    """
    vocab = index.vocabulary()
    if not vocab or not word:
        return []
    key = word.casefold()
    if key in index.postings:
        return []
    return get_close_matches(key, vocab, n=n, cutoff=cutoff)


def multi_suggestions(
    index: InvertedIndex,
    missing_terms: Sequence[str],
    *,
    n_per_term: int = 3,
) -> dict[str, list[str]]:
    """Batch wrapper preserving input order of ``missing_terms``."""
    out: dict[str, list[str]] = {}
    for t in missing_terms:
        if not t:
            continue
        sugs = suggest_terms(index, t, n=n_per_term)
        if sugs:
            out[t] = sugs
    return out


def parse_query_tokens(raw_parts: Sequence[str]) -> list[str]:
    """Lower-case tokenizer reuse for CLI queries."""
    from .indexer import tokenize

    joined = " ".join(p for p in raw_parts if p is not None and str(p).strip() != "")
    return [tok for tok, _ in tokenize(joined)]
