"""
Micro-benchmarks for index construction and conjunctive retrieval.

Run from the ``code`` directory::

    python benchmark.py

This is intentionally tiny (no external harness): it prints wall-clock timings
so the demonstration video can reference reproducible numbers on a laptop.
"""

from __future__ import annotations

import random
import string
import tempfile
import time
from pathlib import Path

from src.indexer import (
    InvertedIndex,
    build_index_from_text_documents,
    load_index,
    save_index,
    tokenize,
)
from src import search as search_mod


def _random_doc(vocab: list[str], n_words: int) -> str:
    return " ".join(random.choice(vocab) for _ in range(n_words))


def bench_tokenize(iters: int = 500) -> float:
    blob = " ".join(_random_doc(list(string.ascii_lowercase), 400) for _ in range(20))
    t0 = time.perf_counter()
    for _ in range(iters):
        tokenize(blob)
    return (time.perf_counter() - t0) / iters


def bench_intersection(n_docs: int = 5_000, avg_posting: int = 120) -> float:
    """Synthetic workload: intersect three frequent terms (posting-list AND)."""
    vocab = [f"w{i}" for i in range(500)]
    idx = InvertedIndex()
    for d in range(n_docs):
        words = random.sample(vocab, k=min(len(vocab), avg_posting))
        pairs = [(w, i) for i, w in enumerate(words)]
        idx.add_document(f"https://example.test/doc/{d}", pairs)

    terms = [vocab[0], vocab[1], vocab[2]]
    t0 = time.perf_counter()
    hits = 50
    for _ in range(hits):
        search_mod.conjunctive_doc_ids(idx, terms)
    return (time.perf_counter() - t0) / hits


def bench_serialise(n_docs: int = 800) -> tuple[float, float, Path]:
    vocab = [f"t{i}" for i in range(200)]
    pairs = []
    for d in range(n_docs):
        words = random.sample(vocab, k=80)
        text = " ".join(words)
        pairs.append((f"https://example.test/{d}", text))
    idx = build_index_from_text_documents(pairs)

    tmp = Path(tempfile.mkdtemp()) / "idx.json"
    t0 = time.perf_counter()
    save_index(idx, tmp, compress=True)
    save_s = time.perf_counter() - t0

    written = tmp.with_suffix(".json.gz")
    t0 = time.perf_counter()
    load_index(written)
    load_s = time.perf_counter() - t0
    return save_s, load_s, written


def bench_rank_modes(n_docs: int = 2_000, iters: int = 40) -> None:
    """Wall time for ranking conjunctive candidates with each strategy."""
    vocab = [f"w{i}" for i in range(400)]
    idx = InvertedIndex()
    for d in range(n_docs):
        words = random.sample(vocab, k=60)
        pairs = [(w, i) for i, w in enumerate(words)]
        idx.add_document(f"https://example.test/{d}", pairs)
    cand = search_mod.conjunctive_doc_ids(idx, [vocab[0], vocab[1], vocab[2]])
    for mode in ("tfidf", "bm25", "hybrid", "bm25_proximity"):
        rank_mode = search_mod.normalise_rank_mode(mode)
        t0 = time.perf_counter()
        for _ in range(iters):
            search_mod.rank_with_mode(idx, cand, [vocab[0], vocab[1], vocab[2]], rank_mode)
        elapsed = (time.perf_counter() - t0) / iters
        print(f"rank_with_mode({mode!r}, mean of {iters} iters): {elapsed:.6f}s")


def main() -> None:
    random.seed(42)
    print("=== coursework2 micro-benchmarks ===")
    print(f"tokenize(mean of 500 iters, long blob): {bench_tokenize():.6f}s")
    print(f"AND intersection (mean of 50 iters): {bench_intersection():.6f}s")
    s_save, s_load, p = bench_serialise()
    print(f"save_index gzip ({p.name}): {s_save:.4f}s")
    print(f"load_index: {s_load:.4f}s")
    print("\nRanking (synthetic conjunctive set, all modes):")
    bench_rank_modes()
    print("\nComplexity cheat-sheet:")
    print("- tokenize: O(|text|) regex scan")
    print("- build: O(total tokens)")
    print("- AND: ~O(sum posting list sizes); rerank O(|hits| * |terms|)")


if __name__ == "__main__":
    main()
