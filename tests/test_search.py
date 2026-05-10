from __future__ import annotations

from src.indexer import build_index_from_text_documents
from src import search as search_mod


def test_parse_query_tokens_strips_noise() -> None:
    assert search_mod.parse_query_tokens(["  Good,", "friends!!"]) == ["good", "friends"]


def test_print_payload_structure() -> None:
    idx = build_index_from_text_documents([("https://z", "hello hello world")])
    payload = search_mod.print_term_postings(idx, "hello")
    assert payload["term"] == "hello"
    assert payload["doc_freq"] == 1
    assert len(payload["postings"]) == 1
    assert payload["postings"][0]["tf"] == 2


def test_bm25_prefers_shorter_document_same_tf() -> None:
    """BM25 length normalisation: same tf=1 favours a shorter document."""
    filler = " ".join(["x"] * 80)
    idx = build_index_from_text_documents(
        [
            ("d0", f"cat {filler}"),
            ("d1", "cat"),
        ],
    )
    s0, _ = search_mod.bm25_score_for_document(idx, 0, ["cat"])
    s1, _ = search_mod.bm25_score_for_document(idx, 1, ["cat"])
    assert s1 > s0


def test_min_cooccurrence_span() -> None:
    idx = build_index_from_text_documents(
        [
            ("close", "alpha beta gamma"),
            ("far", "alpha z z z z beta"),
        ],
    )
    assert search_mod.min_cooccurrence_position_span(idx, 0, ["alpha", "beta"]) == 1
    assert search_mod.min_cooccurrence_position_span(idx, 1, ["alpha", "beta"]) == 5


def test_proximity_ranker_prefers_tighter_spans() -> None:
    idx = build_index_from_text_documents(
        [
            ("u0", "alpha beta gamma delta"),
            ("u1", "alpha z z z z beta"),
        ],
    )
    hits = search_mod.rank_documents_bm25_proximity(idx, {0, 1}, ["alpha", "beta"])
    assert hits[0].doc_id == 0


def test_rank_with_mode_hybrid() -> None:
    idx = build_index_from_text_documents(
        [
            ("a", "foo bar"),
            ("b", "foo baz"),
        ],
    )
    h = search_mod.rank_with_mode(idx, {0, 1}, ["foo"], "hybrid")
    assert len(h) == 2


def test_normalise_rank_mode_unknown_falls_back() -> None:
    assert search_mod.normalise_rank_mode("not-a-mode") == "tfidf"
    assert search_mod.normalise_rank_mode("okapi") == "bm25"
