from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.indexer import (
    InvertedIndex,
    build_index_from_text_documents,
    load_index,
    save_index,
    tokenize,
)
from src import search as search_mod


def test_tokenize_case_folding() -> None:
    assert [t for t, _ in tokenize("Good FRIENDS")] == ["good", "friends"]


def test_tokenize_apostrophe_and_hyphen() -> None:
    toks = [t for t, _ in tokenize("don't well-known")]
    assert "don't" in toks
    assert "well" in toks and "known" in toks


def test_tokenize_empty_and_punct_only() -> None:
    assert tokenize("") == []
    assert tokenize("   ## $$  ") == []


def test_index_roundtrip(tmp_path: Path) -> None:
    idx = build_index_from_text_documents(
        [
            ("https://a.test/1", "alpha beta"),
            ("https://a.test/2", "beta beta gamma"),
        ],
    )
    path = tmp_path / "idx.json"
    save_index(idx, path, compress=False)
    back = load_index(path)
    assert back.num_documents == 2
    assert set(back.postings["beta"].keys()) == {"0", "1"}


def test_gzip_roundtrip(tmp_path: Path) -> None:
    idx = build_index_from_text_documents([("https://x", "hello world")])
    path = tmp_path / "idx.json"
    save_index(idx, path, compress=True)
    gz = path.with_suffix(".json.gz")
    assert gz.is_file()
    back = load_index(gz)
    assert back.postings["hello"]


def test_conjunctive_and_semantics() -> None:
    idx = build_index_from_text_documents(
        [
            ("d0", "one two"),
            ("d1", "two three"),
            ("d2", "one two three"),
        ],
    )
    hits = search_mod.conjunctive_doc_ids(idx, ["one", "three"])
    assert hits == {2}


def test_conjunctive_duplicate_terms() -> None:
    idx = build_index_from_text_documents([("d0", "foo bar"), ("d1", "foo")])
    assert search_mod.conjunctive_doc_ids(idx, ["foo", "foo"]) == {0, 1}


def test_ranking_prefers_higher_tf_idf() -> None:
    idx = InvertedIndex()
    idx.add_document("d0", [("t", 0)] * 1)
    idx.add_document("d1", [("t", 0)] * 5)
    ranked = search_mod.rank_documents(idx, {0, 1}, ["t"])
    assert ranked[0].doc_id == 1


def test_suggestions_only_for_unknown_terms() -> None:
    idx = build_index_from_text_documents([("u", "creativity innovate innovation")])
    assert search_mod.suggest_terms(idx, "creativty")  # missing letter
    assert not search_mod.suggest_terms(idx, "creativity")


def test_unknown_schema_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": "nope", "documents": [], "postings": {}, "doc_freq": {}}))
    with pytest.raises(ValueError):
        load_index(p)
