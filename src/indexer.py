"""
Tokenisation and inverted-index construction with TF-IDF statistics.

Tokenisation is O(n) in text length; building the index is O(W) over all
token occurrences W. Storing positions adds memory linear in W.

IDF uses the textbook smoothed form::

    idf(t) = log((N + 1) / (df(t) + 1)) + 1

so rare terms receive higher weight while avoiding log-zeros.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "xjco3011.index.v1"

# Word tokens: letters/digits plus interior apostrophe (don't, o'clock).
_TOKEN_RE = re.compile(r"[0-9]+|[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    """Minimal document record stored in the serialised index."""

    doc_id: int
    url: str
    token_count: int


class InvertedIndex:
    """
    In-memory inverted index with positions and TF-IDF helpers.

    :ivar documents: Ordered document metadata aligned by ``doc_id``.
    :ivar postings: Mapping ``term -> doc_id(str) -> {tf, positions}``.
    :ivar doc_freq: Document frequency per term.
    """

    __slots__ = ("documents", "postings", "doc_freq", "_vocab_sorted")

    def __init__(self) -> None:
        self.documents: list[IndexedDocument] = []
        self.postings: dict[str, dict[str, dict[str, Any]]] = {}
        self.doc_freq: dict[str, int] = {}
        self._vocab_sorted: list[str] | None = None

    @property
    def num_documents(self) -> int:
        return len(self.documents)

    def vocabulary(self) -> list[str]:
        if self._vocab_sorted is None:
            self._vocab_sorted = sorted(self.postings.keys())
        return list(self._vocab_sorted)

    def invalidate_vocabulary_cache(self) -> None:
        self._vocab_sorted = None

    def idf(self, term: str) -> float:
        """Inverse document frequency with smoothing."""
        N = self.num_documents
        if N == 0:
            return 0.0
        df = self.doc_freq.get(term, 0)
        return math.log((N + 1) / (df + 1)) + 1.0

    def tf_idf_weight(self, term: str, tf: int) -> float:
        """Classic lnc-style local weight: ``(1 + log(tf)) * idf`` for ``tf > 0``."""
        if tf <= 0:
            return 0.0
        return (1.0 + math.log(tf)) * self.idf(term)

    def add_document(self, url: str, tokens_with_positions: Sequence[tuple[str, int]]) -> None:
        """
        Add a document described by ordered ``(token, position)`` pairs.

        Tokens must already be normalised (typically lower-cased).
        """
        doc_id = len(self.documents)
        if not tokens_with_positions:
            self.documents.append(IndexedDocument(doc_id=doc_id, url=url, token_count=0))
            return

        self.documents.append(
            IndexedDocument(doc_id=doc_id, url=url, token_count=len(tokens_with_positions)),
        )

        counts: dict[str, int] = {}
        posmap: dict[str, list[int]] = {}
        for tok, pos in tokens_with_positions:
            counts[tok] = counts.get(tok, 0) + 1
            posmap.setdefault(tok, []).append(pos)

        for term, tf in counts.items():
            self.postings.setdefault(term, {})
            self.postings[term][str(doc_id)] = {"tf": tf, "positions": posmap[term]}
            self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self.invalidate_vocabulary_cache()

    def to_serialisable(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "documents": [asdict(d) for d in self.documents],
            "postings": self.postings,
            "doc_freq": self.doc_freq,
        }

    @staticmethod
    def from_serialisable(payload: Mapping[str, Any]) -> InvertedIndex:
        if payload.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported index schema: {payload.get('schema')!r}")
        idx = InvertedIndex()
        for row in payload["documents"]:
            idx.documents.append(
                IndexedDocument(
                    doc_id=int(row["doc_id"]),
                    url=str(row["url"]),
                    token_count=int(row["token_count"]),
                ),
            )
        idx.postings = {k: dict(v) for k, v in payload["postings"].items()}
        idx.doc_freq = dict(payload["doc_freq"])
        idx.invalidate_vocabulary_cache()
        return idx


def tokenize(text: str) -> list[tuple[str, int]]:
    """
    Case-fold and yield ``(token, position)`` for each word token.

    Punctuation-only shards are ignored; hyphenated words become separate
    tokens (e.g. ``well-known`` → ``well``, ``known``).

    :param text: Raw UTF-8 text (quotes, author names, tags, etc.).
    """
    if not text:
        return []
    lowered = text.casefold()
    out: list[tuple[str, int]] = []
    pos = 0
    for m in _TOKEN_RE.finditer(lowered):
        tok = m.group(0)
        if not tok:
            continue
        out.append((tok, pos))
        pos += 1
    return out


def build_index_from_text_documents(url_to_text: Iterable[tuple[str, str]]) -> InvertedIndex:
    """Convenience builder from ``(url, plain_text)`` streams."""
    idx = InvertedIndex()
    for url, text in url_to_text:
        idx.add_document(url, tokenize(text))
    return idx


def save_index(index: InvertedIndex, path: str | Path, *, compress: bool = True) -> None:
    """
    Persist ``index`` to ``path`` as JSON (optionally gzip-compressed).

    Compression keeps Minerva submissions small while remaining a single file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(index.to_serialisable(), ensure_ascii=False, indent=2)
    if compress:
        path = path.with_suffix(path.suffix + ".gz") if not str(path).endswith(".gz") else path
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(data)
    else:
        with path.open("w", encoding="utf-8") as fh:
            fh.write(data)


def load_index(path: str | Path) -> InvertedIndex:
    """Load an index saved via :func:`save_index`."""
    path = Path(path)
    opener: Any
    if str(path).endswith(".gz"):
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: path.open("r", encoding="utf-8")
    with opener() as fh:
        payload = json.load(fh)
    return InvertedIndex.from_serialisable(payload)


def iter_index_file_candidates(primary: str | Path) -> Iterator[Path]:
    """Yield likely on-disk filenames (plain + ``.gz`` variants)."""
    p = Path(primary)
    yield p
    if not str(p).endswith(".gz"):
        yield Path(str(p) + ".gz")
