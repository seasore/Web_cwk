"""
Interactive CLI shell for building, loading, and querying the index.

Environment variables (advanced / CI):
- ``SEARCH_ENGINE_INDEX``: filesystem path for the serialised index (default ``data/index.json``).
- ``SEARCH_ENGINE_POLITENESS``: crawl delay in seconds (default ``6``; keep ≥6 for coursework compliance).
- ``SEARCH_ENGINE_RANKER``: ``tfidf`` (default), ``bm25``, ``hybrid``, or ``bm25_proximity`` — see ``src.search``.
"""
from __future__ import annotations

import cmd
import json
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

from . import crawler
from .indexer import InvertedIndex, build_index_from_text_documents, iter_index_file_candidates, load_index, save_index
from . import search as search_mod

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "index.json"


def _resolve_index_path() -> Path:
    raw = os.environ.get("SEARCH_ENGINE_INDEX", str(DEFAULT_INDEX_PATH))
    return Path(raw).expanduser()


def _politeness_s() -> float:
    raw = os.environ.get("SEARCH_ENGINE_POLITENESS", "6")
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("Invalid SEARCH_ENGINE_POLITENESS=%r — falling back to 6s", raw)
        return 6.0


def _locate_existing_index(primary: Path) -> Path:
    for cand in iter_index_file_candidates(primary):
        if cand.is_file():
            return cand
    return primary


def _rank_mode() -> search_mod.RankMode:
    return search_mod.normalise_rank_mode(os.environ.get("SEARCH_ENGINE_RANKER"))


class SearchShell(cmd.Cmd):
    """``cmd.Cmd`` REPL implementing *build/load/print/find*."""

    intro = textwrap.dedent(
        """
        Quotes search engine — type help or ? to list commands.
        Commands: build | load | print <term> | find <terms...>
        Multi-word search uses implicit AND — type: find we go (not the word AND).
        Optional: SEARCH_ENGINE_RANKER=tfidf|bm25|hybrid|bm25_proximity.
        """,
    ).strip()
    prompt = "> "

    def __init__(self, completekey: str = "tab", stdin: Any = None, stdout: Any = None) -> None:
        super().__init__(completekey=completekey, stdin=stdin, stdout=stdout)
        self.index: InvertedIndex | None = None
        self.index_path = _resolve_index_path()

    def do_EOF(self, _arg: str) -> bool:  # Ctrl+D
        self.stdout.write("\n")
        return True

    def do_exit(self, _arg: str) -> bool:
        """Quit the shell."""
        return True

    def do_quit(self, arg: str) -> bool:
        return self.do_exit(arg)

    def do_build(self, _arg: str) -> None:
        """
        Crawl https://quotes.toscrape.com, build the inverted index, and save it.

        The on-disk file defaults to ``data/index.json.gz`` (gzip-wrapped JSON).
        """
        polite = _politeness_s()
        if polite < crawler.DEFAULT_POLITENESS_SECONDS:
            self.stdout.write(
                f"[warn] politeness={polite}s is below the coursework {crawler.DEFAULT_POLITENESS_SECONDS:.0f}s "
                "requirement — only use this for local debugging.\n",
            )

        self.stdout.write("[build] crawling (network + politeness window) …\n")
        try:
            result = crawler.crawl_quotes_site(politeness_seconds=polite)
        except Exception as exc:  # noqa: BLE001 — crawl stack: HTTP, HTML, I/O, etc.
            self.stdout.write(f"[err] crawl failed ({type(exc).__name__}): {exc}\n")
            return

        if result.errors:
            self.stdout.write(f"[warn] crawl finished with {len(result.errors)} failed GET attempts.\n")

        if not result.documents:
            self.stdout.write("[err] crawl produced zero quote blocks — check network or site HTML.\n")
            return

        pairs = [(d.url, d.merged_text()) for d in result.documents]
        idx = build_index_from_text_documents(pairs)
        out = self.index_path
        try:
            save_index(idx, out, compress=True)
        except OSError as exc:
            self.stdout.write(f"[err] cannot write index file ({type(exc).__name__}): {exc}\n")
            return
        self.index = idx
        resolved = _locate_existing_index(out)
        self.stdout.write(
            f"[build] indexed {idx.num_documents} quote blocks from {result.pages_fetched} listing pages → {resolved}\n",
        )

    def do_load(self, _arg: str) -> None:
        """Load a previously built index (accepts either ``.json`` or ``.json.gz``)."""
        pathish = self.index_path
        chosen = _locate_existing_index(pathish)
        if not chosen.is_file():
            self.stdout.write(
                f"[err] no index file at {pathish} (also tried .gz variant). Run build first.\n",
            )
            return
        try:
            self.index = load_index(chosen)
        except json.JSONDecodeError as exc:
            self.stdout.write(
                f"[err] index file is not valid JSON (line {exc.lineno}, col {exc.colno}): {exc.msg}\n",
            )
            return
        except UnicodeDecodeError as exc:
            self.stdout.write(f"[err] index file is not readable UTF-8 text ({type(exc).__name__}).\n")
            return
        except ValueError as exc:
            self.stdout.write(f"[err] index schema or structure invalid: {exc}\n")
            return
        except OSError as exc:
            self.stdout.write(f"[err] cannot read index file ({type(exc).__name__}): {exc}\n")
            return
        self.stdout.write(f"[load] ready — {self.index.num_documents} documents from {chosen}\n")

    def do_print(self, arg: str) -> None:
        """Pretty-print inverted list entries for a single term: ``print indifference``."""
        word = arg.strip()
        if not self._require_index():
            return
        assert self.index is not None
        if not word:
            self.stdout.write("[err] print requires a term, e.g. `print nonsense`.\n")
            return
        tokens = search_mod.parse_query_tokens([word])
        if not tokens:
            self.stdout.write(f"[err] `{word}` contains no searchable tokens after normalisation.\n")
            return
        if len(tokens) > 1:
            self.stdout.write(
                f"[info] using first normalised token {tokens[0]!r} (query contained multiple tokens).\n",
            )
        term = tokens[0]
        payload = search_mod.print_term_postings(self.index, term)
        self.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def do_find(self, arg: str) -> None:
        """
        Conjunctive search with implicit AND (e.g. ``find good friends``).

        Ranking is controlled by ``SEARCH_ENGINE_RANKER``
        (``tfidf``, ``bm25``, ``hybrid``, ``bm25_proximity``).
        """
        if not self._require_index():
            return
        assert self.index is not None

        parts = arg.split()
        if not parts:
            self.stdout.write("[err] find needs at least one term, e.g. `find indifference`.\n")
            return

        terms = search_mod.parse_query_tokens(parts)
        if not terms:
            self.stdout.write("[err] query contains no alphanumeric tokens after normalisation.\n")
            self._hint_suggestions_for_raw(parts)
            return

        missing = [t for t in dict.fromkeys(terms) if t not in self.index.postings]
        if missing:
            self.stdout.write(f"[info] unknown term(s): {missing!r}\n")
            sug_map = search_mod.multi_suggestions(self.index, missing)
            if sug_map:
                self.stdout.write("[info] suggestions: " + json.dumps(sug_map, ensure_ascii=False) + "\n")
            self.stdout.write("[err] AND-query requires every term to exist in the index.\n")
            return

        candidates = search_mod.conjunctive_doc_ids(self.index, terms)
        if not candidates:
            self.stdout.write(
                "[info] no single quote contains all terms (AND is within one indexed document).\n",
            )
            self.stdout.write("[]\n")
            return

        mode = _rank_mode()
        if mode != "tfidf":
            self.stdout.write(f"[info] ranker={mode} (SEARCH_ENGINE_RANKER).\n")

        ranked = search_mod.rank_with_mode(self.index, candidates, terms, mode)
        lines = [h.url for h in ranked]
        self.stdout.write(json.dumps(lines, ensure_ascii=False, indent=2) + "\n")

    def _hint_suggestions_for_raw(self, parts: list[str]) -> None:
        if not self.index:
            return
        blob = " ".join(parts)
        tokens = [blob] if blob else []
        if not tokens:
            return
        sug_map = search_mod.multi_suggestions(self.index, tokens)
        if sug_map:
            self.stdout.write("[info] suggestions: " + json.dumps(sug_map, ensure_ascii=False) + "\n")

    def _require_index(self) -> bool:
        if self.index is None:
            self.stdout.write("[err] index not loaded — run load (or build) first.\n")
            return False
        return True

    def default(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        self.stdout.write(f"[err] unknown command: {stripped!r}. Type help.\n")


def main(argv: list[str] | None = None) -> int:
    """Entry point: interactive shell only (argv reserved for future batch flags)."""
    _ = argv
    logging.basicConfig(level=os.environ.get("SEARCH_ENGINE_LOG_LEVEL", "WARNING"))
    try:
        SearchShell().cmdloop()
    except KeyboardInterrupt:
        sys.stdout.write("\n[exit] Ctrl+C — use exit or quit for a clean quit.\n")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
