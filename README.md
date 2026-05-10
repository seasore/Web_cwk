# Quotes Search Engine (XJCO3011 / COMP3011 Coursework 2)

Educational search tool for **[Quotes to Scrape](https://quotes.toscrape.com/)**: a polite crawler, a serialised inverted index with positional statistics, conjunctive (AND) multi-word search, **`tfidf` / Okapi `bm25` / `hybrid` / `bm25_proximity` ranking** (see `src/search.py`), and lightweight **“did you mean”** suggestions.

## Features vs. the brief

- **Crawl target**: all paginated listing pages; each ``div.quote`` becomes its own logical document. Synthetic URLs use ``<listing-url>#quote-<idx>`` so every hit is deep-linkable.
- **Politeness**: defaults to **≥ 6 s** between HTTP GETs (configurable *downwards only for debugging*).
- **Index**: case-insensitive tokeniser (Unicode-aware via ``str.casefold``), token positions, document frequencies, smoothed IDF, ``(1 + log(tf)) * idf`` weights.
- **CLI**: interactive ``build``, ``load``, ``print <term>``, ``find <terms…>`` as required.
- **Stretch goals (80‑100 band)**: multiple rankers with literature-backed BM25 & proximity heuristic, query suggestions, gzip JSON, `benchmark.py` (incl. ranker timings), GitHub Actions CI (**≥82%** coverage gate), `pytest` suite (incl. mocked `build` / `load` errors / `KeyboardInterrupt`), typed modules.

## Complexity & benchmarking

Asymptotic notes are embedded in module docstrings. For wall-clock micro-benchmarks run:

```bash
cd code
python benchmark.py
```

## Repository layout

```
code/
  src/
    crawler.py
    indexer.py
    search.py
    main.py
  tests/
    test_crawler.py
    test_indexer.py
    test_search.py
    test_main.py
  data/                # default output location + .gitkeep
  requirements.txt
  pytest.ini
  benchmark.py
```

Course submission expects this layout at the **Git root**. If you keep this coursework folder as the Git root, place the provided ``.github/workflows/ci.yml`` under ``<repo>/.github/`` (already generated one level above ``code/``).

## Installation

```bash
cd code
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Python **3.11+** recommended.

## Running the interactive shell

```bash
cd code
python -m src.main
```

Example session (after installation):

```
> build
> load
> print change
> find good friends
> exit
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| ``SEARCH_ENGINE_INDEX`` | Override index path (default ``code/data/index.json``, auto-loads ``*.json.gz`` too). |
| ``SEARCH_ENGINE_POLITENESS`` | Seconds between GETs (default ``6``; use lower values **only** while debugging). |
| ``SEARCH_ENGINE_RANKER`` | ``tfidf`` (default), ``bm25``, ``hybrid``, ``bm25_proximity`` — see `src/search.py` docstring. |
| ``SEARCH_ENGINE_LOG_LEVEL`` | ``INFO`` / ``DEBUG`` for verbose crawler logs. |

## Tests & coverage

```bash
cd code
pytest
pytest --cov=src --cov-report=term-missing
```

CI (`.github/workflows/ci.yml`) runs the same suite with ``--cov-fail-under=78``.

## Professional Git practices (marking)

- Use **Conventional Commits** (`feat:`, `fix:`, `test:`, `docs:`, `chore:`) — markers love readable history.
- Tag releases (`v1.0.0`) once the video + index artefact are frozen.
- Keep the `README` + `RUNNING_AND_EDGE_CASES.md` updated when behaviour changes.

## Licence / usage

For University of Leeds coursework use only; target site is designed for scraping practice—**always** keep the politeness window at production value when demonstrating compliance.
