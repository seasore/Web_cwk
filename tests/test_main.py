from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

from src.crawler import CrawlDocument, CrawlResult
from src.indexer import build_index_from_text_documents, save_index
from src.main import SearchShell, main


def test_shell_find_requires_loaded_index() -> None:
    stdin = StringIO("find hello\nexit\n")
    stdout = StringIO()
    shell = SearchShell(stdin=stdin, stdout=stdout)
    shell.use_rawinput = False
    shell.cmdloop()
    out = stdout.getvalue()
    assert "index not loaded" in out.lower()


def test_shell_load_print_find(tmp_path: Path, monkeypatch) -> None:
    idx = build_index_from_text_documents(
        [
            ("https://example.test/#quote-0", "good friends forever"),
            ("https://example.test/#quote-1", "only good"),
        ],
    )
    path = tmp_path / "idx.json"
    save_index(idx, path, compress=False)

    monkeypatch.setenv("SEARCH_ENGINE_INDEX", str(path))

    stdin = StringIO(
        "load\n"
        "print friends\n"
        "find good friends\n"
        "find nosuchterm\n"
        "print\n"
        "find\n"
        "exit\n",
    )
    stdout = StringIO()
    shell = SearchShell(stdin=stdin, stdout=stdout)
    shell.use_rawinput = False
    shell.cmdloop()
    out = stdout.getvalue()
    assert "[load]" in out
    assert "friends" in out
    assert "example.test" in out
    assert "unknown term" in out.lower() or "nosuchterm" in out
    assert "[err] print requires" in out
    assert "[err] find needs" in out


def test_find_empty_and_explains_same_document_rule(tmp_path: Path, monkeypatch) -> None:
    """Multi-term AND requires every token in the *same* quote block."""
    idx = build_index_from_text_documents(
        [
            ("https://example.test/#quote-0", "only alpha here"),
            ("https://example.test/#quote-1", "only beta here"),
        ],
    )
    path = tmp_path / "idx.json"
    save_index(idx, path, compress=False)
    monkeypatch.setenv("SEARCH_ENGINE_INDEX", str(path))

    stdin = StringIO("load\nfind alpha beta\nexit\n")
    stdout = StringIO()
    shell = SearchShell(stdin=stdin, stdout=stdout)
    shell.use_rawinput = False
    shell.cmdloop()
    out = stdout.getvalue()
    assert "no single quote contains all terms" in out
    assert "[]" in out


def test_unknown_command_reports_error() -> None:
    stdin = StringIO("typo_command foo\nexit\n")
    stdout = StringIO()
    shell = SearchShell(stdin=stdin, stdout=stdout)
    shell.use_rawinput = False
    shell.cmdloop()
    assert "unknown command" in stdout.getvalue().lower()


def test_main_handles_keyboardinterrupt(monkeypatch) -> None:
    class Boom(SearchShell):
        def cmdloop(self, intro=None) -> None:  # type: ignore[override]
            raise KeyboardInterrupt

    monkeypatch.setattr("src.main.SearchShell", Boom)
    assert main() == 130


def test_build_saves_mocked_crawl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_ENGINE_INDEX", str(tmp_path / "idx.json"))
    docs = [
        CrawlDocument(
            url="https://quotes.toscrape.com/#quote-0",
            quote_text="hello world",
            author="A",
            tags=["t"],
            source_page="https://quotes.toscrape.com/",
        ),
    ]
    fake = CrawlResult(documents=docs, pages_fetched=1, errors=[])

    with patch("src.main.crawler.crawl_quotes_site", return_value=fake):
        stdin = StringIO("build\nexit\n")
        stdout = StringIO()
        shell = SearchShell(stdin=stdin, stdout=stdout)
        shell.use_rawinput = False
        shell.cmdloop()

    out = stdout.getvalue()
    assert "[build] indexed 1 quote blocks" in out
    gz = tmp_path / "idx.json.gz"
    assert gz.is_file()


def test_build_zero_documents_aborts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_ENGINE_INDEX", str(tmp_path / "idx.json"))
    fake = CrawlResult(documents=[], pages_fetched=0, errors=["failed_to_fetch:x"])
    with patch("src.main.crawler.crawl_quotes_site", return_value=fake):
        stdin = StringIO("build\nexit\n")
        stdout = StringIO()
        shell = SearchShell(stdin=stdin, stdout=stdout)
        shell.use_rawinput = False
        shell.cmdloop()
    assert "zero quote blocks" in stdout.getvalue().lower()
    assert not (tmp_path / "idx.json.gz").is_file()


def test_build_save_oserror(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_ENGINE_INDEX", str(tmp_path / "idx.json"))
    doc = CrawlDocument(
        url="u",
        quote_text="x",
        author="y",
        tags=[],
        source_page="p",
    )
    fake = CrawlResult(documents=[doc], pages_fetched=1, errors=[])

    def boom(*_a, **_kw) -> None:
        raise OSError("disk full")

    with (
        patch("src.main.crawler.crawl_quotes_site", return_value=fake),
        patch("src.main.save_index", side_effect=boom),
    ):
        stdin = StringIO("build\nexit\n")
        stdout = StringIO()
        shell = SearchShell(stdin=stdin, stdout=stdout)
        shell.use_rawinput = False
        shell.cmdloop()
    assert "cannot write index" in stdout.getvalue().lower()


def test_build_crawl_exception(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_ENGINE_INDEX", str(tmp_path / "idx.json"))
    with patch("src.main.crawler.crawl_quotes_site", side_effect=RuntimeError("boom")):
        stdin = StringIO("build\nexit\n")
        stdout = StringIO()
        shell = SearchShell(stdin=stdin, stdout=stdout)
        shell.use_rawinput = False
        shell.cmdloop()
    assert "crawl failed" in stdout.getvalue().lower()


def test_load_invalid_json(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("SEARCH_ENGINE_INDEX", str(p))
    stdin = StringIO("load\nexit\n")
    stdout = StringIO()
    shell = SearchShell(stdin=stdin, stdout=stdout)
    shell.use_rawinput = False
    shell.cmdloop()
    assert "not valid json" in stdout.getvalue().lower()


def test_find_shows_ranker_when_bm25(tmp_path: Path, monkeypatch) -> None:
    idx = build_index_from_text_documents(
        [("https://example.test/#quote-0", "alpha beta gamma beta")],
    )
    path = tmp_path / "idx.json"
    save_index(idx, path, compress=False)
    monkeypatch.setenv("SEARCH_ENGINE_INDEX", str(path))
    monkeypatch.setenv("SEARCH_ENGINE_RANKER", "bm25")

    stdin = StringIO("load\nfind alpha beta\nexit\n")
    stdout = StringIO()
    shell = SearchShell(stdin=stdin, stdout=stdout)
    shell.use_rawinput = False
    shell.cmdloop()
    out = stdout.getvalue()
    assert "ranker=bm25" in out
    assert "example.test" in out


def test_load_wrong_schema(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "bad.json"
    p.write_text(
        '{"schema": "wrong", "documents": [], "postings": {}, "doc_freq": {}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_ENGINE_INDEX", str(p))
    stdin = StringIO("load\nexit\n")
    stdout = StringIO()
    shell = SearchShell(stdin=stdin, stdout=stdout)
    shell.use_rawinput = False
    shell.cmdloop()
    assert "schema" in stdout.getvalue().lower()
