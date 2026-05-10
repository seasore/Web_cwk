from __future__ import annotations

from unittest.mock import MagicMock

from src.crawler import crawl_quotes_site


def _minimal_listing(page_href: str | None) -> str:
    nxt = f'<li class="next"><a href="{page_href}"></a></li>' if page_href else ""
    return f"""
    <html><body>
      <div class="quote" itemscope>
        <span class="text" itemprop="text">“Alpha quoted text.”</span>
        <span>by <small class="author" itemprop="author">Author One</small>
          <a href="/author/Author-One">(about)</a>
        </span>
        <div class="tags">
          Tags:
          <meta class="keywords" itemprop="keywords" content="humor,life" />
          <a class="tag" href="/tag/humor/page/1/">humor</a>
        </div>
      </div>
      {nxt}
    </body></html>
    """


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.text = text
        self.status_code = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("boom")


class _SessionFactory:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self) -> MagicMock:
        sess = MagicMock()
        sess.headers = {}

        def get(url: str, timeout: int = 0) -> _FakeResp:
            self.calls.append(url)
            if len(self.calls) == 1:
                return _FakeResp(_minimal_listing("/page/2/"))
            if len(self.calls) == 2:
                return _FakeResp(_minimal_listing(None))
            raise AssertionError("unexpected fetch")

        sess.get.side_effect = get
        sess.close = MagicMock()
        return sess


def test_crawler_respects_single_pass_pagination(monkeypatch) -> None:
    factory = _SessionFactory()
    monkeypatch.setattr("time.sleep", lambda _: None)
    res = crawl_quotes_site(politeness_seconds=0.0, session=factory())
    assert len(res.documents) == 2  # two listing pages, one quote each
    assert res.pages_fetched == 2
    assert "quoted" in res.documents[0].merged_text().lower()
