"""
HTTP crawling for quotes.toscrape.com with politeness and resilience.

Complexity (per fetched listing page):
- BeautifulSoup parses the HTML once → O(|HTML|).
- Per-page quote extraction is O(k) for k quote blocks on that page.
- Full site crawl: O(L · |HTML|) for L listing pages; total documents O(total quotes).

Network wait dominates runtime: with politeness ``P`` seconds, a lower bound is
``P * (L - 1)`` between successive requests.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://quotes.toscrape.com"
DEFAULT_POLITENESS_SECONDS = 6.0
REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0


@dataclass(slots=True)
class CrawlDocument:
    """One indexed document: a single quote block from a listing page."""

    url: str
    quote_text: str
    author: str
    tags: list[str]
    source_page: str
    raw_context: str = ""

    def merged_text(self) -> str:
        """Concatenate fields that should participate in search."""
        tags_joined = " ".join(self.tags)
        return " ".join(
            p
            for p in (
                self.quote_text,
                self.author,
                tags_joined,
                self.raw_context,
            )
            if p
        )


@dataclass(slots=True)
class CrawlResult:
    """Outcome of a full crawl."""

    documents: list[CrawlDocument] = field(default_factory=list)
    pages_fetched: int = 0
    errors: list[str] = field(default_factory=list)


def _same_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "quotes.toscrape.com"


def _normalise_listing_url(href: str, current: str) -> str | None:
    if not href:
        return None
    full = urljoin(current, href)
    if not _same_host(full):
        return None
    parsed = urlparse(full)
    if parsed.scheme not in ("http", "https"):
        return None
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    canon = f"{parsed.scheme}://{parsed.netloc}{path}"
    if "/tag/" in path or "/author/" in path or path.rstrip("/") == "/login":
        return None
    if not (path == "" or path == "/" or path.startswith("/page/")):
        return None
    return canon if path and path != "/" else f"{parsed.scheme}://{parsed.netloc}/"


def _fetch_html(session: requests.Session, url: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT_S)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            logger.warning("Fetch failed (%s/%s) for %s: %s", attempt, MAX_RETRIES, url, exc)
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_BACKOFF_S * attempt)
    return None


def _extract_next_listing_url(html: str, current_url: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    nxt = soup.select_one("li.next a")
    if not nxt:
        return None
    href = nxt.get("href")
    if not href:
        return None
    return _normalise_listing_url(href, current_url)


def _keywords_from_meta(div: BeautifulSoup) -> list[str]:
    meta = div.select_one('meta.keywords[itemprop="keywords"]')
    if not meta:
        return []
    content = (meta.get("content") or "").strip()
    if not content:
        return []
    return [p.strip() for p in content.split(",") if p.strip()]


def _parse_quote_blocks(html: str, listing_url: str) -> list[CrawlDocument]:
    """Parse all ``div.quote`` units from a *listing* HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    docs: list[CrawlDocument] = []
    listing_url = listing_url.split("#")[0]
    for idx, div in enumerate(soup.select("div.quote")):
        text_el = div.select_one("span.text")
        auth_el = div.select_one("small.author")
        if not text_el or not auth_el:
            continue
        raw = text_el.get_text(" ", strip=True)
        quote_text = raw.strip("“”\"'")
        author = auth_el.get_text(strip=True)
        tag_texts = [a.get_text(strip=True) for a in div.select("a.tag")]
        keywords = _keywords_from_meta(div)
        tag_set: list[str] = []
        seen: set[str] = set()
        for t in tag_texts + keywords:
            low = t.lower()
            if low in seen:
                continue
            seen.add(low)
            tag_set.append(t)

        about = div.select_one('span a[href^="/author/"]')
        raw_ctx = ""
        if about and about.get("href"):
            raw_ctx = urljoin(BASE_URL, about["href"])

        doc_url = f"{listing_url}#quote-{idx}"
        docs.append(
            CrawlDocument(
                url=doc_url,
                quote_text=quote_text,
                author=author,
                tags=tag_set,
                source_page=listing_url,
                raw_context=raw_ctx,
            ),
        )
    return docs


def crawl_quotes_site(
    *,
    politeness_seconds: float = DEFAULT_POLITENESS_SECONDS,
    session: requests.Session | None = None,
    base_url_or_session: Any = None,
) -> CrawlResult:
    """
    Crawl all paginated listing pages on quotes.toscrape.com.

    Each ``div.quote`` becomes its own document; the synthetic URL uses the
    listing URL plus ``#quote-<index>`` so multi-word search can point to an
    atomic quote.

    :param politeness_seconds: Minimum idle time between consecutive GETs.
    :param session: Optional ``requests.Session`` for testing / tuning.
    :param base_url_or_session: Ignored; reserved for test doubles.
    """
    _ = base_url_or_session
    owns_session = session is None
    sess = session or requests.Session()
    sess.headers.setdefault(
        "User-Agent",
        "XJCO3011-Coursework2-EducationalBot/1.0 (+https://quotes.toscrape.com)",
    )

    result = CrawlResult()
    seen_listings: set[str] = set()
    queue: list[str] = [f"{BASE_URL}/"]

    last_request_mono: float | None = None

    def polite_sleep() -> None:
        nonlocal last_request_mono
        if last_request_mono is None:
            return
        elapsed = time.monotonic() - last_request_mono
        wait = politeness_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

    try:
        while queue:
            page_url = queue.pop(0)
            key = page_url.split("#")[0].rstrip("/") or "/"
            if key in seen_listings:
                continue
            seen_listings.add(key)

            polite_sleep()
            html = _fetch_html(sess, page_url)
            last_request_mono = time.monotonic()
            if html is None:
                result.errors.append(f"failed_to_fetch:{page_url}")
                continue

            result.pages_fetched += 1
            for doc in _parse_quote_blocks(html, page_url):
                result.documents.append(doc)

            nxt = _extract_next_listing_url(html, page_url)
            if nxt:
                nk = nxt.rstrip("/") or "/"
                if nk not in seen_listings and nxt not in queue:
                    queue.append(nxt)
    finally:
        if owns_session:
            sess.close()

    return result


def crawl_quotes_site_mockable(
    politeness_seconds: float = DEFAULT_POLITENESS_SECONDS,
    session_factory: Any = requests.Session,
) -> CrawlResult:
    """Helper for tests that inject a ``requests.Session`` subclass."""
    sess = session_factory()
    return crawl_quotes_site(politeness_seconds=politeness_seconds, session=sess)


_QUOTE_FRAG_RE = re.compile(r"#quote-\d+\s*$")

def is_synthetic_quote_url(url: str) -> bool:
    """Return True if ``url`` uses the coursework synthetic fragment scheme."""
    return bool(_QUOTE_FRAG_RE.search(url))
