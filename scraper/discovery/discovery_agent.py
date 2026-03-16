"""Tier-2 Internet Discovery Agent.

Activated when Tier-1 (Smartprix, 91mobiles) cannot retrieve sufficient
product data. Searches the open internet within a strict allowed-domains
whitelist, scrapes candidates, and returns the most complete RawProductData.

Allowed domains (whitelist):
    gsmarena.com, notebookcheck.net, amazon.in, flipkart.com,
    and major brand official sites.

Flow:
    1. Build search query from product name + category
    2. Search DuckDuckGo HTML (no API key required)
    3. Filter ALL results to ALLOWED_DOMAINS — off-whitelist URLs are dropped
    4. Scrape up to MAX_DISCOVERY_PAGES candidates using the best scraper
    5. Return the RawProductData with the most complete specs

Normalization:
    Raw data passes through the same pipeline as Tier-1:
      raw HTML → structured extraction → Pydantic validation → normalized JSON
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

from scraper.playwright_scraper.base_scraper import (
    BaseScraper,
    PAGE_TIMEOUT_MS,
    RawProductData,
)

logger = logging.getLogger(__name__)

# ── Allowed domains whitelist ─────────────────────────────────────────────────
# CRITICAL: Only URLs matching these domains are scraped.
# This prevents Tier-2 from crawling irrelevant or harmful sites.

ALLOWED_DOMAINS: frozenset[str] = frozenset({
    # Structured comparison / review sites
    "gsmarena.com",
    "notebookcheck.net",
    # Retail sources (price data)
    "amazon.in",
    "flipkart.com",
    # Brand official sites
    "apple.com",
    "samsung.com",
    "oneplus.com",
    "mi.com",
    "xiaomi.com",
    "realme.com",
    "motorola.com",
    "nokia.com",
    "oppo.com",
    "vivo.com",
    "asus.com",
    "dell.com",
    "lenovo.com",
    "hp.com",
    "acer.com",
    "msi.com",
    "lg.com",
    "sony.com",
    "google.com",
    "microsoft.com",
})

MAX_DISCOVERY_PAGES: int = 3
SEARCH_TIMEOUT_SECONDS: int = 10
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"


# ── Domain utilities ──────────────────────────────────────────────────────────


def _extract_domain(url: str) -> str:
    """Extract root domain from a URL (strips 'www.' prefix).

    Examples:
        "https://www.amazon.in/dp/B09..." → "amazon.in"
        "https://gsmarena.com/..."        → "gsmarena.com"
    """
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname
    except Exception:
        return ""


def _is_allowed(url: str, allowed_domains: frozenset[str]) -> bool:
    """Return True if the URL's domain is in the allowed whitelist."""
    domain = _extract_domain(url)
    if not domain:
        return False
    for allowed in allowed_domains:
        if domain == allowed or domain.endswith("." + allowed):
            return True
    return False


# ── Discovery Agent ───────────────────────────────────────────────────────────


class DiscoveryAgent:
    """Tier-2 open-internet discovery agent with strict domain allowlisting.

    Searches for product pages within ALLOWED_DOMAINS only.
    Off-whitelist URLs are silently dropped — never scraped.
    """

    def __init__(
        self,
        category: str,
        allowed_domains: frozenset[str] = ALLOWED_DOMAINS,
        max_pages: int = MAX_DISCOVERY_PAGES,
    ) -> None:
        self.category = category
        self.allowed_domains = allowed_domains
        self.max_pages = max_pages

    async def discover(self, product_name: str) -> RawProductData | None:
        """Full discovery flow: search → filter → scrape → return best result.

        Args:
            product_name: Human-readable product name (e.g., "Samsung Galaxy S24").

        Returns:
            RawProductData with the most complete specs found, or None if all fail.
        """
        logger.info(
            "tier2_discovery_start",
            extra={"product": product_name, "category": self.category},
        )

        query = self._build_search_query(product_name)
        candidate_urls = await self._search_duckduckgo(query)

        if not candidate_urls:
            logger.warning("tier2_no_candidates", extra={"query": query})
            return None

        best: RawProductData | None = None
        best_score = -1

        for url in candidate_urls[: self.max_pages]:
            try:
                raw = await self._scrape_candidate(url)
                if raw is None or raw.error:
                    continue

                score = self._completeness_score(raw)
                if score > best_score:
                    best_score = score
                    best = raw

                if score >= 5:
                    break  # sufficient data found

            except Exception as exc:
                logger.warning(
                    "tier2_candidate_failed",
                    extra={"url": url, "error": str(exc)},
                )
                continue

        if best:
            logger.info(
                "tier2_discovery_complete",
                extra={
                    "product": product_name,
                    "best_url": best.url,
                    "spec_count": len(best.specs),
                    "completeness_score": best_score,
                },
            )
        else:
            logger.warning(
                "tier2_all_candidates_failed",
                extra={"product": product_name},
            )

        return best

    def _build_search_query(self, product_name: str) -> str:
        """Build a targeted search query for product spec pages."""
        if self.category == "laptop":
            return f"{product_name} laptop specifications review"
        return f"{product_name} smartphone specifications"

    async def _search_duckduckgo(self, query: str) -> list[str]:
        """Query DuckDuckGo HTML and return allowed URLs from results."""
        try:
            async with httpx.AsyncClient(
                timeout=SEARCH_TIMEOUT_SECONDS,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            ) as client:
                response = await client.post(
                    DUCKDUCKGO_HTML_URL,
                    data={"q": query, "b": "", "kl": "in-en"},
                )
                response.raise_for_status()
                urls = self._parse_ddg_results(response.text)
                logger.info(
                    "tier2_ddg_search_complete",
                    extra={"query": query, "allowed_urls_found": len(urls)},
                )
                return urls

        except Exception as exc:
            logger.error("tier2_duckduckgo_failed", extra={"error": str(exc)})
            return []

    def _parse_ddg_results(self, html: str) -> list[str]:
        """Parse DuckDuckGo HTML response, extracting only allowed-domain URLs."""
        soup = BeautifulSoup(html, "lxml")
        urls: list[str] = []

        # DuckDuckGo HTML result links use class="result__a" or "result__url"
        for tag in soup.select("a.result__a, a.result__url, h2.result__title a"):
            href = tag.get("href", "")
            if not href:
                continue

            # DDG sometimes wraps URLs in a redirect; extract the actual URL
            if "uddg=" in href:
                m = re.search(r"uddg=([^&]+)", href)
                if m:
                    href = urllib.parse.unquote(m.group(1))

            # Only include URLs within allowed domains
            if href.startswith("http") and _is_allowed(href, self.allowed_domains):
                if href not in urls:
                    urls.append(href)

        return urls

    async def _scrape_candidate(self, url: str) -> RawProductData | None:
        """Scrape a candidate URL with the most appropriate scraper.

        Routes to dedicated scrapers for known domains;
        falls back to GenericProductScraper for others.
        """
        domain = _extract_domain(url)

        if domain in ("amazon.in",):
            from scraper.playwright_scraper.amazon_scraper import AmazonScraper
            scraper: BaseScraper = AmazonScraper(category=self.category)

        elif domain in ("flipkart.com",):
            from scraper.playwright_scraper.flipkart_scraper import FlipkartScraper
            scraper = FlipkartScraper(category=self.category)

        else:
            scraper = GenericProductScraper(category=self.category)

        return await scraper._scrape_with_retry(url)

    @staticmethod
    def _completeness_score(raw: RawProductData) -> int:
        """Score completeness of extracted data (higher = more complete).

        Used to select the best result when multiple candidates are scraped.
        Each point counts:
          +1 for title, +1 for brand, +1 for price
          +1 per valid spec field (capped at 10)
        """
        score = 0
        if raw.title:
            score += 1
        if raw.brand:
            score += 1
        if raw.price is not None:
            score += 1
        valid_specs = sum(1 for v in raw.specs.values() if v and str(v).strip())
        score += min(valid_specs, 10)
        return score


# ── Generic Product Scraper ───────────────────────────────────────────────────


class GenericProductScraper(BaseScraper):
    """Generic Playwright scraper for unknown product pages.

    Uses heuristic extraction in this priority order:
      1. Schema.org JSON-LD (most reliable for structured product pages)
      2. Open Graph meta tags (title, description)
      3. Generic spec table detection (th/td, dl/dt/dd, colon-separated li)

    Used by DiscoveryAgent for domains without a dedicated scraper
    (e.g., gsmarena.com, notebookcheck.net, brand sites).
    """

    def get_domain(self) -> str:
        return "generic"

    async def scrape(self, url: str) -> RawProductData:
        """Heuristically extract product data from any product page URL."""
        from playwright.async_api import async_playwright

        assert isinstance(url, str) and url.strip(), "url must be a non-empty string"
        scraped_at = datetime.now(timezone.utc).isoformat()
        vendor = _extract_domain(url) or "unknown"

        async with async_playwright() as playwright:
            browser = await self._launch_browser(playwright)
            try:
                page = await self._new_stealth_page(browser)
                page.set_default_timeout(PAGE_TIMEOUT_MS)

                logger.info("generic_scrape_start", extra={"url": url, "vendor": vendor})
                await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
                await self._slow_scroll(page)

                if await self._detect_captcha(page):
                    logger.warning("generic_captcha_detected", extra={"url": url})
                    return RawProductData(
                        url=url, vendor=vendor, category=self.category,
                        scraped_at=scraped_at, error="captcha_detected",
                    )

                html = await page.content()
                soup = BeautifulSoup(html, "lxml")

                # 1. Schema.org JSON-LD (richest source)
                title, brand, price_raw, specs = self._extract_jsonld(soup)

                # 2. Open Graph / HTML fallback for title
                if not title:
                    og_title = self._extract_og(soup, "og:title")
                    if og_title:
                        title = og_title
                    else:
                        h1 = soup.find("h1")
                        title = h1.get_text(strip=True) if h1 else ""

                # 3. Generic spec tables (supplement JSON-LD specs)
                if len(specs) < 3:
                    specs.update(self._extract_generic_specs(soup))

                price = self._parse_price_inr(price_raw)
                model_name = title.strip()
                if brand and model_name.lower().startswith(brand.lower()):
                    model_name = model_name[len(brand):].strip(" -–")

                raw = RawProductData(
                    url=url,
                    vendor=vendor,
                    category=self.category,
                    title=title.strip(),
                    brand=brand.strip(),
                    model_name=model_name,
                    price=price,
                    currency="INR",
                    availability=True,
                    specs=specs,
                    scraped_at=scraped_at,
                )

                logger.info(
                    "generic_scrape_complete",
                    extra={"url": url, "title": title, "spec_count": len(specs)},
                )
                return raw

            except Exception as exc:
                logger.error("generic_scrape_error", extra={"url": url, "error": str(exc)})
                return RawProductData(
                    url=url, vendor=vendor, category=self.category,
                    scraped_at=scraped_at, error=str(exc),
                )
            finally:
                await browser.close()

    # ── Extraction helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_jsonld(soup: Any) -> tuple[str, str, str, dict[str, str]]:
        """Extract product data from Schema.org JSON-LD embedded in the page.

        Returns:
            (title, brand, price_raw_str, specs_dict)
        """
        title, brand, price_raw = "", "", ""
        specs: dict[str, str] = {}

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")

                # Unwrap @graph arrays
                if isinstance(data, dict) and "@graph" in data:
                    items = data["@graph"]
                elif isinstance(data, list):
                    items = data
                else:
                    items = [data]

                for item in items:
                    if not isinstance(item, dict):
                        continue

                    item_type = item.get("@type", "")
                    # Accept Product or tech-review types
                    if item_type not in ("Product", "MobileApplication", "SoftwareApplication"):
                        continue

                    title = title or str(item.get("name", ""))

                    raw_brand = item.get("brand", {})
                    if isinstance(raw_brand, dict):
                        brand = brand or str(raw_brand.get("name", ""))
                    elif isinstance(raw_brand, str):
                        brand = brand or raw_brand

                    offers = item.get("offers", {})
                    if isinstance(offers, list) and offers:
                        offers = offers[0]
                    if isinstance(offers, dict):
                        price_raw = price_raw or str(offers.get("price", ""))

                    # additionalProperty → spec key/value pairs
                    for prop in item.get("additionalProperty", []):
                        if isinstance(prop, dict):
                            name = str(prop.get("name", "")).strip()
                            value = str(prop.get("value", "")).strip()
                            if name and value:
                                specs[name] = value

            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        return title, brand, price_raw, specs

    @staticmethod
    def _extract_og(soup: Any, property_name: str) -> str:
        """Extract Open Graph meta tag content by property name."""
        tag = soup.find("meta", property=property_name)
        return tag.get("content", "").strip() if tag else ""

    @staticmethod
    def _extract_generic_specs(soup: Any) -> dict[str, str]:
        """Extract specs from generic HTML structures.

        Attempts three strategies in order:
          1. <table> with th/td pairs
          2. <dl> / <dt> / <dd> definition lists
          3. Colon-separated <li> text items
        """
        specs: dict[str, str] = {}

        # Strategy 1: HTML tables with th/td key-value pairs
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).rstrip(":")
                    val = cells[1].get_text(strip=True)
                    if key and val and len(key) < 80 and len(val) < 300:
                        specs[key] = val
            if len(specs) >= 5:
                break

        # Strategy 2: <dl> definition lists
        if len(specs) < 3:
            for dl in soup.find_all("dl"):
                dts = dl.find_all("dt")
                dds = dl.find_all("dd")
                for dt, dd in zip(dts, dds):
                    key = dt.get_text(strip=True).rstrip(":")
                    val = dd.get_text(strip=True)
                    if key and val and len(key) < 80:
                        specs[key] = val

        # Strategy 3: colon-separated <li> text
        if len(specs) < 3:
            for li in soup.find_all("li"):
                text = li.get_text(strip=True)
                if ":" in text and len(text) < 200:
                    parts = text.split(":", 1)
                    key, val = parts[0].strip(), parts[1].strip()
                    if key and val and len(key) < 60 and len(val) < 200:
                        specs[key] = val

        return specs
