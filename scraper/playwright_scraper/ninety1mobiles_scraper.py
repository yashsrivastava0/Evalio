"""91mobiles (91mobiles.com) structured catalog scraper — Tier-1.

Fallback Tier-1 source when Smartprix cannot find the product.
Provides clean, structured spec tables across phones and laptops.

Two-step flow:
  1. search_product(name, category) → canonical 91mobiles product URL
  2. scrape(url) → RawProductData with structured specs

Purpose (Tier-1):
  - Extract clean product specifications
  - Obtain canonical product metadata
  - Populate products and variants tables

Category is ALWAYS passed from the caller — never inferred.
"""

from __future__ import annotations

import logging
import urllib.parse
from datetime import datetime, timezone

from playwright.async_api import async_playwright

from scraper.playwright_scraper.base_scraper import (
    BaseScraper,
    PAGE_TIMEOUT_MS,
    RawProductData,
)

logger = logging.getLogger(__name__)

NINETYONE_BASE = "https://www.91mobiles.com"
NINETYONE_SEARCH = "https://www.91mobiles.com/search/?q={q}"


class NinetyOneMobilesScraper(BaseScraper):
    """Tier-1 scraper for 91mobiles.com structured catalog."""

    # ── Search result selectors ───────────────────────────────────────────────

    _SEARCH_RESULT_SELECTORS = [
        "a.search-results__link",
        "div.result-item a",
        "ul.result-list li a",
        "div.product-item a",
        "a.prod-name-link",
        "div.auto-suggest a",
        "li.search-result a",
        "div.search-result-item a",
        "a[href*='/price/']",
        "a[href*='/laptops/']",
    ]

    # ── Product detail selectors ──────────────────────────────────────────────

    _TITLE_SELECTORS = [
        "h1.product-name",
        "h1.prod-name",
        "div.product-header h1",
        "div.prod-details h1",
        "div.productName h1",
        "h1.title",
        "h1",
    ]

    _PRICE_SELECTORS = [
        "div.price-box .price",
        "span.latest-price",
        "div.buy-box .price",
        "span.price-val",
        "div.prod-price",
        "span.price",
        "div.best-price .price",
    ]

    _SPEC_TABLE_SELECTORS = [
        "ul.specs-sheet li",
        "table.specs-table tr",
        "div.spec-row",
        "ul.specs-list li",
        "div.specs-item",
        "table.specification-table tr",
        "div.specsWrap li",
        "table.specsTable tr",
        "ul.specsList li",
    ]

    _BRAND_SELECTORS = [
        "a.brand-name",
        "span.brand",
        "div.brand a",
        "span.manufacturer",
        "a.manufacturer",
    ]

    def get_domain(self) -> str:
        return "91mobiles.com"

    async def search_product(self, product_name: str, category: str) -> str | None:
        """Search 91mobiles for a product → return detail page URL or None.

        Args:
            product_name: Human-readable product name (e.g., "OnePlus 12").
            category: "smartphone" | "laptop".

        Returns:
            Full URL of the first matching product detail page, or None if not found.
        """
        query = urllib.parse.quote_plus(product_name)
        search_url = NINETYONE_SEARCH.format(q=query)

        async with async_playwright() as playwright:
            browser = await self._launch_browser(playwright)
            try:
                page = await self._new_stealth_page(browser)
                page.set_default_timeout(PAGE_TIMEOUT_MS)

                logger.info(
                    "91mobiles_search_start",
                    extra={"query": product_name, "url": search_url},
                )
                await page.goto(search_url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
                await self._slow_scroll(page)

                if await self._detect_captcha(page):
                    logger.warning("91mobiles_captcha_on_search", extra={"url": search_url})
                    return None

                for sel in self._SEARCH_RESULT_SELECTORS:
                    try:
                        el = await page.query_selector(sel)
                        if el:
                            href = await el.get_attribute("href")
                            if href and href not in ("/", ""):
                                if href.startswith("/"):
                                    href = NINETYONE_BASE + href
                                if href.startswith("http"):
                                    logger.info(
                                        "91mobiles_product_url_found",
                                        extra={"url": href},
                                    )
                                    return href
                    except Exception:
                        continue

                logger.warning("91mobiles_no_result", extra={"query": product_name})
                return None

            except Exception as exc:
                logger.error("91mobiles_search_error", extra={"error": str(exc)})
                return None
            finally:
                await browser.close()

    async def scrape(self, url: str) -> RawProductData:
        """Scrape a 91mobiles product detail page.

        Args:
            url: Full 91mobiles product URL.

        Returns:
            RawProductData with extracted specs. error field set on failure.
        """
        assert isinstance(url, str) and url.strip(), "url must be a non-empty string"
        scraped_at = datetime.now(timezone.utc).isoformat()

        async with async_playwright() as playwright:
            browser = await self._launch_browser(playwright)
            try:
                page = await self._new_stealth_page(browser)
                page.set_default_timeout(PAGE_TIMEOUT_MS)

                logger.info("91mobiles_scrape_start", extra={"url": url})
                await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
                await self._slow_scroll(page)

                if await self._detect_captcha(page):
                    logger.warning("91mobiles_captcha_on_detail", extra={"url": url})
                    html = await page.content()
                    raw = RawProductData(
                        url=url, vendor=self.get_domain(), category=self.category,
                        scraped_at=scraped_at, error="captcha_detected",
                    )
                    self._save_error_snapshot("91m-captcha", html)
                    return raw

                title = await self._extract_text(page, self._TITLE_SELECTORS)
                brand_raw = await self._extract_text(page, self._BRAND_SELECTORS)
                brand = brand_raw.strip() or (title.split()[0] if title else "")
                price_raw = await self._extract_text(page, self._PRICE_SELECTORS)
                price = self._parse_price_inr(price_raw)
                specs = await self._extract_specs(page)
                model_name = self._infer_model_name(title, brand)

                raw = RawProductData(
                    url=url,
                    vendor=self.get_domain(),
                    category=self.category,
                    title=title,
                    brand=brand,
                    model_name=model_name,
                    price=price,
                    currency="INR",
                    availability=True,   # 91mobiles is an info site, not a store
                    specs=specs,
                    scraped_at=scraped_at,
                )

                if not self._validate_extraction(raw):
                    html = await page.content()
                    self._save_error_snapshot(
                        f"91m-sparse-{model_name[:20].replace(' ', '-')}", html
                    )
                    raw.error = "sparse_specs"

                logger.info(
                    "91mobiles_scrape_complete",
                    extra={"title": title, "spec_count": len(specs)},
                )
                return raw

            except Exception as exc:
                logger.error("91mobiles_scrape_error", extra={"url": url, "error": str(exc)})
                return RawProductData(
                    url=url, vendor=self.get_domain(), category=self.category,
                    scraped_at=scraped_at, error=str(exc),
                )
            finally:
                await browser.close()

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _extract_text(self, page: object, selectors: list[str]) -> str:
        """Try selectors in order; return first non-empty text found."""
        for sel in selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    text = await el.inner_text()
                    if text and text.strip():
                        return text.strip()
            except Exception:
                continue
        return ""

    async def _extract_specs(self, page: object) -> dict[str, str]:
        """Extract spec key-value pairs from 91mobiles spec sections."""
        specs: dict[str, str] = {}

        for sel in self._SPEC_TABLE_SELECTORS:
            try:
                items = await page.query_selector_all(sel)
                if not items:
                    continue

                for item in items:
                    try:
                        # Strategy 1: table row (th/td)
                        cells = await item.query_selector_all("th, td")
                        if len(cells) >= 2:
                            key = (await cells[0].inner_text()).strip().rstrip(":")
                            val = (await cells[1].inner_text()).strip()
                            if key and val:
                                specs[key] = val
                            continue

                        # Strategy 2: label/value spans
                        spans = await item.query_selector_all(
                            "span, strong, div.spec-val, div.spec-key"
                        )
                        texts = []
                        for s in spans:
                            t = (await s.inner_text()).strip()
                            if t:
                                texts.append(t)
                        if len(texts) >= 2:
                            specs[texts[0].rstrip(":")] = texts[1]
                            continue

                        # Strategy 3: colon-separated plain text
                        raw_text = (await item.inner_text()).strip()
                        if ":" in raw_text:
                            parts = raw_text.split(":", 1)
                            key, val = parts[0].strip(), parts[1].strip()
                            if key and val and len(key) < 80:
                                specs[key] = val

                    except Exception:
                        continue

                if len(specs) >= 3:
                    break

            except Exception:
                continue

        return specs

    @staticmethod
    def _infer_model_name(title: str, brand: str) -> str:
        """Strip brand prefix from title."""
        if not title:
            return ""
        if brand and title.lower().startswith(brand.lower()):
            return title[len(brand):].strip(" -–")
        return title.strip()
