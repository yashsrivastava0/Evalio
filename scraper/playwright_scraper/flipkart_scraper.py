"""Flipkart (flipkart.com) product page scraper.

Targets single product detail pages only.
Category is passed by the caller — never inferred from DOM.

Selector strategy:
  Layered fallback selectors per field. Flipkart's CSS class names are
  obfuscated and change frequently; multiple alternatives are provided.

Anti-bot:
  - Stealth Playwright via BaseScraper._new_stealth_page
  - Random user agent rotation
  - Slow scroll to trigger lazy content
  - CAPTCHA / bot detection → graceful abort
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from playwright.async_api import async_playwright

from scraper.playwright_scraper.base_scraper import (
    BaseScraper,
    PAGE_TIMEOUT_MS,
    RawProductData,
)

logger = logging.getLogger(__name__)


class FlipkartScraper(BaseScraper):
    """Scraper for flipkart.com product detail pages."""

    # ── Selector layers ───────────────────────────────────────────────────────

    _TITLE_SELECTORS = [
        "span.B_NuCI",
        "._35KyD6",
        "h1.yhB1nd",
        "div.CfWCLH h1",
        "div._6EBuvT span",
    ]

    _PRICE_SELECTORS = [
        "div._30jeq3._16Jk6d",          # current selling price
        "div.Nx9bqj",                    # alternate layout
        "._16Jk6d",
        "div._25b18c ._30jeq3",
        "div.CEmiEU div._30jeq3",
    ]

    _AVAILABILITY_SELECTORS = [
        "div._16FRp0",
        "._8I9MaK._8oPTRq",
        "div.mb-4 ._8VnyE",
        "span._1dVbu9",
    ]

    _SPEC_TABLE_SELECTORS = [
        "div._14cfVK table tr",          # standard spec table
        "div.RmoJUa table tr",           # alternate spec table
        "div._3k-BhJ table tr",
        "section._4gvKMe table tr",
        "div._2418kt table tr",
    ]

    _REVIEW_COUNT_SELECTORS = [
        "span._2_R_DZ label",
        "div._3LWZlK",
        "span.T3g4Yq",
        "span.Y1HWO0",
    ]

    _RATING_SELECTORS = [
        "div._3LWZlK",
        "span._1lRcqv .a-text-bold",
        "div.col.col-5-12 div._3LWZlK",
    ]

    _BRAND_SELECTORS = [
        "span._1lRcqv ._2Kn22P",
        "a.G6XhRU",
        "div.G6XhRU",
    ]

    def get_domain(self) -> str:
        return "flipkart.com"

    async def scrape(self, url: str) -> RawProductData:
        """Scrape a single Flipkart product page.

        Args:
            url: Full Flipkart product URL. Must be a single detail page — no
                 search pages, category listings, or comparison pages.

        Returns:
            RawProductData. error field set on failure.
        """
        assert isinstance(url, str) and url.strip(), "url must be a non-empty string"
        scraped_at = datetime.now(timezone.utc).isoformat()

        async with async_playwright() as playwright:
            browser = await self._launch_browser(playwright)
            try:
                page = await self._new_stealth_page(browser)
                page.set_default_timeout(PAGE_TIMEOUT_MS)

                logger.info("flipkart_scrape_start", extra={"url": url})
                await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)

                # Dismiss login popup if present (common on first visit)
                try:
                    close_btn = await page.query_selector("button._2KpZ6l._2doB4z")
                    if close_btn:
                        await close_btn.click()
                except Exception:
                    pass

                await self._slow_scroll(page)

                # Bot / CAPTCHA guard
                if await self._detect_captcha(page):
                    logger.warning("flipkart_captcha_detected", extra={"url": url})
                    html = await page.content()
                    raw = RawProductData(
                        url=url, vendor=self.get_domain(), category=self.category,
                        scraped_at=scraped_at, error="captcha_detected",
                    )
                    self._save_error_snapshot("flipkart-captcha", html)
                    return raw

                # ── Extract fields ────────────────────────────────────────────

                title = await self._extract_text(page, self._TITLE_SELECTORS)
                brand_raw = await self._extract_text(page, self._BRAND_SELECTORS)
                brand = brand_raw.strip()
                price_raw = await self._extract_text(page, self._PRICE_SELECTORS)
                price = self._parse_price_inr(price_raw)
                availability = await self._extract_availability(page)
                specs = await self._extract_specs(page)
                review_count = await self._extract_review_count(page)
                rating = await self._extract_rating(page)
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
                    availability=availability,
                    specs=specs,
                    review_count=review_count,
                    rating=rating,
                    scraped_at=scraped_at,
                )

                if not self._validate_extraction(raw):
                    html = await page.content()
                    self._save_error_snapshot(
                        f"fk-sparse-{model_name[:20].replace(' ', '-')}", html
                    )
                    raw.error = "sparse_specs"

                logger.info(
                    "flipkart_scrape_complete",
                    extra={"title": title, "price": price, "spec_count": len(specs)},
                )
                return raw

            except Exception as exc:
                logger.error("flipkart_scrape_error", extra={"url": url, "error": str(exc)})
                return RawProductData(
                    url=url, vendor=self.get_domain(), category=self.category,
                    scraped_at=scraped_at, error=str(exc),
                )
            finally:
                await browser.close()

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _extract_text(self, page: object, selectors: list[str]) -> str:
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

    async def _extract_availability(self, page: object) -> bool:
        """Return False if explicit out-of-stock indicator found."""
        for sel in self._AVAILABILITY_SELECTORS:
            try:
                el = await page.query_selector(sel)
                if el:
                    text = (await el.inner_text()).lower()
                    if any(s in text for s in ["sold out", "out of stock", "notify me"]):
                        return False
                    if "add to cart" in text or "buy now" in text:
                        return True
            except Exception:
                continue
        return True  # default optimistic

    async def _extract_specs(self, page: object) -> dict[str, str]:
        """Extract spec key-value pairs from the specification tables."""
        specs: dict[str, str] = {}

        for sel in self._SPEC_TABLE_SELECTORS:
            try:
                rows = await page.query_selector_all(sel)
                if not rows:
                    continue
                for row in rows:
                    try:
                        cells = await row.query_selector_all("td, th")
                        texts = [
                            (await c.inner_text()).strip()
                            for c in cells
                            if (await c.inner_text()).strip()
                        ]
                        if len(texts) >= 2:
                            key = texts[0].rstrip(":").strip()
                            val = texts[1].strip()
                            if key and val:
                                specs[key] = val
                    except Exception:
                        continue
                if len(specs) >= 3:
                    break
            except Exception:
                continue

        return specs

    async def _extract_review_count(self, page: object) -> int:
        for sel in self._REVIEW_COUNT_SELECTORS:
            try:
                el = await page.query_selector(sel)
                if el:
                    text = await el.inner_text()
                    m = re.search(r"([\d,]+)", text)
                    if m:
                        return int(m.group(1).replace(",", ""))
            except Exception:
                continue
        return 0

    async def _extract_rating(self, page: object) -> float | None:
        for sel in self._RATING_SELECTORS:
            try:
                el = await page.query_selector(sel)
                if el:
                    text = await el.inner_text()
                    m = re.search(r"(\d+\.?\d*)", text)
                    if m:
                        val = float(m.group(1))
                        if 0 < val <= 5:
                            return val
            except Exception:
                continue
        return None

    @staticmethod
    def _infer_model_name(title: str, brand: str) -> str:
        if not title:
            return ""
        if brand and title.lower().startswith(brand.lower()):
            return title[len(brand):].strip(" -–")
        return title.strip()
