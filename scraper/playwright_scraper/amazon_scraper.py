"""Amazon India (amazon.in) product page scraper.

Targets single product detail pages (ASIN pages) only.
Category is passed by the caller — never inferred from DOM.

Selector strategy:
  All selectors use layered fallbacks. If primary selector misses,
  secondary and tertiary selectors are tried in order. If all fail,
  the field remains at its default (empty / None).

Anti-bot:
  - Stealth Playwright via BaseScraper._new_stealth_page
  - Random user agent rotation
  - Slow scroll to trigger lazy content
  - CAPTCHA detection → graceful abort
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from playwright.async_api import async_playwright

from scraper.playwright_scraper.base_scraper import (
    BaseScraper,
    PAGE_TIMEOUT_MS,
    RawProductData,
)

logger = logging.getLogger(__name__)


class AmazonScraper(BaseScraper):
    """Scraper for amazon.in product detail pages."""

    # ── Selector layers (primary, fallback1, fallback2) ─────────────────────

    _TITLE_SELECTORS = [
        "#productTitle",
        "#title span",
        "h1.a-size-large",
    ]

    _PRICE_SELECTORS = [
        "#corePriceDisplay_desktop_feature_div .a-price-whole",
        "#priceblock_ourprice",
        "#priceblock_dealprice",
        ".a-price.reinventPricePriceToPayMargin .a-price-whole",
        "#apex_desktop_newAccordionRow .a-price-whole",
        ".priceToPay .a-price-whole",
    ]

    _AVAILABILITY_SELECTORS = [
        "#availability span",
        "#outOfStock .a-color-price",
        "#deliveryBlockMessage",
    ]

    _SPEC_TABLE_SELECTORS = [
        "#productDetails_techSpec_section_1 tr",
        "#tech-specs-items-rendered tr",
        "#productDetails_detailBullets_sections1 tr",
        ".a-expander-content tr",
        "#detailBullets_feature_div li",
    ]

    _REVIEW_COUNT_SELECTORS = [
        "#acrCustomerReviewText",
        "#acrCustomerReviewLink",
        "span[data-hook='total-review-count']",
    ]

    _RATING_SELECTORS = [
        "span[data-hook='rating-out-of-text']",
        "#acrPopover span.a-icon-alt",
        ".reviewNumericalSummary .a-icon-alt",
    ]

    _BRAND_SELECTORS = [
        "#bylineInfo",
        "#brand",
        "a#bylineInfo",
        "tr.po-brand td.a-span9 span",
    ]

    def get_domain(self) -> str:
        return "amazon.in"

    async def scrape(self, url: str) -> RawProductData:
        """Scrape a single Amazon.in product page.

        Args:
            url: Full ASIN product URL (amazon.in/dp/XXXXXXXX or similar).
                 Must be a single product detail page — no search/listing pages.

        Returns:
            RawProductData with extracted fields. error field populated on failure.
        """
        assert isinstance(url, str) and url.strip(), "url must be a non-empty string"
        scraped_at = datetime.now(timezone.utc).isoformat()

        async with async_playwright() as playwright:
            browser = await self._launch_browser(playwright)
            try:
                page = await self._new_stealth_page(browser)
                page.set_default_timeout(PAGE_TIMEOUT_MS)

                logger.info("amazon_scrape_start", extra={"url": url})
                await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
                await self._slow_scroll(page)

                # CAPTCHA guard
                if await self._detect_captcha(page):
                    logger.warning("amazon_captcha_detected", extra={"url": url})
                    html = await page.content()
                    raw = RawProductData(
                        url=url, vendor=self.get_domain(), category=self.category,
                        scraped_at=scraped_at, error="captcha_detected",
                    )
                    self._save_error_snapshot("amazon-captcha", html)
                    return raw

                # ── Extract fields ────────────────────────────────────────────

                title = await self._extract_text(page, self._TITLE_SELECTORS)
                brand_raw = await self._extract_text(page, self._BRAND_SELECTORS)
                brand = self._clean_brand(brand_raw)
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
                        f"amazon-sparse-{model_name[:20].replace(' ', '-')}", html
                    )
                    raw.error = "sparse_specs"

                logger.info(
                    "amazon_scrape_complete",
                    extra={"title": title, "price": price, "spec_count": len(specs)},
                )
                return raw

            except Exception as exc:
                logger.error("amazon_scrape_error", extra={"url": url, "error": str(exc)})
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

    async def _extract_availability(self, page: object) -> bool:
        """Return True if product appears available."""
        for sel in self._AVAILABILITY_SELECTORS:
            try:
                el = await page.query_selector(sel)
                if el:
                    text = (await el.inner_text()).lower()
                    if "in stock" in text or "in stock" in text:
                        return True
                    if any(s in text for s in ["out of stock", "unavailable", "currently unavailable"]):
                        return False
            except Exception:
                continue
        return True  # default optimistic

    async def _extract_specs(self, page: object) -> dict[str, str]:
        """Extract spec key-value pairs from the technical specifications table."""
        specs: dict[str, str] = {}

        for sel in self._SPEC_TABLE_SELECTORS:
            try:
                rows = await page.query_selector_all(sel)
                if not rows:
                    continue
                for row in rows:
                    try:
                        cells = await row.query_selector_all("th, td, span")
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
                    break  # found enough, stop trying selectors
            except Exception:
                continue

        return specs

    async def _extract_review_count(self, page: object) -> int:
        """Extract integer review count."""
        for sel in self._REVIEW_COUNT_SELECTORS:
            try:
                el = await page.query_selector(sel)
                if el:
                    text = await el.inner_text()
                    import re
                    m = re.search(r"([\d,]+)", text)
                    if m:
                        return int(m.group(1).replace(",", ""))
            except Exception:
                continue
        return 0

    async def _extract_rating(self, page: object) -> float | None:
        """Extract star rating as float."""
        for sel in self._RATING_SELECTORS:
            try:
                el = await page.query_selector(sel)
                if el:
                    text = await el.inner_text()
                    import re
                    m = re.search(r"(\d+\.?\d*)", text)
                    if m:
                        return float(m.group(1))
            except Exception:
                continue
        return None

    @staticmethod
    def _clean_brand(raw: str) -> str:
        """Strip 'Visit ...' / 'Brand: ...' prefixes from brand field."""
        if not raw:
            return ""
        import re
        cleaned = re.sub(r"^(visit the |brand:\s*)", "", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*store$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _infer_model_name(title: str, brand: str) -> str:
        """Best-effort model name: strip brand from title start."""
        if not title:
            return ""
        if brand and title.lower().startswith(brand.lower()):
            return title[len(brand):].strip(" -–")
        return title.strip()
