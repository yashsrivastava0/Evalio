"""Hardened abstract base scraper with Playwright stealth helpers.

Subclass this for each vendor. Category is ALWAYS passed from the caller
(DecisionMatrix → fetch_trigger → Celery task → scraper). Never inferred.

Architectural contracts:
  1. scrape(url) MUST target exactly ONE product page. No pagination.
  2. _validate_extraction(raw) MUST be called before any DB write.
     If it returns False → caller must abort insert and set fetch_status=failed.
  3. _parse_price_inr(raw_str) handles all INR formatting edge cases.
  4. max 2 concurrent scrapes enforced via class-level asyncio.Semaphore.
  5. Retry with exponential backoff (max 3 retries, base_delay 2s).
  6. PAGE_TIMEOUT_MS = 15_000 (15 seconds hard timeout per page).
"""

from __future__ import annotations

import abc
import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PAGE_TIMEOUT_MS: int = 15_000
MIN_VALID_SPEC_FIELDS: int = 3        # circuit breaker threshold
MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 2.0         # seconds; exponential → 2, 4, 8

# Class-level semaphore: max 2 concurrent Playwright scrapes across all workers
_SCRAPE_SEMAPHORE = asyncio.Semaphore(2)

# ── User Agent Pool ───────────────────────────────────────────────────────────

_USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.76",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
]


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class RawProductData:
    """Raw scraped product data before normalization.

    Fields are populated on a best-effort basis.
    Missing fields remain at their default (empty / None).
    _validate_extraction() checks minimum field population before DB write.
    """

    url: str
    vendor: str
    category: str                              # passed by caller, never inferred
    title: str = ""
    brand: str = ""
    model_name: str = ""
    price: float | None = None
    currency: str = "INR"
    availability: bool = True
    specs: dict[str, Any] = field(default_factory=dict)
    review_count: int = 0
    rating: float | None = None
    images: list[str] = field(default_factory=list)
    raw_html: str = ""
    scraped_at: str = ""
    error: str | None = None


@dataclass
class ScraperError:
    """Structured error log entry emitted on any scrape failure."""

    product_slug: str
    source: str                # vendor domain
    error_type: str            # e.g. "timeout", "captcha", "layout_mismatch", "sparse_specs"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    detail: str = ""


# ── Abstract Base ─────────────────────────────────────────────────────────────


class BaseScraper(abc.ABC):
    """Abstract base for all Playwright scrapers.

    Each vendor implements scrape() and get_domain().
    Shared helpers: stealth page launch, price parsing, CAPTCHA detection,
    slow scroll, retry decorator, extraction validation.
    """

    def __init__(
        self,
        category: str,
        proxy_url: str = "",
        rate_limit_per_minute: int = 10,
    ) -> None:
        # category is ALWAYS passed by the caller — never inferred from DOM.
        self.category = category
        self.proxy_url = proxy_url
        self.rate_limit_per_minute = rate_limit_per_minute

    # ── Abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    async def scrape(self, url: str) -> RawProductData:
        """Scrape exactly ONE product URL. No crawling, no pagination."""
        ...

    @abc.abstractmethod
    def get_domain(self) -> str:
        """Return the target domain (e.g., 'amazon.in')."""
        ...

    # ── Playwright helpers ────────────────────────────────────────────────────

    async def _launch_browser(self, playwright: Any) -> Any:
        """Launch headless Chromium with stealth args and optional proxy."""
        launch_args = {
            "headless": True,
            "args": [
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--disable-infobars",
                "--disable-extensions",
                "--lang=en-IN",
            ],
        }
        if self.proxy_url:
            launch_args["proxy"] = {"server": self.proxy_url}

        return await playwright.chromium.launch(**launch_args)

    async def _new_stealth_page(self, browser: Any) -> Any:
        """Create a new page with randomized user-agent and stealth headers."""
        ua = random.choice(_USER_AGENTS)
        context = await browser.new_context(
            user_agent=ua,
            locale="en-IN",
            timezone_id="Asia/Kolkata",
            extra_http_headers={
                "Accept-Language": "en-IN,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            },
            viewport={"width": 1280 + random.randint(-80, 80),
                      "height": 800 + random.randint(-40, 40)},
        )
        page = await context.new_page()

        # Mask navigator.webdriver property
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        return page

    async def _slow_scroll(self, page: Any) -> None:
        """Simulate human-like slow scroll to trigger lazy-loaded content."""
        for _ in range(5):
            await page.evaluate("window.scrollBy(0, 300)")
            await asyncio.sleep(random.uniform(0.3, 0.7))
        await asyncio.sleep(random.uniform(0.5, 1.2))

    async def _detect_captcha(self, page: Any) -> bool:
        """Detect CAPTCHA presence — abort gracefully if found."""
        content = (await page.content()).lower()
        captcha_signals = [
            "captcha", "robot check", "verify you are human",
            "security check", "unusual traffic", "automated access",
        ]
        return any(s in content for s in captcha_signals)

    # ── Price parser ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_price_inr(raw: str) -> float | None:
        """Parse Indian Rupee price strings to float.

        Handles:
          "₹1,24,999"  → 124999.0
          "Rs. 49,999" → 49999.0
          "49999"      → 49999.0
          "Currently unavailable" → None
          Price ranges (e.g. "₹10,000 - ₹15,000") → None

        Returns None on any ambiguity or unavailability.
        """
        if not raw or not isinstance(raw, str):
            return None

        cleaned = raw.strip().lower()

        # Unavailability signals
        unavailable_signals = [
            "unavailable", "out of stock", "not available",
            "currently", "sold out", "coming soon",
        ]
        if any(s in cleaned for s in unavailable_signals):
            return None

        # Price range — ambiguous, reject
        if "-" in raw and re.search(r"\d.*-.*\d", raw):
            return None

        # Strip currency symbols and whitespace (keep decimal point)
        cleaned = re.sub(r"[₹$€£\s]", "", raw, flags=re.IGNORECASE)
        # Strip "Rs." / "rs" prefix
        cleaned = re.sub(r"^rs\.?\s*", "", cleaned, flags=re.IGNORECASE)
        # Remove commas (Indian number format: 1,24,999)
        cleaned = cleaned.replace(",", "")

        try:
            value = float(cleaned)
            # Sanity check: price must be positive and reasonable
            if value <= 0 or value > 10_000_000:
                return None
            return value
        except ValueError:
            return None

    # ── Extraction validator (circuit breaker) ────────────────────────────────

    @staticmethod
    def _validate_extraction(raw: RawProductData) -> bool:
        """Circuit breaker: require minimum valid spec fields for DB write.

        Returns False if fewer than MIN_VALID_SPEC_FIELDS spec entries exist.
        Caller must abort insert and set fetch_status=failed on False.
        """
        valid_spec_count = sum(
            1 for v in raw.specs.values()
            if v is not None and str(v).strip() != ""
        )
        if valid_spec_count < MIN_VALID_SPEC_FIELDS:
            logger.warning(
                "sparse_specs_circuit_break",
                extra={
                    "valid_fields": valid_spec_count,
                    "threshold": MIN_VALID_SPEC_FIELDS,
                    "url": raw.url,
                },
            )
            return False
        return True

    # ── Retry helper ──────────────────────────────────────────────────────────

    async def _scrape_with_retry(self, url: str) -> RawProductData:
        """Wrap scrape() with exponential backoff retry (max MAX_RETRIES)."""
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with _SCRAPE_SEMAPHORE:
                    return await self.scrape(url)
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    delay += random.uniform(0, 1)  # jitter
                    logger.warning(
                        "scrape_retry",
                        extra={
                            "attempt": attempt,
                            "delay": delay,
                            "url": url,
                            "error": str(exc),
                        },
                    )
                    await asyncio.sleep(delay)

        ts = datetime.now(timezone.utc).isoformat()
        empty = RawProductData(
            url=url,
            vendor=self.get_domain(),
            category=self.category,
            scraped_at=ts,
            error=str(last_error),
        )
        return empty

    # ── HTML error snapshot ───────────────────────────────────────────────────

    @staticmethod
    def _save_error_snapshot(slug: str, html: str, snapshot_dir: str = "/tmp/scraper_errors") -> None:
        """Save raw HTML for debugging when extraction fails.

        In Docker: /tmp is ephemeral. Mount a volume at /tmp/scraper_errors
        in docker-compose (profiles: [scraping]) for persistence in dev.
        """
        import os
        from pathlib import Path

        os.makedirs(snapshot_dir, exist_ok=True)
        ts = int(time.time())
        path = Path(snapshot_dir) / f"{slug}_{ts}.html"
        try:
            path.write_text(html, encoding="utf-8")
            logger.info("html_snapshot_saved", extra={"path": str(path)})
        except OSError as e:
            logger.warning("html_snapshot_failed", extra={"error": str(e)})
