"""Variant builder — assembles Product and Variant dicts from scraper output.

Generates collision-safe slugs and clean dicts for idempotent DB upsert.
Returns plain Python dicts (never ORM instances) for DB-agnostic use.

Slug format: {brand}-{model_name}-{category}-{release_year|na}
  Example: "apple-iphone-15-pro-smartphone-2023"
  Category is embedded to prevent cross-category collisions.
  Release year is embedded to prevent cross-year collisions for model refreshes.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any


# ── Slug generation ───────────────────────────────────────────────────────────


def _slugify(text: str) -> str:
    """Convert text to URL-safe lowercase slug with hyphens.

    1. Unicode NFKD normalization (handles accented chars)
    2. ASCII-only
    3. Lowercase
    4. Replace non-alphanumeric with hyphens
    5. Collapse consecutive hyphens
    6. Strip leading/trailing hyphens
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def generate_slug(
    brand: str,
    model_name: str,
    category: str,
    release_year: int | str | None = None,
) -> str:
    """Generate a collision-safe product slug.

    Format: {brand}-{model_name}-{category}-{release_year|na}

    The category field eliminates cross-category collisions.
    The release_year field eliminates year-on-year model refresh collisions.

    Args:
        brand: Brand name (e.g. "Apple", "Samsung").
        model_name: Model/title (e.g. "iPhone 15 Pro", "Galaxy S24 Ultra").
        category: Canonical category ("smartphone" or "laptop").
        release_year: 4-digit year or None → "na" suffix.

    Returns:
        Lowercase hyphenated slug string.
    """
    year_part = str(release_year) if release_year else "na"
    combined = f"{brand} {model_name} {category} {year_part}"
    slug = _slugify(combined)
    # Safety: cap slug length at 128 chars
    return slug[:128]


def _extract_release_year(launch_date_str: str | None) -> int | None:
    """Parse 4-digit year from a launch date string."""
    if not launch_date_str:
        return None
    m = re.search(r"(20\d{2})", launch_date_str)
    return int(m.group(1)) if m else None


# ── Product dict builder ──────────────────────────────────────────────────────


def build_product_dict(
    raw: Any,                     # RawProductData
    specs: Any,                   # NormalizedSpecs
    category: str,
) -> dict[str, Any]:
    """Assemble the 'products' table upsert dict.

    Args:
        raw: RawProductData from scraper.
        specs: NormalizedSpecs from normalizer.
        category: Canonical category string (smartphone|laptop).

    Returns:
        Dict with keys matching the Product ORM model.
        product_slug is collision-safe (brand+model+category+year).
    """
    release_year = _extract_release_year(specs.launch_date_str)
    slug = generate_slug(raw.brand, raw.model_name, category, release_year)

    # Serialize normalized specs into jsonb payload
    specs_jsonb: dict[str, Any] = {}
    for field_name in [
        "ram_gb", "storage_gb", "battery_mah", "battery_wh",
        "display_inches", "refresh_rate_hz", "cpu_model", "gpu_model",
        "camera_mp_main", "weight_g", "build_material", "launch_date_str",
    ]:
        val = getattr(specs, field_name, None)
        if val is not None:
            specs_jsonb[field_name] = val

    # Release date from launch_date_str (best-effort ISO)
    release_date = None
    if specs.launch_date_str:
        try:
            # Try common formats: "January 2023", "2023-01-15", "2023"
            for fmt in ("%Y-%m-%d", "%B %Y", "%Y"):
                try:
                    release_date = datetime.strptime(specs.launch_date_str, fmt).replace(
                        tzinfo=timezone.utc
                    )
                    break
                except ValueError:
                    continue
        except Exception:
            pass

    return {
        "product_slug": slug,
        "brand": raw.brand or "Unknown",
        "model_name": raw.model_name or raw.title or "Unknown",
        "canonical_title": raw.title or f"{raw.brand} {raw.model_name}".strip(),
        "category": category,
        "release_date": release_date,
        "specs_jsonb": specs_jsonb,
    }


# ── Variant dict builder ──────────────────────────────────────────────────────


def build_variant_dict(
    raw: Any,         # RawProductData
    product_id: int,
    vendor: str,
) -> dict[str, Any]:
    """Assemble the 'variants' table upsert dict.

    Args:
        raw: RawProductData from scraper.
        product_id: DB ID of the already-upserted parent product.
        vendor: Vendor domain (e.g. "amazon.in", "flipkart.com").

    Returns:
        Dict with keys matching the Variant ORM model.
    """
    # Extract RAM/storage from raw specs for variant-level storage
    ram_gb: int | None = None
    storage_gb: int | None = None
    color: str | None = None

    # Try to pull from raw.specs directly for variant-level differentiation
    from scraper.normalization.spec_normalizer import _parse_ram, _parse_storage
    for k, v in (raw.specs or {}).items():
        kl = k.lower()
        if "ram" in kl or "memory" in kl:
            ram_gb = ram_gb or _parse_ram(str(v))
        if "storage" in kl or "rom" in kl:
            storage_gb = storage_gb or _parse_storage(str(v))
        if "color" in kl or "colour" in kl:
            color = color or str(v).strip()

    return {
        "product_id": product_id,
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "color": color,
        "price_inr": raw.price,
        "currency": raw.currency or "INR",
        "availability_bool": raw.availability,
        "last_price_update": datetime.now(timezone.utc),
        "affiliate_url": raw.url,
        "vendor": vendor,
        "sku": None,    # populated later from vendor API if available
    }
