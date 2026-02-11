from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from pystac_client import Client

try:
    import planetary_computer as pc  # type: ignore
except Exception:  # pragma: no cover
    pc = None  # type: ignore


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _append_sas_token(href: str) -> str:
    """
    If a user provides a Planetary Computer SAS token via env var, append it to the URL.
    This is a fallback for environments where signing is required and the default signing fails.
    """
    token = os.getenv("PC_SAS_TOKEN")
    if not token:
        return href

    # Token may include leading '?'
    token = token.lstrip("?")
    joiner = "&" if "?" in href else "?"
    return f"{href}{joiner}{token}"


def search_sentinel1_rtc(
    bbox: Tuple[float, float, float, float],
    days_lookback: int = 10,
    limit: int = 5,
    orbit_state: Optional[str] = None,
) -> List[dict]:
    """
    Returns a list of STAC items (as dict-like objects) for the sentinel-1-rtc collection.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_lookback)

    catalog = Client.open(STAC_URL)
    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=list(bbox),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        limit=limit,
        query={"sat:orbit_state": {"eq": orbit_state}} if orbit_state else None,
    )

    return [it for it in search.get_items()]


def sign_item(item):
    """
    Sign a STAC item for asset access (Planetary Computer). If signing isn't available, returns item.
    """
    if pc is None:
        return item
    try:
        return pc.sign(item)
    except Exception:
        return item


def pick_asset_href(item_signed, prefer: str = "vv") -> Tuple[str, str]:
    """
    Returns (asset_key, href) for an item. Prefers VV when available.
    """
    assets = getattr(item_signed, "assets", None)
    if not assets:
        raise ValueError("STAC item has no assets")

    keys = list(assets.keys())
    asset_key = prefer if prefer in keys else keys[0]
    href = assets[asset_key].href

    # Optional user token fallback
    href = _append_sas_token(href)
    return asset_key, href
