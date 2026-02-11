#!/usr/bin/env python3
"""
SAR101 - Sentinel-1 RTC mini-lab (STAC-native streaming)

What it does:
- Queries Sentinel-1 RTC items via STAC (Microsoft Planetary Computer)
- Picks two most recent scenes over Breda AOI (t0 older, t1 newer)
- Streams VV (preferred) or first available asset as Cloud-Optimized GeoTIFF (COG)
- Reads a center window
- Converts backscatter to dB
- Creates:
  - Water-like mask (threshold on t1 in dB)
  - Change detection (ratio in dB): ratio_db = db_t1 - db_t0
  - Change mask (threshold on ratio_db)
- Writes GeoTIFFs + GeoPackage polygon layers for masks

No Python osgeo bindings required. Uses rasterio/fiona only.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.windows import Window
from rasterio.features import shapes

import fiona
from fiona.crs import to_string as crs_to_string

from pystac_client import Client
import planetary_computer as pc


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# Breda AOI (lon/lat bbox): [minLon, minLat, maxLon, maxLat]
DEFAULT_BBOX_BREDA = [4.70, 51.53, 4.90, 51.66]


@dataclass
class Scene:
    item_id: str
    dt: datetime
    href: str
    asset_key: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAR101: Sentinel-1 RTC STAC streaming + masks + change detection")
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX_BREDA,
                   metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                   help="AOI bounding box in lon/lat (EPSG:4326). Default is around Breda.")
    p.add_argument("--days", type=int, default=30, help="Look back window in days (default: 30).")
    p.add_argument("--limit", type=int, default=20, help="Max STAC items to fetch (default: 20).")
    p.add_argument("--collection", type=str, default="sentinel-1-rtc", help="STAC collection (default: sentinel-1-rtc).")
    p.add_argument("--prefer-polarization", type=str, default="vv", choices=["vv", "vh"],
                   help="Preferred polarization asset key to use if available (default: vv).")
    p.add_argument("--window-size", type=int, default=1024, help="Square window size to read (default: 1024).")
    p.add_argument("--water-thr-db", type=float, default=-20.0,
                   help="Threshold in dB for a rough water-like mask on t1 (default: -20).")
    p.add_argument("--change-thr-db", type=float, default=2.0,
                   help="Threshold in dB for change mask on ratio_db (db_t1 - db_t0) (default: 2).")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory (default: outputs).")
    return p.parse_args()


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def stac_search_two_most_recent(
    bbox: List[float],
    days: int,
    limit: int,
    collection: str,
    prefer_pol: str,
) -> Tuple[Scene, Scene]:
    """
    Returns (t0, t1) where t0 is older, t1 is newer.
    Attempts to choose two most recent items; if multiple assets exist, picks preferred polarization.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    catalog = Client.open(STAC_URL)

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        limit=limit,
    )
    items = list(search.get_items())
    if len(items) < 2:
        raise RuntimeError(
            f"Need at least 2 scenes, found {len(items)}. Try increasing --days or expanding --bbox."
        )

    # Sort by datetime ascending
    def item_dt(it) -> datetime:
        dt = it.properties.get("datetime")
        if isinstance(dt, str):
            return datetime.fromisoformat(dt.replace("Z", "+00:00"))
        return dt

    items_sorted = sorted(items, key=item_dt)

    # Take the two most recent
    t0_item = items_sorted[-2]
    t1_item = items_sorted[-1]

    t0 = scene_from_item(t0_item, prefer_pol)
    t1 = scene_from_item(t1_item, prefer_pol)

    return t0, t1


def scene_from_item(item: Any, prefer_pol: str) -> Scene:
    signed = pc.sign(item)
    assets = signed.assets
    asset_keys = list(assets.keys())

    # Prefer vv/vh if present; otherwise pick first raster-like asset
    chosen_key = None
    if prefer_pol in assets:
        chosen_key = prefer_pol
    else:
        # common variations (some collections use uppercase or different naming)
        for k in asset_keys:
            if k.lower() == prefer_pol:
                chosen_key = k
                break
    if chosen_key is None:
        chosen_key = asset_keys[0]

    href = assets[chosen_key].href

    dt = signed.properties.get("datetime")
    if isinstance(dt, str):
        dt_parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    else:
        dt_parsed = dt

    return Scene(item_id=signed.id, dt=dt_parsed, href=href, asset_key=chosen_key)


def open_streaming_raster(href: str) -> rasterio.io.DatasetReader:
    """
    Open a remote COG with efficient HTTP range requests.
    """
    # rasterio uses GDAL; /vsicurl/ enables streaming range reads
    vsicurl_href = href
    if not href.startswith("/vsicurl/"):
        vsicurl_href = "/vsicurl/" + href
    return rasterio.open(vsicurl_href)


def read_center_window(ds: rasterio.io.DatasetReader, size: int) -> Tuple[np.ndarray, rasterio.Affine]:
    w = ds.width
    h = ds.height
    half = size // 2
    col_off = max(0, w // 2 - half)
    row_off = max(0, h // 2 - half)

    # Clamp if near edges
    if col_off + size > w:
        col_off = max(0, w - size)
    if row_off + size > h:
        row_off = max(0, h - size)

    window = Window(col_off=col_off, row_off=row_off, width=size, height=size)
    arr = ds.read(1, window=window).astype("float32")
    transform = ds.window_transform(window)
    return arr, transform


def to_db(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32")
    arr[arr <= 0] = np.nan
    return 10.0 * np.log10(arr)


def write_geotiff(path: str, arr: np.ndarray, transform: rasterio.Affine, crs: Any, dtype: str, nodata: Optional[float] = None) -> None:
    height, width = arr.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def save_quicklook_png(path: str, arr: np.ndarray, title: str, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    plt.figure()
    plt.imshow(arr, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def polygonize_mask_to_gpkg(
    gpkg_path: str,
    layer_name: str,
    mask: np.ndarray,
    transform: rasterio.Affine,
    crs: Any,
    keep_value: int = 1,
) -> None:
    """
    Writes polygons where mask == keep_value into a *new* GeoPackage.
    Avoids Fiona append-mode issues (NULL pointer error) on some macOS builds.
    """
    schema = {"geometry": "Polygon", "properties": {"value": "int"}}

    # Fiona CRS handling: prefer rasterio CRS WKT (stable across Fiona versions)
    crs_wkt = None
    try:
        crs_wkt = crs.to_wkt()
    except Exception:
        # fallback: Fiona can sometimes take rasterio CRS directly
        crs_wkt = crs

    # Always create a fresh dataset (no append mode)
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)

    with fiona.open(
        gpkg_path,
        mode="w",
        driver="GPKG",
        layer=layer_name,
        crs_wkt=crs_wkt,
        schema=schema,
    ) as dst:
        # rasterio.features.shapes yields GeoJSON-like geometries
        for geom, val in shapes(mask, mask=(mask == keep_value), transform=transform):
            dst.write({"geometry": geom, "properties": {"value": int(val)}})


def main() -> None:
    args = parse_args()
    ensure_outdir(args.outdir)

    bbox = list(args.bbox)

    print(f"AOI bbox (lon/lat): {bbox}")
    print(f"Searching STAC collection '{args.collection}' last {args.days} days (limit {args.limit}) ...")

    t0, t1 = stac_search_two_most_recent(
        bbox=bbox,
        days=args.days,
        limit=args.limit,
        collection=args.collection,
        prefer_pol=args.prefer_polarization,
    )

    print("\nSelected scenes:")
    print(f"  t0 (older): {t0.item_id}  {t0.dt.isoformat()}  asset={t0.asset_key}")
    print(f"  t1 (newer): {t1.item_id}  {t1.dt.isoformat()}  asset={t1.asset_key}")

    # Open both scenes streaming
    with open_streaming_raster(t0.href) as ds0, open_streaming_raster(t1.href) as ds1:
        # If CRSs differ (rare for RTC), we bail for simplicity
        if ds0.crs != ds1.crs:
            raise RuntimeError(f"CRS mismatch: t0={ds0.crs} t1={ds1.crs}. Use a preprocessing step to align.")
        crs = ds0.crs

        # Read same-size center window from each
        arr0, transform0 = read_center_window(ds0, args.window_size)
        arr1, transform1 = read_center_window(ds1, args.window_size)

        # For simplicity, require same transform (center windows may differ slightly if raster dims differ)
        # If transforms differ, we still proceed using t1 transform for outputs, assuming same pixel grid
        transform = transform1

    # Convert to dB
    db0 = to_db(arr0)
    db1 = to_db(arr1)

    # Change detection ratio in dB (equivalent to 10log10(arr1/arr0))
    ratio_db = db1 - db0

    # Masks
    water_mask_t1 = (db1 < args.water_thr_db).astype("uint8")
    change_mask = (np.abs(ratio_db) > args.change_thr_db).astype("uint8")

    # Output paths
    out_db1_png = os.path.join(args.outdir, "quicklook_t1_backscatter_db.png")
    out_ratio_png = os.path.join(args.outdir, "quicklook_ratio_db_t1_minus_t0.png")

    out_water_tif = os.path.join(args.outdir, "water_mask_t1.tif")
    out_ratio_tif = os.path.join(args.outdir, "ratio_db_t1_minus_t0.tif")
    out_change_tif = os.path.join(args.outdir, "change_mask_ratio_db.tif")

    # out_gpkg = os.path.join(args.outdir, "sar101_masks.gpkg")

    # Quicklooks (auto vmin/vmax to see contrast; tweak if you want consistent scaling)
    save_quicklook_png(out_db1_png, db1, title=f"S1 RTC {t1.asset_key.upper()} backscatter (dB) t1")
    save_quicklook_png(out_ratio_png, ratio_db, title="Change detection ratio_db = db(t1) - db(t0)")

    # Write rasters
    write_geotiff(out_water_tif, water_mask_t1, transform, crs, dtype="uint8", nodata=0)
    write_geotiff(out_change_tif, change_mask, transform, crs, dtype="uint8", nodata=0)
    write_geotiff(out_ratio_tif, ratio_db.astype("float32"), transform, crs, dtype="float32", nodata=np.nan)

    # # Polygonize to GeoPackage (two layers)
    # if os.path.exists(out_gpkg):
    #     os.remove(out_gpkg)
    # polygonize_mask_to_gpkg(out_gpkg, "water_polys_t1", water_mask_t1, transform, crs, keep_value=1)
    # polygonize_mask_to_gpkg(out_gpkg, "change_polys_ratio_db", change_mask, transform, crs, keep_value=1)

    out_gpkg_water = os.path.join(args.outdir, "water_polys_t1.gpkg")
    out_gpkg_change = os.path.join(args.outdir, "change_polys_ratio_db.gpkg")

    polygonize_mask_to_gpkg(out_gpkg_water, "water_polys_t1", water_mask_t1, transform, crs, keep_value=1)
    polygonize_mask_to_gpkg(out_gpkg_change, "change_polys_ratio_db", change_mask, transform, crs, keep_value=1)

    print("\nWrote outputs:")
    print(f"  {out_db1_png}")
    print(f"  {out_ratio_png}")
    print(f"  {out_water_tif}")
    print(f"  {out_ratio_tif}")
    print(f"  {out_change_tif}")
    print(f"  {out_gpkg_water}")
    print(f"  {out_gpkg_change}")

    print("\nTips:")
    print("  - Open outputs/*.gpkg in QGIS.")
    print("  - The thresholds are rough; tune --water-thr-db and --change-thr-db for your AOI/time period.")


if __name__ == "__main__":
    main()
