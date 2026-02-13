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
from dataclasses import dataclass, is_dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.warp import transform as rio_transform

import fiona
from fiona.crs import to_string as crs_to_string

from pystac_client import Client
import planetary_computer as pc
import logging
from logging.handlers import RotatingFileHandler
import inspect
import json
from pathlib import Path #as _Path
import socket
import getpass
import time

import uuid

# Optional dependency: iceye-audit-log (https://github.com/iceye-ltd/python-audit-log)
try:
    import audit_log.log as _iceye_audit_mod  # type: ignore
    from audit_log.log import log as _iceye_audit_log  # type: ignore
except Exception:  # pragma: no cover
    _iceye_audit_mod = None
    _iceye_audit_log = None


@dataclass
class PrincipalCompat:
    """Dataclass-compatible principal for iceye-audit-log.

    `iceye-audit-log` serializes the given principal using `dataclasses.asdict()`.
    If we pass a plain dict, it crashes with:
      TypeError: asdict() should be called on dataclass instances

    This small compatibility wrapper keeps sar101.py usable both with and without
    the external audit log dependency.
    """

    user: Optional[str] = None
    host: Optional[str] = None
    ip: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# Breda AOI (lon/lat bbox): [minLon, minLat, maxLon, maxLat]
DEFAULT_BBOX_BREDA = [4.70, 51.53, 4.90, 51.66]

# Nijmegen AOI (Brakkenstein area) (lon/lat bbox): [minLon, minLat, maxLon, maxLat]
DEFAULT_BBOX_NIJMEGEN_BRAKKENSTEIN = [5.845, 51.812, 5.885, 51.835]

# Port of Rotterdam AOI (incl. Europoort/Maasvlakte-ish) (lon/lat bbox): [minLon, minLat, maxLon, maxLat]
DEFAULT_BBOX_PORT_OF_ROTTERDAM = [3.98, 51.86, 4.55, 52.03]

DEFAULT_BBOX = DEFAULT_BBOX_NIJMEGEN_BRAKKENSTEIN


def _bbox_center(b: List[float]) -> Tuple[float, float]:
    """(lon, lat) center of a lon/lat bbox."""
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance (km) using a simple haversine."""
    r = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def log_stac_search_diagnostics(
    *,
    logger: logging.Logger,
    collection: str,
    bbox: List[float],
    start: datetime,
    end: datetime,
    items: List[Any],
    top_n: int = 10,
) -> None:
    """Log basic sanity checks: input bbox vs returned item footprints/bboxes."""
    in_lon, in_lat = _bbox_center(bbox)
    logger.info(
        "STAC search diagnostics: collection=%s bbox=%s center=(%.6f,%.6f) datetime=%s/%s items=%d",
        collection,
        bbox,
        in_lon,
        in_lat,
        start.isoformat(),
        end.isoformat(),
        len(items),
    )

    # Sort newest first for easier eyeballing
    def _item_dt(it) -> datetime:
        dt = it.properties.get("datetime")
        if isinstance(dt, str):
            return datetime.fromisoformat(dt.replace("Z", "+00:00"))
        return dt

    items_sorted = sorted(items, key=_item_dt, reverse=True)

    for it in items_sorted[:top_n]:
        ib = getattr(it, "bbox", None)
        geom = getattr(it, "geometry", None)
        dt = _item_dt(it)

        # Item bbox center if bbox present
        if ib and len(ib) == 4:
            i_lon, i_lat = _bbox_center(list(ib))
            dist_km = _haversine_km(in_lon, in_lat, i_lon, i_lat)
        else:
            i_lon = i_lat = float("nan")
            dist_km = float("nan")

        logger.info(
            "STAC item: id=%s dt=%s item_bbox=%s item_bbox_center=(%.6f,%.6f) center_dist_km=%.1f has_geometry=%s",
            it.id,
            dt.isoformat(),
            ib,
            i_lon,
            i_lat,
            dist_km,
            bool(geom),
        )

    # Extra hint: this script reads the *scene center* unless you change the windowing.
    logger.info(
        "NOTE: SAR101 currently reads a center window of the *full raster scene*. "
        "If a returned item covers a large swath, the center window may be far from your input bbox."
    )

def repo_root() -> Path:
    """Repo root assumed to be the parent of the `src/` directory."""
    return Path(__file__).resolve().parents[1]


def cleanup_old_logs(logs_dir: Path, max_days: int = 7) -> None:
    """Delete audit log files older than max_days. Best-effort (never breaks pipeline)."""
    cutoff = time.time() - (max_days * 24 * 60 * 60)
    for p in logs_dir.glob("sar101-audit.log*"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except Exception:
            pass


def setup_file_logging() -> Path:
    """
    Configure local file logging under ./logs with:
    - size-based rotation at 10 MB
    - retention cleanup older than 7 days
    """
    logs_dir = repo_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "sar101-audit.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Also log to console for quick debugging (once).
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(
            fmt="%(asctime)sZ %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        root_logger.addHandler(sh)

    # Avoid duplicate handlers on repeated runs (e.g., notebooks)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith(str(log_path))
               for h in root_logger.handlers):
        handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=20,             # adjust if needed
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            fmt="%(asctime)sZ %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    cleanup_old_logs(logs_dir, max_days=7)
    return log_path


def audit(event: str, **fields) -> None:
    """
    Write an audit event to:
      1) local rotating file log (JSON line) via stdlib logging
      2) (optional) ICEYE iceye-audit-log (https://github.com/iceye-ltd/python-audit-log)

    Notes about iceye-audit-log:
    - Its public API is `from audit_log.log import log`.
    - The function signature is positional-first and differs from earlier experiments.
      Current releases (e.g. 0.2.x) expect at least:
        log(event, resource_type, resource_id, result, principal, ...)
      and will try to pull `request_id` from a ContextVar if not provided.

    This wrapper always logs locally and then best-effort calls iceye-audit-log
    **without ever breaking** the SAR101 pipeline.
    """

    # --- 1) Always log locally as JSON ---
    payload = {"event": event, **fields}
    logging.getLogger("sar101.audit").info(json.dumps(payload, default=str))

    # --- 2) Optional call into iceye-audit-log ---
    if _iceye_audit_log is None:
        return

    # Required/standard fields
    # iceye-audit-log expects `principal` to be a *dataclass instance* because it
    # calls `dataclasses.asdict(principal)` internally.
    raw_principal = fields.get("principal")
    if raw_principal is None:
        raw_principal = {
            "user": getpass.getuser(),
            "host": socket.gethostname(),
        }

    if is_dataclass(raw_principal) and not isinstance(raw_principal, type):
        principal = raw_principal
    elif isinstance(raw_principal, dict):
        principal = PrincipalCompat(
            user=raw_principal.get("user") or raw_principal.get("username") or raw_principal.get("name"),
            host=raw_principal.get("host") or socket.gethostname(),
            ip=raw_principal.get("ip"),
            extra={k: v for k, v in raw_principal.items() if k not in {"user", "username", "name", "host", "ip"}},
        )
    else:
        # treat as a simple identifier (e.g., username string)
        principal = PrincipalCompat(user=str(raw_principal), host=socket.gethostname())
    resource_type = fields.get("resource_type") or "sar101"
    resource_id = (
        fields.get("resource_id")
        or fields.get("run_id")
        or fields.get("t1_id")
        or fields.get("item_id")
        or event
    )
    result = fields.get("result")
    if result is None:
        result = "FAILURE" if event in ("run_failed",) or event.endswith("_failed") else "SUCCESS"

    # IMPORTANT: avoid ContextVar LookupError in iceye-audit-log by always providing request_id.
    request_id = str(fields.get("run_id") or uuid.uuid4())

    # Add all context as details (if supported by the lib)
    details = dict(fields)
    details.setdefault("event", event)

    try:
        # If module exposes REQ_ID ContextVar, set it too (belt-and-suspenders).
        if _iceye_audit_mod is not None and hasattr(_iceye_audit_mod, "REQ_ID"):
            try:
                _iceye_audit_mod.REQ_ID.set(request_id)  # type: ignore[attr-defined]
            except Exception:
                pass

        sig = inspect.signature(_iceye_audit_log)
        params = sig.parameters

        # Build kwargs only for supported optional params
        opt_kwargs = {}
        if "request_id" in params:
            opt_kwargs["request_id"] = request_id
        if "details" in params:
            opt_kwargs["details"] = details
        # Some versions may accept "reason"; pass through if user provided.
        if "reason" in params and "reason" in fields:
            opt_kwargs["reason"] = fields["reason"]

        # Prefer keyword names if present; otherwise call positionally.
        if "action" in params:
            # (older/alternate API)
            _iceye_audit_log(
                action=event,
                resource_type=resource_type,
                resource_id=resource_id,
                result=result,
                principal=principal,
                **opt_kwargs,
            )
        elif "event" in params:
            _iceye_audit_log(
                event=event,
                resource_type=resource_type,
                resource_id=resource_id,
                result=result,
                principal=principal,
                **opt_kwargs,
            )
        else:
            # Current API (0.2.x): positional required args
            _iceye_audit_log(event, resource_type, resource_id, result, principal, **opt_kwargs)

    except Exception:
        logging.getLogger("sar101.audit").exception("iceye-audit-log call failed")

@dataclass
class Scene:
    item_id: str
    dt: datetime
    href: str
    asset_key: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAR101: Sentinel-1 RTC STAC streaming + masks + change detection")
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
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
    items = list(search.items())

    # Diagnostics: show what we asked for vs what came back.
    # This is especially useful if you *think* you're landing in the wrong place.
    # NOTE: later in the pipeline we read the *scene center window* (not an AOI window),
    # so it is totally possible to query Nijmegen and then visualize something that looks like
    # Harderwijk (center of the returned scene). This log makes that obvious.
    try:
        log_stac_search_diagnostics(
            logger=logging.getLogger("sar101.stac"),
            collection=collection,
            bbox=bbox,
            start=start,
            end=end,
            items=items,
            top_n=min(10, len(items)),
        )
    except Exception:
        logging.getLogger("sar101.stac").exception("Failed to log STAC search diagnostics")
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


def read_aoi_center_window(
    ds: rasterio.io.DatasetReader,
    size: int,
    bbox_lonlat: List[float],
    *,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, rasterio.Affine]:
    """Read a square window centered on the *AOI bbox center*.

    - bbox_lonlat is expected in EPSG:4326 lon/lat.
    - The AOI center is reprojected into the dataset CRS before indexing.

    This is the high-impact fix that ensures the extracted chip actually corresponds
    to your intended AOI (instead of the scene center).
    """
    if logger is None:
        logger = logging.getLogger("sar101.window")

    # AOI center in lon/lat
    aoi_lon, aoi_lat = _bbox_center(bbox_lonlat)

    # Reproject AOI center into dataset CRS (x/y)
    if ds.crs is None:
        raise RuntimeError("Dataset has no CRS; cannot place AOI window.")

    if str(ds.crs).upper() in ("EPSG:4326", "WGS84"):
        x, y = aoi_lon, aoi_lat
    else:
        xs, ys = rio_transform("EPSG:4326", ds.crs, [aoi_lon], [aoi_lat])
        x, y = xs[0], ys[0]

    # Convert to raster indices
    row, col = ds.index(x, y)

    w = ds.width
    h = ds.height
    half = size // 2

    col_off = int(col - half)
    row_off = int(row - half)

    # Clamp offsets within raster bounds
    col_off = max(0, min(col_off, max(0, w - size)))
    row_off = max(0, min(row_off, max(0, h - size)))

    window = Window(col_off=col_off, row_off=row_off, width=size, height=size)
    arr = ds.read(1, window=window).astype("float32")
    transform = ds.window_transform(window)

    # Helpful diagnostics for "why am I seeing Harderwijk" moments.
    try:
        left, bottom, right, top = rasterio.windows.bounds(window, ds.transform)
        logger.info(
            "AOI-center window: aoi_center_lonlat=(%.6f,%.6f) ds_crs=%s aoi_center_xy=(%.3f,%.3f) "
            "rowcol=(%d,%d) window_off=(%d,%d) window_bounds_in_ds_crs=(%.3f,%.3f,%.3f,%.3f)",
            aoi_lon,
            aoi_lat,
            str(ds.crs),
            x,
            y,
            row,
            col,
            col_off,
            row_off,
            left,
            bottom,
            right,
            top,
        )
    except Exception:
        logger.exception("Failed to compute/log AOI window diagnostics")

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
    try:
        args = parse_args()

        log_path = setup_file_logging()

        run_id = str(uuid.uuid4())
        audit("run_started", run_id=run_id, user=getpass.getuser(),
            host=socket.gethostname(),
            outdir=args.outdir,
            bbox=list(args.bbox),
            days=args.days,
            limit=args.limit,
            collection=args.collection,
            prefer_polarization=args.prefer_polarization,
            window_size=args.window_size,
            water_thr_db=args.water_thr_db,
            change_thr_db=args.change_thr_db,
            audit_log_file=str(log_path),
        )

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

        audit("open_streaming_rasters", run_id=run_id, t0_href=t0.href, t1_href=t1.href)

        # Open both scenes streaming
        with open_streaming_raster(t0.href) as ds0, open_streaming_raster(t1.href) as ds1:
            # If CRSs differ (rare for RTC), we bail for simplicity
            if ds0.crs != ds1.crs:
                raise RuntimeError(f"CRS mismatch: t0={ds0.crs} t1={ds1.crs}. Use a preprocessing step to align.")
            crs = ds0.crs

            # Read same-size window centered on the *AOI bbox center* (reprojected to dataset CRS)
            arr0, transform0 = read_aoi_center_window(ds0, args.window_size, bbox)
            arr1, transform1 = read_aoi_center_window(ds1, args.window_size, bbox)

            audit(
                "read_aoi_center_windows",
                run_id=run_id,
                window_size=args.window_size,
                crs=str(crs),
                aoi_bbox_lonlat=bbox,
                aoi_center_lonlat=_bbox_center(bbox),
            )

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


        audit(
            "masks_computed",
            run_id=run_id, water_thr_db=args.water_thr_db,
            change_thr_db=args.change_thr_db,
            water_pct=float(water_mask_t1.mean() * 100.0),
            change_pct=float(change_mask.mean() * 100.0),
        )
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

        audit(
            "output_paths_resolved",
            run_id=run_id, quicklook_t1=out_db1_png,
            quicklook_ratio=out_ratio_png,
            water_mask_tif=out_water_tif,
            ratio_tif=out_ratio_tif,
            change_mask_tif=out_change_tif,
            water_gpkg=out_gpkg_water,
            change_gpkg=out_gpkg_change,
        )

        polygonize_mask_to_gpkg(out_gpkg_water, "water_polys_t1", water_mask_t1, transform, crs, keep_value=1)
        polygonize_mask_to_gpkg(out_gpkg_change, "change_polys_ratio_db", change_mask, transform, crs, keep_value=1)


        audit(
            "outputs_written",
            run_id=run_id, quicklook_t1=out_db1_png,
            quicklook_ratio=out_ratio_png,
            water_mask_tif=out_water_tif,
            ratio_tif=out_ratio_tif,
            change_mask_tif=out_change_tif,
            water_gpkg=out_gpkg_water,
            change_gpkg=out_gpkg_change,
        )
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
    except Exception as e:
        audit("run_failed", run_id=run_id, error=str(e), error_type=type(e).__name__)
        raise
    else:
        audit("run_completed")


if __name__ == "__main__":
    main()