from __future__ import annotations

import datetime as dt
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from zoneinfo import ZoneInfo

import pyart
from netCDF4 import num2date

import cartopy.crs as ccrs  # required by Py-ART map display


# ------------------------------ CONFIG ------------------------------ #

SITE = "KBOX"
BUCKET_BASE = "https://unidata-nexrad-level2.s3.amazonaws.com"

# Extent tuned for Southern New England / Cape Cod visibility
EXTENT = {
    "min_lon": -74.6,
    "max_lon": -69.0,
    "min_lat": 40.2,
    "max_lat": 43.3,
}

# Output sizing
OUT_DIR = Path("output")
OUT_W = 1600
OUT_H = 1600
DPI = 200
FIGSIZE_IN = (OUT_W / DPI, OUT_H / DPI)  # 8x8 at 200 dpi -> 1600x1600

# Overlay PNG (drawn ON TOP of radar)
OVERLAY_PATH = Path("overlays") / "sne_states_with_dbz.png"

# Plot settings
FIELD = "reflectivity"
VMIN = 0.0
VMAX = 75.0
CMAP = "pyart_NWSRef"  # NWSRef colormap name in Py-ART

# Branding text
BRAND_LINE1 = "KBOX Local Radar"
BRAND_LINE2 = "CapeCodWeather.net"

# Eastern Time with EST/EDT automatically
TZ_LOCAL = ZoneInfo("America/New_York")


# ------------------------------ HELPERS ------------------------------ #

def utc_today_prefix() -> str:
    """Return YYYY/MM/DD/SITE/ for today's UTC date."""
    now_utc = dt.datetime.now(dt.timezone.utc)
    return f"{now_utc:%Y/%m/%d}/{SITE}/"


def list_s3_keys(prefix: str) -> List[str]:
    """
    List keys under the public S3 bucket prefix using XML listing.
    Handles pagination via IsTruncated/NextMarker.
    """
    keys: List[str] = []
    marker = None

    while True:
        params = {"prefix": prefix}
        if marker:
            params["marker"] = marker

        r = requests.get(BUCKET_BASE, params=params, timeout=30)
        r.raise_for_status()

        root = ET.fromstring(r.text)

        def findall(tag: str):
            return root.findall(f".//{{*}}{tag}")

        for c in findall("Contents"):
            k = c.findtext("{*}Key")
            if k:
                keys.append(k)

        is_truncated_text = root.findtext(".//{*}IsTruncated")
        is_truncated = (is_truncated_text or "").strip().lower() == "true"

        if not is_truncated:
            break

        next_marker = root.findtext(".//{*}NextMarker")
        marker = next_marker if next_marker else (keys[-1] if keys else None)
        if not marker:
            break

    return keys


_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def key_timestamp_utc(key: str) -> dt.datetime | None:
    """Extract UTC timestamp from key names containing YYYYMMDD_HHMMSS."""
    m = _TS_RE.search(key)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    try:
        return dt.datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def filter_level2_keys(keys: List[str], prefix: str) -> List[Tuple[str, dt.datetime]]:
    """Filter keys that look like Level II archives, sortable by embedded timestamp."""
    items: List[Tuple[str, dt.datetime]] = []
    for k in keys:
        if not k.startswith(prefix):
            continue
        if k.endswith("/"):
            continue
        ts = key_timestamp_utc(k)
        if ts is None:
            continue
        if f"/{SITE}" not in k:
            continue
        items.append((k, ts))

    items.sort(key=lambda x: x[1])
    return items


def download_key_to_tmp(key: str, tmpdir: Path) -> Path:
    url = f"{BUCKET_BASE}/{key}"
    local = tmpdir / Path(key).name
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return local


def radar_valid_time_local(radar) -> str:
    """
    Use the last ray time as validity time.
    Returns: 'Valid: Dec 28, 2025 • 5:54 PM EST'
    """
    tdata = radar.time["data"]
    units = radar.time["units"]
    t_last = float(tdata[-1])

    dt_utc = num2date(t_last, units)

    if getattr(dt_utc, "tzinfo", None) is None:
        dt_utc = dt_utc.replace(tzinfo=dt.timezone.utc)
    else:
        dt_utc = dt_utc.astimezone(dt.timezone.utc)

    dt_local = dt_utc.astimezone(TZ_LOCAL)
    tzabbr = dt_local.tzname() or "ET"

    month = dt_local.strftime("%b")
    day = dt_local.day
    year = dt_local.year
    hour12 = dt_local.strftime("%I").lstrip("0") or "12"
    minute = dt_local.strftime("%M")
    ampm = dt_local.strftime("%p")

    return f"Valid: {month} {day}, {year} • {hour12}:{minute} {ampm} {tzabbr}"


def add_text_branding(ax, valid_line: str) -> None:
    outline = [pe.withStroke(linewidth=3, foreground="black", alpha=0.85)]

    ax.text(
        0.02, 0.98,
        f"{BRAND_LINE1}\n{BRAND_LINE2}",
        transform=ax.transAxes,
        ha="left", va="top",
        color="white",
        fontsize=18,
        fontweight="bold",
        path_effects=outline,
        zorder=20,
    )

    ax.text(
        0.98, 0.02,
        valid_line,
        transform=ax.transAxes,
        ha="right", va="bottom",
        color="white",
        fontsize=16,
        path_effects=outline,
        zorder=20,
    )


def render_reflectivity_png(level2_path: Path, out_path: Path) -> None:
    radar = pyart.io.read_nexrad_archive(str(level2_path))

    if FIELD not in radar.fields:
        raise RuntimeError(f"Field '{FIELD}' not found in {level2_path.name}. Available: {list(radar.fields.keys())}")

    valid_line = radar_valid_time_local(radar)

    projection = ccrs.Mercator()

    fig = plt.figure(figsize=FIGSIZE_IN, dpi=DPI)
    fig.patch.set_facecolor("black")

    ax = plt.axes(projection=projection)
    ax.set_facecolor("black")

    # zorder stability fix
    ax.set_zorder(0)

    display = pyart.graph.RadarMapDisplay(radar)

    display.plot_ppi_map(
        FIELD,
        sweep=0,
        vmin=VMIN,
        vmax=VMAX,
        cmap=CMAP,
        projection=projection,
        ax=ax,
        colorbar_flag=False,
        title_flag=False,
        min_lon=EXTENT["min_lon"],
        max_lon=EXTENT["max_lon"],
        min_lat=EXTENT["min_lat"],
        max_lat=EXTENT["max_lat"],
        lat_lines=None,
        lon_lines=None,
        resolution="10m",
    )

    # Try to keep radar underneath overlay
    if ax.collections:
        try:
            ax.collections[-1].set_zorder(1)
        except Exception:
            pass

    # Remove axes junk
    try:
        ax.outline_patch.set_visible(False)
    except Exception:
        pass
    ax.set_xticks([])
    ax.set_yticks([])

    # Overlay PNG
    if not OVERLAY_PATH.exists():
        raise FileNotFoundError(f"Overlay not found: {OVERLAY_PATH}")
    overlay = plt.imread(str(OVERLAY_PATH))
    ax.imshow(
        overlay,
        origin="upper",
        transform=ax.transAxes,
        extent=(0, 1, 0, 1),
        zorder=10,
    )

    add_text_branding(ax, valid_line)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=DPI, bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    prefix = utc_today_prefix()
    print(f"[INFO] Listing keys for prefix: {prefix}")

    keys = list_s3_keys(prefix)
    items = filter_level2_keys(keys, prefix)

    if len(items) < 12:
        raise RuntimeError(f"Found only {len(items)} timestamped keys for {prefix}; need 12.")

    latest_12 = items[-12:]  # ascending
    latest_12_keys = [k for (k, _ts) in latest_12]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for idx, key in enumerate(latest_12_keys):
            frame_name = f"radar_{idx:02d}.png"
            out_path = OUT_DIR / frame_name

            print(f"[INFO] ({idx+1}/12) Downloading: {key}")
            local = download_key_to_tmp(key, tmpdir)

            print(f"[INFO] Rendering: {frame_name}")
            render_reflectivity_png(local, out_path)

    print("[OK] Wrote 12 frames to output/")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise

