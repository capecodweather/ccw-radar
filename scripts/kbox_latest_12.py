from __future__ import annotations

import datetime as dt
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from zoneinfo import ZoneInfo

import pyart
from netCDF4 import num2date
import cartopy.crs as ccrs


# ---------------- CONFIG ---------------- #

SITE = "KBOX"
BUCKET_BASE = "https://unidata-nexrad-level2.s3.amazonaws.com"

EXTENT = {
    "min_lon": -74.6,
    "max_lon": -69.0,
    "min_lat": 40.2,
    "max_lat": 43.3,
}

OUT_DIR = Path("output")
DPI = 200
FIGSIZE_IN = (8, 8)

OVERLAY_PATH = Path("overlays") / "sne_states_with_dbz.png"

FIELD = "reflectivity"
VMIN = 0.0
VMAX = 75.0
CMAP = "NWSRef"

BRAND_LINE1 = "KBOX Local Radar"
BRAND_LINE2 = "CapeCodWeather.net"

TZ_LOCAL = ZoneInfo("America/New_York")


# ---------------- HELPERS ---------------- #

def date_prefix(date: dt.date) -> str:
    return f"{date:%Y/%m/%d}/{SITE}/"


def list_s3_keys(prefix: str) -> List[str]:
    keys = []
    marker = None

    while True:
        params = {"prefix": prefix}
        if marker:
            params["marker"] = marker

        r = requests.get(BUCKET_BASE, params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.text)

        for c in root.findall(".//{*}Contents"):
            k = c.findtext("{*}Key")
            if k:
                keys.append(k)

        truncated = root.findtext(".//{*}IsTruncated") == "true"
        if not truncated:
            break

        marker = root.findtext(".//{*}NextMarker") or keys[-1]

    return keys


_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def key_timestamp_utc(key: str) -> dt.datetime | None:
    m = _TS_RE.search(key)
    if not m:
        return None
    try:
        return dt.datetime.strptime(
            m.group(1) + m.group(2),
            "%Y%m%d%H%M%S"
        ).replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def collect_recent_keys() -> List[Tuple[str, dt.datetime]]:
    now_utc = dt.datetime.now(dt.timezone.utc)
    today = now_utc.date()
    yesterday = today - dt.timedelta(days=1)

    all_items: List[Tuple[str, dt.datetime]] = []

    for day in (today, yesterday):
        prefix = date_prefix(day)
        print(f"[INFO] Listing keys for prefix: {prefix}")
        for k in list_s3_keys(prefix):
            ts = key_timestamp_utc(k)
            if ts:
                all_items.append((k, ts))

    all_items.sort(key=lambda x: x[1])
    return all_items


def download_key(key: str, tmpdir: Path) -> Path:
    url = f"{BUCKET_BASE}/{key}"
    out = tmpdir / Path(key).name
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out


def radar_valid_time_local(radar) -> str:
    """
    SAFELY convert radar time (cftime or datetime) to Eastern Time string.
    """
    t_last = float(radar.time["data"][-1])
    dt_any = num2date(t_last, radar.time["units"])

    # Build real Python datetime explicitly (works for cftime)
    dt_utc = dt.datetime(
        dt_any.year,
        dt_any.month,
        dt_any.day,
        dt_any.hour,
        dt_any.minute,
        int(dt_any.second),
        tzinfo=dt.timezone.utc,
    )

    dt_local = dt_utc.astimezone(TZ_LOCAL)

    month = dt_local.strftime("%b")
    day = dt_local.day
    year = dt_local.year
    hour12 = dt_local.strftime("%I").lstrip("0") or "12"
    minute = dt_local.strftime("%M")
    ampm = dt_local.strftime("%p")
    tzabbr = dt_local.tzname() or "ET"

    return f"Valid: {month} {day}, {year} • {hour12}:{minute} {ampm} {tzabbr}"


def add_labels(ax, valid_line: str):
    peff = [pe.withStroke(linewidth=3, foreground="black", alpha=0.85)]

    ax.text(
        0.02, 0.98,
        f"{BRAND_LINE1}\n{BRAND_LINE2}",
        transform=ax.transAxes,
        ha="left", va="top",
        color="white",
        fontsize=18,
        fontweight="bold",
        path_effects=peff,
        zorder=20,
    )

    ax.text(
        0.98, 0.02,
        valid_line,
        transform=ax.transAxes,
        ha="right", va="bottom",
        color="white",
        fontsize=16,
        path_effects=peff,
        zorder=20,
    )


def render_png(level2: Path, out: Path):
    radar = pyart.io.read_nexrad_archive(str(level2))
    valid_line = radar_valid_time_local(radar)

    fig = plt.figure(figsize=FIGSIZE_IN, dpi=DPI)
    fig.patch.set_facecolor("black")

    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_facecolor("black")
    ax.set_zorder(0)

    display = pyart.graph.RadarMapDisplay(radar)
    display.plot_ppi_map(
        FIELD,
        0,
        vmin=VMIN,
        vmax=VMAX,
        cmap=CMAP,
        ax=ax,
        projection=ccrs.Mercator(),
        colorbar_flag=False,
        title_flag=False,
        min_lon=EXTENT["min_lon"],
        max_lon=EXTENT["max_lon"],
        min_lat=EXTENT["min_lat"],
        max_lat=EXTENT["max_lat"],
    )

    overlay = plt.imread(OVERLAY_PATH)
    ax.imshow(
        overlay,
        transform=ax.transAxes,
        extent=(0, 1, 0, 1),
        zorder=10,
    )

    add_labels(ax, valid_line)

    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------------- MAIN ---------------- #

def main():
    items = collect_recent_keys()

    if len(items) < 12:
        raise RuntimeError(f"Only found {len(items)} total scans")

    latest = items[-12:]

    OUT_DIR.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, (key, _) in enumerate(latest):
            print(f"[INFO] Rendering frame {i+1}/12: {key}")
            local = download_key(key, td)
            render_png(local, OUT_DIR / f"radar_{i:02d}.png")

    print("[OK] Generated 12 frames")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
