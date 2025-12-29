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
FIGSIZE_IN = (8, 8)  # 1600x1600

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

        if root.findtext(".//{*}IsTruncated") != "true":
            break

        marker = root.findtext(".//{*}NextMarker") or keys[-1]

    return keys


_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def key_timestamp_utc(key: str) -> dt.datetime | None:
    if "_MDM" in key or "_V06" not in key:
        return None
    m = _TS_RE.search(key)
    if not m:
        return None
    return dt.datetime.strptime(
        m.group(1) + m.group(2),
        "%Y%m%d%H%M%S"
    ).replace(tzinfo=dt.timezone.utc)


def collect_recent_keys() -> List[Tuple[str, dt.datetime]]:
    now = dt.datetime.now(dt.timezone.utc)
    days = [now.date(), now.date() - dt.timedelta(days=1)]

    items: List[Tuple[str, dt.datetime]] = []
    for d in days:
        prefix = date_prefix(d)
        for k in list_s3_keys(prefix):
            ts = key_timestamp_utc(k)
            if ts:
                items.append((k, ts))

    items.sort(key=lambda x: x[1])
    return items


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
    t = num2date(radar.time["data"][-1], radar.time["units"])
    dt_utc = dt.datetime(
        t.year, t.month, t.day, t.hour, t.minute, int(t.second),
        tzinfo=dt.timezone.utc
    )
    dt_local = dt_utc.astimezone(TZ_LOCAL)
    return dt_local.strftime("Valid: %b %d, %Y • %I:%M %p %Z").replace(" 0", " ")


def add_labels(ax, valid_line: str):
    stroke = [pe.withStroke(linewidth=2, foreground="black")]

    ax.text(
        0.02, 0.98,
        f"{BRAND_LINE1}\n{BRAND_LINE2}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        color="white",
        path_effects=stroke,
        zorder=20,
    )

    ax.text(
        0.98, 0.02,
        valid_line,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        color="white",
        path_effects=stroke,
        zorder=20,
    )


def render_png(level2: Path, out: Path):
    try:
        radar = pyart.io.read_nexrad_archive(str(level2))
    except OSError:
        return False

    fig = plt.figure(figsize=FIGSIZE_IN, dpi=DPI)
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent(
        [EXTENT["min_lon"], EXTENT["max_lon"],
         EXTENT["min_lat"], EXTENT["max_lat"]],
        crs=ccrs.PlateCarree()
    )
    ax.axis("off")

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
        lat_lines=None,
        lon_lines=None,
    )

    overlay = plt.imread(OVERLAY_PATH)
    ax.imshow(
        overlay,
        transform=ccrs.PlateCarree(),
        extent=(
            EXTENT["min_lon"], EXTENT["max_lon"],
            EXTENT["min_lat"], EXTENT["max_lat"],
        ),
        zorder=10,
    )

    add_labels(ax, radar_valid_time_local(radar))

    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return True


# ---------------- MAIN ---------------- #

def main():
    items = collect_recent_keys()
    OUT_DIR.mkdir(exist_ok=True)

    written = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for key, _ in reversed(items):
            if written == 12:
                break
            local = download_key(key, td)
            if render_png(local, OUT_DIR / f"radar_{11-written:02d}.png"):
                written += 1

    if written < 12:
        raise RuntimeError("Could not generate 12 frames")

    print("[OK] Generated 12 aligned frames")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
