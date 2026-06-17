from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pyart
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RADAR = "KBOX"
BASE_URL = "https://unidata-nexrad-level2.s3.amazonaws.com"
MAX_FRAMES = 12

IMAGE_PX = 2400
DPI = 100
MAX_RANGE_KM = 230.0

DBZ_MIN = 0
DBZ_MAX = 75

WORK_DIR = Path("render")
OVERLAY_PATH = Path("overlays/sne_states_with_dbz.png")
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def list_keys(session: requests.Session, prefix: str) -> list[str]:
    response = session.get(f"{BASE_URL}/", params={"prefix": prefix}, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    return [node.text for node in root.findall("s3:Contents/s3:Key", S3_NS) if node.text]


def find_latest_keys(session: requests.Session) -> list[str]:
    today = datetime.now(ZoneInfo("UTC")).date()
    prefixes = [f"{today - timedelta(days=days):%Y/%m/%d}/{RADAR}/" for days in range(3)]

    keys: list[str] = []
    for prefix in prefixes:
        try:
            keys.extend(
                key
                for key in list_keys(session, prefix)
                if "_MDM" not in key and (key.endswith("_V06") or key.endswith("_V07"))
            )
        except Exception as exc:  # Keep one bad listing from killing the whole run.
            print(f"Warning: failed to list {prefix}: {exc}")

    return sorted(set(keys))[-MAX_FRAMES:]


def timestamp_from_key(key: str) -> str:
    match = re.search(r"(\d{8})_(\d{6})", key)
    if not match:
        return "Time unknown"

    ymd, hms = match.groups()
    dt_utc = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo("UTC"))
    dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
    return dt_et.strftime("%b %d, %Y %I:%M %p ET")


def download_scan(session: requests.Session, key: str, destination: Path) -> None:
    with session.get(f"{BASE_URL}/{key}", stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as scan_file:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    scan_file.write(chunk)


def render_scan(key: str, scan_path: Path, output_path: Path, overlay: Image.Image) -> None:
    radar = pyart.io.read_nexrad_archive(scan_path)
    if "reflectivity" not in radar.fields:
        raise ValueError("scan has no reflectivity field")

    scan_slice = radar.get_slice(0)
    reflectivity = radar.fields["reflectivity"]["data"][scan_slice]
    azimuth = np.deg2rad(radar.azimuth["data"][scan_slice])
    range_km = radar.range["data"] / 1000.0

    x = np.outer(np.sin(azimuth), range_km)
    y = np.outer(np.cos(azimuth), range_km)

    cmap = pyart.graph.cm.NWSRef
    norm = mpl.colors.Normalize(vmin=DBZ_MIN, vmax=DBZ_MAX)

    fig, ax = plt.subplots(figsize=(IMAGE_PX / DPI, IMAGE_PX / DPI), dpi=DPI)
    ax.set_position([0, 0, 1, 1])

    ax.pcolormesh(x, y, reflectivity, cmap=cmap, norm=norm, shading="auto", zorder=1)
    ax.imshow(
        overlay,
        extent=[-MAX_RANGE_KM, MAX_RANGE_KM, -MAX_RANGE_KM, MAX_RANGE_KM],
        interpolation="nearest",
        zorder=10,
    )
    ax.text(
        0.99,
        0.01,
        timestamp_from_key(key),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=20,
        color="white",
        path_effects=[pe.withStroke(linewidth=4, foreground="black")],
        zorder=20,
    )

    ax.set_xlim(-MAX_RANGE_KM, MAX_RANGE_KM)
    ax.set_ylim(-MAX_RANGE_KM, MAX_RANGE_KM)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(output_path, dpi=DPI, pad_inches=0)
    plt.close(fig)


def main() -> None:
    if not OVERLAY_PATH.exists():
        raise FileNotFoundError(f"Missing overlay: {OVERLAY_PATH}")

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    overlay = Image.open(OVERLAY_PATH).convert("RGBA")
    overlay = overlay.resize((IMAGE_PX, IMAGE_PX), Image.LANCZOS)

    session = build_session()
    keys = find_latest_keys(session)
    if not keys:
        raise RuntimeError(f"No {RADAR} radar scans found")

    if len(keys) < MAX_FRAMES:
        print(f"Warning: found only {len(keys)} scans; rendering what is available")

    rendered = 0
    for key in keys:
        scan_path = WORK_DIR / "scan"
        output_path = WORK_DIR / f"radar_{rendered:02d}.png"
        try:
            print(f"Rendering {key}")
            download_scan(session, key, scan_path)
            render_scan(key, scan_path, output_path, overlay)
            rendered += 1
        except Exception as exc:
            print(f"Warning: skipped {key}: {exc}")
        finally:
            scan_path.unlink(missing_ok=True)

    if rendered == 0:
        raise RuntimeError("No radar frames rendered")

    print(f"Radar render complete: {rendered} frame(s).")


if __name__ == "__main__":
    main()
