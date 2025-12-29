#!/usr/bin/env python3
"""
RADAR SANITY CHECK

Purpose:
- Prove that KBOX radar data is loading and drawable
- No Cartopy
- No overlay
- No timestamps
"""

import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import requests
import re
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyart


SITE = "KBOX"
BUCKET = "https://unidata-nexrad-level2.s3.amazonaws.com"
OUT = Path("output")
OUT.mkdir(exist_ok=True)

_TS = re.compile(r"(\d{8})_(\d{6})")


def latest_key():
    now = dt.datetime.utcnow()
    prefix = f"{now:%Y/%m/%d}/{SITE}/"
    r = requests.get(BUCKET, params={"prefix": prefix})
    r.raise_for_status()
    root = ET.fromstring(r.text)

    keys = []
    for c in root.findall(".//{*}Contents"):
        k = c.findtext("{*}Key")
        if k and "_V06" in k and "_MDM" not in k and _TS.search(k):
            keys.append(k)

    if not keys:
        raise RuntimeError("No Level II files found")

    return sorted(keys)[-1]


def main():
    key = latest_key()
    print("[INFO] Using:", key)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        local = td / "radar"
        with requests.get(f"{BUCKET}/{key}", stream=True) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)

        radar = pyart.io.read_nexrad_archive(str(local))

        fig = plt.figure(figsize=(8, 8), dpi=200)
        ax = plt.gca()
        ax.set_facecolor("black")

        display = pyart.graph.RadarDisplay(radar)
        display.plot(
            "reflectivity",
            sweep=0,
            vmin=0,
            vmax=75,
            cmap="NWSRef",
            ax=ax,
            colorbar_flag=True,
            title="KBOX Reflectivity (sanity check)",
        )

        plt.savefig(OUT / "sanity.png", dpi=200)
        plt.close(fig)

    print("[OK] Wrote output/sanity.png")


if __name__ == "__main__":
    main()
