"""Confirm the tile preview is north-up and east-right by comparing
known Mars locations in the raw DEM against the same spots in the preview."""

import math
from pathlib import Path

import rasterio
from PIL import Image
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw" / "Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
PREVIEW = REPO_ROOT / "data" / "processed" / "previews" / "elevation_shaded.png"

MARS_RADIUS_M = 3396190.0

# name, longitude (east-positive, degrees), latitude (degrees)
SITES = [
    ("Olympus Mons summit", -133.8, 18.7),
    ("Valles Marineris", -60.0, -10.0),
    ("Hellas Basin floor", 70.0, -42.0),
]


def to_metres(lon, lat):
    return (
        math.radians(lon) * MARS_RADIUS_M,
        math.radians(lat) * MARS_RADIUS_M,
    )


def main():
    preview = Image.open(PREVIEW).convert("RGB")
    pw, ph = preview.size

    with rasterio.open(RAW) as src:
        print(f"{'Site':<22}{'Raw elev (m)':>14}{'Preview RGB':>22}")
        print("-" * 58)
        for name, lon, lat in SITES:
            x, y = to_metres(lon, lat)
            row, col = src.index(x, y)
            elev = float(src.read(1, window=Window(col, row, 1, 1))[0, 0])

            px = int((lon + 180.0) / 360.0 * pw)
            py = int((90.0 - lat) / 180.0 * ph)
            rgb = preview.getpixel((min(px, pw - 1), min(py, ph - 1)))

            print(f"{name:<22}{elev:>14.0f}{str(rgb):>22}")


if __name__ == "__main__":
    main()