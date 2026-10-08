"""Render MOLA elevation as a hillshaded, colour-ramped XYZ tile pyramid.

Output layout (standard XYZ, y=0 at the north edge):
    <output>/<z>/<x>/<y>.png
    <output>/manifest.json
    <output>/preview_z<n>.png   (a stitched overview for a quick visual check)

The global map is 2:1 (equirectangular), so zoom level z has
2**(z+1) columns and 2**z rows of 256 px tiles.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

TILE_SIZE = 256
SUN_AZIMUTH_DEG = 315.0  # clockwise from north; 315 = north-west
SUN_ALTITUDE_DEG = 45.0
AMBIENT = 0.3  # shadowed slopes keep 30% brightness so they never go black

# Elevation in metres -> RGB. Mars has no water, so 0 m is just a reference level.
COLOUR_STOPS = [
    (-8000, 52, 36, 62),
    (-4000, 98, 58, 60),
    (-2000, 140, 74, 52),
    (0, 178, 104, 68),
    (2000, 204, 136, 92),
    (5000, 222, 170, 126),
    (10000, 234, 205, 176),
    (15000, 242, 230, 214),
    (21500, 252, 248, 240),
]
STOPS_M = np.array([s[0] for s in COLOUR_STOPS], dtype=np.float64)
STOPS_RGB = np.array([s[1:] for s in COLOUR_STOPS], dtype=np.float64)


def read_padded(src, col0, row0, ncols, nrows):
    """Read a window that may extend past the raster edges.

    Columns wrap around the globe (longitude is continuous).
    Rows are edge-padded at the poles.
    """
    width, height = src.width, src.height
    out = np.empty((nrows, ncols), dtype=np.float32)

    r_lo = max(row0, 0)
    r_hi = min(row0 + nrows, height)
    top = r_lo - row0
    n_rows = r_hi - r_lo

    filled = 0
    c = col0 % width
    while filled < ncols:
        take = min(width - c, ncols - filled)
        window = Window(c, r_lo, take, n_rows)
        out[top:top + n_rows, filled:filled + take] = src.read(1, window=window)
        filled += take
        c = 0

    if top > 0:
        out[:top] = out[top:top + 1]
    bottom = top + n_rows
    if bottom < nrows:
        out[bottom:] = out[bottom - 1:bottom]
    return out


def render_tile(src, z, x, y):
    cols, rows = 2 ** (z + 1), 2 ** z
    span_x = src.width // cols   # source pixels covered by one tile, horizontally
    span_y = src.height // rows  # source pixels covered by one tile, vertically

    # Read a 1-output-pixel halo on every side so the shading has neighbours at the tile edges.
    halo_x = math.ceil(span_x / TILE_SIZE)
    halo_y = math.ceil(span_y / TILE_SIZE)
    elev = read_padded(
        src,
        x * span_x - halo_x,
        y * span_y - halo_y,
        span_x + 2 * halo_x,
        span_y + 2 * halo_y,
    )
    if src.nodata is not None:
        elev[elev == src.nodata] = 0.0

    # Downsample to (TILE_SIZE + 2) with area averaging.
    ring = TILE_SIZE + 2
    elev = np.asarray(
        Image.fromarray(elev).resize((ring, ring), Image.Resampling.BOX),
        dtype=np.float64,
    )

    # Ground distance per output pixel, in metres.
    px_m_x = src.res[0] * span_x / TILE_SIZE
    px_m_y = src.res[1] * span_y / TILE_SIZE
    d_row, d_col = np.gradient(elev, px_m_y, px_m_x)

    # Surface normal in (east, north, up). Row index grows southward,
    # so the north-facing slope is -d/d(row).
    nx = -d_col
    ny = d_row
    norm = np.sqrt(nx ** 2 + ny ** 2 + 1.0)

    az = math.radians(SUN_AZIMUTH_DEG)
    alt = math.radians(SUN_ALTITUDE_DEG)
    lx = math.sin(az) * math.cos(alt)
    ly = math.cos(az) * math.cos(alt)
    lz = math.sin(alt)
    shade = np.clip((nx * lx + ny * ly + lz) / norm, 0.0, 1.0)

    # Drop the halo ring.
    elev = elev[1:-1, 1:-1]
    shade = shade[1:-1, 1:-1]

    rgb = np.stack(
        [np.interp(elev, STOPS_M, STOPS_RGB[:, ch]) for ch in range(3)],
        axis=-1,
    )
    rgb *= (AMBIENT + (1.0 - AMBIENT) * shade)[..., None]
    return Image.fromarray(np.clip(rgb + 0.5, 0, 255).astype(np.uint8), "RGB")


def make_preview(out_dir, z):
    cols, rows = 2 ** (z + 1), 2 ** z
    canvas = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    for x in range(cols):
        for y in range(rows):
            tile = Image.open(out_dir / str(z) / str(x) / f"{y}.png")
            canvas.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))
    path = out_dir / f"preview_z{z}.png"
    canvas.save(path, optimize=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="Tile MOLA DEM into a shaded XYZ pyramid.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "Mars_MGS_MOLA_DEM_mosaic_global_463m.tif",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "tiles",
    )
    parser.add_argument("--min-zoom", type=int, default=0)
    parser.add_argument("--max-zoom", type=int, default=6)
    args = parser.parse_args()

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.input) as src:
        for z in range(args.min_zoom, args.max_zoom + 1):
            cols, rows = 2 ** (z + 1), 2 ** z
            if src.width % cols or src.height % rows:
                raise SystemExit(
                    f"Zoom {z} does not divide the raster evenly "
                    f"({src.width} x {src.height} px). Pick a different --max-zoom."
                )

        total = sum(2 ** (2 * z + 1) for z in range(args.min_zoom, args.max_zoom + 1))
        levels = {}

        with tqdm(total=total, unit="tile") as bar:
            for z in range(args.min_zoom, args.max_zoom + 1):
                cols, rows = 2 ** (z + 1), 2 ** z
                span_x = src.width // cols
                levels[str(z)] = {
                    "columns": cols,
                    "rows": rows,
                    "metres_per_pixel": round(src.res[0] * span_x / TILE_SIZE, 2),
                }
                for x in range(cols):
                    col_dir = out_dir / str(z) / str(x)
                    col_dir.mkdir(parents=True, exist_ok=True)
                    for y in range(rows):
                        render_tile(src, z, x, y).save(col_dir / f"{y}.png", optimize=True)
                        bar.update(1)

        manifest = {
            "name": "Mars MOLA shaded elevation",
            "source_file": args.input.name,
            "source": "MOLA gridded DEM (Mars Global Surveyor), 463 m global mosaic, USGS",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "crs": str(src.crs),
            "tile_size": TILE_SIZE,
            "tile_scheme": "XYZ: y=0 at north. Zoom z has 2^(z+1) columns x 2^z rows.",
            "tile_url_template": "{z}/{x}/{y}.png",
            "min_zoom": args.min_zoom,
            "max_zoom": args.max_zoom,
            "bounds_m": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "levels": levels,
            "hillshade": {
                "azimuth_deg": SUN_AZIMUTH_DEG,
                "altitude_deg": SUN_ALTITUDE_DEG,
                "ambient": AMBIENT,
            },
            "colour_stops": [
                {"elevation_m": s[0], "rgb": list(s[1:])} for s in COLOUR_STOPS
            ],
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    preview_z = max(args.min_zoom, min(3, args.max_zoom))
    preview_path = make_preview(out_dir, preview_z)

    print(f"Tiles written:  {total}")
    print(f"Manifest:       {out_dir / 'manifest.json'}")
    print(f"Preview:        {preview_path}")


if __name__ == "__main__":
    main()