"""Build XYZ map tile pyramids for the layers listed in apps/web/public/layers.json.

Run from the pipeline folder with the venv active:

    python src/tile_raster.py                        # every layer whose source file exists
    python src/tile_raster.py --layer elevation_shaded

Output goes straight into the web app:

    apps/web/public/tiles/<layer-id>/<z>/<x>/<y>.png
    apps/web/public/tiles/<layer-id>/manifest.json
    data/processed/previews/<layer-id>.png   (overview for a quick visual check)

Zoom level z has 2**(z+1) columns and 2**z rows of tiles over the whole planet.
Each tile is resampled from its source raster, so any source resolution works,
and regional sources land on the same global grid.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "apps" / "web" / "public" / "layers.json"
RAW_DIR = REPO_ROOT / "data" / "raw"
TILE_ROOT = REPO_ROOT / "apps" / "web" / "public" / "tiles"
PREVIEW_DIR = REPO_ROOT / "data" / "processed" / "previews"

PREVIEW_ZOOM = 3
STRETCH_MAX_PX = 2048  # imagery contrast is measured on a downsampled copy


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


# ---------- tile geometry ----------

def level_resolution(cfg, z):
    """Metres per pixel at zoom z."""
    left, _, right, _ = cfg["global_extent_m"]
    return (right - left) / (2 ** (z + 1) * cfg["tile_size"])


def tile_bounds(cfg, z, x, y):
    """(left, bottom, right, top) in metres for tile x, y at zoom z. y=0 is the north edge."""
    left, _, _, top = cfg["global_extent_m"]
    size = cfg["tile_size"] * level_resolution(cfg, z)
    t_left = left + x * size
    t_top = top - y * size
    return (t_left, t_top - size, t_left + size, t_top)


# ---------- reading ----------

def read_resampled(src, bounds, size):
    """Read bounds from src as a size x size float32 array, resampled by area averaging.

    Pixels outside the source raster are NaN. Also returns whether the bounds
    overlap the source at all, so callers can skip empty tiles.
    """
    left, bottom, right, top = bounds
    s_left, s_bottom, s_right, s_top = src.bounds
    il, ir = max(left, s_left), min(right, s_right)
    ib, it = max(bottom, s_bottom), min(top, s_top)
    out = np.full((size, size), np.nan, dtype=np.float32)
    if il >= ir or ib >= it:
        return out, False

    px = (right - left) / size
    col0 = round((il - left) / px)
    row0 = round((top - it) / px)
    cols = min(max(1, round((ir - il) / px)), size - col0)
    rows = min(max(1, round((it - ib) / px)), size - row0)

    window = from_bounds(il, ib, ir, it, transform=src.transform)
    data = src.read(
        1, window=window, out_shape=(rows, cols), resampling=Resampling.average
    ).astype(np.float32)
    if src.nodata is not None:
        data[data == src.nodata] = np.nan
    out[row0:row0 + rows, col0:col0 + cols] = data
    return out, True


def fill_outside_halo(a):
    """The one-pixel halo around a tile sits outside the planet at the seams and poles.
    Copy the nearest inside row or column there, so shading has no false cliff."""
    if np.isnan(a[:, 0]).any():
        a[:, 0] = a[:, 1]
    if np.isnan(a[:, -1]).any():
        a[:, -1] = a[:, -2]
    if np.isnan(a[0, :]).any():
        a[0, :] = a[1, :]
    if np.isnan(a[-1, :]).any():
        a[-1, :] = a[-2, :]
    return a


# ---------- rendering ----------

def shade_elevation(elev, res, layer):
    """Hillshade plus colour ramp. elev is (tile + 2) square with a one-pixel halo."""
    d_row, d_col = np.gradient(elev, res, res)
    nx = -d_col
    ny = d_row
    norm = np.sqrt(nx ** 2 + ny ** 2 + 1.0)

    sh = layer["shading"]
    az = math.radians(sh["azimuth_deg"])
    alt = math.radians(sh["altitude_deg"])
    lx = math.sin(az) * math.cos(alt)
    ly = math.cos(az) * math.cos(alt)
    lz = math.sin(alt)
    shade = np.clip((nx * lx + ny * ly + lz) / norm, 0.0, 1.0)

    elev = elev[1:-1, 1:-1]
    shade = shade[1:-1, 1:-1]

    stops = layer["colour_stops"]
    stops_m = np.array([s["elevation_m"] for s in stops], dtype=np.float64)
    stops_rgb = np.array([s["rgb"] for s in stops], dtype=np.float64)
    rgb = np.stack(
        [np.interp(elev, stops_m, stops_rgb[:, ch]) for ch in range(3)], axis=-1
    )
    rgb *= (sh["ambient"] + (1.0 - sh["ambient"]) * shade)[..., None]

    rgba = np.empty(elev.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return Image.fromarray(rgba, "RGBA")


def contrast_stretch(src):
    """Low and high values for imagery, measured on a downsampled copy."""
    scale = min(1.0, STRETCH_MAX_PX / max(src.width, src.height))
    oh = max(1, round(src.height * scale))
    ow = max(1, round(src.width * scale))
    small = src.read(1, out_shape=(oh, ow), resampling=Resampling.average).astype(np.float32)
    if src.nodata is not None:
        small[small == src.nodata] = np.nan
    lo, hi = np.nanpercentile(small, [2, 98])
    return float(lo), float(hi)


def render_imagery(vals, stretch):
    lo, hi = stretch
    valid = np.isfinite(vals)
    scaled = np.clip((vals - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    grey = (np.nan_to_num(scaled, nan=0.0) * 255 + 0.5).astype(np.uint8)

    rgba = np.empty(vals.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = grey[..., None]
    rgba[..., 3] = np.where(valid, 255, 0)
    return Image.fromarray(rgba, "RGBA")


def render_tile(src, cfg, layer, z, x, y, stretch):
    """Return the tile image, or None if the source has no data for it."""
    ts = cfg["tile_size"]
    res = level_resolution(cfg, z)
    left, bottom, right, top = tile_bounds(cfg, z, x, y)

    if layer["kind"] == "elevation":
        halo = (left - res, bottom - res, right + res, top + res)
        elev, has_data = read_resampled(src, halo, ts + 2)
        if not has_data:
            return None
        elev = fill_outside_halo(elev)
        elev = np.nan_to_num(elev, nan=0.0).astype(np.float64)
        return shade_elevation(elev, res, layer)

    vals, has_data = read_resampled(src, (left, bottom, right, top), ts)
    if not has_data:
        return None
    return render_imagery(vals, stretch)


# ---------- output ----------

def make_preview(cfg, layer, layer_dir):
    z = min(PREVIEW_ZOOM, layer["max_zoom"])
    cols, rows = 2 ** (z + 1), 2 ** z
    ts = cfg["tile_size"]
    canvas = Image.new("RGBA", (cols * ts, rows * ts), (0, 0, 0, 0))
    for x in range(cols):
        for y in range(rows):
            path = layer_dir / str(z) / str(x) / f"{y}.png"
            if path.exists():
                canvas.paste(Image.open(path).convert("RGBA"), (x * ts, y * ts))
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = PREVIEW_DIR / f"{layer['id']}.png"
    canvas.save(out, optimize=True)
    return out


def build_layer(cfg, layer):
    src_path = RAW_DIR / layer["source_file"]
    if not src_path.exists():
        print(f"  skipped: {src_path.name} not found in data/raw")
        return

    layer_dir = TILE_ROOT / layer["id"]
    layer_dir.mkdir(parents=True, exist_ok=True)
    min_z, max_z = layer["min_zoom"], layer["max_zoom"]
    total = sum(2 ** (2 * z + 1) for z in range(min_z, max_z + 1))
    levels = {}
    written = 0

    with rasterio.open(src_path) as src:
        stretch = contrast_stretch(src) if layer["kind"] == "imagery" else None
        crs = str(src.crs)
        coverage = list(src.bounds)

        with tqdm(total=total, unit="tile", desc=layer["id"]) as bar:
            for z in range(min_z, max_z + 1):
                cols, rows = 2 ** (z + 1), 2 ** z
                levels[str(z)] = {
                    "columns": cols,
                    "rows": rows,
                    "metres_per_pixel": round(level_resolution(cfg, z), 2),
                }
                for x in range(cols):
                    col_dir = layer_dir / str(z) / str(x)
                    col_dir.mkdir(parents=True, exist_ok=True)
                    for y in range(rows):
                        img = render_tile(src, cfg, layer, z, x, y, stretch)
                        if img is not None:
                            img.save(col_dir / f"{y}.png", optimize=True)
                            written += 1
                        bar.update(1)

    manifest = {
        "id": layer["id"],
        "name": layer["name"],
        "kind": layer["kind"],
        "source_file": layer["source_file"],
        "source_credit": layer["source_credit"],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crs": crs,
        "tile_size": cfg["tile_size"],
        "global_extent_m": cfg["global_extent_m"],
        "coverage_m": coverage,
        "tile_url_template": f"/tiles/{layer['id']}/{{z}}/{{x}}/{{y}}.png",
        "min_zoom": min_z,
        "max_zoom": max_z,
        "tiles_written": written,
        "levels": levels,
    }
    (layer_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    preview = make_preview(cfg, layer, layer_dir)
    print(f"  tiles written: {written}")
    print(f"  preview:       {preview}")


def main():
    parser = argparse.ArgumentParser(description="Build map tile pyramids from layers.json.")
    parser.add_argument("--layer", help="Build only this layer id (default: all layers)")
    args = parser.parse_args()

    cfg = load_config()
    layers = cfg["layers"]
    if args.layer:
        layers = [layer for layer in layers if layer["id"] == args.layer]
        if not layers:
            raise SystemExit(f"No layer with id '{args.layer}' in {CONFIG_PATH.name}")

    for layer in layers:
        print(f"\n{layer['id']} (zoom {layer['min_zoom']}-{layer['max_zoom']})")
        build_layer(cfg, layer)
    print("\nDone.")


if __name__ == "__main__":
    main()