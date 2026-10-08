import sys
from pathlib import Path

import numpy as np
import rasterio


def inspect(path: Path) -> None:
    with rasterio.open(path) as src:
        print(f"File:        {path.name}")
        print(f"Driver:      {src.driver}")
        print(f"Size:        {src.width} x {src.height} pixels")
        print(f"Bands:       {src.count}")
        print(f"Data type:   {src.dtypes[0]}")
        print(f"CRS:         {src.crs}")
        print(f"Bounds:      {src.bounds}")
        print(f"Resolution:  {src.res}")
        print(f"NoData:      {src.nodata}")

        nodata = src.nodata
        total_px = src.width * src.height
        valid_px = 0
        elev_sum = 0.0
        elev_min = np.inf
        elev_max = -np.inf

        # Read one block at a time so memory use stays small.
        for _, window in src.block_windows(1):
            block = src.read(1, window=window).astype(np.float64)
            if nodata is not None:
                mask = block != nodata
            else:
                mask = np.isfinite(block)
            values = block[mask]
            if values.size == 0:
                continue
            valid_px += values.size
            elev_sum += values.sum()
            elev_min = min(elev_min, values.min())
            elev_max = max(elev_max, values.max())

        print(f"Min elev:    {elev_min:.1f} m")
        print(f"Max elev:    {elev_max:.1f} m")
        print(f"Mean elev:   {elev_sum / valid_px:.1f} m")
        print(f"Valid px:    {valid_px} of {total_px}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/inspect_dem.py <path-to-tif>")
        sys.exit(1)
    inspect(Path(sys.argv[1]))