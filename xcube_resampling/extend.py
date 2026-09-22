# The MIT License (MIT)
# Copyright (c) 2025 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .constants import FloatInt
from .utils import _grid_spacing, _validate_bbox, clip_dataset_by_bbox


def extend_dataset(
    ds: xr.Dataset,
    bbox: Sequence[FloatInt],
    *,
    x_dim: str | None = "x",
    y_dim: str | None = "y",
    tile_size: tuple[int, int] | None = None,
) -> xr.Dataset:
    """Extend a dataset to cover the requested bounding box.

    The dataset is assumed to have regular one-dimensional spatial coordinates.
    The requested bounding box may differ from the dataset extent by less than
    one pixel. Missing pixels are padded with NaN.

    Args:
        bbox: Bounding box `(xmin, ymin, xmax, ymax)` in the same CRS.
        ds: Dataset with `x_dim` and `y_dim` coordinates in a given CRS.
        x_dim: Optional name of the horizontal coordinate. Defaults to ``"x"``.
        y_dim: Optional name of the vertical coordinate. Defaults to ``"y"``.
        tile_size: Optional spatial output chunk size as `(x, y)`.

    Returns:
        Dataset extended to cover `bbox`.
    """
    bbox = _validate_bbox(bbox)
    if x_dim not in ds.coords or y_dim not in ds.coords:
        raise ValueError(
            f"First dataset must contain coordinates {x_dim!r} and {y_dim!r}."
        )
    if ds[x_dim].ndim != 1 or ds[y_dim].ndim != 1:
        raise ValueError(f"Only 1-D {x_dim!r} and {y_dim!r} coordinates are supported.")
    if ds.sizes[x_dim] < 2 or ds.sizes[y_dim] < 2:
        raise ValueError(
            f"{x_dim!r} and {y_dim!r} coordinates must contain at least two values."
        )

    x = ds[x_dim].values
    y = ds[y_dim].values
    x_res = _grid_spacing(x, x_dim)
    y_res = _grid_spacing(y, y_dim)
    y_increasing = y_res > 0
    y_res = abs(y_res)

    # Number of pixels to add at each side.
    xmin, ymin, xmax, ymax = bbox
    y_max = max(y[0], y[-1])
    y_min = min(y[0], y[-1])
    x_min = x[0]
    x_max = x[-1]

    nx_left = max(0, int(np.ceil((x_min - xmin) / x_res)) - 1)
    nx_right = max(0, int(np.ceil((xmax - x_max) / x_res)) - 1)
    ny_bottom = max(0, int(np.ceil((y_min - ymin) / y_res)) - 1)
    ny_top = max(0, int(np.ceil((ymax - y_max) / y_res)) - 1)

    if nx_left == nx_right == ny_bottom == ny_top == 0:
        ds = clip_dataset_by_bbox(ds, bbox, spatial_coords=(x_dim, y_dim))
        if tile_size is not None:
            ds = ds.chunk({x_dim: tile_size[0], y_dim: tile_size[1]})
        return ds

    # Build the target coordinate vectors by extending the existing ones
    x_start = x_min - x_res * nx_left
    x_end = x_max + x_res * nx_right
    y_start = y_min - y_res * ny_bottom
    y_end = y_max + y_res * ny_top

    new_x = np.arange(x_start, x_end + (x_res / 2), x_res)
    new_y = np.arange(y_start, y_end + (y_res / 2), y_res)
    if not y_increasing:
        new_y = new_y[::-1]

    # Allow for small floating-point errors when matching the original grid.
    extended = ds.reindex(
        {x_dim: new_x, y_dim: new_y},
        method="nearest",
        tolerance=min(x_res, y_res) * 1e-3,
    )
    extended = clip_dataset_by_bbox(extended, bbox, spatial_coords=(x_dim, y_dim))
    if tile_size is not None:
        extended = extended.chunk({x_dim: tile_size[0], y_dim: tile_size[1]})

    return extended
