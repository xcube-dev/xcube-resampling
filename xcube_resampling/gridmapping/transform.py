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

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from xcube_resampling.constants import LOG

from .base import DEFAULT_TOLERANCE, GridMapping
from .coords import new_grid_mapping_from_coords
from .helpers import (
    _assert_valid_xy_names,
    _normalize_crs,
)


def transform_grid_mapping(
    grid_mapping: GridMapping,
    crs: str | pyproj.crs.CRS,
    *,
    tile_size: int | tuple[int, int] = None,
    xy_var_names: tuple[str, str] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> GridMapping:
    target_crs = _normalize_crs(crs)
    if xy_var_names:
        _assert_valid_xy_names(xy_var_names, name="xy_var_names")

    source_crs = grid_mapping.crs
    if source_crs == target_crs:
        LOG.warning(
            "Skipping grid mapping transformation: source and target CRS "
            "are identical (%s).",
            source_crs,
        )
        if tile_size is not None or xy_var_names is not None:
            return grid_mapping.derive(tile_size=tile_size, xy_var_names=xy_var_names)
        return grid_mapping

    if grid_mapping.x_coords.ndim == 1:
        x = da.asarray(grid_mapping.x_coords.data).rechunk(grid_mapping.tile_width)
        y = da.asarray(grid_mapping.y_coords.data).rechunk(grid_mapping.tile_height)
        xx, yy = da.meshgrid(x, y)
    else:
        xx = da.asarray(grid_mapping.x_coords.data)
        yy = da.asarray(grid_mapping.y_coords.data)

    transformer = pyproj.transformer.Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )

    def transform_block(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        x, y = transformer.transform(xx, yy)
        return np.stack([x, y])

    xy_transformed = da.map_blocks(
        transform_block,
        xx,
        yy,
        dtype=np.float32,
        chunks=(2, *xx.chunks),
    )
    xx_transformed = xy_transformed[0]
    yy_transformed = xy_transformed[1]

    xy_var_names = xy_var_names or ("transformed_x", "transformed_y")

    if tile_size is None:
        tile_size = grid_mapping.tile_size

    yx_dim = (grid_mapping.xy_dim_names[1], grid_mapping.xy_dim_names[0])
    return new_grid_mapping_from_coords(
        x_coords=xr.DataArray(xx_transformed, dims=yx_dim, name=xy_var_names[0]),
        y_coords=xr.DataArray(yy_transformed, dims=yx_dim, name=xy_var_names[1]),
        crs=target_crs,
        tile_size=tile_size,
        tolerance=tolerance,
    )
