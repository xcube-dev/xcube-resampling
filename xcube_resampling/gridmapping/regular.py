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

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from xcube_resampling.constants import FloatInt

from .assertions import assert_true
from .base import GridMapping
from .helpers import (
    _default_xy_dim_names,
    _default_xy_var_names,
    _normalize_crs,
    _normalize_int_pair,
    _normalize_number_pair,
    _to_int_or_float,
    _round_sequence,
)


class RegularGridMapping(GridMapping):
    def __init__(self, **kwargs):
        kwargs.pop("is_regular", None)
        super().__init__(is_regular=True, **kwargs)
        self._xy_coords = None

    def _new_x_coords(self) -> xr.DataArray:
        self._assert_regular()
        return xr.DataArray(
            da.linspace(self.x_min, self.x_max, self.width, chunks=self.tile_width),
            dims=self.xy_dim_names[0],
        )

    def _new_y_coords(self) -> xr.DataArray:
        self._assert_regular()
        if self.is_j_axis_up:
            return xr.DataArray(
                da.linspace(
                    self.y_min, self.y_max, self.height, chunks=self.tile_height
                ),
                dims=self.xy_dim_names[1],
            )
        else:
            return xr.DataArray(
                da.linspace(
                    self.y_max, self.y_min, self.height, chunks=self.tile_height
                ),
                dims=self.xy_dim_names[1],
            )

    def _new_xy_coords(self) -> xr.DataArray:
        self._assert_regular()
        x_coords_1d = da.asarray(self.x_coords.data).rechunk(self.tile_width)
        y_coords_1d = da.expand_dims(
            da.asarray(self.y_coords.data).rechunk(self.tile_height), 1
        )
        y_coords_2d, x_coords_2d = da.broadcast_arrays(y_coords_1d, x_coords_1d)
        xy_coords = da.concatenate(
            [da.expand_dims(x_coords_2d, 0), da.expand_dims(y_coords_2d, 0)]
        )
        xy_coords = da.rechunk(
            xy_coords, chunks=(2, xy_coords.chunksize[1], xy_coords.chunksize[2])
        )
        xy_coords = xr.DataArray(
            xy_coords,
            dims=("coord", self.y_coords.dims[0], self.x_coords.dims[0]),
            name="xy_coords",
        )
        xy_coords.name = "xy_coords"
        return xy_coords


def new_regular_grid_mapping(
    size: int | tuple[int, int],
    xy_min: tuple[float, float],
    xy_res: float | tuple[float, float],
    crs: str | pyproj.crs.CRS,
    *,
    tile_size: int | tuple[int, int] = None,
    is_j_axis_up: bool = False,
) -> GridMapping:
    width, height = _normalize_int_pair(size, name="size")
    assert_true(width > 1 and height > 1, "invalid size")

    x_min, y_min = _normalize_number_pair(xy_min, name="xy_min")

    x_res, y_res = _normalize_number_pair(xy_res, name="xy_res")
    assert_true(x_res > 0 and y_res > 0, "invalid xy_res")

    crs = _normalize_crs(crs)

    x_min = _to_int_or_float(x_min)
    y_min = _to_int_or_float(y_min)
    x_max = _to_int_or_float(x_min + x_res * (width - 1))
    y_max = _to_int_or_float(y_min + y_res * (height - 1))
    xy_bbox = x_min - x_res / 2, y_min - y_res / 2, x_max + x_res / 2, y_max + y_res / 2
    xy_bbox = _round_sequence(xy_bbox)
    if crs.is_geographic:
        if xy_bbox[1] < -90:
            raise ValueError("invalid xy_bbox (south)")
        if xy_bbox[3] > 90:
            raise ValueError("invalid xy_bbox (north)")

    return RegularGridMapping(
        crs=crs,
        size=(width, height),
        tile_size=tile_size or (width, height),
        xy_bbox=xy_bbox,
        xy_res=(x_res, y_res),
        xy_var_names=_default_xy_var_names(crs),
        xy_dim_names=_default_xy_dim_names(crs),
        is_lon_360=(x_max > 180) and crs.is_geographic,
        is_j_axis_up=is_j_axis_up,
    )


def to_regular_grid_mapping(
    grid_mapping: GridMapping,
    *,
    tile_size: int | tuple[int, int] = None,
    is_j_axis_up: bool = False,
) -> GridMapping:
    if grid_mapping.is_regular:
        if tile_size is not None or is_j_axis_up != grid_mapping.is_j_axis_up:
            return grid_mapping.derive(tile_size=tile_size, is_j_axis_up=is_j_axis_up)
        return grid_mapping

    x_res, y_res = grid_mapping.xy_res
    x_min = grid_mapping.xy_bbox[0] + x_res / 2
    y_min = grid_mapping.xy_bbox[1] + y_res / 2
    x_max = grid_mapping.xy_bbox[2] - x_res / 2
    y_max = grid_mapping.xy_bbox[3] - y_res / 2

    width = round((x_max - x_min + x_res) / x_res)
    height = round((y_max - y_min + y_res) / y_res)
    width = width if width >= 2 else 2
    height = height if height >= 2 else 2

    if tile_size is None:
        tile_size = grid_mapping.tile_size
    return new_regular_grid_mapping(
        size=(width, height),
        xy_min=(x_min, y_min),
        xy_res=(x_res, y_res),
        crs=grid_mapping.crs,
        tile_size=tile_size,
        is_j_axis_up=is_j_axis_up,
    )


def new_regular_grid_mapping_from_bbox(
    bbox: Sequence[FloatInt],
    xy_res: FloatInt | tuple[FloatInt, FloatInt],
    crs: str | pyproj.CRS,
    tile_size: int | tuple[int, int] = 1024,
    is_j_axis_up: bool = False,
) -> GridMapping:
    if not isinstance(xy_res, tuple):
        xy_res = (xy_res, xy_res)
    x_size = int(np.ceil((bbox[2] - bbox[0]) / xy_res[0]))
    y_size = int(np.ceil(abs(bbox[3] - bbox[1]) / xy_res[1]))
    return new_regular_grid_mapping(
        size=(x_size, y_size),
        xy_min=(bbox[0] + xy_res[0] / 2, bbox[1] + xy_res[1] / 2),
        xy_res=xy_res,
        crs=crs,
        tile_size=tile_size,
        is_j_axis_up=is_j_axis_up,
    )
