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

import abc

import dask
import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from xcube_resampling.constants import FloatInt

from .assertions import assert_instance, assert_true
from .base import DEFAULT_TOLERANCE, GridMapping
from .helpers import (
    _assert_valid_xy_names,
    _default_xy_var_names,
    _normalize_crs,
    _normalize_int_pair,
    _to_int_or_float,
    from_lon_360,
    round_to_fraction,
    _round_sequence,
    to_lon_360,
)

_ER = 6371000


class CoordsGridMapping(GridMapping, abc.ABC):
    """
    Grid mapping constructed from 1D/2D coordinate
    variables and a CRS.
    """

    @property
    def x_coords(self):
        assert isinstance(self._x_coords, xr.DataArray)
        return self._x_coords

    @property
    def y_coords(self):
        assert isinstance(self._y_coords, xr.DataArray)
        return self._y_coords

    def _new_x_coords(self) -> xr.DataArray:
        # Should never come here
        return self._x_coords

    def _new_y_coords(self) -> xr.DataArray:
        # Should never come here
        return self._y_coords


class Coords1DGridMapping(CoordsGridMapping):
    """Grid mapping constructed from
    1D coordinate variables and a CRS.
    """

    def _new_xy_coords(self) -> xr.DataArray:
        y, x = xr.broadcast(self._y_coords, self._x_coords)
        tmp = xr.concat([x, y], dim="coord")
        return tmp.chunk(
            {dim: size for (dim, size) in zip(tmp.dims, self.xy_coords_chunks)}
        )


class Coords2DGridMapping(CoordsGridMapping):
    """Grid mapping constructed from
    2D coordinate variables and a CRS.
    """

    def _new_xy_coords(self) -> xr.DataArray:
        tmp = xr.concat([self._x_coords, self._y_coords], dim="coord")
        return tmp.chunk(
            {dim: size for (dim, size) in zip(tmp.dims, self.xy_coords_chunks)}
        )


def new_grid_mapping_from_coords(
    x_coords: xr.DataArray,
    y_coords: xr.DataArray,
    crs: str | pyproj.crs.CRS,
    *,
    xy_bbox: tuple[FloatInt, FloatInt, FloatInt, FloatInt] = None,
    tile_size: int | tuple[int, int] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> GridMapping:
    crs = _normalize_crs(crs)
    assert_instance(x_coords, xr.DataArray, name="x_coords")
    assert_instance(y_coords, xr.DataArray, name="y_coords")
    assert_true(
        x_coords.ndim in (1, 2), "x_coords and y_coords must be either 1D or 2D arrays"
    )
    assert_instance(tolerance, float, name="tolerance")
    assert_true(tolerance > 0.0, "tolerance must be greater zero")

    if x_coords.name and y_coords.name:
        xy_var_names = str(x_coords.name), str(y_coords.name)
    else:
        xy_var_names = _default_xy_var_names(crs)

    tile_size = _normalize_int_pair(tile_size, default=None)

    if x_coords.ndim == 1 and y_coords.ndim == 1:
        gm = new_1d_grid_mapping_from_coords(
            x_coords,
            y_coords,
            crs,
            xy_var_names,
            xy_bbox=xy_bbox,
            tile_size=tile_size,
            tolerance=tolerance,
        )
    else:
        gm = new_2d_grid_mapping_from_coords(
            x_coords,
            y_coords,
            crs,
            xy_var_names,
            xy_bbox=xy_bbox,
            tile_size=tile_size,
            tolerance=tolerance,
        )
    return gm


def new_1d_grid_mapping_from_coords(
    x_coords: xr.DataArray,
    y_coords: xr.DataArray,
    crs: str | pyproj.crs.CRS,
    xy_var_names: tuple[str, str],
    *,
    xy_bbox: tuple[FloatInt, FloatInt, FloatInt, FloatInt] = None,
    tile_size: int | tuple[int, int] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Coords1DGridMapping:

    assert_true(
        x_coords.size >= 2 and y_coords.size >= 2,
        "sizes of x_coords and y_coords 1D arrays must be >= 2",
    )

    is_lon_360 = None
    if crs.is_geographic:
        is_lon_360 = bool(da.any(x_coords > 180))
    y_diff = np.diff(y_coords)
    x_diff = np.diff(x_coords)
    if np.any(np.nanmax(abs(x_diff)) > 180) and not is_lon_360 and crs.is_geographic:
        x_coords = to_lon_360(x_coords)
        x_diff = np.diff(x_coords)
        is_lon_360 = True

    x_res = x_diff[0]
    y_res = y_diff[0]
    is_regular = (
        da.allclose(x_diff, x_res, atol=tolerance)
        and da.allclose(y_diff, y_res, atol=tolerance)
    ).compute()
    if is_regular:
        x_res = round_to_fraction(float(x_res), 5, 0.1)
        y_res = round_to_fraction(abs(float(y_res)), 5, 0.1)
    else:
        x_res = round_to_fraction(float(np.nanmedian(x_diff, axis=0)), 5, 0.1)
        y_res = round_to_fraction(abs(float(np.nanmedian(y_diff, axis=0))), 5, 0.1)

    if (
        tile_size is None
        and x_coords.chunks is not None
        and y_coords.chunks is not None
    ):
        tile_size = (max(0, *x_coords.chunks[0]), max(0, *y_coords.chunks[0]))

    # Guess j axis direction
    is_j_axis_up = bool(y_coords[0] < y_coords[-1])

    if xy_bbox is None:
        (x_vals, y_vals) = dask.compute(x_coords[[0, -1]], y_coords[[0, -1]])
        x_min, x_max = x_vals.values
        y_min, y_max = y_vals.values
        if not is_j_axis_up:
            y_min, y_max = y_max, y_min
        x_pad = x_res / 2
        y_pad = y_res / 2
        x_min = _to_int_or_float(x_min - x_pad)
        y_min = _to_int_or_float(y_min - y_pad)
        x_max = _to_int_or_float(x_max + x_pad)
        y_max = _to_int_or_float(y_max + y_pad)
        xy_bbox = _round_sequence((x_min, y_min, x_max, y_max))

    return Coords1DGridMapping(
        x_coords=x_coords,
        y_coords=y_coords,
        crs=crs,
        size=(x_coords.size, y_coords.size),
        tile_size=tile_size,
        xy_bbox=xy_bbox,
        xy_res=(x_res, y_res),
        xy_var_names=xy_var_names,
        xy_dim_names=(str(x_coords.dims[0]), str(y_coords.dims[0])),
        is_regular=is_regular,
        is_lon_360=is_lon_360,
        is_j_axis_up=is_j_axis_up,
    )


def new_2d_grid_mapping_from_coords(
    x_coords: xr.DataArray,
    y_coords: xr.DataArray,
    crs: str | pyproj.crs.CRS,
    xy_var_names: tuple[str, str],
    *,
    xy_bbox: tuple[FloatInt, FloatInt, FloatInt, FloatInt] = None,
    tile_size: int | tuple[int, int] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Coords2DGridMapping:

    assert_true(
        x_coords.shape == y_coords.shape,
        "shapes of x_coords and y_coords 2D arrays must be equal",
    )
    assert_true(
        x_coords.dims == y_coords.dims,
        "dimensions of x_coords and y_coords 2D arrays must be equal",
    )

    height, width = x_coords.shape
    ydim, xdim = x_coords.dims
    x_da, y_da = dask.compute(
        x_coords.isel({ydim: [0, -1], xdim: [0, 1, -1]}),
        y_coords.isel({ydim: [0, 1, -1], xdim: [0, -1]}),
    )
    x_vals = x_da.values
    y_vals = y_da.values
    x00 = x_vals[0, 0]
    x00_diff = x_vals[0, 1] - x_vals[0, 0]
    x01 = x_vals[0, 2]
    x10 = x_vals[1, 0]
    x11 = x_vals[1, 2]
    y00 = y_vals[0, 0]
    y01 = y_vals[0, 1]
    y00_diff = abs(y_vals[0, 0] - y_vals[1, 0])
    y10 = y_vals[2, 0]
    y11 = y_vals[2, 1]

    is_j_axis_up = y00 < y10
    is_lon_360 = None
    if crs.is_geographic:
        is_lon_360 = bool(max(x00, x01, x10, x11) > 180)
    if max(x00, x10) > min(x01, x11):
        x_coords = to_lon_360(x_coords)
        x_da = x_coords.isel({ydim: [0, 1, -1], xdim: [0, -1]}).compute()
        x_vals = x_da.values
        x00 = x_vals[0, 0]
        x00_diff = x_vals[1, 0] - x_vals[0, 0]
        x01 = x_vals[0, 1]
        x10 = x_vals[2, 0]
        x11 = x_vals[2, 1]
        is_lon_360 = True

    x_res = abs(np.mean([x00, x10]) - np.mean([x11, x01])) / (width - 1)
    y_res = abs(np.mean([y00, y01]) - np.mean([y11, y10])) / (height - 1)
    assert_true(
        x_res > 0 and y_res > 0,
        "internal error: x_res and y_res could not be determined",
        exception_type=RuntimeError,
    )
    is_regular = (
        np.isclose(x00_diff, x_res, atol=tolerance)
        and np.isclose(x00, x10, atol=tolerance)
        and np.isclose(x01, x11, atol=tolerance)
        and np.isclose(y00_diff, y_res, atol=tolerance)
        and np.isclose(y00, y01, atol=tolerance)
        and np.isclose(y10, y11, atol=tolerance)
    )
    x_res = round_to_fraction(x_res, 5, 0.1)
    y_res = round_to_fraction(y_res, 5, 0.1)

    if tile_size is None and x_coords.chunks is not None:
        j_chunks, i_chunks = x_coords.chunks
        tile_size = max(0, *i_chunks), max(0, *j_chunks)

    if tile_size is not None:
        tile_width, tile_height = tile_size
        x_coords = x_coords.chunk(
            {x_coords.dims[0]: tile_height, x_coords.dims[1]: tile_width}
        )
        y_coords = y_coords.chunk(
            {y_coords.dims[0]: tile_height, y_coords.dims[1]: tile_width}
        )

    if xy_bbox is None:
        x_pad, y_pad = x_res / 2, y_res / 2
        x_min = min(x00, x10)
        x_max = max(x01, x11)
        if is_j_axis_up:
            y_min = min(y00, y01)
            y_max = max(y10, y11)
        else:
            y_min = min(y10, y11)
            y_max = max(y00, y01)
        x_min = _to_int_or_float(x_min - x_pad)
        y_min = _to_int_or_float(y_min - y_pad)
        x_max = _to_int_or_float(x_max + x_pad)
        y_max = _to_int_or_float(y_max + y_pad)
        xy_bbox = _round_sequence((x_min, y_min, x_max, y_max))

    return Coords2DGridMapping(
        x_coords=x_coords,
        y_coords=y_coords,
        crs=crs,
        size=(width, height),
        tile_size=tile_size,
        xy_bbox=xy_bbox,
        xy_res=(x_res, y_res),
        xy_var_names=xy_var_names,
        xy_dim_names=(str(x_coords.dims[1]), str(x_coords.dims[0])),
        is_regular=is_regular,
        is_lon_360=is_lon_360,
        is_j_axis_up=is_j_axis_up,
    )


def _abs_no_zero(array: xr.DataArray | da.Array | np.ndarray):
    array = da.fabs(array)
    return da.where(da.isclose(array, 0), np.nan, array)


def _abs_no_nan(array: da.Array | np.ndarray):
    array = da.fabs(array)
    return da.where(da.logical_or(da.isnan(array), da.isclose(array, 0)), 0, array)


def grid_mapping_to_coords(
    grid_mapping: GridMapping,
    xy_var_names: tuple[str, str] = None,
    xy_dim_names: tuple[str, str] = None,
    reuse_coords: bool = False,
    exclude_bounds: bool = False,
) -> dict[str, xr.DataArray]:
    """Get CF-compliant axis coordinate variables and cell
    boundary coordinate variables.

    Defined only for grid mappings with regular x,y coordinates.

    Args:
        grid_mapping: A regular grid mapping.
        xy_var_names: Optional coordinate variable names (x_var_name,
            y_var_name).
        xy_dim_names: Optional coordinate dimensions names (x_dim_name,
            y_dim_name).
        reuse_coords: Whether to either reuse target coordinate arrays
            from target_gm or to compute new ones.
        exclude_bounds: If True, do not create bounds coordinates.
            Defaults to False. Ignored if *reuse_coords* is True.

    Returns:
        dictionary with coordinate variables
    """

    if xy_var_names:
        _assert_valid_xy_names(xy_var_names, name="xy_var_names")
    if xy_dim_names:
        _assert_valid_xy_names(xy_dim_names, name="xy_dim_names")

    if reuse_coords:
        try:
            # noinspection PyUnresolvedReferences
            x, y = grid_mapping.x_coords, grid_mapping.y_coords
        except AttributeError:
            x, y = None, None
        if (
            isinstance(x, xr.DataArray)
            and isinstance(y, xr.DataArray)
            and x.ndim == 1
            and y.ndim == 1
            and x.size == grid_mapping.width
            and y.size == grid_mapping.height
        ):
            return {
                name: xr.DataArray(coord.values, dims=dim, attrs=coord.attrs)
                for name, dim, coord in zip(xy_var_names, xy_dim_names, (x, y))
            }

    x_name, y_name = xy_var_names or grid_mapping.xy_var_names
    x_dim_name, y_dim_name = xy_dim_names or grid_mapping.xy_dim_names
    w, h = grid_mapping.size
    x1, y1, x2, y2 = grid_mapping.xy_bbox
    x_res, y_res = grid_mapping.xy_res
    x_res_05 = x_res / 2
    y_res_05 = y_res / 2

    dtype = np.float64

    x_data = np.linspace(x1 + x_res_05, x2 - x_res_05, w, dtype=dtype)
    if grid_mapping.is_lon_360:
        x_data = from_lon_360(x_data)

    if grid_mapping.is_j_axis_up:
        y_data = np.linspace(y1 + y_res_05, y2 - y_res_05, h, dtype=dtype)
    else:
        y_data = np.linspace(y2 - y_res_05, y1 + y_res_05, h, dtype=dtype)

    if grid_mapping.crs.is_geographic:
        x_attrs = dict(
            long_name="longitude coordinate",
            standard_name="longitude",
            units="degrees_east",
        )
        y_attrs = dict(
            long_name="latitude coordinate",
            standard_name="latitude",
            units="degrees_north",
        )
    else:
        x_attrs = dict(
            long_name="x coordinate of projection",
            standard_name="projection_x_coordinate",
        )
        y_attrs = dict(
            long_name="y coordinate of projection",
            standard_name="projection_y_coordinate",
        )

    x_coords = xr.DataArray(x_data, dims=x_dim_name, attrs=x_attrs)
    y_coords = xr.DataArray(y_data, dims=y_dim_name, attrs=y_attrs)
    coords = {
        x_name: x_coords,
        y_name: y_coords,
    }
    if not exclude_bounds:
        x_bnds_0_data = np.linspace(x1, x2 - x_res, w, dtype=dtype)
        x_bnds_1_data = np.linspace(x1 + x_res, x2, w, dtype=dtype)

        if grid_mapping.is_lon_360:
            x_bnds_0_data = from_lon_360(x_bnds_0_data)
            x_bnds_1_data = from_lon_360(x_bnds_1_data)

        if grid_mapping.is_j_axis_up:
            y_bnds_0_data = np.linspace(y1, y2 - y_res, h, dtype=dtype)
            y_bnds_1_data = np.linspace(y1 + y_res, y2, h, dtype=dtype)
        else:
            y_bnds_0_data = np.linspace(y2, y1 + y_res, h, dtype=dtype)
            y_bnds_1_data = np.linspace(y2 - y_res, y1, h, dtype=dtype)

        bnds_dim_name = "bnds"
        x_bnds_name = f"{x_name}_{bnds_dim_name}"
        y_bnds_name = f"{y_name}_{bnds_dim_name}"
        # Note, according to CF, bounds variables are not required to have
        # any attributes, so we don't pass any.
        x_bnds_coords = xr.DataArray(
            list(zip(x_bnds_0_data, x_bnds_1_data)), dims=[x_dim_name, bnds_dim_name]
        )
        y_bnds_coords = xr.DataArray(
            list(zip(y_bnds_0_data, y_bnds_1_data)), dims=[y_dim_name, bnds_dim_name]
        )
        x_coords.attrs.update(bounds=x_bnds_name)
        y_coords.attrs.update(bounds=y_bnds_name)
        coords.update(
            {
                x_bnds_name: x_bnds_coords,
                y_bnds_name: y_bnds_coords,
            }
        )

    return coords
