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

import math
from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr
import dask.array as da
from dask_image import ndinterp

from .gridmapping import GridMapping
from .constants import Aggregator, AffineTransformMatrix
from .utils import (
    _can_apply_affine_transform,
    _get_agg_method,
    _get_recover_nan,
    _get_spline_order,
)


def affine_transform_dataset(
    source_ds: xr.Dataset,
    target_gm: GridMapping,
    source_gm: GridMapping | None = None,
    spline_orders: int | Mapping[np.dtype | str, int] | None = None,
    agg_methods: Aggregator | Mapping[np.dtype | str, Aggregator] | None = None,
    recover_nans: bool | Mapping[np.dtype | str, bool] = False,
) -> xr.Dataset:
    if source_gm is None:
        source_gm = GridMapping.from_dataset(source_ds)

    assert _can_apply_affine_transform(source_gm, target_gm), (
        f"Affine transformation cannot be applied to source CRS "
        f"{source_gm.crs.name!r} and target CRS {target_gm.crs.name!r}"
    )

    x_name, y_name = source_gm.xy_dim_names
    coords = source_ds.coords.to_dataset()
    coords = coords.drop_vars((x_name, y_name))
    x_name, y_name = target_gm.xy_dim_names
    coords[x_name] = target_gm.x_coords
    coords[y_name] = target_gm.y_coords
    coords["spatial_ref"] = xr.DataArray(0, attrs=target_gm.crs.to_cf())
    target_ds = xr.Dataset(coords=coords, attrs=source_ds.attrs)

    affine_matrix = target_gm.ij_transform_to(source_gm)
    xy_dims = (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0])
    for var_name, data_array in source_ds.items():
        if data_array.dims[-2:] != xy_dims:
            continue

        if isinstance(data_array.data, np.ndarray):
            is_numpy_array = True
            array = da.asarray(data_array.data)
        else:
            is_numpy_array = False
            array = data_array.data

        output_shape = array.shape[-2:] + (target_gm.size[1], target_gm.size[0])
        output_chunks = tuple(chunks[0] for chunks in array.chunks[-2:]) + (
            target_gm.tile_size[1],
            target_gm.tile_size[0],
        )
        resampled_array = _resample_array(
            array,
            affine_matrix,
            output_shape,
            output_chunks,
            _get_spline_order(spline_orders, var_name, data_array),
            _get_agg_method(agg_methods, var_name, data_array),
            _get_recover_nan(recover_nans, var_name, data_array),
        )
        if is_numpy_array:
            resampled_array = resampled_array.compute()
        target_ds[var_name] = (data_array.dims, resampled_array)
        target_ds[var_name].attrs = data_array.attrs

    return target_ds


def _resample_array(
    array: da.Array,
    affine_matrix: AffineTransformMatrix,
    output_shape: Sequence[int],
    output_chunks: Sequence[int],
    spline_order: int,
    agg_method: Aggregator,
    recover_nan: bool = False,
) -> da.Array:
    if affine_matrix[0][0] > 1 or affine_matrix[1][0] > 1:
        array = _downscale(
            array,
            affine_matrix,
            output_shape,
            output_chunks,
            agg_method,
            spline_order,
            recover_nan,
        )
    else:
        array = _upscale(
            array, affine_matrix, output_shape, output_chunks, spline_order, recover_nan
        )
    return array


def _downscale(
    array: da.Array,
    affine_matrix: AffineTransformMatrix,
    output_shape: Sequence[int],
    output_chunks: Sequence[int],
    agg_method: Aggregator,
    spline_order: int,
    recover_nan: bool,
) -> da.Array:
    ((i_scale, _, i_off), (_, j_scale, j_off)) = affine_matrix
    j_divisor = math.ceil(abs(j_scale))
    i_divisor = math.ceil(abs(i_scale))
    output_shape = tuple(output_shape[-2:]) + (
        output_shape[-2] * j_divisor,
        output_shape[-1] * i_divisor,
    )
    array = _upscale(
        array, affine_matrix, output_shape, output_chunks, spline_order, recover_nan
    )
    axes = {array.ndim - 2: j_divisor, array.ndim - 1: i_divisor}
    # noinspection PyTypeChecker
    array = da.coarsen(agg_method, array, axes)
    array = array.rechunk(output_chunks)

    return array


def _upscale(
    array: da.Array,
    affine_matrix: AffineTransformMatrix,
    output_shape: Sequence[int],
    output_chunks: Sequence[int],
    spline_order: int,
    recover_nan: bool,
) -> da.Array:
    ((i_scale, _, i_off), (_, j_scale, j_off)) = affine_matrix
    offset = (array.ndim - 2) * (0,) + (j_off, i_off)
    scale = (array.ndim - 2) * (1,) + (j_scale, i_scale)
    matrix = np.diag(scale)
    kwargs = dict(
        offset=offset,
        order=spline_order,
        output_shape=output_shape,
        output_chunks=output_chunks,
        mode="constant",
    )
    if recover_nan and spline_order > 0:
        # We can "recover" values that are neighbours to NaN values
        # that would otherwise become NaN too.
        mask = da.isnan(array)
        # First check if there are NaN values ar all
        if da.any(mask):
            # 1. replace NaN by zero
            filled_im = da.where(mask, 0.0, array)
            # 2. transform the zero-filled image
            scaled_im = ndinterp.affine_transform(filled_im, matrix, **kwargs, cval=0.0)
            # 3. transform the inverted mask
            scaled_norm = ndinterp.affine_transform(
                1.0 - mask, matrix, **kwargs, cval=0.0
            )
            # 4. put back NaN where there was zero,
            #    otherwise decode using scaled mask
            return da.where(
                da.isclose(scaled_norm, 0.0), np.nan, scaled_im / scaled_norm
            )

    # No dealing with NaN required
    return ndinterp.affine_transform(array, matrix, **kwargs, cval=np.nan)
