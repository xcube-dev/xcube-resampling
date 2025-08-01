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
from typing import Sequence, Iterable

import numpy as np
import xarray as xr
import dask.array as da
from dask_image import ndinterp

from .gridmapping import GridMapping
from .constants import (
    AggMethods,
    AggFunction,
    AffineTransformMatrix,
    SplineOrder,
    SplineOrders,
    RecoverNans,
    FillValues,
    FloatInt,
)
from .utils import (
    normalize_grid_mapping,
    _can_apply_affine_transform,
    _get_agg_method,
    _get_recover_nan,
    _get_spline_order,
    _get_fill_value,
    _select_variables,
)


def affine_transform_dataset(
    source_ds: xr.Dataset,
    target_gm: GridMapping,
    source_gm: GridMapping | None = None,
    variables: str | Iterable[str] | None = None,
    spline_orders: SplineOrders | None = None,
    agg_methods: AggMethods | None = None,
    recover_nans: RecoverNans = False,
    fill_values: FillValues | None = None,
) -> xr.Dataset:
    if source_gm is None:
        source_gm = GridMapping.from_dataset(source_ds)
    source_ds = normalize_grid_mapping(source_ds, source_gm)

    assert _can_apply_affine_transform(source_gm, target_gm), (
        f"Affine transformation cannot be applied to source CRS "
        f"{source_gm.crs.name!r} and target CRS {target_gm.crs.name!r}"
    )

    source_ds = _select_variables(source_ds, variables)

    target_ds = resample_dataset(
        source_ds,
        target_gm.ij_transform_to(source_gm),
        (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0]),
        target_gm.size,
        target_gm.tile_size,
        spline_orders,
        agg_methods,
        recover_nans,
        fill_values,
    )

    # assign coordinates from target grid-mapping
    x_name, y_name = target_gm.xy_dim_names
    coords = {x_name: target_gm.x_coords, y_name: target_gm.y_coords}
    target_ds = target_ds.assign_coords(coords)

    return target_ds


def resample_dataset(
    dataset: xr.Dataset,
    affine_matrix: AffineTransformMatrix,
    yx_dims: tuple[str, str],
    target_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    spline_orders: SplineOrders | None = None,
    agg_methods: AggMethods | None = None,
    recover_nans: RecoverNans = False,
    fill_values: FillValues | None = None,
) -> xr.Dataset:
    data_vars = dict()
    coords = dict()
    for var_name, data_array in dataset.items():
        new_data_array = None
        if data_array.dims[-2:] == yx_dims:
            if isinstance(data_array.data, np.ndarray):
                is_numpy_array = True
                array = da.asarray(data_array.data)
            else:
                is_numpy_array = False
                array = data_array.data

            output_shape = array.shape[:-2] + (target_size[1], target_size[0])
            output_chunks = tuple(chunks[0] for chunks in array.chunks[:-2]) + (
                target_tile_size[1],
                target_tile_size[0],
            )
            resampled_array = _resample_array(
                array,
                affine_matrix,
                output_shape,
                output_chunks,
                _get_spline_order(spline_orders, var_name, data_array),
                _get_agg_method(agg_methods, var_name, data_array),
                _get_recover_nan(recover_nans, var_name, data_array),
                _get_fill_value(fill_values, var_name, data_array),
            )
            if is_numpy_array:
                resampled_array = resampled_array.compute()
            new_data_array = xr.DataArray(
                data=resampled_array, dims=data_array.dims, attrs=data_array.attrs
            )
        elif yx_dims[0] not in data_array.dims and yx_dims[1] not in data_array.dims:
            new_data_array = data_array
        if new_data_array is not None:
            if var_name in dataset.coords:
                coords[var_name] = new_data_array
            elif var_name in dataset.data_vars:
                data_vars[var_name] = new_data_array

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=dataset.attrs)


def _resample_array(
    array: da.Array,
    affine_matrix: AffineTransformMatrix,
    output_shape: Sequence[int],
    output_chunks: Sequence[int],
    spline_order: SplineOrder,
    agg_method: AggFunction,
    recover_nan: bool,
    fill_value: FloatInt,
) -> da.Array:
    if (affine_matrix[0][0] > 1 or affine_matrix[1][0] > 1) and spline_order != 0:
        array = _downscale(
            array,
            affine_matrix,
            output_shape,
            output_chunks,
            agg_method,
            spline_order,
            recover_nan,
            fill_value,
        )
    else:
        array = _upscale(
            array,
            affine_matrix,
            output_shape,
            output_chunks,
            spline_order,
            recover_nan,
            fill_value,
        )
    return array


def _downscale(
    array: da.Array,
    affine_matrix: AffineTransformMatrix,
    output_shape: Sequence[int],
    output_chunks: Sequence[int],
    agg_method: AggFunction,
    spline_order: SplineOrder,
    recover_nan: bool,
    fill_value: FloatInt,
) -> da.Array:
    ((i_scale, _, i_off), (_, j_scale, j_off)) = affine_matrix
    j_divisor = math.ceil(abs(j_scale))
    i_divisor = math.ceil(abs(i_scale))
    affine_matrix = (
        (i_scale / i_divisor, affine_matrix[0][1], affine_matrix[0][2]),
        (affine_matrix[1][0], j_scale / j_divisor, affine_matrix[1][2]),
    )
    output_shape = tuple(output_shape[:-2]) + (
        output_shape[-2] * j_divisor,
        output_shape[-1] * i_divisor,
    )

    array = _upscale(
        array,
        affine_matrix,
        output_shape,
        output_chunks,
        spline_order,
        recover_nan,
        fill_value,
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
    spline_order: SplineOrder,
    recover_nan: bool,
    fill_value: FloatInt,
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
        cval=fill_value,
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

    return ndinterp.affine_transform(array, matrix, **kwargs)
