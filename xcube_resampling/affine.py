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
from collections.abc import Iterable, Sequence

import dask.array as da
import numpy as np
import xarray as xr
from dask_image import ndinterp

from .constants import (
    AffineTransformMatrix,
    AggFunction,
    FillValues,
    FloatInt,
    PreventNaNPropagations,
    SpatialAggMethods,
    SpatialInterpMethodInt,
    SpatialInterpMethods,
)
from .gridmapping import GridMapping
from .utils import (
    _can_apply_affine_transform,
    _get_fill_value,
    _get_prevent_nan_propagation,
    _get_spatial_agg_method,
    _get_spatial_interp_method_int,
    _select_variables,
    normalize_grid_mapping,
)


def affine_transform_dataset(
    source_ds: xr.Dataset,
    target_gm: GridMapping,
    *,
    source_gm: GridMapping | None = None,
    variables: str | Iterable[str] | None = None,
    interp_methods: SpatialInterpMethods | None = None,
    agg_methods: SpatialAggMethods | None = None,
    prevent_nan_propagations: PreventNaNPropagations = False,
    fill_values: FillValues | None = None,
) -> xr.Dataset:
    """
    Apply an affine transformation to the spatial dimensions of a dataset,
    transforming it from the source grid mapping to the target grid mapping.

    Args:
        source_ds: The input dataset to be transformed.
        target_gm: The target grid mapping defining the spatial reference and
            output geometry.
        source_gm: The grid mapping of the input dataset. If None, it is inferred
            from the dataset.
        variables: Optional variable(s) to transform. If None, all variables are used.
        interp_methods: Optional interpolation method to be used for upsampling spatial
            data variables. Can be a single interpolation method for all variables or a
            dictionary mapping variable names or dtypes to interpolation method.
            Supported methods include:

            - `0` (nearest neighbor)
            - `1` (linear / bilinear)
            - `"nearest"`
            - `"bilinear"`

            The default is `0` for integer arrays, else `1`.
        agg_methods: Optional aggregation methods for downsampling spatial variables.
            Can be a single method for all variables or a dictionary mapping variable
            names or dtypes to methods. Supported methods include:
                "center", "count", "first", "last", "max", "mean", "median",
                "mode", "min", "prod", "std", "sum", and "var".
            Defaults to "center" for integer arrays, else "mean".
        prevent_nan_propagations: Optional boolean or mapping to prevent NaN
            propagation during upsampling (only applies when interpolation method
            is not nearest). Can be a single boolean or a dictionary mapping
            variable names or dtypes to booleans. Defaults to False.
        fill_values: Optional fill value(s) for areas outside the input bounds.
            Can be a single value or a dictionary mapping variable names or dtypes
            to fill values. If not provided, defaults are:

            - float: NaN
            - uint8: 255
            - uint16: 65535
            - other integers: -1

    Returns:
        A new dataset resampled and aligned to the target grid mapping.
            Data variables without spatial dimensions are copied to the output.
            Data variables with only one spatial dimension are ignored.
    """
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
        interp_methods,
        agg_methods,
        prevent_nan_propagations,
        fill_values,
    )

    # assign coordinates from target grid-mapping
    x_name, y_name = target_gm.xy_var_names
    target_ds = target_ds.assign_coords(
        {x_name: target_gm.x_coords, y_name: target_gm.y_coords}
    )

    return target_ds


def resample_dataset(
    dataset: xr.Dataset,
    affine_matrix: AffineTransformMatrix,
    yx_dims: tuple[str, str],
    target_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    interp_methods: SpatialInterpMethods | None = None,
    agg_methods: SpatialAggMethods | None = None,
    prevent_nan_propagations: PreventNaNPropagations = False,
    fill_values: FillValues | None = None,
) -> xr.Dataset:
    """
    Resample a dataset's spatial variables using an affine transformation.

    Applies resampling to 2D or 3D data arrays with spatial dimensions matching
    `yx_dims`. Variables that do not include the specified spatial dimensions
    are copied unchanged.

    Args:
        dataset: The input dataset containing spatial and non-spatial variables.
        affine_matrix: Affine transformation matrix mapping target to source coordinates.
        yx_dims: Tuple specifying the names of the spatial dimensions (y, x).
        target_size: The shape (height, width) of the resampled output in pixels.
        target_tile_size: Chunk size (height, width) for tiled output arrays.
        interp_methods: Optional interpolation method to be used for upsampling spatial
            data variables. Can be a single interpolation method for all variables or a
            dictionary mapping variable names or dtypes to interpolation method.
            Supported methods include:

            - `0` (nearest neighbor)
            - `1` (linear / bilinear)
            - `"nearest"`
            - `"bilinear"`

            The default is `0` for integer arrays, else `1`.
        agg_methods: Optional aggregation methods for downsampling spatial variables.
            Can be a single method for all variables or a dictionary mapping variable
            names or dtypes to methods. Supported methods include:
                "center", "count", "first", "last", "max", "mean", "median",
                "mode", "min", "prod", "std", "sum", and "var".
            Defaults to "center" for integer arrays, else "mean".
        prevent_nan_propagations: Optional boolean or mapping to prevent NaN
            propagation during upsampling (only applies when interpolation method
            is not nearest). Can be a single boolean or a dictionary mapping
            variable names or dtypes to booleans. Defaults to False.
        fill_values: Optional value(s) to use for regions outside source extent.
            Can be a single value or a dictionary mapping variable names or dtypes
            to specific fill values. If not provided, defaults are:

            - float: NaN
            - uint8: 255
            - uint16: 65535
            - other integers: -1

    Returns:
        A new dataset with spatial variables resampled to the target
            geometry. Non-spatial variables are preserved. Variables with only one
            spatial dimension are excluded.
    """
    data_vars = dict()
    coords = dict()
    for var_name, data_array in dataset.variables.items():
        data_array = xr.DataArray(data_array)
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
                _get_spatial_interp_method_int(interp_methods, var_name, data_array),
                _get_spatial_agg_method(agg_methods, var_name, data_array),
                _get_prevent_nan_propagation(
                    prevent_nan_propagations, var_name, data_array
                ),
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
    interp_method: SpatialInterpMethodInt,
    agg_method: AggFunction,
    prevent_nan_propagation: bool,
    fill_value: FloatInt,
) -> da.Array:
    if (affine_matrix[0][0] > 1 or affine_matrix[1][0] > 1) and interp_method != 0:
        array = _downscale(
            array,
            affine_matrix,
            output_shape,
            output_chunks,
            agg_method,
            interp_method,
            prevent_nan_propagation,
            fill_value,
        )
    else:
        array = _upscale(
            array,
            affine_matrix,
            output_shape,
            output_chunks,
            interp_method,
            prevent_nan_propagation,
            fill_value,
        )
    return array


def _downscale(
    array: da.Array,
    affine_matrix: AffineTransformMatrix,
    output_shape: Sequence[int],
    output_chunks: Sequence[int],
    agg_method: AggFunction,
    interp_method: SpatialInterpMethodInt,
    prevent_nan_propagation: bool,
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
        interp_method,
        prevent_nan_propagation,
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
    interp_method: SpatialInterpMethodInt,
    prevent_nan_propagation: bool,
    fill_value: FloatInt,
) -> da.Array:
    ((i_scale, _, i_off), (_, j_scale, j_off)) = affine_matrix
    offset = (array.ndim - 2) * (0,) + (j_off, i_off)
    scale = (array.ndim - 2) * (1,) + (j_scale, i_scale)
    matrix = np.diag(scale)
    if interp_method > 1:
        raise ValueError(
            "interp_methods must be one of 0, 1, 'nearest', 'bilinear'. "
            "Higher order is not supported for 3D arrays in affine transforms, "
            "as it causes unintended blending across the non-spatial (e.g., time) "
            "dimension."
        )
    kwargs = dict(
        offset=offset,
        order=interp_method,
        output_shape=output_shape,
        output_chunks=output_chunks,
        mode="constant",
        cval=fill_value,
    )
    if prevent_nan_propagation and interp_method > 0:
        # Prevent NaNs from spreading to nearby pixels during interpolation
        mask = da.isnan(array)
        # First check if there are NaN values ar all
        if da.any(mask):
            # 1. replace NaN by zero
            filled_im = da.where(mask, 0.0, array)
            # 2. transform the zero-filled image
            scaled_im = ndinterp.affine_transform(filled_im, matrix, **kwargs)
            # 3. transform the inverted mask
            scaled_norm = ndinterp.affine_transform(1.0 - mask, matrix, **kwargs)
            # 4. put back NaN where there was zero,
            #    otherwise decode using scaled mask
            return da.where(
                da.isclose(scaled_norm, 0.0), np.nan, scaled_im / scaled_norm
            )

    return ndinterp.affine_transform(array, matrix, **kwargs)
