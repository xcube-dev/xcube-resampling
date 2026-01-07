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

from collections.abc import Iterable

import xarray as xr

from .affine import affine_transform_dataset
from .constants import (
    LOG,
    FillValues,
    PreventNaNPropagations,
    SpatialAggMethods,
    SpatialInterpMethods,
)
from .gridmapping import GridMapping
from .rectify import rectify_dataset
from .reproject import reproject_dataset
from .utils import _can_apply_affine_transform


def resample_in_space(
    source_ds: xr.Dataset,
    *,
    target_gm: GridMapping | None = None,
    source_gm: GridMapping | None = None,
    variables: str | Iterable[str] | None = None,
    interp_methods: SpatialInterpMethods | None = None,
    agg_methods: SpatialAggMethods | None = None,
    prevent_nan_propagations: PreventNaNPropagations = False,
    fill_values: FillValues | None = None,
    tile_size: int | tuple[int, int] | None = None,
) -> xr.Dataset:
    """
    Resample the spatial dimensions of a dataset to match a target grid mapping.

    Depending on the regularity and compatibility of the source and target grid
    mappings, this function will either rectify, reproject, or affine-transform the
    spatial dimensions of `source_ds`.

    Args:
        source_ds: The input xarray.Dataset. Data variables must have dimensions
            in the following order: optional third dimension followed by the
            y-dimension (e.g., "y" or "lat") and the x-dimension (e.g., "x" or "lon").
        target_gm: The target GridMapping to which the dataset should be resampled.
            Must be regular. If not provided, a default regular grid is derived
            from `source_gm` using `to_regular(tile_size)`.
        source_gm: The GridMapping describing the source dataset's spatial layout.
            If not provided, it is inferred from `source_ds` using
            `GridMapping.from_dataset(source_ds)`.
        variables: A single variable name or iterable of variable names to be
            resampled. If None, all data variables will be processed.
        interp_methods: Optional interpolation method to be used for upsampling spatial
            data variables. Can be a single interpolation method for all variables or a
            dictionary mapping variable names or dtypes to interpolation method.
            Supported methods include:

            - `0` (nearest neighbor)
            - `1` (linear / bilinear)
            - `"nearest"`
            - `"triangular"`
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
        fill_values: Optional fill value(s) for areas outside input coverage.
            Can be a single value or dictionary by variable or type. If not provided,
            defaults based on data type are used:

            - float: NaN
            - uint8: 255
            - uint16: 65535
            - other ints: -1

        tile_size: Optional tile size used when generating a regular grid from
            an irregular source grid mapping. Only used if `target_gm` is not provided.

    Returns:
        A new dataset that has been spatially resampled to match the target grid
            mapping.

    Notes:
        - If `source_gm` is not provided, it is inferred from `source_ds`.
        - If `target_gm` is not provided, and the source is irregular, it is
          derived from `source_gm.to_regular(tile_size=tile_size)`.
        - If both grid mappings are regular and approximately equal, the original
          dataset is returned unchanged.
        - If the transformation can be represented as an affine mapping, it is
          applied directly for performance.
        - If the source is irregular, rectification is applied.
        - Otherwise, a reprojection is performed.
        - See the [xcube-resampling documentation](https://xcube-dev.github.io/xcube-resampling/)
          for more details.
    """
    if source_gm is None:
        source_gm = GridMapping.from_dataset(source_ds)

    if not source_gm.is_regular:
        return rectify_dataset(
            source_ds,
            target_gm=target_gm,
            source_gm=source_gm,
            variables=variables,
            interp_methods=interp_methods,
            agg_methods=agg_methods,
            prevent_nan_propagations=prevent_nan_propagations,
            fill_values=fill_values,
            tile_size=tile_size,
        )
    else:
        if target_gm is None:
            LOG.warning(
                "If source grid mapping is regular `target_gm` must be given. "
                "Source dataset is returned."
            )
            return source_ds
        GridMapping.assert_regular(target_gm, name="target_gm")
        if source_gm.is_close(target_gm):
            return source_ds

        if _can_apply_affine_transform(source_gm, target_gm):
            return affine_transform_dataset(
                source_ds,
                target_gm,
                source_gm=source_gm,
                variables=variables,
                interp_methods=interp_methods,
                agg_methods=agg_methods,
                prevent_nan_propagations=prevent_nan_propagations,
                fill_values=fill_values,
            )
        else:
            return reproject_dataset(
                source_ds,
                target_gm,
                source_gm=source_gm,
                variables=variables,
                interp_methods=interp_methods,
                agg_methods=agg_methods,
                prevent_nan_propagations=prevent_nan_propagations,
                fill_values=fill_values,
            )
