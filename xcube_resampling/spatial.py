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
    AggMethods,
    FillValues,
    RecoverNans,
    SplineOrders,
)
from .gridmapping import GridMapping
from .rectify import rectify_dataset
from .reproject import reproject_dataset
from .utils import _can_apply_affine_transform


def resample_in_space(
    source_ds: xr.Dataset,
    target_gm: GridMapping | None = None,
    source_gm: GridMapping | None = None,
    variables: str | Iterable[str] | None = None,
    spline_orders: SplineOrders | None = None,
    agg_methods: AggMethods | None = None,
    recover_nans: RecoverNans = False,
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
        spline_orders: Spline orders used for upsampling spatial data variables.
            Can be a single spline order (0=nearest, 1=linear, 2=bilinear, 3=cubic),
            or a dictionary mapping variable names or data dtypes to spline orders.
            Defaults to 3 for floating-point data and 0 for integers.
        agg_methods: Aggregation methods for downsampling spatial data variables.
            Can be a single method (e.g., `np.mean`) or a dictionary mapping variable
            names or dtypes to aggregation methods. These are passed to
            `dask.array.coarsen`.
        recover_nans: Whether to apply a special algorithm to recover values that
            would otherwise become NaN during resampling. This can be a single boolean
            or a dictionary mapping variable names or dtypes to booleans. Defaults to
            False.
        fill_values: Fill values to use for areas where data is unavailable.
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
            spline_orders=spline_orders,
            agg_methods=agg_methods,
            recover_nans=recover_nans,
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
                spline_orders=spline_orders,
                agg_methods=agg_methods,
                recover_nans=recover_nans,
                fill_values=fill_values,
            )
        else:
            return reproject_dataset(
                source_ds,
                target_gm,
                source_gm=source_gm,
                variables=variables,
                spline_orders=spline_orders,
                agg_methods=agg_methods,
                recover_nans=recover_nans,
                fill_values=fill_values,
            )
