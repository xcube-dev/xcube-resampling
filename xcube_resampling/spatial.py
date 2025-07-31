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

from collections.abc import Mapping, Sequence

import xarray as xr
import numpy as np

from .affine import affine_transform_dataset
from .rectify import rectify_dataset
from .gridmapping import GridMapping
from .constants import LOG, Aggregator
from .reproject import reproject_dataset
from .utils import _can_apply_affine_transform


def resample_in_space(
    source_ds: xr.Dataset,
    target_gm: GridMapping | None = None,
    source_gm: GridMapping | None = None,
    spline_orders: int | Mapping[np.dtype | str, int] | None = None,
    agg_methods: Aggregator | Mapping[np.dtype | str, Aggregator] | None = None,
    recover_nans: bool | Mapping[np.dtype | str, bool] = False,
    fill_values: int | float | Mapping[np.dtype | str, int | float] | None = None,
    tile_size: int | Sequence[int, int] | None = None,
) -> xr.Dataset:
    """
    Resample a dataset *source_ds* in the spatial dimensions.

    Args:
        source_ds: The source dataset. Data variables must have
            dimensions in the following order: optional 3rd dimension followed
            by the y-dimension (e.g., `y` or `lat`) followed by the
            x-dimension (e.g., `x` or `lon`).
        source_gm: The source grid mapping.
        target_gm: The target grid mapping. Must be regular.
        spline_orders: Spline orders to be used for upsampling
            spatial data variables. It can be a single spline order
            for all variables or a dictionary that maps a variable name or a data dtype
            to the spline order. A spline order is given by one of `0`
            (nearest neighbor), `1` (linear), `2` (bi-linear), or `3` (cubic).
            The default is `3` fo floating point datasets and `0` for integer datasets.
        agg_methods: Aggregation methods to be used for downsampling
            spatial data variables. It can be a single aggregation method for all
            variables or a dictionary that maps a variable name or a data dtype to the
            aggregation method. The aggregation method is a function like `np.sum`,
            `np.mean` which is propagated to [`dask.array.coarsen`](https://docs.dask.org/en/stable/generated/dask.array.coarsen.html).
        recover_nans: If true, whether a special algorithm shall be used that is able
            to recover values that would otherwise yield NaN during resampling. Default
            is False for all variable types since this may require considerable CPU
            resources on top. It can be a single aggregation method for all
            variables or a dictionary that maps a variable name or a data dtype to a
            boolean.
        fill_values: fill values
        tile_size: tile size in target gridmapping; only used if source dataset is
            irregular and *target_gm* is not assigned.

    Returns:
        The spatially resampled dataset, or None if the requested output area does
        not intersect with *dataset*.

    Notes:
        - If the source grid mapping *source_gm* is not given, it is derived from *dataset*:
          `source_gm = GridMapping.from_dataset(source_ds)`.
        - If the target grid mapping *target_gm* is not given, it is derived from
          *ref_ds* as `target_gm = GridMapping.from_dataset(ref_ds)`; if *ref_ds* is
          not given, *target_gm* is derived from *source_gm* as
          `target_gm = source_gm.to_regular()`.
        - If *source_gm* is almost equal to *target_gm*, this function is a no-op
          and *dataset* is returned unchanged.
        - further information is given in the [xcube-resample documentation]()
    """
    if source_gm is None:
        source_gm = GridMapping.from_dataset(source_ds)

    if not source_gm.is_regular:
        return rectify_dataset(
            source_ds,
            target_gm=target_gm,
            source_gm=source_gm,
            spline_orders=spline_orders,
            agg_methods=agg_methods,
            recover_nans=recover_nans,
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
                spline_orders=spline_orders,
                agg_methods=agg_methods,
                recover_nans=recover_nans,
            )
        else:
            return reproject_dataset(
                source_ds,
                target_gm,
                source_gm=source_gm,
                spline_orders=spline_orders,
                agg_methods=agg_methods,
                recover_nans=recover_nans,
                fill_values=fill_values,
            )
