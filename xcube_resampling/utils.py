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

from collections.abc import Mapping, Callable, Hashable, Sequence
import xarray as xr
from xcube.core.gridmapping import GridMapping
import numpy as np

from .constants import (
    LOG,
    Aggregator,
    FILLVALUE_FLOAT,
    FILLVALUE_UINT8,
    FILLVALUE_UINT16,
    FILLVALUE_INT,
)


def get_spatial_dims(ds: xr.Dataset) -> (str, str):
    """
    Determine the names of the horizontal spatial dimensions in a xarray dataset.

    This function checks for common naming conventions for horizontal spatial
    dimensions, either ("lon", "lat") or ("x", "y"), and returns them as a tuple.

    Args:
        ds: An xarray.Dataset whose dimensions will be inspected.

    Returns:
        A tuple (x_coord, y_coord) containing the names of the horizontal spatial
        dimensions. For example, ("lon", "lat") or ("x", "y").

    Raises:
        KeyError: If the dataset does not contain recognizable spatial
                  dimension names, i.e., ("lon", "lat") or ("x", "y").
    """
    if "lat" in ds and "lon" in ds:
        x_coord, y_coord = "lon", "lat"
    elif "y" in ds and "x" in ds:
        x_coord, y_coord = "x", "y"
    else:
        raise KeyError(
            f"No standard spatial dimensions found in dataset. "
            f"Expected pairs ('lon', 'lat') or ('x', 'y'), but found: {list(ds.dims)}."
        )
    return x_coord, y_coord


def clip_dataset_by_bbox(
    ds: xr.Dataset,
    bbox: tuple[float | int] | list[float | int],
    spatial_dims: Sequence[str, str] | None = None,
) -> xr.Dataset:
    """
    Clip a xarray Dataset to a given bounding box.

    The function selects a spatial subset of the dataset based on the provided
    bounding box. It automatically handles datasets with increasing or decreasing
    y-axis orientation.

    Args:
        ds: The input dataset to clip.
        bbox: Bounding box in the format (min_x, min_y, max_x, max_y).
        spatial_dims: A sequence of two spatial dimension names in the form
                      (x_dim, y_dim), e.g. ('lon', 'lat'). If None, these
                      will be inferred automatically.

    Returns:
        A subset of the original xarray.Dataset clipped to the bounding box.

    Raises:
        ValueError: If `bbox` is not a 4-element tuple or list.
        KeyError: If spatial dimensions cannot be found in the dataset.
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox of length 4, got: {bbox}")

    if spatial_dims is None:
        spatial_dims = get_spatial_dims(ds)
    x_dim, y_dim = spatial_dims

    if ds[y_dim][-1] - ds[y_dim][0] < 0:
        ds = ds.sel({x_dim: slice(bbox[0], bbox[2]), y_dim: slice(bbox[3], bbox[1])})
    else:
        ds = ds.sel({x_dim: slice(bbox[0], bbox[2]), y_dim: slice(bbox[1], bbox[3])})

    if any(size == 0 for size in ds.sizes.values()):
        LOG.warning(
            "Clipped dataset contains at least one zero-sized dimension. "
            f"Check if the bounding box {bbox} overlaps with the dataset extent."
        )
    return ds


def _can_apply_affine_transform(source_gm: GridMapping, target_gm: GridMapping) -> bool:
    GridMapping.assert_regular(source_gm, name="source_gm")
    GridMapping.assert_regular(target_gm, name="target_gm")
    geographic = source_gm.crs.is_geographic and target_gm.crs.is_geographic
    return geographic or source_gm.crs.equals(target_gm.crs)


def _get_spline_order(
    spline_orders: int | Mapping[np.dtype | str, int] | None,
    key: Hashable,
    var: xr.DataArray,
) -> int:

    def assign_defaults(data_type: np.dtype) -> int:
        return 0 if np.issubdtype(data_type, np.integer) else 3

    if isinstance(spline_orders, Mapping):
        spline_order = spline_orders.get(str(key), spline_orders.get(var.dtype))
        if spline_order is None:
            LOG.warning(
                f"Spline order could not be derived from the mapping `spline_orders` "
                f"for data variable {key!r} with data type {var.dtype!r}. Defaults "
                f"are assigned."
            )
            spline_order = assign_defaults(var.dtype)
    elif isinstance(spline_orders, int):
        spline_order = spline_orders
    else:
        spline_order = assign_defaults(var.dtype)

    return spline_order


def _get_agg_method(
    agg_methods: Aggregator | Mapping[np.dtype | str, Aggregator] | None,
    key: Hashable,
    var: xr.DataArray,
) -> Callable:
    def assign_defaults(data_type: np.dtype) -> Callable:
        return np.mean if np.issubdtype(data_type, np.integer) else np.mean

    if isinstance(agg_methods, Mapping):
        agg_method = agg_methods.get(str(key), agg_methods.get(var.dtype))
        if agg_method is None:
            LOG.warning(
                f"Aggregation method could not be derived from the mapping `agg_methods` "
                f"for data variable {key!r} with data type {var.dtype!r}. Defaults "
                f"are assigned."
            )
            agg_method = assign_defaults(var.dtype)
    elif isinstance(agg_methods, Aggregator):
        agg_method = agg_methods
    else:
        agg_method = assign_defaults(var.dtype)

    return agg_method


def _get_recover_nan(
    recover_nans: bool | Mapping[np.dtype | str, bool],
    key: Hashable,
    var: xr.DataArray,
) -> bool:
    if isinstance(recover_nans, Mapping):
        recover_nan = recover_nans.get(str(key), recover_nans.get(var.dtype))
        if recover_nan is None:
            LOG.warning(
                f"The method to recover nan could not be derived from the mapping "
                f"`recover_nans`  for data variable {key!r} with data type "
                f"{var.dtype!r}. Defaults are assigned."
            )
            recover_nan = False
    elif isinstance(recover_nans, bool):
        recover_nan = recover_nans
    else:
        recover_nan = False

    return recover_nan


def _get_fill_value(
    fill_values: int | float | Mapping[np.dtype | str, int | float] | None,
    key: Hashable,
    var: xr.DataArray,
) -> int:

    def assign_defaults(data_type: np.dtype) -> int:
        if data_type == np.uint8:
            fill_value = FILLVALUE_UINT8
        elif data_type == np.uint16:
            fill_value = FILLVALUE_UINT16
        elif np.issubdtype(data_type, np.integer):
            fill_value = FILLVALUE_INT
        else:
            fill_value = FILLVALUE_FLOAT
        return fill_value

    if isinstance(fill_values, Mapping):
        fill_value = fill_values.get(str(key), fill_values.get(var.dtype))
        if fill_value is None:
            LOG.warning(
                f"Fill value could not be derived from the mapping `fill_values` "
                f"for data variable {key!r} with data type {var.dtype!r}. Defaults "
                f"are assigned."
            )
            fill_value = assign_defaults(var.dtype)
    elif fill_values is not None:
        fill_value = fill_values
    else:
        fill_value = assign_defaults(var.dtype)

    return fill_value
