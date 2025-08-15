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

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

from .constants import (
    AGG_METHODS,
    FILLVALUE_FLOAT,
    FILLVALUE_INT,
    FILLVALUE_UINT8,
    FILLVALUE_UINT16,
    INTERP_METHOD_MAPPING,
    LOG,
    AggMethod,
    AggMethods,
    FloatInt,
    InterpMethod,
    InterpMethodStr,
    InterpMethodInt,
    InterpMethods,
    RecoverNans,
)
from .gridmapping import GridMapping


def get_spatial_dims(ds: xr.Dataset) -> (str, str):
    """
    Identify the names of horizontal spatial dimensions in an xarray dataset.

    This function checks for standard dimension name pairs used for horizontal
    spatial referencing: either ("lon", "lat") or ("x", "y"). It returns the
    detected pair as a tuple in the order (x_dim, y_dim).

    Args:
        ds: The xarray.Dataset to inspect.

    Returns:
        A tuple (x_dim, y_dim) containing the names of the horizontal spatial
        dimensions, e.g., ("lon", "lat") or ("x", "y").

    Raises:
        KeyError: If no recognized spatial dimension pair is found in the dataset.
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
    bbox: Sequence[FloatInt],
    spatial_dims: tuple[str, str] | None = None,
) -> xr.Dataset:
    """
    Clip a xarray Dataset to a specified bounding box.

    This function extracts a spatial subset of the dataset based on the given
    bounding box. It handles both increasing and decreasing orientation of the
    y-axis automatically to ensure correct spatial clipping.

    Args:
        ds: The input xarray.Dataset to be clipped.
        bbox: A sequence of four numbers representing the bounding box in the form
              (min_x, min_y, max_x, max_y).
        spatial_dims: Optional tuple of two spatial dimension names (x_dim, y_dim),
              e.g., ('lon', 'lat'). If None, the dimensions are inferred automatically.

    Returns:
        A spatial subset of the input dataset clipped to the bounding box.

    Raises:
        ValueError: If `bbox` does not contain exactly four elements.
        KeyError: If spatial dimension names cannot be determined from the dataset.

    Notes:
        If the bounding box does not overlap with the dataset extent, the returned
        dataset may contain one or more zero-sized dimensions.
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


def normalize_grid_mapping(ds: xr.Dataset, gm: GridMapping) -> xr.Dataset:
    """
    Normalize the grid mapping of a dataset to use a standard "spatial_ref" coordinate.

    This function standardizes geospatial metadata by replacing any existing grid
    mapping variable with a unified "spatial_ref" coordinate. It updates the
    "grid_mapping" attribute of all data variables to reference "spatial_ref",
    removes the original grid mapping variable (if present), and adds a new
    "spatial_ref" coordinate with CF-compliant CRS attributes.

    Args:
        ds: The input xarray.Dataset with geospatial grid mapping metadata.
        gm: The GridMapping object associated with the dataset.

    Returns:
        A new dataset with a standardized "spatial_ref" coordinate for grid mapping.
    """
    gm_name = _get_grid_mapping_name(ds)
    if gm_name is not None:
        ds = ds.drop_vars(gm_name)
    ds = ds.assign_coords(spatial_ref=xr.DataArray(0, attrs=gm.crs.to_cf()))
    for var in ds.data_vars:
        ds[var].attrs["grid_mapping"] = "spatial_ref"

    return ds


def _select_variables(
    ds: xr.Dataset, variables: str | Iterable[str] | None = None
) -> xr.Dataset:
    if variables is not None:
        if isinstance(variables, str):
            variables = [variables]
        ds = ds[variables]
    return ds


def _get_grid_mapping_name(ds: xr.Dataset) -> str | None:
    gm_names = []
    for var in ds.data_vars:
        if "grid_mapping" in ds[var].attrs:
            gm_names.append(ds[var].attrs["grid_mapping"])
    if "crs" in ds:
        gm_names.append("crs")
    if "spatial_ref" in ds.coords:
        gm_names.append("spatial_ref")
    gm_names = np.unique(gm_names)
    assert len(gm_names) <= 1, "Multiple grid mapping names found."
    if len(gm_names) == 1:
        return str(gm_names[0])
    else:
        return None


def _can_apply_affine_transform(source_gm: GridMapping, target_gm: GridMapping) -> bool:
    GridMapping.assert_regular(source_gm, name="source_gm")
    GridMapping.assert_regular(target_gm, name="target_gm")
    return _is_equal_crs(source_gm, target_gm)


def _is_equal_crs(source_gm: GridMapping, target_gm: GridMapping) -> bool:
    geographic = source_gm.crs.is_geographic and target_gm.crs.is_geographic
    return geographic or source_gm.crs.equals(target_gm.crs)


def _get_interp_method(
    interp_methods: InterpMethods | None,
    key: Hashable,
    var: xr.DataArray,
) -> InterpMethod:
    def assign_defaults(data_type: np.dtype) -> InterpMethod:
        return 0 if np.issubdtype(data_type, np.integer) else 1

    if isinstance(interp_methods, Mapping):
        interp_method = interp_methods.get(str(key), interp_methods.get(var.dtype))
        if interp_method is None:
            LOG.warning(
                f"Interpolation method could not be derived from the mapping "
                f"`interp_methods` for data variable {key!r} with data type "
                f"{var.dtype!r}. Defaults are assigned."
            )
            interp_method = assign_defaults(var.dtype)
    elif isinstance(interp_methods, int) or isinstance(interp_methods, str):
        interp_method = interp_methods
    else:
        interp_method = assign_defaults(var.dtype)

    return interp_method


def _get_interp_method_int(
    interp_methods: InterpMethods | None,
    key: Hashable,
    var: xr.DataArray,
) -> InterpMethodInt:
    interp_method = _get_interp_method(interp_methods, key, var)
    if isinstance(interp_method, str):
        interp_method = INTERP_METHOD_MAPPING[interp_method]
    return interp_method


def _get_interp_method_str(
    interp_methods: InterpMethods | None,
    key: Hashable,
    var: xr.DataArray,
) -> InterpMethodStr:
    interp_method = _get_interp_method(interp_methods, key, var)
    if isinstance(interp_method, int):
        interp_method = INTERP_METHOD_MAPPING[interp_method]
    return interp_method


def _prep_interp_methods_downscale(
    interp_methods: InterpMethods | None,
) -> InterpMethods | None:
    if interp_methods == "triangular":
        return "bilinear"
    elif (
        isinstance(interp_methods, Mapping) and "triangular" in interp_methods.values()
    ):
        return {
            k: ("bilinear" if v == "triangular" else v)
            for k, v in interp_methods.items()
        }
    return interp_methods


def _get_agg_method(
    agg_methods: AggMethods | None,
    key: Hashable,
    var: xr.DataArray,
) -> Callable:
    def assign_defaults(data_type: np.dtype) -> AggMethod:
        return "center" if np.issubdtype(data_type, np.integer) else "mean"

    if isinstance(agg_methods, Mapping):
        agg_method = agg_methods.get(str(key), agg_methods.get(var.dtype))
        if agg_method is None:
            LOG.warning(
                f"Aggregation method could not be derived from the mapping `agg_methods` "
                f"for data variable {key!r} with data type {var.dtype!r}. Defaults "
                f"are assigned."
            )
            agg_method = assign_defaults(var.dtype)
    elif isinstance(agg_methods, str):
        agg_method = agg_methods
    else:
        agg_method = assign_defaults(var.dtype)

    return AGG_METHODS[agg_method]


def _get_recover_nan(
    recover_nans: RecoverNans | None,
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
    # noinspection PyShadowingNames
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
