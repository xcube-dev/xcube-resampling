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
import pyproj
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
    InterpMethodInt,
    InterpMethods,
    InterpMethodStr,
    RecoverNans,
)
from .gridmapping import GridMapping

# noinspection PyProtectedMember
from .gridmapping.helpers import _normalize_crs


def get_spatial_coords(ds: xr.Dataset) -> (str, str):
    """
    Identify the names of horizontal spatial coordinate in an xarray dataset.

    This function checks for standard coordinate name pairs used for horizontal
    spatial referencing: either, ("longitude", "latitude"), ("lon", "lat") or
    ("x", "y"). It returns the detected pair as a tuple in the order (x_coord, y_coord).

    Args:
        ds: The xarray.Dataset to inspect.

    Returns:
        A tuple (x_coord, y_coord) containing the names of the horizontal spatial
        dimensions, e.g., ("longitude", "latitude"), ("lon", "lat") or ("x", "y").

    Raises:
        KeyError: If no recognized spatial dimension pair is found in the dataset.
    """
    if "transformed_x" in ds and "transformed_y" in ds:
        x_coord, y_coord = "transformed_x", "transformed_y"
    elif "latitude" in ds and "longitude" in ds:
        x_coord, y_coord = "longitude", "latitude"
    elif "lat" in ds and "lon" in ds:
        x_coord, y_coord = "lon", "lat"
    elif "y" in ds and "x" in ds:
        x_coord, y_coord = "x", "y"
    else:
        raise KeyError(
            f"No standard spatial coordinates found in dataset. "
            f"Expected pairs ('lon', 'lat') or ('x', 'y'), but found: {list(ds.dims)}."
        )
    return x_coord, y_coord


def clip_dataset_by_bbox(
    ds: xr.Dataset,
    bbox: Sequence[FloatInt],
    spatial_coords: tuple[str, str] | None = None,
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
        spatial_coords: Optional tuple of two spatial coordinate names (x_dim, y_dim),
            e.g., ('lon', 'lat'). If None, the coordinates are inferred automatically.

    Returns:
        A spatial subset of the input dataset clipped to the bounding box.

    Raises:
        ValueError: If `bbox` does not contain exactly four elements.
        KeyError: If spatial coordinate names cannot be determined from the dataset.

    Notes:
        If the bounding box does not overlap with the dataset extent, the returned
        dataset may contain one or more zero-sized dimensions.
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox of length 4, got: {bbox}")

    if spatial_coords is None:
        spatial_coords = get_spatial_coords(ds)
    x_coord, y_coord = spatial_coords

    if ds[x_coord].ndim == 2 and ds[y_coord].ndim == 2:
        ds = _clip_2dcoord_dataset_by_bbox(ds, bbox, x_coord, y_coord)
    elif ds[x_coord].ndim == 1 and ds[y_coord].ndim == 1:
        ds = _clip_1dcoord_dataset_by_bbox(ds, bbox, x_coord, y_coord)
    else:
        raise ValueError(
            f"Unsupported coordinate dimensions: x_coord.ndim={ds[x_coord].ndim}, "
            f"y_coord.ndim={ds[y_coord].ndim}. Expected both 1D or both 2D."
        )

    if any(size == 0 for size in ds.sizes.values()):
        LOG.warning(
            "Clipped dataset contains at least one zero-sized dimension. "
            f"Check if the bounding box {bbox} overlaps with the dataset extent."
        )
    return ds


def _clip_2dcoord_dataset_by_bbox(
    ds: xr.Dataset,
    bbox: Sequence[FloatInt],
    x_coord: str,
    y_coord: str,
) -> xr.Dataset:
    mask = (
        (ds[x_coord] >= bbox[0])
        & (ds[x_coord] <= bbox[2])
        & (ds[y_coord] >= bbox[1])
        & (ds[y_coord] <= bbox[3])
    )
    # Explicitly load the mask into memory here to compute row/column indices using
    # NumPy. This avoids duplicating computations if we stay in Dask for the following
    # operations:
    #   rows = np.any(mask, axis=1)
    #   cols = np.any(mask, axis=0)
    # Note: This will load the entire mask and break chunking, so it is a conscious
    # choice.
    mask = mask.values

    # Find bounding rectangle in index space
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    idxs = np.where(rows)[0]
    if idxs.size == 0:
        row_min, row_max = 0, -1
    else:
        row_min, row_max = idxs[[0, -1]]
    idxs = np.where(cols)[0]
    if idxs.size == 0:
        col_min, col_max = 0, -1
    else:
        col_min, col_max = idxs[[0, -1]]

    y_dim, x_dim = ds[x_coord].dims
    return ds.isel(
        {
            y_dim: slice(row_min, row_max + 1),
            x_dim: slice(col_min, col_max + 1),
        }
    )


def _clip_1dcoord_dataset_by_bbox(
    ds: xr.Dataset,
    bbox: Sequence[FloatInt],
    x_coord: str,
    y_coord: str,
) -> xr.Dataset:
    if ds[y_coord][-1] - ds[y_coord][0] < 0:
        return ds.sel(
            {x_coord: slice(bbox[0], bbox[2]), y_coord: slice(bbox[3], bbox[1])}
        )
    else:
        return ds.sel(
            {x_coord: slice(bbox[0], bbox[2]), y_coord: slice(bbox[1], bbox[3])}
        )


def reproject_bbox(
    source_bbox: Sequence[FloatInt],
    source_crs: pyproj.CRS | str,
    target_crs: pyproj.CRS | str,
) -> Sequence[FloatInt]:
    """Reprojects a bounding box from a source CRS to a target CRS, with optional
    buffering.

    The function transforms a bounding box defined in the source coordinate reference
    system (CRS) to the target CRS using `pyproj`. If the source and target CRS are
    the same, no transformation is performed. An optional buffer (as a fraction of
    width/height) can be applied to expand the resulting bounding box.

    Args:
        source_bbox: The bounding box to reproject, in the form
            (min_x, min_y, max_x, max_y).
        source_crs: The source CRS, as a `pyproj.CRS` or string.
        target_crs: The target CRS, as a `pyproj.CRS` or string.

    Returns:
        A tuple representing the reprojected (and optionally buffered) bounding box:
        (min_x, min_y, max_x, max_y).
    """
    source_crs = _normalize_crs(source_crs)
    target_crs = _normalize_crs(target_crs)
    if source_crs != target_crs:
        t = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
        target_bbox = t.transform_bounds(*source_bbox, densify_pts=21)
    else:
        target_bbox = source_bbox

    return target_bbox


def bbox_overlap(
    source_bbox: Sequence[FloatInt], target_bbox: Sequence[FloatInt]
) -> float:
    """
    Calculate the fraction of the source bounding box that overlaps with the target
    bounding box.

    Args:
        source_bbox: (min_x, min_y, max_x, max_y)
        target_bbox: (min_x, min_y, max_x, max_y)

    Returns:
        float in [0, 1]
    """
    inter_min_x = max(source_bbox[0], target_bbox[0])
    inter_min_y = max(source_bbox[1], target_bbox[1])
    inter_max_x = min(source_bbox[2], target_bbox[2])
    inter_max_y = min(source_bbox[3], target_bbox[3])

    inter_w = max(0, inter_max_x - inter_min_x)
    inter_h = max(0, inter_max_y - inter_min_y)
    inter_area = inter_w * inter_h
    area_source = (source_bbox[2] - source_bbox[0]) * (source_bbox[3] - source_bbox[1])

    return inter_area / area_source


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
