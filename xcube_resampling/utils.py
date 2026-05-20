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

import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from .constants import (
    AGG_METHODS,
    FILLVALUE_FLOAT,
    FILLVALUE_INT,
    FILLVALUE_UINT8,
    FILLVALUE_UINT16,
    FILLVALUE_UINT32,
    INTERP_METHOD_MAPPING,
    LOG,
    FillValues,
    FloatInt,
    PreventNaNPropagations,
    SpatialAggMethod,
    SpatialAggMethods,
    SpatialInterpMethod,
    SpatialInterpMethodInt,
    SpatialInterpMethods,
    SpatialInterpMethodStr,
)
from .gridmapping import GridMapping

# noinspection PyProtectedMember
from .gridmapping.helpers import _normalize_crs


def get_spatial_coords(ds: xr.Dataset) -> tuple[str, str]:
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


def get_utm_crs(lon: float, lat: float) -> pyproj.CRS:
    """
    Returns the UTM CRS for a given latitude and longitude as a pyproj.CRS object.

    Args:
        lon: Longitude in decimal degrees.
        lat: Latitude in decimal degrees.

    Returns:
        CRS object corresponding to the correct UTM zone.
    """
    # Standard UTM zone calculation
    zone_number = int((lon + 180) // 6) + 1

    # Norway exception (Western Norway and Svalbard)
    if 56.0 <= lat <= 64.0 and 3.0 <= lon <= 12.0:
        zone_number = 32  # Western Norway
    if 72.0 <= lat <= 84.0:
        # Svalbard zones (special UTM divisions)
        if 0 <= lon < 9:
            zone_number = 31
        elif 9 <= lon < 21:
            zone_number = 33
        elif 21 <= lon < 33:
            zone_number = 35
        elif 33 <= lon < 42:
            zone_number = 37

    # Northern or Southern hemisphere
    epsg = 32600 + zone_number if lat >= 0 else 32700 + zone_number
    return pyproj.CRS.from_epsg(epsg)


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
    bounding box relative to the area of the source bounding box.

    Args:
        source_bbox: (min_x, min_y, max_x, max_y)
        target_bbox: (min_x, min_y, max_x, max_y)

    Returns:
        float in [0, 1]
    """
    source_bboxes = _split_bbox_antimeridian(source_bbox)
    target_bboxes = _split_bbox_antimeridian(target_bbox)

    inter_area = 0.0
    for source_bbox in source_bboxes:
        for target_bbox in target_bboxes:
            inter_min_x = max(source_bbox[0], target_bbox[0])
            inter_min_y = max(source_bbox[1], target_bbox[1])
            inter_max_x = min(source_bbox[2], target_bbox[2])
            inter_max_y = min(source_bbox[3], target_bbox[3])

            inter_w = max(0, inter_max_x - inter_min_x)
            inter_h = max(0, inter_max_y - inter_min_y)
            inter_area += inter_w * inter_h
    area_source = 0.0
    for source_bbox in source_bboxes:
        area_source += (source_bbox[2] - source_bbox[0]) * (
            source_bbox[3] - source_bbox[1]
        )

    return inter_area / area_source


def _split_bbox_antimeridian(bbox: Sequence[FloatInt]) -> Sequence[Sequence[FloatInt]]:
    min_x, min_y, max_x, max_y = bbox

    if max_x < min_x:
        return [(min_x, min_y, 180, max_y), (-180, min_y, max_x, max_y)]
    return [bbox]


def transform_resolution(
    ref_point: tuple[FloatInt, FloatInt],
    resolution: FloatInt | tuple[FloatInt, FloatInt],
    src_crs: str | pyproj.CRS,
    dst_crs: str | pyproj.CRS,
) -> tuple[FloatInt, FloatInt]:
    """Estimate local spatial resolution in the destination CRS using
    finite differences.

    Args:
        ref_point: Reference point (easting, northing) in the source CRS.
        resolution: Spatial resolution in the source CRS. Can be a single number
            (applied equally to both axes) or a tuple `(x_res, y_res)`.
        src_crs: Source coordinate reference system.
        dst_crs: Destination coordinate reference system.

    Returns:
        tuple: Estimated local resolution in the destination CRS as `(x_res, y_res)`.
    """
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    if not isinstance(resolution, tuple):
        resolution = (resolution, resolution)

    # reference point
    x0, y0 = transformer.transform(*ref_point)

    # step in x direction
    x1, y1 = transformer.transform(ref_point[0] + resolution[0], ref_point[1])

    # step in y direction
    x2, y2 = transformer.transform(ref_point[0], ref_point[1] + resolution[1])

    # Euclidean distances
    res_x = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    res_y = np.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)

    return res_x, res_y


def resolution_meters_to_degrees(
    resolution: FloatInt | tuple[FloatInt, FloatInt], latitude: FloatInt
) -> tuple[FloatInt, FloatInt]:
    """Convert spatial resolution from meters to degrees in longitude and latitude
    at a given geographic latitude.

    This function is deprecated and will be removed in a future release.
    Use `transform_resolution` instead for CRS-aware and more accurate conversions.

    Args:
        resolution: Spatial resolution in meters. Can be a single number
            (applied equally to both axes) or a tuple ``(x_res, y_res)``.
        latitude: Latitude in degrees at which to compute the longitude scaling.

    Returns:
        A tuple `(lon_res_deg, lat_res_deg)` giving the approximate spatial
        resolution in degrees for the latitude and longitude directions.

    Notes:
        - 1 degree of latitude ≈ 111,320 meters (constant approximation).
        - 1 degree of longitude ≈ 111,320 * cos(latitude) meters.

    """
    warnings.warn(
        "resolution_meters_to_degrees is deprecated and will be removed in a future release. "
        "Use `transform_resolution` instead for accurate CRS-based resolution handling.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
    return (
        resolution[0] / (111320 * np.cos(np.deg2rad(latitude))),
        resolution[1] / 111320,
    )


def resolution_degrees_to_meters(
    resolution: FloatInt | tuple[FloatInt, FloatInt], latitude: FloatInt
) -> tuple[FloatInt, FloatInt]:
    """Convert spatial resolution from degrees to meters at a given geographic latitude.

    This function is deprecated and will be removed in a future release.
    Use `transform_resolution` instead for CRS-aware and more accurate conversions.

    Args:
        resolution: Spatial resolution in degrees. Can be a single number
            (applied equally to both axes) or a tuple ``(lon_res, lat_res)``.
        latitude: Latitude in degrees at which to compute the longitude scaling.

    Returns:
        A tuple `(x_res, y_res)` giving the approximate spatial
        resolution in degrees for the latitude and longitude directions.

    Notes:
        - 1 degree of latitude ≈ 111,320 meters (constant approximation).
        - 1 degree of longitude ≈ 111,320 * cos(latitude) meters.

    """
    warnings.warn(
        "resolution_degrees_to_meters is deprecated and will be removed in a future release. "
        "Use `transform_resolution` instead for accurate CRS-based resolution handling.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
    return (
        resolution[0] * (111320 * np.cos(np.deg2rad(latitude))),
        resolution[1] * 111320,
    )


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


def _get_spatial_interp_method(
    interp_methods: SpatialInterpMethods | None,
    key: Hashable,
    var: xr.DataArray,
) -> SpatialInterpMethod:
    def assign_defaults(data_type: np.dtype) -> SpatialInterpMethod:
        if np.issubdtype(data_type, np.bool_):
            return 0
        if np.issubdtype(data_type, np.integer):
            return 0
        return 1

    if isinstance(interp_methods, Mapping):
        interp_method = interp_methods.get(str(key), interp_methods.get(var.dtype))
        if interp_method is None:
            LOG.debug(
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


def _get_spatial_interp_method_int(
    interp_methods: SpatialInterpMethods | None,
    key: Hashable,
    var: xr.DataArray,
) -> SpatialInterpMethodInt:
    interp_method = _get_spatial_interp_method(interp_methods, key, var)
    if isinstance(interp_method, str):
        interp_method = INTERP_METHOD_MAPPING[interp_method]
    return interp_method


def _get_spatial_interp_method_str(
    interp_methods: SpatialInterpMethods | None,
    key: Hashable,
    var: xr.DataArray,
) -> SpatialInterpMethodStr:
    interp_method = _get_spatial_interp_method(interp_methods, key, var)
    if isinstance(interp_method, int):
        interp_method = INTERP_METHOD_MAPPING[interp_method]
    return interp_method


def _prep_spatial_interp_methods_downscale(
    interp_methods: SpatialInterpMethods | None,
) -> SpatialInterpMethods | None:
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


def _get_spatial_interp_methods(
    ds: xr.Dataset, interp_methods: SpatialInterpMethods | None
) -> Mapping[Hashable, SpatialInterpMethod]:
    interp_methods_new = {}
    for name, var in ds.data_vars.items():
        interp_methods_new[name] = _get_spatial_interp_method(interp_methods, name, var)
    return interp_methods_new


def _get_spatial_agg_method(
    agg_methods: SpatialAggMethods | None,
    key: Hashable,
    var: xr.DataArray,
) -> Callable:
    def assign_defaults(data_type: np.dtype) -> SpatialAggMethod:
        if np.issubdtype(data_type, np.bool_):
            return "center"
        if np.issubdtype(data_type, np.integer):
            return "center"
        return "mean"

    if isinstance(agg_methods, Mapping):
        agg_method = agg_methods.get(str(key), agg_methods.get(var.dtype))
        if agg_method is None:
            LOG.debug(
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


def _get_prevent_nan_propagation(
    prevent_nan_propagations: PreventNaNPropagations | None,
    key: Hashable,
    var: xr.DataArray,
) -> bool:
    if isinstance(prevent_nan_propagations, Mapping):
        prevent_nan_propagation = prevent_nan_propagations.get(
            str(key), prevent_nan_propagations.get(var.dtype)
        )
        if prevent_nan_propagation is None:
            LOG.debug(
                f"The method to prevent NaN propagation could not be derived from "
                f"the mapping `prevent_nan_propagations`  for data variable {key!r} "
                f"with data type {var.dtype!r}. Defaults are assigned."
            )
            prevent_nan_propagation = False
    elif isinstance(prevent_nan_propagations, bool):
        prevent_nan_propagation = prevent_nan_propagations
    else:
        prevent_nan_propagation = False

    return prevent_nan_propagation


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
        elif data_type == np.uint32:
            fill_value = FILLVALUE_UINT32
        elif np.issubdtype(data_type, np.integer):
            fill_value = FILLVALUE_INT
        elif np.issubdtype(data_type, np.bool_):
            fill_value = 0
        else:
            fill_value = FILLVALUE_FLOAT
        return fill_value

    if isinstance(fill_values, Mapping):
        fill_value = fill_values.get(str(key), fill_values.get(var.dtype))
        if fill_value is None:
            LOG.debug(
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


def _sample_array_at_indices(
    data: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    interp_method: SpatialInterpMethodStr,
) -> np.ndarray:
    """
    Sample a 3d array at fractional indices (iy, ix).
    """
    if interp_method == "nearest":
        ix_i = np.ceil(ix - 0.5).astype(np.intp)
        iy_i = np.ceil(iy - 0.5).astype(np.intp)
        return data[:, iy_i, ix_i]

    ix_floor = np.floor(ix).astype(np.intp)
    iy_floor = np.floor(iy).astype(np.intp)
    ix_ceil = np.ceil(ix).astype(np.intp)
    iy_ceil = np.ceil(iy).astype(np.intp)

    dx = ix - ix_floor
    dy = iy - iy_floor

    v00 = data[:, iy_floor, ix_floor]
    v01 = data[:, iy_floor, ix_ceil]
    v10 = data[:, iy_ceil, ix_floor]
    v11 = data[:, iy_ceil, ix_ceil]

    if interp_method == "bilinear":
        u0 = v00 + dx * (v01 - v00)
        u1 = v10 + dx * (v11 - v10)
        return u0 + dy * (u1 - u0)

    if interp_method == "triangular":
        dx3 = dx[np.newaxis, ...]
        dy3 = dy[np.newaxis, ...]

        tri1 = v00 + dx3 * (v01 - v00) + dy3 * (v10 - v00)
        tri2 = v11 + (1 - dx3) * (v10 - v11) + (1 - dy3) * (v01 - v11)
        mask3 = (dx + dy < 1.0)[np.newaxis, ...]
        return np.where(mask3, tri1, tri2)

    raise NotImplementedError(
        f"interp_methods must be one of 0, 1, 'nearest', 'bilinear', 'triangular', "
        f"was '{interp_method}'."
    )


def _create_empty_dataset(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    target_gm: GridMapping,
    fill_values: FillValues | None = None,
) -> xr.Dataset:
    x_name, y_name = source_gm.xy_var_names
    coords = source_ds.coords.to_dataset()
    coords = coords.drop_vars((x_name, y_name), errors="ignore")
    x_name, y_name = target_gm.xy_var_names
    coords[x_name] = target_gm.x_coords
    coords[y_name] = target_gm.y_coords
    coords["spatial_ref"] = xr.DataArray(0, attrs=target_gm.crs.to_cf())
    target_ds = xr.Dataset(coords=coords, attrs=source_ds.attrs)
    for key, data in source_ds.data_vars.items():
        shape = list(source_ds[key].shape)
        shape[-1] = target_gm.width
        shape[-2] = target_gm.height
        dims = list(source_ds[key].dims)
        dims[-1] = target_gm.xy_var_names[0]
        dims[-2] = target_gm.xy_var_names[1]
        if source_ds[key].ndim == 3:
            chunks = (
                source_ds[key].chunks[0][0],
                target_gm.height,
                target_gm.width,
            )
        else:
            chunks = (target_gm.height, target_gm.width)
        target_ds[key] = xr.DataArray(
            da.full(
                shape,
                fill_value=_get_fill_value(fill_values, key, data),
                chunks=chunks,
            ),
            dims=dims,
            attrs=source_ds[key].attrs,
        )
    return target_ds


def _clip_if_needed(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    target_gm: GridMapping,
    fill_values: FillValues | None,
    transformer: pyproj.Transformer | None = None,
) -> tuple[xr.Dataset, GridMapping, bool]:
    """
    Clip the source dataset to the target bounding box if overlap is limited.
    """
    if transformer is None:
        target_bbox = target_gm.xy_bbox
        buffer_x = 2 * max(source_gm.x_res, target_gm.x_res)
        buffer_y = 2 * max(source_gm.y_res, target_gm.y_res)
    else:
        target_bbox = transformer.transform_bounds(*target_gm.xy_bbox)
        target_x_res = (target_bbox[2] - target_bbox[0]) / target_gm.width
        target_y_res = (target_bbox[3] - target_bbox[1]) / target_gm.height
        buffer_x = 2 * max(source_gm.x_res, target_x_res)
        buffer_y = 2 * max(source_gm.y_res, target_y_res)

    overlap = bbox_overlap(source_gm.xy_bbox, target_bbox)
    if overlap == 0.0:
        LOG.info(
            "Target grid mapping does not overlap with the source grid mapping. "
            "Returning empty target dataset."
        )
        empty_ds = _create_empty_dataset(source_ds, source_gm, target_gm, fill_values)
        return empty_ds, target_gm, True
    if overlap >= 0.8:
        return source_ds, source_gm, False

    # Expand target bbox slightly to ensure coverage
    bbox = [
        target_bbox[0] - buffer_x,
        target_bbox[1] - buffer_y,
        target_bbox[2] + buffer_x,
        target_bbox[3] + buffer_y,
    ]
    clipped = clip_dataset_by_bbox(source_ds, bbox)

    if any(clipped.sizes[source_gm.xy_dim_names[i]] < 2 for i in range(2)):
        LOG.warning(
            "Clipped dataset contains a spatial dimension with size < 2. "
            "Returning empty target dataset."
        )
        empty_ds = _create_empty_dataset(source_ds, source_gm, target_gm, fill_values)
        return empty_ds, target_gm, True

    return clipped, GridMapping.from_dataset(clipped), False


@dataclass(frozen=True)
class SourceTileIndexing:
    ij_bboxes: np.ndarray  # (4, ny_tiles, nx_tiles)
    pad_width: tuple[tuple[int, int], tuple[int, int]]
    output_size: tuple[int, int]
    tile_size: tuple[int, int]


def _reorganize_tiled_array(
    array: da.Array,
    indexing: SourceTileIndexing,
    fill_value: FloatInt,
) -> da.Array:
    """
    Reorganize a 2D or 3D array into uniformly sized tiles based on source index bboxes.
    """

    is_3d = array.ndim == 3
    if not is_3d:
        array = array[None, ...]

    pad_width = ((0, 0),) + indexing.pad_width
    padded = da.pad(array, pad_width, mode="constant", constant_values=fill_value)

    out_shape = (
        array.shape[0],
        indexing.output_size[0],
        indexing.output_size[1],
    )
    chunks = (
        array.chunks[0],
        indexing.tile_size[0],
        indexing.tile_size[1],
    )

    out = da.zeros(out_shape, chunks=chunks, dtype=array.dtype)

    ny, nx = indexing.ij_bboxes.shape[1:]
    th, tw = indexing.tile_size

    for j in range(ny):
        for i in range(nx):
            bbox = indexing.ij_bboxes[:, j, i]

            y0, y1 = j * th, (j + 1) * th
            x0, x1 = i * tw, (i + 1) * tw

            if bbox[0] == -1:
                out[:, y0:y1, x0:x1] = da.full(
                    (array.shape[0], th, tw),
                    fill_value,
                    chunks=chunks,
                    dtype=array.dtype,
                )
            else:
                out[:, y0:y1, x0:x1] = padded[
                    :,
                    bbox[1] : bbox[3],
                    bbox[0] : bbox[2],
                ]

    return out if is_3d else out[0]


def _map_to_source_indices(pixel_target_ij, indexing):
    """Convert tile-local fractional source pixel indices to original source indices.

    Applies per-tile source index offsets and removes padding introduced during
    tiling, returning fractional (ix, iy) indices in the original source grid.
    """
    offsets_x = da.zeros_like(pixel_target_ij[0])
    offsets_y = da.zeros_like(pixel_target_ij[1])

    ny, nx = indexing.ij_bboxes.shape[1:]
    th, tw = indexing.tile_size

    for j in range(ny):
        for i in range(nx):
            bbox = indexing.ij_bboxes[:, j, i]
            offsets_x[th * j : th * (j + 1), tw * i : tw * (i + 1)] = bbox[0]
            offsets_y[th * j : th * (j + 1), tw * i : tw * (i + 1)] = bbox[1]

    offsets_x -= indexing.pad_width[1][0]
    offsets_y -= indexing.pad_width[0][0]

    return da.stack([pixel_target_ij[0] + offsets_x, pixel_target_ij[1] + offsets_y])
