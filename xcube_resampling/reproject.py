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
from collections.abc import Hashable, Iterable

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from .affine import affine_transform_dataset
from .constants import (
    SCALE_LIMIT,
    FillValues,
    FloatInt,
    PreventNaNPropagations,
    SpatialAggMethods,
    SpatialInterpMethods,
    SpatialInterpMethodStr,
)
from .gridmapping import GridMapping
from .utils import (
    _get_fill_value,
    _get_spatial_interp_method_str,
    _prep_spatial_interp_methods_downscale,
    _select_variables,
    clip_dataset_by_bbox,
    normalize_grid_mapping,
)


def reproject_dataset(
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
    Reproject a dataset from one coordinate reference system (CRS) to another.

    This function transforms a dataset’s 2D spatial variables to match a new CRS and
    grid layout defined by `target_gm`. It handles interpolation, optional
    downsampling, and fill values for areas outside the input bounds.

    Args:
        source_ds: The input dataset to be reprojected.
        target_gm: The target grid mapping that defines the spatial reference and
            grid layout in the target CRS.
        source_gm: Optional source grid mapping of the input dataset. If not
            provided, it is inferred from the dataset.
        variables: Optional variable name or list of variable names to reproject.
            If None, all suitable variables are processed.
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
        fill_values: Optional fill value(s) to assign outside source coverage.
            Can be a single value or dictionary by variable or type. If not set,
            defaults are:

            - float: NaN
            - uint8: 255
            - uint16: 65535
            - other integers: -1

    Returns:
        A new dataset with variables reprojected to the target CRS and
            grid. Variables without 2D spatial dimensions are copied as-is.
            1D spatial coordinate variables are ignored in the output.
    """

    if source_gm is None:
        source_gm = GridMapping.from_dataset(source_ds)
    if source_gm.is_j_axis_up:
        v_var = source_gm.xy_var_names[1]
        source_ds = source_ds.isel({v_var: slice(None, None, -1)})
        source_gm = GridMapping.from_dataset(source_ds)

    source_ds = normalize_grid_mapping(source_ds, source_gm)

    source_ds = _select_variables(source_ds, variables)

    transformer = pyproj.Transformer.from_crs(
        target_gm.crs, source_gm.crs, always_xy=True
    )

    # If source has higher resolution than target, downscale first, then reproject
    source_ds, source_gm = _downscale_source_dataset(
        source_ds,
        source_gm,
        target_gm,
        transformer,
        interp_methods,
        agg_methods,
        prevent_nan_propagations,
    )

    # For each bounding box in the target grid mapping:
    # - determine the indices of the bbox in the source dataset
    # - extract the corresponding coordinates for each bbox in the source dataset
    # - compute the pad_width to handle areas requested by target_gm that exceed the
    #   bounds of source_gm.
    scr_ij_bboxes, x_coords, y_coords, pad_width = _get_scr_bboxes_indices(
        transformer, source_gm, target_gm
    )

    # transform grid points from target grid mapping to source grid mapping
    source_xx, source_yy = _transform_gridpoints(transformer, target_gm)

    # reproject dataset
    x_name, y_name = source_gm.xy_var_names
    coords = source_ds.coords.to_dataset()
    coords = coords.drop_vars((x_name, y_name), errors="ignore")
    x_name, y_name = target_gm.xy_var_names
    coords[x_name] = target_gm.x_coords
    coords[y_name] = target_gm.y_coords
    coords["spatial_ref"] = xr.DataArray(0, attrs=target_gm.crs.to_cf())
    target_ds = xr.Dataset(coords=coords, attrs=source_ds.attrs)

    yx_dims = (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0])
    for var_name, data_array in source_ds.items():
        if data_array.dims[-2:] == yx_dims:
            assert len(data_array.dims) in (
                2,
                3,
            ), f"Data variable {var_name} has {len(data_array.dims)} dimensions."

            target_ds[var_name] = _reproject_data_array(
                data_array,
                var_name,
                source_gm,
                target_gm,
                source_xx,
                source_yy,
                x_coords,
                y_coords,
                scr_ij_bboxes,
                pad_width,
                interp_methods,
                fill_values,
            )
        elif yx_dims[0] not in data_array.dims and yx_dims[1] not in data_array.dims:
            target_ds[var_name] = data_array

    return target_ds


def _reproject_data_array(
    data_array: xr.DataArray,
    var_name: Hashable,
    source_gm: GridMapping,
    target_gm: GridMapping,
    source_xx: da.Array,
    source_yy: da.Array,
    x_coords: da.Array,
    y_coords: da.Array,
    scr_ij_bboxes: np.ndarray,
    pad_width: tuple[tuple[int]],
    interp_methods: SpatialInterpMethods | None = None,
    fill_values: FillValues | None = None,
) -> xr.DataArray:
    data_array_expanded = False
    if len(data_array.dims) == 2:
        data_array = data_array.expand_dims({"dummy": 1})
        data_array_expanded = True

    if isinstance(data_array.data, np.ndarray):
        is_numpy_array = True
        array = da.asarray(data_array.data)
    else:
        is_numpy_array = False
        array = data_array.data

    # reorganize data array slice to align with the
    # chunks of source_xx and source_yy
    fill_value = _get_fill_value(fill_values, var_name, data_array)
    interp_method = _get_spatial_interp_method_str(interp_methods, var_name, data_array)
    scr_data = _reorganize_data_array_slice(
        array,
        x_coords,
        y_coords,
        scr_ij_bboxes,
        pad_width,
        fill_value,
    )
    slices_reprojected = []
    # calculate reprojection of each chunk along the 1st (non-spatial) dimension.
    dim0_end = 0
    for chunk_size in array.chunks[0]:
        dim0_start = dim0_end
        dim0_end = dim0_start + chunk_size

        data_reprojected = da.map_blocks(
            _reproject_block,
            source_xx,
            source_yy,
            scr_data[dim0_start:dim0_end],
            x_coords,
            y_coords,
            dtype=data_array.dtype,
            chunks=(
                scr_data[dim0_start:dim0_end].shape[0],
                source_yy.chunks[0][0],
                source_yy.chunks[1][0],
            ),
            scr_x_res=source_gm.x_res,
            scr_y_res=source_gm.y_res,
            interp_method=interp_method,
        )
        data_reprojected = data_reprojected[:, : target_gm.height, : target_gm.width]
        slices_reprojected.append(data_reprojected)
    array_reprojected = da.concatenate(slices_reprojected, axis=0)
    if is_numpy_array:
        array_reprojected = array_reprojected.compute()
    if data_array_expanded:
        array_reprojected = array_reprojected[0, :, :]
        dims = (target_gm.xy_dim_names[1], target_gm.xy_dim_names[0])
    else:
        dims = (
            data_array.dims[0],
            target_gm.xy_dim_names[1],
            target_gm.xy_dim_names[0],
        )
    return xr.DataArray(data=array_reprojected, dims=dims, attrs=data_array.attrs)


def _reproject_block(
    source_xx: np.ndarray,
    source_yy: np.ndarray,
    scr_data: np.ndarray,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    scr_x_res: int | float,
    scr_y_res: int | float,
    interp_method: SpatialInterpMethodStr,
) -> np.ndarray:
    ix = (source_xx - x_coord[0]) / scr_x_res
    iy = (source_yy - y_coord[0]) / -scr_y_res

    if interp_method == "nearest":
        ix = np.rint(ix).astype(np.int16)
        iy = np.rint(iy).astype(np.int16)
        data_reprojected = scr_data[:, iy, ix]
    elif interp_method == "triangular":
        ix_ceil = np.ceil(ix).astype(np.int16)
        ix_floor = np.floor(ix).astype(np.int16)
        iy_ceil = np.ceil(iy).astype(np.int16)
        iy_floor = np.floor(iy).astype(np.int16)
        diff_ix = ix - ix_floor
        diff_iy = iy - iy_floor
        value_00 = scr_data[:, iy_floor, ix_floor]
        value_01 = scr_data[:, iy_floor, ix_ceil]
        value_10 = scr_data[:, iy_ceil, ix_floor]
        value_11 = scr_data[:, iy_ceil, ix_ceil]
        mask = diff_ix + diff_iy < 1.0
        mask_3d = np.repeat(mask[np.newaxis, :, :], scr_data.shape[0], axis=0)
        diff_ix = np.repeat(diff_ix[np.newaxis, :, :], scr_data.shape[0], axis=0)
        diff_iy = np.repeat(diff_iy[np.newaxis, :, :], scr_data.shape[0], axis=0)
        data_reprojected = np.zeros(
            (scr_data.shape[0], iy.shape[0], iy.shape[1]), dtype=scr_data.dtype
        )
        # Closest triangle
        data_reprojected[mask_3d] = (
            value_00[mask_3d]
            + diff_ix[mask_3d] * (value_01[mask_3d] - value_00[mask_3d])
            + diff_iy[mask_3d] * (value_10[mask_3d] - value_00[mask_3d])
        )
        # Opposite triangle
        data_reprojected[~mask_3d] = (
            value_11[~mask_3d]
            + (1.0 - diff_ix[~mask_3d]) * (value_10[~mask_3d] - value_11[~mask_3d])
            + (1.0 - diff_iy[~mask_3d]) * (value_01[~mask_3d] - value_11[~mask_3d])
        )
    elif interp_method == "bilinear":
        ix_ceil = np.ceil(ix).astype(np.int16)
        ix_floor = np.floor(ix).astype(np.int16)
        iy_ceil = np.ceil(iy).astype(np.int16)
        iy_floor = np.floor(iy).astype(np.int16)
        diff_ix = ix - ix_floor
        diff_iy = iy - iy_floor
        value_00 = scr_data[:, iy_floor, ix_floor]
        value_01 = scr_data[:, iy_floor, ix_ceil]
        value_10 = scr_data[:, iy_ceil, ix_floor]
        value_11 = scr_data[:, iy_ceil, ix_ceil]
        value_u0 = value_00 + diff_ix * (value_01 - value_00)
        value_u1 = value_10 + diff_ix * (value_11 - value_10)
        data_reprojected = value_u0 + diff_iy * (value_u1 - value_u0)
    else:
        raise NotImplementedError(
            f"interp_methods must be one of 0, 1, 'nearest', 'bilinear', 'triangular', "
            f"was '{interp_method}'."
        )

    return data_reprojected


def _downscale_source_dataset(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    target_gm: GridMapping,
    transformer: pyproj.Transformer,
    interp_methods: SpatialInterpMethods | None,
    agg_methods: SpatialAggMethods | None,
    prevent_nan_propagations: PreventNaNPropagations,
) -> (xr.Dataset, GridMapping):
    bbox_trans = transformer.transform_bounds(*target_gm.xy_bbox)
    xres_trans = (bbox_trans[2] - bbox_trans[0]) / target_gm.width
    yres_trans = (bbox_trans[3] - bbox_trans[1]) / target_gm.height
    x_scale = source_gm.x_res / xres_trans
    y_scale = source_gm.y_res / yres_trans
    if x_scale < SCALE_LIMIT or y_scale < SCALE_LIMIT:
        # clip source dataset to the transformed bounding box defined by
        # target grid mapping, so that affine_transform_dataset is not that heavy
        bbox_trans = (
            bbox_trans[0] - 2 * xres_trans,
            bbox_trans[1] - 2 * yres_trans,
            bbox_trans[2] + 2 * xres_trans,
            bbox_trans[3] + 2 * yres_trans,
        )
        source_ds = clip_dataset_by_bbox(source_ds, bbox_trans, source_gm.xy_dim_names)
        source_gm = GridMapping.from_dataset(source_ds)
        w, h = np.floor(x_scale * source_gm.width), np.floor(y_scale * source_gm.height)
        downscaled_size = (w if w >= 2 else 2, h if h >= 2 else 2)
        downscale_target_gm = GridMapping.regular(
            size=downscaled_size,
            xy_min=(
                source_gm.xy_bbox[0] + xres_trans / 2,
                source_gm.xy_bbox[1] + yres_trans / 2,
            ),
            xy_res=(xres_trans, yres_trans),
            crs=source_gm.crs,
            tile_size=source_gm.tile_size,
        )
        source_ds = affine_transform_dataset(
            source_ds,
            downscale_target_gm,
            source_gm=source_gm,
            interp_methods=_prep_spatial_interp_methods_downscale(interp_methods),
            agg_methods=agg_methods,
            prevent_nan_propagations=prevent_nan_propagations,
        )
        source_gm = GridMapping.from_dataset(source_ds)

    return source_ds, source_gm


def _get_scr_bboxes_indices(
    transformer: pyproj.Transformer,
    source_gm: GridMapping,
    target_gm: GridMapping,
) -> (np.ndarray, da.Array, da.Array, tuple[tuple[int]]):
    num_tiles_x = math.ceil(target_gm.width / target_gm.tile_width)
    num_tiles_y = math.ceil(target_gm.height / target_gm.tile_height)

    # get bboxes indices in source grid mapping
    origin = source_gm.x_coords.values[0], source_gm.y_coords.values[0]
    scr_ij_bboxes = np.full((4, num_tiles_y, num_tiles_x), -1, dtype=np.int32)
    for idx, xy_bbox in enumerate(target_gm.xy_bboxes):
        j, i = np.unravel_index(idx, (num_tiles_y, num_tiles_x))
        source_xy_bbox = transformer.transform_bounds(*xy_bbox)
        i_min = math.floor((source_xy_bbox[0] - origin[0]) / source_gm.x_res)
        i_max = math.ceil((source_xy_bbox[2] - origin[0]) / source_gm.x_res)
        j_min = math.floor((origin[1] - source_xy_bbox[3]) / source_gm.y_res)
        j_max = math.ceil((origin[1] - source_xy_bbox[1]) / source_gm.y_res)
        scr_ij_bboxes[:, j, i] = [i_min, j_min, i_max, j_max]

    # Extend bounding box indices to match the largest bounding box.
    # This ensures uniform chunk sizes, which are required for da.map_blocks.
    i_diff = scr_ij_bboxes[2] - scr_ij_bboxes[0]
    j_diff = scr_ij_bboxes[3] - scr_ij_bboxes[1]
    i_diff_max = np.max(i_diff) + 1
    j_diff_max = np.max(j_diff) + 1
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            scr_ij_bbox = scr_ij_bboxes[:, j, i]

            i_half = (i_diff_max - i_diff[j, i]) // 2
            i_start = scr_ij_bbox[0] - i_half
            i_end = i_start + i_diff_max

            j_half = (j_diff_max - j_diff[j, i]) // 2
            j_start = scr_ij_bbox[1] - j_half
            j_end = j_start + j_diff_max

            scr_ij_bboxes[:, j, i] = [i_start, j_start, i_end, j_end]

    # gather the coordinates; coordinates will be extended
    # if they are outside the source grid mapping
    x_coords = np.zeros((i_diff_max, num_tiles_y, num_tiles_x), dtype=np.float32)
    y_coords = np.zeros((j_diff_max, num_tiles_y, num_tiles_x), dtype=np.float32)
    i_min = np.min(scr_ij_bboxes[0])
    i_max = np.max(scr_ij_bboxes[2])
    j_min = np.min(scr_ij_bboxes[[1, 3]])
    j_max = np.max(scr_ij_bboxes[[1, 3]])
    x_start = source_gm.x_coords.values[0] + i_min * source_gm.x_res
    x_end = source_gm.x_coords.values[0] + i_max * source_gm.x_res
    x_coord = np.arange(x_start, x_end, source_gm.x_res)
    y_res = source_gm.y_coords.values[1] - source_gm.y_coords.values[0]
    y_start = source_gm.y_coords.values[0] + j_min * y_res
    y_end = source_gm.y_coords.values[0] + j_max * y_res
    y_coord = np.arange(y_start, y_end, y_res)
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            scr_ij_bbox = scr_ij_bboxes[:, j, i]

            i_start = scr_ij_bbox[0] - i_min
            i_end = i_start + i_diff_max
            x_coords[:, j, i] = x_coord[i_start:i_end]

            j_start = scr_ij_bbox[1] - j_min
            j_end = j_start + j_diff_max
            y_coords[:, j, i] = y_coord[j_start:j_end]

    x_coords = da.from_array(x_coords, chunks=(-1, 1, 1))
    y_coords = da.from_array(y_coords, chunks=(-1, 1, 1))

    pad_width = (
        (0, 0),
        (
            -min(0, int(j_min)),
            max(0, int(j_max - source_gm.height)),
        ),
        (
            -min(0, int(i_min)),
            max(0, int(i_max - source_gm.width)),
        ),
    )
    scr_ij_bboxes[[1, 3]] += pad_width[1][0]
    scr_ij_bboxes[[0, 2]] += pad_width[2][0]

    return scr_ij_bboxes, x_coords, y_coords, pad_width


def _transform_gridpoints(
    transformer: pyproj.Transformer, target_gm: GridMapping
) -> (da.Array, da.Array):
    # get meshed coordinates
    target_x = da.from_array(target_gm.x_coords.values, chunks=target_gm.tile_width)
    target_y = da.from_array(target_gm.y_coords.values, chunks=target_gm.tile_height)
    target_xx, target_yy = da.meshgrid(target_x, target_y)

    # get transformed coordinates
    # noinspection PyShadowingNames
    def transform_block(target_xx: np.ndarray, target_yy: np.ndarray):
        trans_xx, trans_yy = transformer.transform(target_xx, target_yy)
        return np.stack([trans_xx, trans_yy])

    source_xx_yy = da.map_blocks(
        transform_block,
        target_xx,
        target_yy,
        dtype=np.float32,
        chunks=(2, target_yy.chunks[0][0], target_yy.chunks[1][0]),
    )
    source_xx = source_xx_yy[0]
    source_yy = source_xx_yy[1]

    return source_xx, source_yy


def _reorganize_data_array_slice(
    array: da.Array,
    x_coords: da.Array,
    y_coords: da.Array,
    scr_ij_bboxes: np.ndarray,
    pad_width: tuple[tuple[int]],
    fill_value: FloatInt,
) -> da.Array:
    data_out = da.zeros(
        (
            array.shape[0],
            y_coords.shape[0] * scr_ij_bboxes.shape[1],
            x_coords.shape[0] * scr_ij_bboxes.shape[2],
        ),
        chunks=(array.chunks[0][0], y_coords.shape[0], x_coords.shape[0]),
        dtype=array.dtype,
    )
    data_in = da.pad(array, pad_width, mode="constant", constant_values=fill_value)
    for i in range(scr_ij_bboxes.shape[2]):
        for j in range(scr_ij_bboxes.shape[1]):
            scr_ij_bbox = scr_ij_bboxes[:, j, i]
            data_out[
                :,
                j * y_coords.shape[0] : (j + 1) * y_coords.shape[0],
                i * x_coords.shape[0] : (i + 1) * x_coords.shape[0],
            ] = data_in[
                :,
                scr_ij_bbox[1] : scr_ij_bbox[3],
                scr_ij_bbox[0] : scr_ij_bbox[2],
            ]

    return data_out
