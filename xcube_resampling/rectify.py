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

from collections.abc import Hashable, Iterable

import dask.array as da
import numba as nb
import numpy as np
import pyproj
import xarray as xr

from .affine import resample_dataset
from .constants import (
    LOG,
    SCALE_LIMIT,
    UV_DELTA,
    FillValues,
    FloatInt,
    PreventNaNPropagations,
    SpatialAggMethods,
    SpatialInterpMethods,
    SpatialInterpMethodStr,
)
from .dask import compute_array_from_func
from .gridmapping import GridMapping
from .utils import (
    _get_fill_value,
    _get_spatial_interp_method_str,
    _is_equal_crs,
    _prep_spatial_interp_methods_downscale,
    _select_variables,
    bbox_overlap,
    clip_dataset_by_bbox,
    normalize_grid_mapping,
)


def rectify_dataset(
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
    output_indices_names: tuple[str, str] | None = None,
) -> xr.Dataset:
    """
    Rectify a dataset with non-regular grid to a regular grid defined by a target
    grid mapping.

    This function transforms spatial coordinates to a regular grid while preserving
    data values. It optionally downsamples high-resolution inputs prior to rectifying.

    Args:
        source_ds: The source dataset with 2D spatial coordinate variables.
        target_gm: Optional target grid mapping defining the output regular grid.
            If not provided, one is derived from the source grid mapping.
        source_gm: Optional grid mapping of the source dataset. If not given, it is
            inferred from the dataset.
        variables: Optional variable(s) to rectify. If None, all eligible variables
            are processed.
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

        tile_size: Optional tile size for inferring a regular grid, if `target_gm` is
            not provided.
        output_indices_names: Optional names for two variables that store the source
            pixel indices for the last and second-last dimension, respectively.

    Returns:
        A new dataset with spatial variables rectified to a regular grid.
            Variables not having 2D spatial dimensions are copied as-is. 1D spatial
            coordinate variables are ignored in the output.
    """
    source_ds = _select_variables(source_ds, variables)

    if source_gm is None:
        source_gm = GridMapping.from_dataset(source_ds)
    source_ds = normalize_grid_mapping(source_ds, source_gm)

    if target_gm is None:
        target_gm = source_gm.to_regular(tile_size=tile_size)

    # transform 2d spatial coordinate of source dataset to target CRS
    if not _is_equal_crs(source_gm, target_gm):
        source_ds = _transform_coords(source_ds, source_gm, target_gm)
        source_gm = GridMapping.from_dataset(source_ds)

    # if the bbox of the target grid mapping overlaps less than 80% with the
    # bounding box of the source grid mapping, clip the source dataset.
    overlap = bbox_overlap(source_gm.xy_bbox, target_gm.xy_bbox)
    if overlap < 1e-5:
        LOG.info(
            "Target grid mapping does not overlap with the source grid mapping. "
            "Target dataset filled with the respective fill value is returned."
        )
        return _create_empty_dataset(source_ds, source_gm, target_gm, fill_values)
    if overlap < 0.8:
        bbox = [
            target_gm.xy_bbox[0] - 2 * source_gm.x_res,
            target_gm.xy_bbox[1] - 2 * source_gm.y_res,
            target_gm.xy_bbox[2] + 2 * source_gm.x_res,
            target_gm.xy_bbox[3] + 2 * source_gm.x_res,
        ]
        source_ds = clip_dataset_by_bbox(source_ds, bbox)
        if any(source_ds.sizes[source_gm.xy_dim_names[i]] < 2 for i in range(2)):
            LOG.warning(
                "Clipped dataset contains at least dimension with size < 2. "
                "Target dataset filled with the respective fill value is returned."
            )
            return _create_empty_dataset(source_ds, source_gm, target_gm, fill_values)
        source_gm = GridMapping.from_dataset(source_ds)

    # If source has higher resolution than target, downscale first, then rectify
    source_ds, source_gm = _downscale_source_dataset(
        source_ds,
        source_gm,
        target_gm,
        interp_methods,
        agg_methods,
        prevent_nan_propagations,
    )

    # calculate source indices in target grid-mapping
    target_source_ij = _compute_target_source_ij(source_gm, target_gm, UV_DELTA)

    # rectify dataset
    x_name, y_name = source_gm.xy_var_names
    coords = source_ds.coords.to_dataset()
    coords = coords.drop_vars((x_name, y_name), errors="ignore")
    x_name, y_name = target_gm.xy_var_names
    target_coords = target_gm.to_coords()
    coords[x_name] = target_coords[x_name]
    coords[y_name] = target_coords[y_name]
    coords["spatial_ref"] = xr.DataArray(0, attrs=target_gm.crs.to_cf())
    target_ds = xr.Dataset(coords=coords, attrs=source_ds.attrs)

    yx_dims = (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0])
    for var_name, data_array in source_ds.data_vars.items():
        if data_array.dims[-2:] == yx_dims:
            assert len(data_array.dims) in (
                2,
                3,
            ), f"Data variable {var_name} has {len(data_array.dims)} dimensions."

            target_ds[var_name] = _rectify_data_array(
                data_array,
                var_name,
                target_gm,
                target_source_ij,
                interp_methods,
                fill_values,
            )

        elif yx_dims[0] not in data_array.dims and yx_dims[1] not in data_array.dims:
            target_ds[var_name] = data_array

    if output_indices_names:
        target_ds[output_indices_names[0]] = ((y_name, x_name), (target_source_ij[0]))
        target_ds[output_indices_names[1]] = ((y_name, x_name), (target_source_ij[1]))

    return target_ds


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


def _transform_coords(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    target_gm: GridMapping,
) -> xr.Dataset:
    source_xx = source_gm.x_coords.data
    source_yy = source_gm.y_coords.data
    if isinstance(source_xx, np.ndarray):
        is_numpy_array = True
        source_xx = da.asarray(source_xx)
        source_yy = da.asarray(source_yy)
    else:
        is_numpy_array = False

    transformer_forward = pyproj.Transformer.from_crs(
        source_gm.crs, target_gm.crs, always_xy=True
    )

    # get transformed coordinates
    # noinspection PyShadowingNames
    def transform_block(source_xx: np.ndarray, source_yy: np.ndarray):
        target_xx, target_yy = transformer_forward.transform(source_xx, source_yy)
        return np.stack([target_xx, target_yy])

    target_xx_yy = da.map_blocks(
        transform_block,
        source_xx,
        source_yy,
        dtype=np.float32,
        chunks=(2, source_yy.chunks[0][0], source_yy.chunks[1][0]),
    )
    target_xx_yy = target_xx_yy[:, : source_gm.height, : source_gm.width]
    source_ds = source_ds.drop_vars(source_gm.xy_var_names)
    yx_dims = (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0])
    yx_var_names = (
        ("lon", "lat")
        if target_gm.crs.is_geographic
        else ("transformed_x", "transformed_y")
    )
    if is_numpy_array:
        target_xx_yy = target_xx_yy.compute()
    source_ds = source_ds.assign_coords(
        {
            "spatial_ref": xr.DataArray(0, attrs=target_gm.crs.to_cf()),
            yx_var_names[0]: (yx_dims, target_xx_yy[0]),
            yx_var_names[1]: (yx_dims, target_xx_yy[1]),
        }
    )

    return source_ds


def _downscale_source_dataset(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    target_gm: GridMapping,
    interp_methods: SpatialInterpMethods | None,
    agg_methods: SpatialAggMethods | None,
    prevent_nan_propagations: PreventNaNPropagations,
) -> (xr.Dataset, GridMapping):
    if interp_methods in [0, "nearest"]:
        return source_ds, source_gm
    x_scale = source_gm.x_res / target_gm.x_res
    y_scale = source_gm.y_res / target_gm.y_res
    if x_scale < SCALE_LIMIT or y_scale < SCALE_LIMIT:
        w, h = np.floor(x_scale * source_gm.width), np.floor(y_scale * source_gm.height)
        downscaled_size = (w if w >= 2 else 2, h if h >= 2 else 2)

        source_ds = resample_dataset(
            source_ds,
            ((1 / x_scale, 0, 0), (0, 1 / y_scale, 0)),
            (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0]),
            downscaled_size,
            source_gm.tile_size,
            _prep_spatial_interp_methods_downscale(interp_methods),
            agg_methods,
            prevent_nan_propagations,
        )
        source_gm = GridMapping.from_dataset(source_ds)

    return source_ds, source_gm


def _rectify_data_array(
    data_array: xr.DataArray,
    var_name: Hashable,
    target_gm: GridMapping,
    target_source_ij: da.Array,
    interp_methods: SpatialInterpMethods | None = None,
    fill_values: FillValues | None = None,
) -> xr.DataArray:
    data_array_expanded = False
    if len(data_array.dims) == 2:
        data_array = data_array.expand_dims({"dummy": 1})
        data_array_expanded = True

    if isinstance(data_array.data, np.ndarray):
        is_numpy_array = True
        data_array = data_array.chunk({dim: -1 for dim in data_array.dims})
    else:
        is_numpy_array = False

    fill_value = _get_fill_value(fill_values, var_name, data_array)
    interp_method = _get_spatial_interp_method_str(interp_methods, var_name, data_array)

    # calculate rectification of each chunk along the 1st (non-spatial) dimension.
    slices_rectified = []
    dim0_end = 0
    for chunk_size in data_array.chunks[0]:
        dim0_start = dim0_end
        dim0_end = dim0_start + chunk_size

        data_rectified = _compute_var_image(
            data_array[dim0_start:dim0_end], target_source_ij, fill_value, interp_method
        )
        slices_rectified.append(data_rectified)
    array_rectified = da.concatenate(slices_rectified, axis=0)
    if is_numpy_array and not target_gm.is_tiled:
        array_rectified = array_rectified.compute()
    if data_array_expanded:
        array_rectified = array_rectified[0, :, :]
        dims = (target_gm.xy_dim_names[1], target_gm.xy_dim_names[0])
    else:
        dims = (
            data_array.dims[0],
            target_gm.xy_dim_names[1],
            target_gm.xy_dim_names[0],
        )

    return xr.DataArray(data=array_rectified, dims=dims, attrs=data_array.attrs)


def _compute_target_source_ij(
    src_geo_coding: GridMapping, output_geom: GridMapping, uv_delta: float
) -> da.Array:
    """Compute dask.array.Array destination image
    with source pixel i,j coords from xarray.DataArray x,y sources.
    """
    dst_width = output_geom.width
    dst_height = output_geom.height
    dst_tile_width = output_geom.tile_width
    dst_tile_height = output_geom.tile_height
    dst_var_shape = 2, dst_height, dst_width
    dst_var_chunks = 2, dst_tile_height, dst_tile_width

    dst_x_min, dst_y_min, dst_x_max, dst_y_max = output_geom.xy_bbox
    dst_x_res, dst_y_res = output_geom.xy_res
    dst_is_j_axis_up = output_geom.is_j_axis_up

    # Compute an empirical xy_border as a function of the
    # number of tiles, because the more tiles we have
    # the smaller the destination xy-bboxes and the higher
    # the risk to not find any source ij-bbox for a given xy-bbox.
    # xy_border will not be larger than half of the
    # coverage of a tile.
    num_tiles_x = dst_width / dst_tile_width
    num_tiles_y = dst_height / dst_tile_height
    xy_border = min(
        min(2 * num_tiles_x * output_geom.x_res, 2 * num_tiles_y * output_geom.y_res),
        min(0.5 * (dst_x_max - dst_x_min), 0.5 * (dst_y_max - dst_y_min)),
    )

    dst_xy_bboxes = output_geom.xy_bboxes
    src_ij_bboxes = src_geo_coding.ij_bboxes_from_xy_bboxes(
        dst_xy_bboxes, xy_border=xy_border, ij_border=1
    )

    return compute_array_from_func(
        _compute_target_source_ij_block,
        dst_var_shape,
        dst_var_chunks,
        np.float64,
        ctx_arg_names=[
            "dtype",
            "block_id",
            "block_shape",
            "block_slices",
        ],
        args=(
            src_geo_coding.xy_coords,
            src_ij_bboxes,
            dst_x_min,
            dst_y_min,
            dst_y_max,
            dst_x_res,
            dst_y_res,
            dst_is_j_axis_up,
            uv_delta,
        ),
        name="ij_pixels",
    )


def _compute_target_source_ij_block(
    dtype: np.dtype,
    block_id: int,
    block_shape: tuple[int, int],
    block_slices: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    src_xy_coords: xr.DataArray,
    src_ij_bboxes: np.ndarray,
    dst_x_min: float,
    dst_y_min: float,
    dst_y_max: float,
    dst_x_res: float,
    dst_y_res: float,
    dst_is_j_axis_up: bool,
    uv_delta: float,
) -> np.ndarray:
    """Compute dask.array.Array destination block with source
    pixel i,j coords from xarray.DataArray x,y sources.
    """
    dst_src_ij_block = np.full(block_shape, np.nan, dtype=dtype)
    _, (dst_y_slice_start, _), (dst_x_slice_start, _) = block_slices
    src_ij_bbox = src_ij_bboxes[block_id]
    src_i_min, src_j_min, src_i_max, src_j_max = src_ij_bbox
    if src_i_min == -1:
        return dst_src_ij_block
    src_xy_values = src_xy_coords[
        :, src_j_min : src_j_max + 1, src_i_min : src_i_max + 1
    ].values
    src_x_values = src_xy_values[0]
    src_y_values = src_xy_values[1]
    dst_x_offset = dst_x_min + dst_x_slice_start * dst_x_res
    if dst_is_j_axis_up:
        dst_y_offset = dst_y_min + dst_y_slice_start * dst_y_res
    else:
        dst_y_offset = dst_y_max - dst_y_slice_start * dst_y_res
    _compute_target_source_ij_sequential(
        src_x_values,
        src_y_values,
        src_i_min,
        src_j_min,
        dst_src_ij_block,
        dst_x_offset,
        dst_y_offset,
        dst_x_res,
        dst_y_res if dst_is_j_axis_up else -dst_y_res,
        uv_delta,
    )
    return dst_src_ij_block


# Extra dask version, because if we use parallel=True
# and nb.prange, we end up in infinite JIT compilation :(
@nb.njit(nogil=True, cache=True)
def _compute_target_source_ij_sequential(
    src_x_image: np.ndarray,
    src_y_image: np.ndarray,
    src_i_min: int,
    src_j_min: int,
    dst_src_ij_images: np.ndarray,
    dst_x_offset: float,
    dst_y_offset: float,
    dst_x_scale: float,
    dst_y_scale: float,
    uv_delta: float,
):
    """Compute numpy.ndarray destination image with source pixel i,j coords
    from numpy.ndarray x,y sources NOT in parallel mode.
    """
    src_height = src_x_image.shape[-2]
    dst_src_ij_images[:, :, :] = np.nan
    for src_j0 in range(src_height - 1):
        _compute_target_source_ij_line(
            src_j0,
            src_x_image,
            src_y_image,
            src_i_min,
            src_j_min,
            dst_src_ij_images,
            dst_x_offset,
            dst_y_offset,
            dst_x_scale,
            dst_y_scale,
            uv_delta,
        )


@nb.njit(nogil=True, cache=True)
def _compute_target_source_ij_line(
    src_j0: int,
    src_x_image: np.ndarray,
    src_y_image: np.ndarray,
    src_i_min: int,
    src_j_min: int,
    dst_src_ij_images: np.ndarray,
    dst_x_offset: float,
    dst_y_offset: float,
    dst_x_scale: float,
    dst_y_scale: float,
    uv_delta: float,
):
    """Compute numpy.ndarray destination image with source
    pixel i,j coords from a numpy.ndarray x,y source line.
    """
    src_width = src_x_image.shape[-1]

    dst_width = dst_src_ij_images.shape[-1]
    dst_height = dst_src_ij_images.shape[-2]

    dst_px = np.zeros(4, dtype=src_x_image.dtype)
    dst_py = np.zeros(4, dtype=src_y_image.dtype)

    u_min = v_min = -uv_delta
    uv_max = 1.0 + 2 * uv_delta

    for src_i0 in range(src_width - 1):
        src_i1 = src_i0 + 1
        src_j1 = src_j0 + 1

        dst_px[0] = dst_p0x = src_x_image[src_j0, src_i0]
        dst_px[1] = dst_p1x = src_x_image[src_j0, src_i1]
        dst_px[2] = dst_p2x = src_x_image[src_j1, src_i0]
        dst_px[3] = dst_p3x = src_x_image[src_j1, src_i1]

        dst_py[0] = dst_p0y = src_y_image[src_j0, src_i0]
        dst_py[1] = dst_p1y = src_y_image[src_j0, src_i1]
        dst_py[2] = dst_p2y = src_y_image[src_j1, src_i0]
        dst_py[3] = dst_p3y = src_y_image[src_j1, src_i1]

        dst_pi = np.floor((dst_px - dst_x_offset) / dst_x_scale).astype(np.int64)
        dst_pj = np.floor((dst_py - dst_y_offset) / dst_y_scale).astype(np.int64)

        dst_i_min = np.min(dst_pi)
        dst_i_max = np.max(dst_pi)
        dst_j_min = np.min(dst_pj)
        dst_j_max = np.max(dst_pj)

        if (
            dst_i_max < 0
            or dst_j_max < 0
            or dst_i_min >= dst_width
            or dst_j_min >= dst_height
        ):
            continue

        if dst_i_min < 0:
            dst_i_min = 0

        if dst_i_max >= dst_width:
            dst_i_max = dst_width - 1

        if dst_j_min < 0:
            dst_j_min = 0

        if dst_j_max >= dst_height:
            dst_j_max = dst_height - 1

        # u from p0 right to p1, v from p0 down to p2
        # noinspection PyTypeChecker
        det_a = _fdet(dst_p0x, dst_p0y, dst_p1x, dst_p1y, dst_p2x, dst_p2y)
        if np.isnan(det_a):
            det_a = 0.0

        # u from p3 left to p2, v from p3 up to p1
        # noinspection PyTypeChecker
        det_b = _fdet(dst_p3x, dst_p3y, dst_p2x, dst_p2y, dst_p1x, dst_p1y)
        if np.isnan(det_b):
            det_b = 0.0

        if det_a == 0.0 and det_b == 0.0:
            # Both the triangles do not exist.
            continue

        for dst_j in range(dst_j_min, dst_j_max + 1):
            dst_y = dst_y_offset + (dst_j + 0.5) * dst_y_scale
            for dst_i in range(dst_i_min, dst_i_max + 1):
                sentinel = dst_src_ij_images[0, dst_j, dst_i]
                if not np.isnan(sentinel):
                    # If we have a source pixel in dst_i, dst_j already,
                    # there is no need to compute another one.
                    # One is as good as the other.
                    continue

                dst_x = dst_x_offset + (dst_i + 0.5) * dst_x_scale

                src_i = src_j = -1

                if det_a != 0.0:
                    # noinspection PyTypeChecker
                    u = _fu(dst_x, dst_y, dst_p0x, dst_p0y, dst_p2x, dst_p2y) / det_a
                    # noinspection PyTypeChecker
                    v = _fv(dst_x, dst_y, dst_p0x, dst_p0y, dst_p1x, dst_p1y) / det_a
                    if u >= u_min and v >= v_min and u + v <= uv_max:
                        src_i = src_i0 + _fclamp(u, 0.0, 1.0)
                        src_j = src_j0 + _fclamp(v, 0.0, 1.0)
                if src_i == -1 and det_b != 0.0:
                    # noinspection PyTypeChecker
                    u = _fu(dst_x, dst_y, dst_p3x, dst_p3y, dst_p1x, dst_p1y) / det_b
                    # noinspection PyTypeChecker
                    v = _fv(dst_x, dst_y, dst_p3x, dst_p3y, dst_p2x, dst_p2y) / det_b
                    if u >= u_min and v >= v_min and u + v <= uv_max:
                        src_i = src_i1 - _fclamp(u, 0.0, 1.0)
                        src_j = src_j1 - _fclamp(v, 0.0, 1.0)
                if src_i != -1:
                    dst_src_ij_images[0, dst_j, dst_i] = src_i_min + src_i
                    dst_src_ij_images[1, dst_j, dst_i] = src_j_min + src_j


def _compute_var_image(
    src_var: xr.DataArray,
    dst_src_ij_images: da.Array,
    fill_value: FloatInt,
    interp_method: SpatialInterpMethodStr,
) -> da.Array:
    """Extract source pixels from xarray.DataArray source
    with dask.array.Array data.
    """
    # Retrieve the chunk size required for `da.map_blocks`, as the resulting array
    # will have a different shape.
    chunksize = src_var.shape[:-2] + tuple(c[0] for c in dst_src_ij_images.chunks[-2:])
    arr = da.map_blocks(
        _compute_var_image_block,
        dst_src_ij_images,
        src_var,
        fill_value,
        interp_method,
        chunksize,
        dtype=src_var.dtype,
        chunks=chunksize,
    )
    arr = arr[..., : dst_src_ij_images.shape[-2], : dst_src_ij_images.shape[-1]]
    return arr


def _compute_var_image_block(
    dst_src_ij_images: np.ndarray,
    src_var_image: xr.DataArray,
    fill_value: FloatInt,
    interp_method: SpatialInterpMethodStr,
    chunksize: tuple[int],
) -> np.ndarray:
    """Extract source pixels from np.ndarray source
    and return a block of a dask array.
    """
    dst_width = dst_src_ij_images.shape[-1]
    dst_height = dst_src_ij_images.shape[-2]
    dst_shape = src_var_image.shape[:-2] + (dst_height, dst_width)
    dst_out = np.full(chunksize, fill_value, dtype=src_var_image.dtype)
    if np.all(np.isnan(dst_src_ij_images[0])):
        return dst_out
    dst_values = np.full(dst_shape, fill_value, dtype=src_var_image.dtype)
    src_bbox = (
        int(np.nanmin(dst_src_ij_images[0])),
        int(np.nanmin(dst_src_ij_images[1])),
        min(int(np.nanmax(dst_src_ij_images[0])) + 2, src_var_image.shape[-1]),
        min(int(np.nanmax(dst_src_ij_images[1])) + 2, src_var_image.shape[-2]),
    )
    src_var_image = src_var_image[
        ..., src_bbox[1] : src_bbox[3], src_bbox[0] : src_bbox[2]
    ].values.astype(np.float64)
    _compute_var_image_sequential(
        src_var_image, dst_src_ij_images, dst_values, src_bbox, interp_method
    )
    dst_out[..., :dst_height, :dst_width] = dst_values
    return dst_out


# Extra dask version, because if we use parallel=True
# and nb.prange, we end up in infinite JIT compilation :(
@nb.njit(nogil=True, cache=True)
def _compute_var_image_sequential(
    src_var_image: np.ndarray,
    dst_src_ij_images: np.ndarray,
    dst_var_image: np.ndarray,
    src_bbox: tuple[int, int, int, int],
    interp_method: SpatialInterpMethodStr,
):
    """Extract source pixels from np.ndarray source
    NOT using numba parallel mode.
    """
    dst_height = dst_var_image.shape[-2]
    for dst_j in range(dst_height):
        _compute_var_image_for_dest_line(
            dst_j,
            src_var_image,
            dst_src_ij_images,
            dst_var_image,
            src_bbox,
            interp_method,
        )


@nb.njit(nogil=True, cache=True)
def _compute_var_image_for_dest_line(
    dst_j: int,
    src_var_image: np.ndarray,
    dst_src_ij_images: np.ndarray,
    dst_var_image: np.ndarray,
    src_bbox: tuple[int, int, int, int],
    interp_method: SpatialInterpMethodStr,
):
    """Extract source pixels from *src_values* np.ndarray
    and write into dst_values np.ndarray.
    """
    src_width = src_var_image.shape[-1]
    src_height = src_var_image.shape[-2]
    dst_width = dst_var_image.shape[-1]
    src_i_min = 0
    src_j_min = 0
    src_i_max = src_width - 1
    src_j_max = src_height - 1
    for dst_i in range(dst_width):
        src_i_f = dst_src_ij_images[0, dst_j, dst_i] - src_bbox[0]
        src_j_f = dst_src_ij_images[1, dst_j, dst_i] - src_bbox[1]
        if np.isnan(src_i_f) or np.isnan(src_j_f):
            continue
        # Note int() is 2x faster than math.floor() and
        # should yield the same results for only positive i,j.
        src_i0 = int(src_i_f)
        src_j0 = int(src_j_f)
        u = src_i_f - src_i0
        v = src_j_f - src_j0
        if interp_method == "nearest":
            if u > 0.5:
                src_i0 = _iclamp(src_i0 + 1, src_i_min, src_i_max)
            if v > 0.5:
                src_j0 = _iclamp(src_j0 + 1, src_j_min, src_j_max)
            dst_var_value = src_var_image[..., src_j0, src_i0]
        elif interp_method == "triangular":
            src_i1 = _iclamp(src_i0 + 1, src_i_min, src_i_max)
            src_j1 = _iclamp(src_j0 + 1, src_j_min, src_j_max)
            value_01 = src_var_image[..., src_j0, src_i1]
            value_10 = src_var_image[..., src_j1, src_i0]
            if u + v < 1.0:
                # Closest triangle
                value_00 = src_var_image[..., src_j0, src_i0]
                dst_var_value = (
                    value_00 + u * (value_01 - value_00) + v * (value_10 - value_00)
                )
            else:
                # Opposite triangle
                value_11 = src_var_image[..., src_j1, src_i1]
                dst_var_value = (
                    value_11
                    + (1.0 - u) * (value_10 - value_11)
                    + (1.0 - v) * (value_01 - value_11)
                )
        elif interp_method == "bilinear":
            src_i1 = _iclamp(src_i0 + 1, src_i_min, src_i_max)
            src_j1 = _iclamp(src_j0 + 1, src_j_min, src_j_max)
            value_00 = src_var_image[..., src_j0, src_i0]
            value_01 = src_var_image[..., src_j0, src_i1]
            value_10 = src_var_image[..., src_j1, src_i0]
            value_11 = src_var_image[..., src_j1, src_i1]
            value_u0 = value_00 + u * (value_01 - value_00)
            value_u1 = value_10 + u * (value_11 - value_10)
            dst_var_value = value_u0 + v * (value_u1 - value_u0)
        else:
            raise NotImplementedError(
                f"interp_methods must be one of 0, 1, 'nearest', 'bilinear', "
                f"'triangular', was '{interp_method}'."
            )

        dst_var_image[..., dst_j, dst_i] = dst_var_value


@nb.njit(
    "float64(float64, float64, float64, float64, float64, float64)",
    nogil=True,
    inline="always",
)
def _fdet(
    px0: float, py0: float, px1: float, py1: float, px2: float, py2: float
) -> float:
    return (px0 - px1) * (py0 - py2) - (px0 - px2) * (py0 - py1)


@nb.njit(
    "float64(float64, float64, float64, float64, float64, float64)",
    nogil=True,
    inline="always",
)
def _fu(px: float, py: float, px0: float, py0: float, px2: float, py2: float) -> float:
    return (px0 - px) * (py0 - py2) - (py0 - py) * (px0 - px2)


@nb.njit(
    "float64(float64, float64, float64, float64, float64, float64)",
    nogil=True,
    inline="always",
)
def _fv(px: float, py: float, px0: float, py0: float, px1: float, py1: float) -> float:
    return (py0 - py) * (px0 - px1) - (px0 - px) * (py0 - py1)


@nb.njit("float64(float64, float64, float64)", nogil=True, inline="always")
def _fclamp(x: float, x_min: float, x_max: float) -> float:
    return x_min if x < x_min else (x_max if x > x_max else x)


@nb.njit("int64(int64, int64, int64)", nogil=True, inline="always")
def _iclamp(x: int, x_min: int, x_max: int) -> int:
    return x_min if x < x_min else (x_max if x > x_max else x)
