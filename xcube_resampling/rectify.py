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
from collections.abc import Mapping, Sequence
from typing import Optional, Union

import dask.array as da
import numba as nb
import numpy as np
import xarray as xr
import pyproj

from xcube.util.dask import compute_array_from_func
from .gridmapping import GridMapping
from .constants import Aggregator, UV_DELTA
from .utils import clip_dataset_by_bbox


def rectify_dataset(
    source_ds: xr.Dataset,
    target_gm: GridMapping | None = None,
    source_gm: GridMapping | None = None,
    spline_orders: int | Mapping[np.dtype | str, int] | None = None,
    agg_methods: Aggregator | Mapping[np.dtype | str, Aggregator] | None = None,
    recover_nans: bool | Mapping[np.dtype | str, bool] = False,
    fill_values: int | float | Mapping[np.dtype | str, int | float] | None = None,
    target_tile_size: int | Sequence[int, int] | None = None,
) -> Optional[xr.Dataset]:
    if source_gm is None:
        source_gm = GridMapping.from_dataset(source_ds)

    if target_gm is None:
        target_gm = source_gm.to_regular(tile_size=target_tile_size)

    # clip source dataset to area which intersects with the target grid mapping
    transformer = pyproj.Transformer.from_crs(
        target_gm.crs, source_gm.crs, always_xy=True
    )
    bbox_trans = transformer.transform_bounds(*target_gm.xy_bbox)
    bbox_trans = [
        bbox_trans[0] - 2 * source_gm.x_res,
        bbox_trans[1] - 2 * source_gm.y_res,
        bbox_trans[2] + 2 * source_gm.x_res,
        bbox_trans[3] + 2 * source_gm.y_res,
    ]
    source_ds = clip_dataset_by_bbox(
        source_ds, bbox_trans, spatial_dims=target_gm.xy_dim_names
    )
    source_gm = GridMapping.from_dataset(source_ds)

    # rectify dataset
    x_name, y_name = source_gm.xy_dim_names
    coords = source_ds.coords.to_dataset()
    coords = coords.drop_vars((x_name, y_name))
    x_name, y_name = target_gm.xy_dim_names
    coords[x_name] = target_gm.x_coords
    coords[y_name] = target_gm.y_coords
    coords["spatial_ref"] = xr.DataArray(0, attrs=target_gm.crs.to_cf())
    ds_out = xr.Dataset(coords=coords, attrs=source_ds.attrs)

    dst_src_ij_array = _compute_ij_images(source_gm, target_gm)

    xy_dims = (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0])
    for var_name, data_array in source_ds.items():
        if data_array.dims[-2:] != xy_dims:
            continue
        # treat 2d arrays as 3d arrays
        assert len(data_array.dims) in (
            2,
            3,
        ), f"Data variable {var_name} has {len(data_array.dims)} dimensions."

        dst_var_dims = src_var.dims[0:-2] + dst_dims
        dst_var_coords = {
            d: src_var.coords[d] for d in dst_var_dims if d in src_var.coords
        }
        # noinspection PyTypeChecker
        dst_var_coords.update(
            {d: dst_ds_coords[d] for d in dst_var_dims if d in dst_ds_coords}
        )
        dst_var_array = _compute_var_image(
            src_var,
            dst_src_ij_array,
            fill_value=np.nan,
            interpolation=interpolation_mode,
        )
        dst_var = xr.DataArray(
            dst_var_array,
            dims=dst_var_dims,
            coords=dst_var_coords,
            attrs=src_var.attrs,
        )
        dst_vars[src_var_name] = dst_var

    if output_ij_names:
        output_i_name, output_j_name = output_ij_names
        dst_ij_coords = {d: dst_ds_coords[d] for d in dst_dims if d in dst_ds_coords}
        dst_vars[output_i_name] = xr.DataArray(
            dst_src_ij_array[0], dims=dst_dims, coords=dst_ij_coords
        )
        dst_vars[output_j_name] = xr.DataArray(
            dst_src_ij_array[1], dims=dst_dims, coords=dst_ij_coords
        )

    return complete_resampled_dataset(
        encode_cf,
        xr.Dataset(dst_vars, coords=dst_ds_coords, attrs=src_attrs),
        target_gm,
        gm_name,
        ref_ds.coords if ref_ds else None,
    )


def _select_variables(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    var_names: Union[None, str, Sequence[str]],
) -> Mapping[str, xr.DataArray]:
    """Select variables from *dataset*.

    Args:
        source_ds: Source dataset.
        source_gm: Optional dataset geo-coding.
        var_names: Optional variable name or sequence of variable names.

    Returns:
        The selected variables as a variable name to ``xr.DataArray``
        mapping
    """
    spatial_var_names = source_gm.xy_var_names
    spatial_shape = tuple(reversed(source_gm.size))
    spatial_dims = tuple(reversed(source_gm.xy_dim_names))
    if var_names is None:
        var_names = [
            var_name
            for var_name, var in source_ds.data_vars.items()
            if var_name not in spatial_var_names
            and _is_2d_spatial_var(var, spatial_shape, spatial_dims)
        ]
    elif isinstance(var_names, str):
        var_names = (var_names,)
    elif len(var_names) == 0:
        raise ValueError(f"empty var_names")
    src_vars = {}
    for var_name in var_names:
        src_var = source_ds[var_name]
        if not _is_2d_spatial_var(src_var, spatial_shape, spatial_dims):
            raise ValueError(
                f"cannot rectify variable {var_name!r}"
                f" as its shape or dimensions"
                f" do not match those of {spatial_var_names[0]!r}"
                f" and {spatial_var_names[1]!r}"
            )
        src_vars[var_name] = src_var
    return src_vars


def _is_2d_spatial_var(var: xr.DataArray, shape, dims) -> bool:
    return var.ndim >= 2 and var.shape[-2:] == shape and var.dims[-2:] == dims


def _compute_ij_images(source_gm: GridMapping, target_gm: GridMapping) -> da.Array:
    """Compute dask.array.Array destination image
    with source pixel i,j coords from xarray.DataArray x,y sources.
    """
    dst_width = target_gm.width
    dst_height = target_gm.height
    dst_tile_width = target_gm.tile_width
    dst_tile_height = target_gm.tile_height
    dst_var_shape = 2, dst_height, dst_width
    dst_var_chunks = 2, dst_tile_height, dst_tile_width

    dst_x_min, dst_y_min, dst_x_max, dst_y_max = target_gm.xy_bbox
    dst_x_res, dst_y_res = target_gm.xy_res
    dst_is_j_axis_up = target_gm.is_j_axis_up

    # Compute an empirical xy_border as a function of the
    # number of tiles, because the more tiles we have
    # the smaller the destination xy-bboxes and the higher
    # the risk to not find any source ij-bbox for a given xy-bbox.
    # xy_border will not be larger than half of the
    # coverage of a tile.
    #
    num_tiles_x = dst_width / dst_tile_width
    num_tiles_y = dst_height / dst_tile_height
    xy_border = min(
        min(2 * num_tiles_x * target_gm.x_res, 2 * num_tiles_y * target_gm.y_res),
        min(0.5 * (dst_x_max - dst_x_min), 0.5 * (dst_y_max - dst_y_min)),
    )

    dst_xy_bboxes = target_gm.xy_bboxes
    src_ij_bboxes = source_gm.ij_bboxes_from_xy_bboxes(
        dst_xy_bboxes, xy_border=xy_border, ij_border=1
    )

    return compute_array_from_func(
        _compute_ij_images_block,
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
            source_gm.xy_coords,
            src_ij_bboxes,
            dst_x_min,
            dst_y_min,
            dst_y_max,
            dst_x_res,
            dst_y_res,
            dst_is_j_axis_up,
        ),
        name="ij_pixels",
    )


def _compute_ij_images_block(
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
    _compute_ij_images_numpy_sequential(
        src_x_values,
        src_y_values,
        src_i_min,
        src_j_min,
        dst_src_ij_block,
        dst_x_offset,
        dst_y_offset,
        dst_x_res,
        dst_y_res if dst_is_j_axis_up else -dst_y_res,
    )
    return dst_src_ij_block


# Extra dask version, because if we use parallel=True
# and nb.prange, we end up in infinite JIT compilation :(
@nb.njit(nogil=True, cache=True)
def _compute_ij_images_numpy_sequential(
    src_x_image: np.ndarray,
    src_y_image: np.ndarray,
    src_i_min: int,
    src_j_min: int,
    dst_src_ij_images: np.ndarray,
    dst_x_offset: float,
    dst_y_offset: float,
    dst_x_scale: float,
    dst_y_scale: float,
):
    """Compute numpy.ndarray destination image with source pixel i,j coords
    from numpy.ndarray x,y sources NOT in parallel mode.
    """
    src_height = src_x_image.shape[-2]
    dst_src_ij_images[:, :, :] = np.nan
    for src_j0 in range(src_height - 1):
        _compute_ij_images_for_source_line(
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
        )


@nb.njit(nogil=True, cache=True)
def _compute_ij_images_for_source_line(
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
):
    """Compute numpy.ndarray destination image with source
    pixel i,j coords from a numpy.ndarray x,y source line.
    """
    src_width = src_x_image.shape[-1]

    dst_width = dst_src_ij_images.shape[-1]
    dst_height = dst_src_ij_images.shape[-2]

    dst_px = np.zeros(4, dtype=src_x_image.dtype)
    dst_py = np.zeros(4, dtype=src_y_image.dtype)

    u_min = v_min = -UV_DELTA
    uv_max = 1.0 + 2 * UV_DELTA

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
    fill_value: Union[int, float, complex] = np.nan,
    interpolation: int = 0,
) -> da.Array:
    """Extract source pixels from xarray.DataArray source
    with dask.array.Array data.
    """
    # If the source variable is not 3D, dummy variables are added to ensure
    # that `_compute_var_image_xarray_dask_block` always operates on 3D chunks
    # when processing each Dask chunk.
    if src_var.ndim == 1:
        src_var = src_var.expand_dims(dim={"dummy0": 1, "dummy1": 1})
    if src_var.ndim == 2:
        src_var = src_var.expand_dims(dim={"dummy": 1})
    # Retrieve the chunk size required for `da.map_blocks`, as the resulting array
    # will have a different shape.
    chunksize = src_var.shape[:-2] + dst_src_ij_images.chunksize[-2:]
    arr = da.map_blocks(
        _compute_var_image_block,
        dst_src_ij_images,
        src_var,
        fill_value,
        interpolation,
        chunksize,
        dtype=src_var.dtype,
        chunks=chunksize,
    )
    arr = arr[..., : dst_src_ij_images.shape[-2], : dst_src_ij_images.shape[-1]]
    if arr.shape[0] == 1:
        arr = arr[0, :, :]
    if arr.shape[0] == 1:
        arr = arr[0, :]
    return arr


def _compute_var_image_block(
    dst_src_ij_images: np.ndarray,
    src_var_image: xr.DataArray,
    fill_value: Union[int, float, complex],
    interpolation: int,
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
    _compute_var_image_numpy_sequential(
        src_var_image, dst_src_ij_images, dst_values, src_bbox, interpolation
    )
    dst_out[..., :dst_height, :dst_width] = dst_values
    return dst_out


# Extra dask version, because if we use parallel=True
# and nb.prange, we end up in infinite JIT compilation :(
@nb.njit(nogil=True, cache=True)
def _compute_var_image_numpy_sequential(
    src_var_image: np.ndarray,
    dst_src_ij_images: np.ndarray,
    dst_var_image: np.ndarray,
    src_bbox: tuple[int, int, int, int],
    interpolation: int,
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
            interpolation,
        )


@nb.njit(nogil=True, cache=True)
def _compute_var_image_for_dest_line(
    dst_j: int,
    src_var_image: np.ndarray,
    dst_src_ij_images: np.ndarray,
    dst_var_image: np.ndarray,
    src_bbox: tuple[int, int, int, int],
    interpolation: int,
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
        if interpolation == 0:
            # interpolation == "nearest"
            if u > 0.5:
                src_i0 = _iclamp(src_i0 + 1, src_i_min, src_i_max)
            if v > 0.5:
                src_j0 = _iclamp(src_j0 + 1, src_j_min, src_j_max)
            dst_var_value = src_var_image[..., src_j0, src_i0]
        elif interpolation == 1:
            # interpolation == "triangular"
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
        else:
            # interpolation == "bilinear"
            src_i1 = _iclamp(src_i0 + 1, src_i_min, src_i_max)
            src_j1 = _iclamp(src_j0 + 1, src_j_min, src_j_max)
            value_00 = src_var_image[..., src_j0, src_i0]
            value_01 = src_var_image[..., src_j0, src_i1]
            value_10 = src_var_image[..., src_j1, src_i0]
            value_11 = src_var_image[..., src_j1, src_i1]
            value_u0 = value_00 + u * (value_01 - value_00)
            value_u1 = value_10 + u * (value_11 - value_10)
            dst_var_value = value_u0 + v * (value_u1 - value_u0)
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
