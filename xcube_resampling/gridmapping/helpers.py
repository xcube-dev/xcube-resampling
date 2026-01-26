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
from collections.abc import Hashable, Sequence
from fractions import Fraction
from typing import Any

import affine
import dask.array as da
import numpy as np
import pyproj.crs
import xarray as xr

from xcube_resampling.constants import AffineTransformMatrix, FloatInt

from .assertions import assert_given, assert_instance, assert_true
from .undefined import UNDEFINED


def _to_int_or_float(x: FloatInt) -> FloatInt:
    """If x is an int or is close to an int return it
    as int otherwise as float. Helps o avoid errors
    introduced by inaccurate floating point ops.
    """
    if isinstance(x, int):
        return x
    xf = float(x)
    xi = round(xf)
    return xi if math.isclose(xi, xf, rel_tol=1e-5) else xf


def _round_sequence(bbox: Sequence[FloatInt], ndigits: int = 7) -> Sequence[FloatInt]:
    return tuple(round(v, ndigits=ndigits) for v in bbox)


def _from_affine(matrix: affine.Affine) -> AffineTransformMatrix:
    return (matrix.a, matrix.b, matrix.c), (matrix.d, matrix.e, matrix.f)


def _to_affine(matrix: AffineTransformMatrix) -> affine.Affine:
    return affine.Affine(*matrix[0], *matrix[1])


def _normalize_crs(crs: str | pyproj.CRS) -> pyproj.CRS:
    if isinstance(crs, pyproj.CRS):
        return crs
    assert_instance(crs, str, "crs")
    return pyproj.CRS.from_string(crs)


def _normalize_int_pair(
    value: Any, name: str = None, default: tuple[int, int] | None = UNDEFINED
) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    elif value is not None:
        x, y = value
        return int(x), int(y)
    elif default != UNDEFINED:
        return default
    else:
        assert_given(name, "name")
        raise ValueError(f"{name} must be an int or a sequence of two ints")


def _normalize_number_pair(
    value: Any, name: str = None, default: tuple[FloatInt, FloatInt] | None = UNDEFINED
) -> tuple[FloatInt, FloatInt]:
    if isinstance(value, (float, int)):
        x, y = value, value
        return _to_int_or_float(x), _to_int_or_float(y)
    elif value is not None:
        x, y = value
        return _to_int_or_float(x), _to_int_or_float(y)
    elif default != UNDEFINED:
        return default
    else:
        assert_given(name, "name")
        raise ValueError(f"{name} must be a number or a sequence of two numbers")


def to_lon_360(lon_var: np.ndarray | da.Array | xr.DataArray):
    if isinstance(lon_var, xr.DataArray):
        return lon_var.where(lon_var >= 0.0, lon_var + 360.0)
    else:
        lon_var = da.asarray(lon_var)
        return da.where(lon_var >= 0.0, lon_var, lon_var + 360.0)


def from_lon_360(lon_var: np.ndarray | da.Array | xr.DataArray):
    if isinstance(lon_var, xr.DataArray):
        return lon_var.where(lon_var <= 180.0, lon_var - 360.0)
    else:
        lon_var = da.asarray(lon_var)
        return da.where(lon_var <= 180.0, lon_var, lon_var - 360.0)


def get_dataset_chunks(dataset: xr.Dataset) -> dict[Hashable, int]:
    """Get the most common chunk sizes for each
    chunked dimension of *dataset*.

    Note: Only data variables are considered.

    Args:
        dataset: A dataset.

    Returns:
        A dictionary that maps dimension names to common chunk sizes.
    """

    # Record the frequencies of chunk sizes for
    # each dimension d in each data variable var
    dim_size_counts: dict[Hashable, dict[int, int]] = {}
    for var_name, var in dataset.data_vars.items():
        if var.chunks:
            for d, c in zip(var.dims, var.chunks):
                # compute max chunk size max_c from
                # e.g.  c = (512, 512, 512, 193)
                max_c = max(0, *c)
                # for dimension d, save the frequencies
                # of the different max_c
                if d not in dim_size_counts:
                    size_counts = {max_c: 1}
                    dim_size_counts[d] = size_counts
                else:
                    size_counts = dim_size_counts[d]
                    if max_c not in size_counts:
                        size_counts[max_c] = 1
                    else:
                        size_counts[max_c] += 1

    # For each dimension d, determine the most frequently
    # seen chunk size max_c
    dim_sizes: dict[Hashable, int] = {}
    for d, size_counts in dim_size_counts.items():
        max_count = 0
        best_max_c = 0
        for max_c, count in size_counts.items():
            if count > max_count:
                # Should always come here, because count=1 is minimum
                max_count = count
                best_max_c = max_c
        assert best_max_c > 0
        dim_sizes[d] = best_max_c

    return dim_sizes


def _default_xy_var_names(crs: pyproj.crs.CRS) -> tuple[str, str]:
    return ("lon", "lat") if crs.is_geographic else ("x", "y")


def _default_xy_dim_names(crs: pyproj.crs.CRS) -> tuple[str, str]:
    return _default_xy_var_names(crs)


def _assert_valid_xy_names(value: Any, name: str = None):
    assert_instance(value, tuple, name=name)
    assert_true(
        len(value) == 2 and all(value) and value[0] != value[1],
        f"invalid {name or 'value'}",
    )


def _assert_valid_xy_coords(xy_coords: Any):
    assert_instance(xy_coords, xr.DataArray, name="xy_coords")
    assert_true(
        xy_coords.ndim == 3
        and xy_coords.shape[0] == 2
        and xy_coords.shape[1] >= 2
        and xy_coords.shape[2] >= 2,
        "xy_coords must have dimensions"
        " (2, height, width) with height >= 2 and width >= 2",
    )


_RESOLUTIONS = {
    10: (1, 0),
    20: (2, 0),
    25: (25, 1),
    50: (5, 0),
    100: (1, -1),
}

_RESOLUTION_SET = {k / 100 for k in _RESOLUTIONS.keys()}


def round_to_fraction(value: float, digits: int = 2, resolution: float = 1) -> Fraction:
    """Round *value* at position given by significant
    *digits* and return result as fraction.

    Args:
        value: The value
        digits: The number of significant digits. Must be an integer >=
            1. Default is 2.
        resolution: The rounding resolution for the least significant
            digit. Must be one of (0.1, 0.2, 0.25, 0.5, 1). Default is
            1.

    Returns:
        The rounded value as fraction.Fraction instance.
    """
    if digits < 1:
        raise ValueError("digits must be a positive integer")
    resolution_key = round(100 * resolution)
    if resolution_key not in _RESOLUTIONS or not math.isclose(
        100 * resolution, resolution_key
    ):
        raise ValueError(f"resolution must be one of {_RESOLUTION_SET}")
    if value == 0:
        return Fraction(0, 1)
    sign = 1
    if value < 0:
        sign = -1
        value = -value
    resolution, resolution_digits = _RESOLUTIONS[resolution_key]
    exponent = math.floor(math.log10(value)) - digits - resolution_digits
    if exponent >= 0:
        magnitude = Fraction(10**exponent, 1)
    else:
        magnitude = Fraction(1, 10**-exponent)
    scaled_value = value / magnitude
    discrete_value = resolution * round(scaled_value / resolution)
    return (sign * discrete_value) * magnitude


def scale_xy_res_and_size(
    xy_res: tuple[float, float], size: tuple[int, int], xy_scale: tuple[float, float]
) -> tuple[tuple[float, float], tuple[int, int]]:
    """Scale given *xy_res* and *size* using *xy_scale*.
    Make sure, size components are not less than 2.
    """
    x_res, y_res = xy_res
    x_scale, y_scale = xy_scale
    w, h = size
    w, h = round(x_scale * w), round(y_scale * h)
    return (
        (x_res / x_scale, y_res / y_scale),
        (w if w >= 2 else 2, h if h >= 2 else 2),
    )
