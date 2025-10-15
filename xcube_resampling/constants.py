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


import logging
from collections.abc import Callable, Hashable, Mapping
from typing import Literal, TypeAlias, Sequence

import numpy as np

from .coarsen import center, first, last, mean, median, mode, std, var


__all__ = [
    "FloatInt",
    "AffineTransformMatrix",
    "SpatialAggMethod",
    "SpatialAggMethods",
    "SpatialInterpMethodInt",
    "SpatialInterpMethodStr",
    "SpatialInterpMethod",
    "SpatialInterpMethods",
    "RecoverNans",
    "FillValues",
    "FILLVALUE_UINT8",
    "FILLVALUE_UINT16",
    "FILLVALUE_INT",
    "FILLVALUE_FLOAT",
]

FloatInt = int | float
"""A type alias representing either a float or an int."""

AffineTransformMatrix = tuple[
    tuple[FloatInt, FloatInt, FloatInt], tuple[FloatInt, FloatInt, FloatInt]
]
"""A 2×3 affine transformation matrix represented as nested tuples."""

SpatialAggMethod: TypeAlias = Literal[
    "center",
    "count",
    "first",
    "last",
    "max",
    "mean",
    "median",
    "mode",
    "min",
    "prod",
    "std",
    "sum",
    "var",
]
"""A literal type representing supported aggregation methods."""

TemporalAggMethod = Literal[
    "all",
    "any",
    "argmax",
    "argmin",
    "count",
    "cumprod",
    "cumsum",
    "first",
    "last",
    "max",
    "min",
    "mean",
    "median",
    "percentile_<p>",
    "prod",
    "std",
    "sum",
    "var"
]

SpatialAggMethods: TypeAlias = SpatialAggMethod | Mapping[np.dtype | str, SpatialAggMethod]
"""An aggregation method or a mapping from dtype to aggregation method."""

TemporalAggMethods: TypeAlias = TemporalAggMethod | Sequence[
    TemporalAggMethod] | Mapping[str, TemporalAggMethod | Sequence[TemporalAggMethod]]

SpatialInterpMethodInt = Literal[0, 1]
"""Interpolation method, as integer code."""
SpatialInterpMethodStr = Literal["nearest", "triangular", "bilinear"]
"""Interpolation method, as string literal."""
SpatialInterpMethod = SpatialInterpMethodInt | SpatialInterpMethodStr
"""Interpolation method, as integer code or string literal."""

SpatialInterpMethods: TypeAlias = SpatialInterpMethod | Mapping[np.dtype | Hashable, SpatialInterpMethod]
"""An interpolation method or a mapping from dtype to interpolation method."""

TemporalInterpMethods: TypeAlias = Literal[
    "linear",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "polynomial"
]

RecoverNans: TypeAlias = bool | Mapping[np.dtype | str, bool]
"""Whether to attempt recovery of NaN values, globally or per dtype."""

FillValues: TypeAlias = FloatInt | Mapping[np.dtype | str, FloatInt]
"""Fill values for missing data, as a scalar or mapping from dtype to value."""

FILLVALUE_UINT8 = 255
"""Default fill value for uint8 arrays."""

FILLVALUE_UINT16 = 65535
"""Default fill value for uint16 arrays."""

FILLVALUE_INT = -1
"""Default fill value for integer arrays."""

FILLVALUE_FLOAT = np.nan
"""Default fill value for floating-point arrays."""

# Internal helpers (not exported in __all__)
AggFunction: TypeAlias = Callable[[np.ndarray, tuple[int, ...] | None], np.ndarray]
AGG_METHODS: dict[SpatialAggMethod, AggFunction] = {
    "center": center,
    "count": np.count_nonzero,
    "first": first,
    "last": last,
    "prod": np.nanprod,
    "max": np.nanmax,
    "mean": mean,
    "median": median,
    "min": np.nanmin,
    "mode": mode,
    "std": std,
    "sum": np.nansum,
    "var": var,
}
INTERP_METHOD_MAPPING = {0: "nearest", 1: "bilinear", "nearest": 0, "bilinear": 1}

SCALE_LIMIT = 0.95
UV_DELTA = 1e-3

LOG = logging.getLogger("xcube.resampling")
