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
from typing import Annotated, Literal, Sequence, TypeAlias

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
    "TemporalAggMethod",
    "TemporalAggMethods",
    "TemporalInterpMethods",
    "TemporalInterpMethod",
    "PreventNaNPropagations",
    "FillValues",
    "FILLVALUE_UINT8",
    "FILLVALUE_UINT16",
    "FILLVALUE_INT",
    "FILLVALUE_FLOAT",
    "LOG",
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
"""A literal type representing supported spatial aggregation methods."""

SpatialAggMethods: TypeAlias = (
    SpatialAggMethod | Mapping[np.dtype | str, SpatialAggMethod]
)
"""An spatial aggregation method or a mapping from variable name or dtype to 
spatial aggregation method.
"""

SpatialInterpMethodInt = Literal[0, 1]
"""Spatial interpolation method, as integer code."""
SpatialInterpMethodStr = Literal["nearest", "triangular", "bilinear"]
"""Spatial interpolation method, as string literal."""
SpatialInterpMethod = SpatialInterpMethodInt | SpatialInterpMethodStr
"""Spatial interpolation method, as integer code or string literal."""
SpatialInterpMethods: TypeAlias = (
    SpatialInterpMethod | Mapping[np.dtype | Hashable, SpatialInterpMethod]
)
"""A spatial interpolation method or a mapping from variable name or dtype to 
interpolation method.
"""

PercentileString: TypeAlias = Annotated[str, "format: percentile_<int>"]
TemporalAggMethod = (
    Literal[
        "all",
        "any",
        "backfill",
        "bfill",
        "count",
        "cumprod",
        "cumsum",
        "ffill",
        "first",
        "last",
        "max",
        "min",
        "mean",
        "median",
        "nearest",
        "pad",
        "prod",
        "std",
        "sum",
        "var",
    ]
    | PercentileString
)
"""A literal type representing supported temporal aggregation methods."""
TemporalAggMethods: TypeAlias = (
    TemporalAggMethod
    | Sequence[TemporalAggMethod]
    | Mapping[np.dtype | str, TemporalAggMethod | Sequence[TemporalAggMethod]]
)
"""An temporal aggregation method, a list of temporal aggregation methods, or a 
mapping from variable name or dtype to temporal aggregation method(s).
"""

TemporalInterpMethod = Literal[
    "linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"
]
"""Temporal interpolation method, as string literal."""
TemporalInterpMethods: TypeAlias = (
    TemporalInterpMethod
    | Sequence[TemporalInterpMethod]
    | Mapping[
        np.dtype | Hashable, TemporalInterpMethod | Sequence[TemporalInterpMethod]
    ]
)
"""A temporal interpolation method, a list of temporal interpolation methods, or a 
mapping from variable name or dtype to interpolation method(s).
"""

PreventNaNPropagations: TypeAlias = bool | Mapping[np.dtype | str, bool]
"""It True, it prevents NaN propagation during upsampling (only applies when 
interpolation method is not nearest). It can be set globally or per data variable
or dtype.
"""

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
