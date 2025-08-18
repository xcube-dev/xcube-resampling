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
from typing import Literal, TypeAlias

import numpy as np

from .coarsen import center, first, last, mean, median, mode, std, var

FloatInt = int | float
AffineTransformMatrix = tuple[
    tuple[FloatInt, FloatInt, FloatInt], tuple[FloatInt, FloatInt, FloatInt]
]
AggMethod: TypeAlias = Literal[
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
AggMethods: TypeAlias = AggMethod | Mapping[np.dtype | str, AggMethod]
AggFunction: TypeAlias = Callable[[np.ndarray, tuple[int, ...] | None], np.ndarray]
AGG_METHODS: dict[AggMethod, AggFunction] = {
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
InterpMethodInt = Literal[0, 1]
InterpMethodStr = Literal["nearest", "triangular", "bilinear"]
InterpMethod = InterpMethodInt | InterpMethodStr
InterpMethods: TypeAlias = InterpMethod | Mapping[np.dtype | Hashable, InterpMethod]
INTERP_METHOD_MAPPING = {0: "nearest", 1: "bilinear", "nearest": 0, "bilinear": 1}
RecoverNans: TypeAlias = bool | Mapping[np.dtype | str, bool]
FillValues: TypeAlias = FloatInt | Mapping[np.dtype | str, FloatInt]

FILLVALUE_UINT8 = 255
FILLVALUE_UINT16 = 65535
FILLVALUE_INT = -1
FILLVALUE_FLOAT = np.nan

SCALE_LIMIT = 0.95
UV_DELTA = 1e-3

LOG = logging.getLogger("xcube.resampling")
