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

from typing import Callable
import logging

import numpy as np
import dask.array as da

Aggregator = Callable[[np.ndarray | da.Array], np.ndarray | da.Array]
FloatInt = int | float
AffineTransformMatrix = tuple[
    tuple[FloatInt, FloatInt, FloatInt], tuple[FloatInt, FloatInt, FloatInt]
]

FILLVALUE_UINT8 = 255
FILLVALUE_UINT16 = 65535
FILLVALUE_INT = -1
FILLVALUE_FLOAT = np.nan

SCALE_LIMIT = 0.95
UV_DELTA = 1e-3

LOG = logging.getLogger("xcube.stac")
