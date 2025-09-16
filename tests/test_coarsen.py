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

"""This module comprises reducer functions to be passed to
`dask.array.coarsen()` that either do not exist in numpy, like `mode`,
or have special implementations compared to their numpy equivalents.
"""


import unittest

import numpy as np

from xcube_resampling import coarsen


class TestCoarsenAllInOne(unittest.TestCase):

    def test_all_reducers(self):
        # Prepare test arrays
        arr_float = np.array([[1.0, 2.0], [3.0, 4.0]])
        arr_int = np.array([[1, 2], [3, 4]])
        arr_mode = np.array([[1, 2, 2], [3, 2, 2]])

        axis = (0, 1)

        # ---- Reducers with specified axis ----
        np.testing.assert_array_equal(coarsen.first(arr_float, axis), np.array(1.0))
        np.testing.assert_array_equal(coarsen.last(arr_float, axis), np.array(4.0))
        np.testing.assert_array_equal(coarsen.center(arr_float, axis), np.array(4.0))
        np.testing.assert_array_equal(coarsen.mean(arr_float, axis), np.array(2.5))
        np.testing.assert_array_equal(coarsen.mean(arr_int, axis), np.array(2))
        np.testing.assert_array_equal(coarsen.median(arr_float, axis), np.array(2.5))
        np.testing.assert_array_almost_equal(
            coarsen.std(arr_float, axis), np.array(np.std(arr_float))
        )
        np.testing.assert_array_equal(coarsen.sum(arr_int, axis), np.array(10))
        np.testing.assert_array_almost_equal(
            coarsen.var(arr_float, axis), np.array(np.var(arr_float))
        )
        np.testing.assert_array_equal(coarsen.mode(arr_mode, axis), np.array(2))

        # ---- Reducers with axis=None (edge block pass-through) ----
        for reducer in [coarsen.first, coarsen.last, coarsen.center, coarsen.mode]:
            np.testing.assert_array_equal(reducer(arr_float, axis=None), arr_float)
