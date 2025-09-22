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


import unittest

from xcube_resampling import dask as xr_dask


class TestNestedList(unittest.TestCase):
    def test_nested_list(self):
        # Create a 2x3 nested list
        nl = xr_dask._NestedList((2, 3), fill_value=0)

        # shape should match
        self.assertEqual(nl.shape, (2, 3))

        # data should be a nested list of zeros
        self.assertEqual(nl.data, [[0, 0, 0], [0, 0, 0]])

        # __len__ should return size of first dimension
        self.assertEqual(len(nl), 2)

        # __getitem__ single index should give row
        self.assertEqual(nl[0], [0, 0, 0])

        # __getitem__ tuple index should give element
        self.assertEqual(nl[1, 2], 0)

        # __setitem__ single index should replace row
        nl[0] = [1, 2, 3]
        self.assertEqual(nl[0], [1, 2, 3])

        # __setitem__ tuple index should replace element
        nl[1, 1] = 42
        self.assertEqual(nl[1, 1], 42)

        # works with slice indexing too
        nl[0:2] = [[7, 8, 9], [10, 11, 12]]
        self.assertEqual(nl[0], [7, 8, 9])
        self.assertEqual(nl[1], [10, 11, 12])

        # Verify nested structure also works for higher dimensions
        nl3 = xr_dask._NestedList((2, 2, 2), fill_value=-1)
        self.assertEqual(nl3.shape, (2, 2, 2))
        self.assertEqual(nl3[1, 1, 1], -1)
        nl3[1, 0, 1] = 99
        self.assertEqual(nl3[1, 0, 1], 99)
