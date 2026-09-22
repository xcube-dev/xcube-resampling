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

import numpy as np
import xarray as xr

from xcube_resampling import extend_dataset


class TestExtendDataset(unittest.TestCase):

    @staticmethod
    def _create_dataset(y_increasing: bool) -> xr.Dataset:
        y = [0.5, 1.5, 2.5] if y_increasing else [2.5, 1.5, 0.5]
        return xr.Dataset(
            {
                "temperature": (
                    ("y", "x"),
                    np.arange(9, dtype=np.float32).reshape(3, 3),
                )
            },
            coords={"x": [0.5, 1.5, 2.5], "y": y},
            attrs={"title": "sample"},
        )

    def test_bbox_larger_than_dataset(self):
        for y_increasing in (True, False):
            with self.subTest(y_increasing=y_increasing):
                ds = self._create_dataset(y_increasing)

                actual = extend_dataset(ds, bbox=(-1.0, -1.0, 4.0, 4.0))

                expected_y = (
                    [-0.5, 0.5, 1.5, 2.5, 3.5]
                    if y_increasing
                    else [3.5, 2.5, 1.5, 0.5, -0.5]
                )
                expected = ds.reindex(x=[-0.5, 0.5, 1.5, 2.5, 3.5], y=expected_y)
                xr.testing.assert_identical(actual, expected)

    def test_bbox_partially_covers_dataset(self):
        for y_increasing in (True, False):
            with self.subTest(y_increasing=y_increasing):
                ds = self._create_dataset(y_increasing)

                actual = extend_dataset(ds, bbox=(-1.0, 1.0, 2.0, 4.0))

                expected_y = [1.5, 2.5, 3.5] if y_increasing else [3.5, 2.5, 1.5]
                expected = ds.reindex(x=[-0.5, 0.5, 1.5], y=expected_y)
                xr.testing.assert_identical(actual, expected)

    def test_bbox_smaller_than_dataset(self):
        for y_increasing in (True, False):
            with self.subTest(y_increasing=y_increasing):
                ds = self._create_dataset(y_increasing)

                actual = extend_dataset(ds, bbox=(1.0, 1.0, 2.0, 2.0), tile_size=(1, 1))

                y_slice = slice(1.0, 2.0) if y_increasing else slice(2.0, 1.0)
                expected = ds.sel(x=slice(1.0, 2.0), y=y_slice)
                xr.testing.assert_identical(actual, expected)
                self.assertEqual(actual.chunksizes["x"], (1,))
                self.assertEqual(actual.chunksizes["y"], (1,))

    def test_custom_dimension_names_and_chunks(self):
        ds = self._create_dataset(y_increasing=False).rename(x="lon", y="lat")

        actual = extend_dataset(
            ds,
            x_dim="lon",
            y_dim="lat",
            bbox=(-1.0, -1.0, 4.0, 4.0),
            tile_size=(2, 3),
        )

        self.assertEqual(actual.chunksizes["lon"], (2, 2, 1))
        self.assertEqual(actual.chunksizes["lat"], (3, 2))

    def test_missing_spatial_coordinate(self):
        ds = xr.Dataset(coords={"x": [0.5, 1.5]})

        with self.assertRaisesRegex(ValueError, "must contain coordinates 'x' and 'y'"):
            extend_dataset(ds, bbox=(0.0, 0.0, 2.0, 2.0))

    def test_two_dimensional_spatial_coordinate(self):
        ds = xr.Dataset(
            coords={
                "x": (("y", "x"), [[0.5, 1.5], [0.5, 1.5]]),
                "y": [0.5, 1.5],
            }
        )

        with self.assertRaisesRegex(ValueError, "Only 1-D 'x' and 'y'"):
            extend_dataset(ds, bbox=(0.0, 0.0, 2.0, 2.0))

    def test_spatial_coordinate_too_short(self):
        ds = xr.Dataset(coords={"x": [0.5], "y": [0.5, 1.5]})

        with self.assertRaisesRegex(ValueError, "must contain at least two values"):
            extend_dataset(ds, bbox=(0.0, 0.0, 2.0, 2.0))

    def test_none_coordinate_name_is_not_allowed(self):
        ds = self._create_dataset(y_increasing=True)

        with self.assertRaisesRegex(ValueError, "coordinates None and 'y'"):
            extend_dataset(ds, (0.0, 0.0, 2.0, 2.0), x_dim=None)
