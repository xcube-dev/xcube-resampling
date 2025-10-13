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

from tests.sampledata import create_7x8x6_dataset_with_regular_coords
from xcube_resampling.temporal import resample_in_time


class ResampleInTimeTest(unittest.TestCase):
    def test_resample_in_time_min_max(self):
        input_cube = create_7x8x6_dataset_with_regular_coords()
        resampled_cube = resample_in_time(input_cube, "2D", ["min", "max"])
        self.assertIn("time", resampled_cube)
        self.assertIn("refl_min", resampled_cube)
        self.assertIn("refl_max", resampled_cube)
        self.assertEqual((4,), resampled_cube.time.shape)
        self.assertEqual(("time", "lat", "lon"), resampled_cube.refl_min.dims)
        self.assertEqual(("time", "lat", "lon"), resampled_cube.refl_max.dims)
        self.assertEqual((4, 6, 8), resampled_cube.refl_min.shape)
        self.assertEqual((4, 6, 8), resampled_cube.refl_max.shape)
        self.assertEqual(
            list(resampled_cube.time.values),
            [
                np.datetime64("2025-08-01T00:00:00.000000000"),
                np.datetime64("2025-08-03T00:00:00.000000000"),
                np.datetime64("2025-08-05T00:00:00.000000000"),
                np.datetime64("2025-08-07T00:00:00.000000000"),
            ],
        )
        np.testing.assert_allclose(
            resampled_cube.refl_min.values[..., 0, 1],
            np.array([-3., -1.,  1.,  3.]),
        )
        np.testing.assert_allclose(
            resampled_cube.refl_max.values[..., 0, 1],
            np.array([-2.,  0.,  2.,  4.]),
        )

    def test_resample_in_time_p90(self):
        input_cube = create_7x8x6_dataset_with_regular_coords()
        resampled_cube = resample_in_time(input_cube, "3D", "percentile_90")
        self.assertIn("time", resampled_cube)
        self.assertIn("refl_p90", resampled_cube)
        self.assertEqual((3,), resampled_cube.time.shape)
        self.assertEqual(("time", "lat", "lon"), resampled_cube.refl_p90.dims)
        self.assertEqual((3, 6, 8), resampled_cube.refl_p90.shape)
        self.assertEqual(
            list(resampled_cube.time.values),
            [
                np.datetime64("2025-08-01T00:00:00.000000000"),
                np.datetime64("2025-08-04T00:00:00.000000000"),
                np.datetime64("2025-08-07T00:00:00.000000000"),
            ],
        )
        np.testing.assert_allclose(
            resampled_cube.refl_p90.values[..., 0, 1],
            np.array([-1.2,  1.8,  3.9]),
        )

    def test_resample_in_time_f_all(self):
        input_cube = create_7x8x6_dataset_with_regular_coords()
        resampled_cube = resample_in_time(input_cube, "all", ["min", "max"])
        self.assertIn("time", resampled_cube)
        self.assertIn("refl_min", resampled_cube)
        self.assertIn("refl_max", resampled_cube)
        self.assertEqual((1,), resampled_cube.time.shape)
        self.assertEqual(("time", "lat", "lon"), resampled_cube.refl_min.dims)
        self.assertEqual(("time", "lat", "lon"), resampled_cube.refl_max.dims)
        self.assertEqual((1, 6, 8), resampled_cube.refl_min.shape)
        self.assertEqual((1, 6, 8), resampled_cube.refl_max.shape)
        self.assertEqual(
            list(resampled_cube.time.values),
            [
                np.datetime64("2025-08-01T00:00:00.000000000"),
            ],
        )
        np.testing.assert_allclose(
            resampled_cube.refl_min.values[..., 0, 1],
            np.array([-3.0]),
        )
        np.testing.assert_allclose(
            resampled_cube.refl_max.values[..., 0, 1],
            np.array([4.0]),
        )
