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
import pandas as pd
import xarray as xr

from tests.sampledata import create_nx8x6_dataset_with_regular_coords
from xcube_resampling.temporal import (
    resample_in_time,
    _guess_resampling_operation,
    _get_agg_method_kwargs,
    _get_temporal_interp_method,
    _get_temporal_agg_method,
)


class ResampleInTimeTest(unittest.TestCase):
    def setUp(self):
        self.regular_time = pd.date_range("2020-01-01", periods=5, freq="D")
        self.irregular_time = pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-07"]
        )
        self.ds_regular = xr.Dataset(
            {"a": ("time", np.arange(5))}, coords={"time": self.regular_time}
        )
        self.ds_irregular = xr.Dataset(
            {
                "a": ("time", np.arange(4)),
                "b": (("y", "x"), np.arange(20).reshape((5, 4))),
            },
            coords={"time": self.irregular_time, "x": np.arange(4), "y": np.arange(5)},
        )
        self.ds_notime = xr.Dataset(
            {"a": (("y", "x"), np.arange(20).reshape((5, 4)))},
            coords={"x": np.arange(4), "y": np.arange(5)},
        )

    def test_resample_in_time_min_max(self):
        input_cube = create_nx8x6_dataset_with_regular_coords(8)
        resampled_cube = resample_in_time(input_cube, "2D", agg_methods=["min", "max"])
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
            np.array([-3.0, -1.0, 1.0, 3.0]),
        )
        np.testing.assert_allclose(
            resampled_cube.refl_max.values[..., 0, 1],
            np.array([-2.0, 0.0, 2.0, 4.0]),
        )

    def test_resample_in_time_p90(self):
        input_cube = create_nx8x6_dataset_with_regular_coords(8)
        resampled_cube = resample_in_time(input_cube, "3D", agg_methods="percentile_90")
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
            np.array([-1.2, 1.8, 3.9]),
        )

    def test_resample_in_time_all(self):
        input_cube = create_nx8x6_dataset_with_regular_coords(8)
        resampled_cube = resample_in_time(input_cube, "all", agg_methods=["min", "max"])
        self.assertIn("time", resampled_cube)
        self.assertIn("refl_min", resampled_cube)
        self.assertIn("refl_max", resampled_cube)
        self.assertIn("ndvi_min", resampled_cube)
        self.assertIn("ndvi_max", resampled_cube)
        self.assertEqual((1,), resampled_cube.time.shape)
        self.assertEqual(("time", "lat", "lon"), resampled_cube.refl_min.dims)
        self.assertEqual(("time", "lat", "lon"), resampled_cube.refl_max.dims)
        self.assertEqual(("time", "lat", "lon"), resampled_cube.ndvi_min.dims)
        self.assertEqual(("time", "lat", "lon"), resampled_cube.ndvi_max.dims)
        self.assertEqual((1, 6, 8), resampled_cube.refl_min.shape)
        self.assertEqual((1, 6, 8), resampled_cube.refl_max.shape)
        self.assertEqual((1, 6, 8), resampled_cube.ndvi_min.shape)
        self.assertEqual((1, 6, 8), resampled_cube.ndvi_max.shape)
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

    def test_resample_in_time_nearest_interpolation(self):
        input_cube = create_nx8x6_dataset_with_regular_coords(4)
        resampled_cube = resample_in_time(input_cube, "6H", interp_methods="nearest")
        self.assertIn("time", resampled_cube)
        self.assertIn("refl_nearest", resampled_cube)
        self.assertIn("ndvi_nearest", resampled_cube)
        self.assertEqual((13,), resampled_cube.time.shape)
        np.testing.assert_allclose(
            resampled_cube.refl_nearest.values[..., 0, 1],
            np.array([-1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2]),
        )

    def test_resample_in_time_linear_interpolation(self):
        input_cube = create_nx8x6_dataset_with_regular_coords(4)
        resampled_cube = resample_in_time(input_cube, "6H", interp_methods="linear")
        self.assertIn("time", resampled_cube)
        self.assertIn("refl_linear", resampled_cube)
        self.assertIn("ndvi_linear", resampled_cube)
        self.assertEqual((13,), resampled_cube.time.shape)
        np.testing.assert_allclose(
            resampled_cube.refl_linear.values[..., 0, 1],
            np.array(
                [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
            ),
        )

    def test_resample_in_time_list_interpolation(self):
        input_cube = create_nx8x6_dataset_with_regular_coords(4)
        resampled_cube = resample_in_time(
            input_cube, "6H", interp_methods=["linear", "nearest"]
        )
        self.assertIn("time", resampled_cube)
        self.assertIn("refl_linear", resampled_cube)
        self.assertIn("refl_nearest", resampled_cube)
        self.assertIn("ndvi_linear", resampled_cube)
        self.assertIn("ndvi_nearest", resampled_cube)
        self.assertEqual((13,), resampled_cube.time.shape)
        np.testing.assert_allclose(
            resampled_cube.refl_linear.values[..., 0, 1],
            np.array(
                [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
            ),
        )

    def test_resample_in_time_variable_selection(self):
        input_cube = create_nx8x6_dataset_with_regular_coords(4)
        resampled_cube = resample_in_time(
            input_cube, "6H", interp_methods="linear", variables="refl"
        )
        self.assertIn("time", resampled_cube)
        self.assertIn("refl_linear", resampled_cube)
        self.assertNotIn("ndvi_linear", resampled_cube)
        self.assertEqual((13,), resampled_cube.time.shape)
        np.testing.assert_allclose(
            resampled_cube.refl_linear.values[..., 0, 1],
            np.array(
                [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
            ),
        )

        resampled_cube = resample_in_time(
            input_cube, "6H", interp_methods="linear", variables="ndvi"
        )
        self.assertIn("time", resampled_cube)
        self.assertNotIn("refl_linear", resampled_cube)
        self.assertIn("ndvi_linear", resampled_cube)
        self.assertEqual((13,), resampled_cube.time.shape)
        np.testing.assert_allclose(
            resampled_cube.ndvi_linear.values[..., 0, 1],
            np.array(
                [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
            ),
        )

    def test_resample_in_time_different_interp_method_per_variable(self):
        input_cube = create_nx8x6_dataset_with_regular_coords(4)
        resampled_cube = resample_in_time(
            input_cube,
            "6H",
            interp_methods={"refl": "linear", "ndvi": "nearest"},
        )
        self.assertIn("refl_linear", resampled_cube)
        self.assertIn("ndvi_nearest", resampled_cube)
        self.assertEqual((13,), resampled_cube.time.shape)
        np.testing.assert_allclose(
            resampled_cube.refl_linear.values[..., 0, 1],
            np.array(
                [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
            ),
        )
        np.testing.assert_allclose(
            resampled_cube.ndvi_nearest.values[..., 0, 1],
            np.array(
                [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
            ),
        )

    def test_resample_in_time_different_agg_method_per_variable(self):
        input_cube = create_nx8x6_dataset_with_regular_coords(4)
        resampled_cube = resample_in_time(
            input_cube, "2D", agg_methods={"refl": "max", "ndvi": "mean"}
        )
        self.assertIn("refl_max", resampled_cube)
        self.assertIn("ndvi_mean", resampled_cube)
        self.assertEqual((2,), resampled_cube.time.shape)
        np.testing.assert_allclose(
            resampled_cube.refl_max.values[..., 0, 1], np.array([0.0, 2.0])
        )
        np.testing.assert_allclose(
            resampled_cube.ndvi_mean.values[..., 0, 1], np.array([-0.5, 1.5])
        )

    def test_irregular_time_series_returns_orignal_dataset(self):
        with self.assertLogs("xcube.resampling", level="WARNING") as cm:
            resampled_cube = resample_in_time(self.ds_irregular, "2D")
        self.assertIn("Could not determine resampling operation.", cm.output[-1])
        self.assertEqual(resampled_cube, self.ds_irregular)
        self.assertEqual((4,), resampled_cube.time.shape)

    def test_irregular_time_series_agg(self):
        resampled_cube = resample_in_time(
            self.ds_irregular, "3D", agg_methods={"a": "max"}
        )
        self.assertIn("a_max", resampled_cube)
        self.assertNotIn("b_max", resampled_cube)
        self.assertEqual((3,), resampled_cube.time.shape)
        np.testing.assert_allclose(resampled_cube.a_max.values[..., 0], np.array([1]))

    def test_irregular_time_series_interp(self):
        resampled_cube = resample_in_time(
            self.ds_irregular, "12h", interp_methods={"a": "nearest"}
        )
        self.assertIn("a_nearest", resampled_cube)
        self.assertNotIn("b_nearest", resampled_cube)
        self.assertEqual((13,), resampled_cube.time.shape)
        np.testing.assert_allclose(
            resampled_cube.a_nearest.values[..., 0], np.array([0])
        )

    def test_no_time_dim(self):
        with self.assertRaises(ValueError) as cm:
            _ = resample_in_time(self.ds_notime, "1D")
        self.assertIn("Dataset must have a 'time' dimension.", str(cm.exception))

    def test_both_agg_and_interp_methods_raises(self):
        with self.assertRaises(ValueError):
            _guess_resampling_operation(
                self.ds_regular, "1D", interp_methods="linear", agg_methods="mean"
            )

    def test_too_few_time_points_raises(self):
        ds_single = xr.Dataset({"a": ("time", [1])}, coords={"time": ["2020-01-01"]})
        with self.assertRaises(ValueError):
            _guess_resampling_operation(ds_single, "1D")

    def test_explicit_agg_methods(self):
        result = _guess_resampling_operation(self.ds_regular, "1D", agg_methods="mean")
        self.assertEqual(result, "agg")

    def test_explicit_interp_methods(self):
        result = _guess_resampling_operation(
            self.ds_regular, "1D", interp_methods="linear"
        )
        self.assertEqual(result, "interp")

    def test_irregular_time_series_returns_none(self):
        result = _guess_resampling_operation(self.ds_irregular, "1D")
        self.assertIsNone(result)

    def test_regular_time_series_returns_none(self):
        result = _guess_resampling_operation(self.ds_regular, "1D")
        self.assertIsNone(result)

    def test_regular_time_series_downsample_returns_agg(self):
        result = _guess_resampling_operation(self.ds_regular, "2D")
        self.assertEqual(result, "agg")

    def test_regular_time_series_upsample_returns_interp(self):
        result = _guess_resampling_operation(self.ds_regular, "12H")
        self.assertEqual(result, "interp")

    def test_resample_in_time_invalid_method(self):
        input_cube = create_nx8x6_dataset_with_regular_coords(4)
        with self.assertRaises(ValueError):
            resample_in_time(input_cube, "6H", interp_methods=["nonlinear", "nearest"])

    def test_get_agg_method_kwargs(self):
        expected = {"tolerance": "1h"}
        result = _get_agg_method_kwargs("backfill", "1h")
        self.assertEqual(expected, result)

        expected = {"tolerance": "1h"}
        result = _get_agg_method_kwargs("backfill", "6h", tolerance="1h")
        self.assertEqual(expected, result)

        expected = {"dim": "time", "keep_attrs": True}
        result = _get_agg_method_kwargs("all", "6h")
        self.assertEqual(expected, result)

        expected = {"dim": "time", "keep_attrs": True}
        result = _get_agg_method_kwargs("all", "6h")
        self.assertEqual(expected, result)

        with self.assertRaises(ValueError):
            _get_agg_method_kwargs("bla", "6H")

    def test_get_temporal_interp_method(self):
        expected = ["nearest"]
        result = _get_temporal_interp_method(None, "a", xr.DataArray([1, 2]))
        self.assertEqual(expected, result)

        expected = ["linear"]
        result = _get_temporal_interp_method(None, "a", xr.DataArray([1.0, 2.0]))
        self.assertEqual(expected, result)

        expected = ["linear"]
        with self.assertLogs("xcube.resampling", level="WARNING") as cm:
            result = _get_temporal_interp_method(
                {"b": "linear"}, "a", xr.DataArray([1.0, 2.0])
            )
        self.assertIn("Interpolation method could not be derived ", cm.output[-1])
        self.assertEqual(expected, result)

    def test_get_temporal_agg_method(self):
        expected = ["nearest"]
        result = _get_temporal_agg_method(None, "a", xr.DataArray([1, 2]))
        self.assertEqual(expected, result)

        expected = ["mean"]
        result = _get_temporal_agg_method(None, "a", xr.DataArray([1.0, 2.0]))
        self.assertEqual(expected, result)

        expected = ["mean"]
        with self.assertLogs("xcube.resampling", level="WARNING") as cm:
            result = _get_temporal_agg_method(
                {"b": "linear"}, "a", xr.DataArray([1.0, 2.0])
            )
        self.assertIn("Aggregation method could not be derived ", cm.output[-1])
        self.assertEqual(expected, result)
