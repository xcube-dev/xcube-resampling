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
from .sampledata import (
    create_2x2_dataset_with_irregular_coords,
    create_2x2_dataset_with_irregular_coords_antimeridian,
)

import numpy as np
import pytest

# noinspection PyUnresolvedReferences
import xarray as xr

from xcube_resampling.gridmapping import CRS_WGS84, GridMapping
from xcube_resampling.rectify import rectify_dataset


# noinspection PyMethodMayBeStatic
class RectifyDatasetTest(unittest.TestCase):
    def test_rectify_2x2_to_default(self):
        source_ds = create_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(4, 4), xy_min=(-1, 49), xy_res=2, crs=CRS_WGS84
        )
        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)

        rad = target_ds.rad
        np.testing.assert_almost_equal(
            rad.values,
            np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 1.0, 2.0, np.nan],
                    [3.0, 3.0, 2.0, np.nan],
                    [np.nan, 4.0, np.nan, np.nan],
                ],
                dtype=rad.dtype,
            ),
        )

    def test_rectify_2x2_to_7x7(self):
        source_ds = create_2x2_dataset_with_irregular_coords()
        # Add offset to "rad" so its values do not lie on a plane
        source_ds["rad"] = source_ds.rad + xr.DataArray(
            np.array([[0.0, 0.0], [0.0, 1.0]]), dims=("y", "x")
        )

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(-0.5, 49.5), xy_res=1.0, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)

        lon, lat, rad = self._assert_shape_and_dim(target_ds, (7, 7))
        np.testing.assert_almost_equal(
            lon.values, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=lon.dtype)
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array([56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0], dtype=lat.dtype),
        )
        np.testing.assert_almost_equal(
            rad.values,
            np.array(
                [
                    [np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 1.0, 1.0, 1.0, 2.0, np.nan, np.nan],
                    [np.nan, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 5.0, 2.0, np.nan, np.nan],
                    [np.nan, 3.0, 5.0, 5.0, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 5.0, np.nan, np.nan, np.nan, np.nan],
                ],
                dtype=rad.dtype,
            ),
        )

    def test_rectify_2x2_to_7x7_spline_orders_1(self):
        source_ds = create_2x2_dataset_with_irregular_coords()
        # Add offset to "rad" so its values do not lie on a plane
        source_ds["rad"] = source_ds.rad + xr.DataArray(
            np.array([[0.0, 0.0], [0.0, 1.0]]), dims=("y", "x")
        )

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(-0.5, 49.5), xy_res=1.0, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=1)

        lon, lat, rad = self._assert_shape_and_dim(target_ds, (7, 7))
        np.testing.assert_almost_equal(
            lon.values, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=lon.dtype)
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array([56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0], dtype=lat.dtype),
        )
        np.testing.assert_almost_equal(
            rad.values,
            np.array(
                [
                    [np.nan, 1.000, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 1.478, 1.391, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 1.957, 1.870, 1.784, 1.697, np.nan, np.nan],
                    [np.nan, 2.435, 2.348, 2.261, 2.174, 2.087, 2.000],
                    [3.000, 3.000, 3.000, 3.000, 3.000, np.nan, np.nan],
                    [np.nan, 4.000, 4.000, 4.000, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 5.000, np.nan, np.nan, np.nan, np.nan],
                ],
                dtype=rad.dtype,
            ),
            decimal=3,
        )

    def test_rectify_2x2_to_7x7_bilinear_interpol(self):
        source_ds = create_2x2_dataset_with_irregular_coords()
        # Add offset to "rad" so its values do not lie on a plane
        source_ds["rad"] = source_ds.rad + xr.DataArray(
            np.array([[0.0, 0.0], [0.0, 1.0]]), dims=("y", "x")
        )

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(-0.5, 49.5), xy_res=1.0, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=2)

        lon, lat, rad = self._assert_shape_and_dim(target_ds, (7, 7))
        np.testing.assert_almost_equal(
            lon.values, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=lon.dtype)
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array([56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0], dtype=lat.dtype),
        )
        np.testing.assert_almost_equal(
            rad.values,
            np.array(
                [
                    [np.nan, 1.000, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 1.488, 1.410, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 1.994, 1.949, 1.858, 1.722, np.nan, np.nan],
                    [np.nan, 2.520, 2.506, 2.448, 2.344, 2.195, 2.000],
                    [3.000, 3.112, 3.163, 3.153, 3.082, np.nan, np.nan],
                    [np.nan, 4.000, 4.041, 4.020, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 5.000, np.nan, np.nan, np.nan, np.nan],
                ],
                dtype=rad.dtype,
            ),
            decimal=3,
        )

    def test_rectify_2x2_to_7x7_invalid_interpol(self):
        source_ds = create_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(-0.5, 49.5), xy_res=1.0, crs=CRS_WGS84
        )

        with pytest.raises(ValueError, match="invalid interpolation: 'bicubic'"):
            rectify_dataset(source_ds, target_gm=target_gm, spline_orders=3)

    def test_rectify_2x2_to_7x7_subset(self):
        source_ds = create_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(1.5, 50.5), xy_res=1.0, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)
        lon, lat, rad = self._assert_shape_and_dim(target_ds, (7, 7))
        np.testing.assert_almost_equal(
            lon.values, np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=lon.dtype)
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array([57.0, 56.0, 55.0, 54.0, 53.0, 52.0, 51.0], dtype=lat.dtype),
        )
        np.testing.assert_almost_equal(
            rad.values,
            np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [1.0, 1.0, 2.0, np.nan, np.nan, np.nan, np.nan],
                    [3.0, 1.0, 2.0, 2.0, 2.0, np.nan, np.nan],
                    [3.0, 4.0, 2.0, np.nan, np.nan, np.nan, np.nan],
                    [4.0, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                dtype=rad.dtype,
            ),
        )

    def test_rectify_2x2_to_13x13(self):
        source_ds = create_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(-0.25, 49.75), xy_res=0.5, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)

        lon, lat, rad = self._assert_shape_and_dim(target_ds, (13, 13))
        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_j_axis_up(self):
        source_ds = create_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13),
            xy_min=(-0.25, 49.75),
            xy_res=0.5,
            crs=CRS_WGS84,
            is_j_axis_up=True,
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)

        lon, lat, rad = self._assert_shape_and_dim(target_ds, (13, 13))
        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    50.0,
                    50.5,
                    51.0,
                    51.5,
                    52.0,
                    52.5,
                    53.0,
                    53.5,
                    54.0,
                    54.5,
                    55.0,
                    55.5,
                    56.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            rad.values, self.expected_rad_13x13(rad.dtype)[::-1]
        )

    def test_rectify_2x2_to_13x13_j_axis_up_dask_5x5(self):
        source_ds = create_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13),
            xy_min=(-0.25, 49.75),
            xy_res=0.5,
            crs=CRS_WGS84,
            tile_size=5,
            is_j_axis_up=True,
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)

        lon, lat, rad = self._assert_shape_and_dim(
            target_ds, (13, 13), chunks=((5, 5, 3), (5, 5, 3))
        )
        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    50.0,
                    50.5,
                    51.0,
                    51.5,
                    52.0,
                    52.5,
                    53.0,
                    53.5,
                    54.0,
                    54.5,
                    55.0,
                    55.5,
                    56.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            rad.values, self.expected_rad_13x13(rad.dtype)[::-1]
        )

    def test_rectify_2x2_to_13x13_dask_7x7(self):
        source_ds = create_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(-0.25, 49.75), xy_res=0.5, crs=CRS_WGS84, tile_size=7
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)

        lon, lat, rad = self._assert_shape_and_dim(
            target_ds, (13, 13), chunks=((7, 6), (7, 6))
        )

        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_dask_5x5(self):
        source_ds = create_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(-0.25, 49.75), xy_res=0.5, crs=CRS_WGS84, tile_size=5
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)

        lon, lat, rad = self._assert_shape_and_dim(
            target_ds, (13, 13), chunks=((5, 5, 3), (5, 5, 3))
        )

        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_dask_3x13(self):
        source_ds = create_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13),
            xy_min=(-0.25, 49.75),
            xy_res=0.5,
            crs=CRS_WGS84,
            tile_size=(3, 13),
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)

        lon, lat, rad = self._assert_shape_and_dim(
            target_ds, (13, 13), chunks=((13,), (3, 3, 3, 3, 1))
        )

        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_dask_13x3(self):
        source_ds = create_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13),
            xy_min=(-0.25, 49.75),
            xy_res=0.5,
            crs=CRS_WGS84,
            tile_size=(13, 3),
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)

        lon, lat, rad = self._assert_shape_and_dim(
            target_ds, (13, 13), chunks=((3, 3, 3, 3, 1), (13,))
        )

        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_antimeridian(self):
        source_ds = create_2x2_dataset_with_irregular_coords_antimeridian()

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(177.75, 49.75), xy_res=0.5, crs=CRS_WGS84
        )

        self.assertEqual(True, target_gm.is_lon_360)

        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)

        self.assertIsNotNone(target_ds)
        lon, lat, rad = self._assert_shape_and_dim(target_ds, (13, 13))
        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [
                    178.0,
                    178.5,
                    179.0,
                    179.5,
                    180.0,
                    -179.5,
                    -179.0,
                    -178.5,
                    -178.0,
                    -177.5,
                    -177.0,
                    -176.5,
                    -176.0,
                ],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_none(self):
        source_ds = create_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(10.0, 50.0), xy_res=0.5, crs=CRS_WGS84
        )
        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)
        self.assertIsNone(target_ds)

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(-10.0, 50.0), xy_res=0.5, crs=CRS_WGS84
        )
        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)
        self.assertIsNone(target_ds)

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(0.0, 58.0), xy_res=0.5, crs=CRS_WGS84
        )
        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)
        self.assertIsNone(target_ds)

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(0.0, 42.0), xy_res=0.5, crs=CRS_WGS84
        )
        target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)
        self.assertIsNone(target_ds)

    def _assert_shape_and_dim(
        self, target_ds, size, chunks=None, var_names=("rad",)
    ) -> tuple[xr.DataArray, ...]:
        w, h = size

        self.assertIn("lon", target_ds)
        lon = target_ds["lon"]
        self.assertEqual((w,), lon.shape)
        self.assertEqual(("lon",), lon.dims)

        self.assertIn("lat", target_ds)
        lat = target_ds["lat"]
        self.assertEqual((h,), lat.shape)
        self.assertEqual(("lat",), lat.dims)

        # noinspection PyShadowingBuiltins
        vars = []
        for var_name in var_names:
            self.assertIn(var_name, target_ds)
            var = target_ds[var_name]
            self.assertEqual((h, w), var.shape)
            self.assertEqual(("lat", "lon"), var.dims)
            self.assertEqual(chunks, var.chunks)
            vars.append(var)

        return lon, lat, *vars

    def expected_rad_13x13(self, dtype):
        return np.array(
            [
                [
                    np.nan,
                    np.nan,
                    1.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    np.nan,
                    np.nan,
                    1.0,
                    1.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    np.nan,
                    np.nan,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    np.nan,
                    np.nan,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    np.nan,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    np.nan,
                    3.0,
                    3.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    np.nan,
                    np.nan,
                ],
                [np.nan, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [
                    np.nan,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    np.nan,
                    np.nan,
                ],
                [
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    2.0,
                    2.0,
                    2.0,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    np.nan,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    2.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    np.nan,
                    np.nan,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    4.0,
                    4.0,
                    4.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    4.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            ],
            dtype=dtype,
        )
