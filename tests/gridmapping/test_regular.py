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
import pyproj
import xarray as xr

from xcube_resampling.gridmapping import CRS_WGS84, GridMapping
from xcube_resampling.gridmapping.regular import RegularGridMapping

GEO_CRS = pyproj.crs.CRS(4326)
NOT_A_GEO_CRS = pyproj.crs.CRS(5243)


# noinspection PyMethodMayBeStatic
class RegularGridMappingTest(unittest.TestCase):
    def test_default_props(self):
        gm = GridMapping.regular((1000, 1000), (10, 53), 0.01, CRS_WGS84)
        self.assertEqual((1000, 1000), gm.size)
        self.assertEqual((1000, 1000), gm.tile_size)
        self.assertEqual(10, gm.x_min)
        self.assertEqual(53, gm.y_min)
        self.assertEqual((0.01, 0.01), gm.xy_res)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_j_axis_up)

    def test_invalid_y(self):
        with self.assertRaises(ValueError) as cm:
            GridMapping.regular((1000, 1000), (10, -90.5), 0.01, CRS_WGS84)
        self.assertEqual("invalid xy_bbox (south)", f"{cm.exception}")

        with self.assertRaises(ValueError) as cm:
            GridMapping.regular((1000, 1000), (10, 53), 0.1, CRS_WGS84)
        self.assertEqual("invalid xy_bbox (north)", f"{cm.exception}")

    def test_xy_bbox(self):
        gm = GridMapping.regular((1000, 1000), (10, 53), 0.01, CRS_WGS84)
        np.testing.assert_allclose(gm.xy_bbox, (9.995, 52.995, 19.995, 62.995))
        self.assertEqual(False, gm.is_lon_360)

    def test_xy_bbox_anti_meridian(self):
        gm = GridMapping.regular((2000, 1000), (174.0, -30.0), 0.005, CRS_WGS84)
        np.testing.assert_allclose(gm.xy_bbox, (173.9975, -30.0025, 183.9975, -25.0025))
        self.assertEqual(True, gm.is_lon_360)

    def test_derive(self):
        gm = GridMapping.regular((1000, 1000), (10, 53), 0.01, CRS_WGS84)
        self.assertEqual((1000, 1000), gm.size)
        self.assertEqual((1000, 1000), gm.tile_size)
        self.assertEqual(False, gm.is_j_axis_up)
        derived_gm = gm.derive(tile_size=500, is_j_axis_up=True)
        self.assertIsNot(gm, derived_gm)
        self.assertIsInstance(derived_gm, RegularGridMapping)
        self.assertEqual((1000, 1000), derived_gm.size)
        self.assertEqual((500, 500), derived_gm.tile_size)
        self.assertEqual(True, derived_gm.is_j_axis_up)

    def test_xy_coords(self):
        gm = GridMapping.regular((8, 4), (10, 53), 0.1, CRS_WGS84).derive(
            tile_size=(4, 2)
        )
        xy_coords = gm.xy_coords
        self.assertIsInstance(xy_coords, xr.DataArray)
        self.assertIs(gm.xy_coords, xy_coords)
        self.assertEqual(("coord", "lat", "lon"), xy_coords.dims)
        self.assertEqual((2, 4, 8), xy_coords.shape)
        self.assertEqual(((2,), (2, 2), (4, 4)), xy_coords.chunks)
        np.testing.assert_almost_equal(
            xy_coords.values[0],
            np.array(
                [
                    [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
                    [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
                    [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
                    [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
                ]
            ),
        )
        np.testing.assert_almost_equal(
            xy_coords.values[1],
            np.array(
                [
                    [53.3, 53.3, 53.3, 53.3, 53.3, 53.3, 53.3, 53.3],
                    [53.2, 53.2, 53.2, 53.2, 53.2, 53.2, 53.2, 53.2],
                    [53.1, 53.1, 53.1, 53.1, 53.1, 53.1, 53.1, 53.1],
                    [53.0, 53.0, 53.0, 53.0, 53.0, 53.0, 53.0, 53.0],
                ]
            ),
        )

    def test_xy_names(self):
        gm = GridMapping.regular((1000, 1000), (10, 53), 0.01, GEO_CRS).derive(
            tile_size=500
        )
        self.assertEqual(("lon", "lat"), gm.xy_var_names)
        self.assertEqual(("lon", "lat"), gm.xy_dim_names)
        gm = GridMapping.regular((1000, 1000), (10, 53), 0.01, NOT_A_GEO_CRS).derive(
            tile_size=500
        )
        self.assertEqual(("x", "y"), gm.xy_var_names)
        self.assertEqual(("x", "y"), gm.xy_dim_names)

    def test_ij_bboxes(self):
        gm = GridMapping.regular(
            size=(2000, 1000), xy_min=(10.0, 20.0), xy_res=0.1, crs=NOT_A_GEO_CRS
        )
        np.testing.assert_almost_equal(
            gm.ij_bboxes, np.array([[0, 0, 2000, 1000]], dtype=np.int64)
        )

        gm = GridMapping.regular(
            size=(2000, 1000), xy_min=(10.0, 20.0), xy_res=0.1, crs=NOT_A_GEO_CRS
        ).derive(tile_size=500)
        np.testing.assert_almost_equal(
            gm.ij_bboxes,
            np.array(
                [
                    [0, 0, 500, 500],
                    [500, 0, 1000, 500],
                    [1000, 0, 1500, 500],
                    [1500, 0, 2000, 500],
                    [0, 500, 500, 1000],
                    [500, 500, 1000, 1000],
                    [1000, 500, 1500, 1000],
                    [1500, 500, 2000, 1000],
                ],
                dtype=np.int64,
            ),
        )

    def test_xy_bboxes(self):
        gm = GridMapping.regular(
            size=(2000, 1000), xy_min=(10.0, 20.0), xy_res=0.1, crs=NOT_A_GEO_CRS
        )
        np.testing.assert_almost_equal(
            gm.xy_bboxes, np.array([[9.95, 19.95, 209.95, 119.95]], dtype=np.float64)
        )

        gm = GridMapping.regular(
            size=(2000, 1000), xy_min=(10.0, 20.0), xy_res=0.1, crs=NOT_A_GEO_CRS
        ).derive(tile_size=500)
        np.testing.assert_almost_equal(
            gm.xy_bboxes,
            np.array(
                [
                    [9.95, 69.95, 59.95, 119.95],
                    [59.95, 69.95, 109.95, 119.95],
                    [109.95, 69.95, 159.95, 119.95],
                    [159.95, 69.95, 209.95, 119.95],
                    [9.95, 19.95, 59.95, 69.95],
                    [59.95, 19.95, 109.95, 69.95],
                    [109.95, 19.95, 159.95, 69.95],
                    [159.95, 19.95, 209.95, 69.95],
                ],
                dtype=np.float64,
            ),
        )

    def test_xy_bboxes_is_j_axis_up(self):
        gm = GridMapping.regular(
            size=(2000, 1000), xy_min=(10.0, 20.0), xy_res=0.1, crs=NOT_A_GEO_CRS
        ).derive(is_j_axis_up=True)
        np.testing.assert_almost_equal(
            gm.xy_bboxes, np.array([[9.95, 19.95, 209.95, 119.95]], dtype=np.float64)
        )

        gm = GridMapping.regular(
            size=(2000, 1000),
            xy_min=(10.0, 20.0),
            xy_res=0.1,
            crs=NOT_A_GEO_CRS,
        ).derive(tile_size=500, is_j_axis_up=True)
        np.testing.assert_almost_equal(
            gm.xy_bboxes,
            np.array(
                [
                    [9.95, 19.95, 59.95, 69.95],
                    [59.95, 19.95, 109.95, 69.95],
                    [109.95, 19.95, 159.95, 69.95],
                    [159.95, 19.95, 209.95, 69.95],
                    [9.95, 69.95, 59.95, 119.95],
                    [59.95, 69.95, 109.95, 119.95],
                    [109.95, 69.95, 159.95, 119.95],
                    [159.95, 69.95, 209.95, 119.95],
                ],
                dtype=np.float64,
            ),
        )

    def test_to_coords(self):
        gm = GridMapping.regular(
            size=(10, 6), xy_min=(-2600.0, 1200.0), xy_res=10.0, crs=NOT_A_GEO_CRS
        )

        cv = gm.to_coords(xy_var_names=("x", "y"))
        self._assert_coord_vars(
            cv,
            (10, 6),
            ("x", "y"),
            (-2600.0, -2510.0),
            (1250.0, 1200.0),
            ("x_bnds", "y_bnds"),
            (
                (-2605.0, -2595.0),
                (-2515.0, -2505.0),
            ),
            (
                (1255.0, 1245.0),
                (1205.0, 1195.0),
            ),
        )

    def test_coord_vars_j_axis_up(self):
        gm = GridMapping.regular(
            size=(10, 6), xy_min=(-2600.0, 1200.0), xy_res=10.0, crs=NOT_A_GEO_CRS
        ).derive(is_j_axis_up=True)

        cv = gm.to_coords(xy_var_names=("x", "y"))
        self._assert_coord_vars(
            cv,
            (10, 6),
            ("x", "y"),
            (-2600.0, -2510.0),
            (1200.0, 1250.0),
            ("x_bnds", "y_bnds"),
            (
                (-2605.0, -2595.0),
                (-2515.0, -2505.0),
            ),
            (
                (1195.0, 1205.0),
                (1245.0, 1255.0),
            ),
        )

    def test_coord_vars_antimeridian(self):
        gm = GridMapping.regular(
            size=(10, 10), xy_min=(172.0, 53.0), xy_res=2.0, crs=GEO_CRS
        )

        cv = gm.to_coords(xy_var_names=("lon", "lat"))
        self._assert_coord_vars(
            cv,
            (10, 10),
            ("lon", "lat"),
            (172.0, -170.0),
            (71.0, 53.0),
            ("lon_bnds", "lat_bnds"),
            (
                (171.0, 173.0),
                (-171.0, -169.0),
            ),
            (
                (72.0, 70.0),
                (54.0, 52.0),
            ),
        )

    def _assert_coord_vars(
        self,
        cv,
        size,
        xy_names,
        x_values,
        y_values,
        xy_bnds_names,
        x_bnds_values,
        y_bnds_values,
    ):
        self.assertIsNotNone(cv)
        self.assertIn(xy_names[0], cv)
        self.assertIn(xy_names[1], cv)
        self.assertIn(xy_bnds_names[0], cv)
        self.assertIn(xy_bnds_names[1], cv)

        x = cv[xy_names[0]]
        self.assertEqual((size[0],), x.shape)
        np.testing.assert_almost_equal(x.values[0], np.array(x_values[0]))
        np.testing.assert_almost_equal(x.values[-1], np.array(x_values[-1]))

        y = cv[xy_names[1]]
        self.assertEqual((size[1],), y.shape)
        np.testing.assert_almost_equal(y.values[0], np.array(y_values[0]))
        np.testing.assert_almost_equal(y.values[-1], np.array(y_values[-1]))

        x_bnds = cv[xy_bnds_names[0]]
        self.assertEqual((size[0], 2), x_bnds.shape)
        np.testing.assert_almost_equal(x_bnds.values[0], np.array(x_bnds_values[0]))
        np.testing.assert_almost_equal(x_bnds.values[-1], np.array(x_bnds_values[-1]))

        y_bnds = cv[xy_bnds_names[1]]
        self.assertEqual((size[1], 2), y_bnds.shape)
        np.testing.assert_almost_equal(y_bnds.values[0], y_bnds_values[0])
        np.testing.assert_almost_equal(y_bnds.values[-1], y_bnds_values[-1])

    def test_to_regular(self):
        gm = GridMapping.regular((1000, 1000), (10, 53), 0.01, CRS_WGS84)
        gm_test = gm.to_regular()
        self.assertEqual(gm_test.size, (1000, 1000))
        self.assertEqual(gm_test.tile_size, (1000, 1000))
        self.assertEqual(gm_test.crs, CRS_WGS84)
        self.assertEqual(gm_test.xy_res, (0.01, 0.01))
        self.assertFalse(gm_test.is_j_axis_up)

        gm_test = gm.to_regular(tile_size=500)
        self.assertEqual(gm_test.size, (1000, 1000))
        self.assertEqual(gm_test.tile_size, (500, 500))
        self.assertEqual(gm_test.crs, CRS_WGS84)
        self.assertEqual(gm_test.xy_res, (0.01, 0.01))
        self.assertFalse(gm_test.is_j_axis_up)

        gm_test = gm.to_regular(is_j_axis_up=True)
        self.assertEqual(gm_test.size, (1000, 1000))
        self.assertEqual(gm_test.tile_size, (1000, 1000))
        self.assertEqual(gm_test.crs, CRS_WGS84)
        self.assertEqual(gm_test.xy_res, (0.01, 0.01))
        self.assertTrue(gm_test.is_j_axis_up)

    def test_create_regular_from_bbox(self):
        gm_test = GridMapping.regular_from_bbox(
            [10, 40, 20, 50], 0.1, "EPSG:4326", tile_size=10
        )
        self.assertEqual(gm_test.size, (100, 100))
        self.assertEqual(gm_test.tile_size, (10, 10))
        self.assertEqual(gm_test.crs, CRS_WGS84)
        self.assertEqual(gm_test.xy_res, (0.1, 0.1))
        self.assertFalse(gm_test.is_j_axis_up)
