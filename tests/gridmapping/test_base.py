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

import affine
import numpy as np
import pyproj
import xarray as xr

from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.gridmapping.coords import Coords2DGridMapping
from xcube_resampling.gridmapping.regular import RegularGridMapping

GEO_CRS = pyproj.crs.CRS(4326)
NOT_A_GEO_CRS = pyproj.crs.CRS(5243)


class _TestGridMapping(GridMapping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rgm = GridMapping.regular(
            size=self.size,
            tile_size=self.tile_size,
            is_j_axis_up=self.is_j_axis_up,
            xy_res=self.xy_res,
            xy_min=(self.xy_bbox[0] + self.x_res / 2, self.xy_bbox[1] + self.x_res / 2),
            crs=self.crs,
        )

    def _new_x_coords(self) -> xr.DataArray:
        return self.rgm.x_coords

    def _new_y_coords(self) -> xr.DataArray:
        return self.rgm.y_coords

    def _new_xy_coords(self) -> xr.DataArray:
        return self.rgm.xy_coords


# noinspection PyMethodMayBeStatic
class GridMappingTest(unittest.TestCase):
    _kwargs = dict(
        size=(720, 360),
        tile_size=(360, 180),
        xy_bbox=(-180.0, -90.0, 180.0, 90.0),
        xy_res=(360 / 720, 360 / 720),
        crs=GEO_CRS,
        xy_var_names=("x", "y"),
        xy_dim_names=("x", "y"),
        is_regular=True,
        is_lon_360=False,
        is_j_axis_up=False,
    )

    def kwargs(self, **kwargs):
        orig_kwargs = dict(self._kwargs)
        orig_kwargs.update(**kwargs)
        if "xy_min" in orig_kwargs:
            # Replace xy_min by xy_bbox.
            # Using xy_min instead of xy_bbox makes it easier for us
            width, height = orig_kwargs["size"]
            try:
                x_res, y_res = orig_kwargs["xy_res"]
            except TypeError:
                x_res, y_res = 2 * (orig_kwargs["xy_res"],)
            x_min, y_min = orig_kwargs.pop("xy_min")
            x_max, y_max = x_min + x_res * (width - 1), y_min + y_res * (height - 1)
            orig_kwargs["xy_bbox"] = (
                x_min - x_res / 2,
                y_min - y_res / 2,
                x_max + x_res / 2,
                y_max + y_res / 2,
            )
        return orig_kwargs

    def test_valid(self):
        gm = _TestGridMapping(**self.kwargs())
        self.assertEqual((720, 360), gm.size)
        self.assertEqual(720, gm.width)
        self.assertEqual(360, gm.height)
        self.assertEqual(True, gm.is_tiled)
        self.assertEqual((360, 180), gm.tile_size)
        self.assertEqual(360, gm.tile_width)
        self.assertEqual(180, gm.tile_height)
        self.assertEqual((0, 0, 720, 360), gm.ij_bbox)
        self.assertEqual((-180.0, -90.0, 180.0, 90.0), gm.xy_bbox)
        self.assertEqual(-179.75, gm.x_min)
        self.assertEqual(-89.75, gm.y_min)
        self.assertEqual(179.75, gm.x_max)
        self.assertEqual(89.75, gm.y_max)
        self.assertEqual((0.5, 0.5), gm.xy_res)
        self.assertEqual(0.5, gm.x_res)
        self.assertEqual(0.5, gm.y_res)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual("degree", gm.spatial_unit_name)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(False, gm.is_j_axis_up)

        self.assertIsInstance(gm.xy_coords, xr.DataArray)
        np.testing.assert_equal(
            gm.ij_bboxes,
            np.array(
                [
                    [0, 0, 360, 180],
                    [360, 0, 720, 180],
                    [0, 180, 360, 360],
                    [360, 180, 720, 360],
                ]
            ),
        )
        np.testing.assert_equal(
            gm.xy_bboxes,
            np.array(
                [
                    [-180.0, 0.0, 0.0, 90.0],
                    [0.0, 0.0, 180.0, 90.0],
                    [-180.0, -90.0, 0.0, 0.0],
                    [0.0, -90.0, 180.0, 0.0],
                ]
            ),
        )

    def test_invalids(self):
        with self.assertRaises(ValueError) as cm:
            _TestGridMapping(**self.kwargs(size=(360, 1)))
        self.assertEqual("invalid size", f"{cm.exception}")

        with self.assertRaises(ValueError) as cm:
            _TestGridMapping(**self.kwargs(size=(360,)))
        self.assertEqual(
            "not enough values to unpack (expected 2, got 1)", f"{cm.exception}"
        )

        with self.assertRaises(ValueError) as cm:
            _TestGridMapping(**self.kwargs(size=None))
        self.assertEqual(
            "size must be an int or a sequence of two ints", f"{cm.exception}"
        )

        with self.assertRaises(ValueError) as cm:
            _TestGridMapping(**self.kwargs(tile_size=0))
        self.assertEqual("invalid tile_size", f"{cm.exception}")

        with self.assertRaises(ValueError) as cm:
            _TestGridMapping(**self.kwargs(xy_res=-0.1))
        self.assertEqual("invalid xy_res", f"{cm.exception}")

    def test_scalars(self):
        gm = _TestGridMapping(**self.kwargs(size=360, tile_size=180, xy_res=0.1))
        self.assertEqual((360, 360), gm.size)
        self.assertEqual((180, 180), gm.tile_size)
        self.assertEqual((0.1, 0.1), gm.xy_res)

    def test_not_tiled(self):
        gm = _TestGridMapping(**self.kwargs(tile_size=None))
        self.assertEqual((720, 360), gm.tile_size)
        self.assertEqual(False, gm.is_tiled)

    def test_ij_to_xy_transform(self):
        image_geom = _TestGridMapping(
            **self.kwargs(size=(1200, 1200), xy_min=(0, 0), xy_res=1, crs=NOT_A_GEO_CRS)
        )
        i2crs = image_geom.ij_to_xy_transform
        self.assertMatrixPoint((0, 0), i2crs, (0, 1199))
        self.assertMatrixPoint((1024, 0), i2crs, (1024, 1199))
        self.assertMatrixPoint((0, 1024), i2crs, (0, 1199 - 1024))
        self.assertMatrixPoint((1024, 1024), i2crs, (1024, 1199 - 1024))
        self.assertEqual(((1, 0, 0), (0.0, -1, 1199)), i2crs)

        image_geom = _TestGridMapping(
            **self.kwargs(size=(1440, 720), xy_min=(-179.875, -89.875), xy_res=0.25)
        )
        i2crs = image_geom.ij_to_xy_transform
        self.assertMatrixPoint((-179.875, 89.875), i2crs, (0, 0))
        self.assertMatrixPoint((0.125, -0.125), i2crs, (720, 360))
        self.assertMatrixPoint((179.875, -89.875), i2crs, (1439, 719))
        self.assertEqual(((0.25, 0.0, -179.875), (0.0, -0.25, 89.875)), i2crs)

        image_geom = _TestGridMapping(
            **self.kwargs(
                size=(1440, 720),
                xy_min=(-179.875, -89.875),
                xy_res=0.25,
                is_j_axis_up=True,
            )
        )
        i2crs = image_geom.ij_to_xy_transform
        self.assertMatrixPoint((-179.875, -89.875), i2crs, (0, 0))
        self.assertMatrixPoint((0.125, 0.125), i2crs, (720, 360))
        self.assertMatrixPoint((179.875, 89.875), i2crs, (1439, 719))
        self.assertEqual(((0.25, 0.0, -179.875), (0.0, 0.25, -89.875)), i2crs)

    def test_xy_to_ij_transform(self):
        image_geom = _TestGridMapping(
            **self.kwargs(size=(1200, 1200), xy_min=(0, 0), xy_res=1, crs=NOT_A_GEO_CRS)
        )
        crs2i = image_geom.xy_to_ij_transform
        self.assertMatrixPoint((0, 0), crs2i, (0, 1199))
        self.assertMatrixPoint((1024, 0), crs2i, (1024, 1199))
        self.assertMatrixPoint((0, 1024), crs2i, (0, 1199 - 1024))
        self.assertMatrixPoint((1024, 1024), crs2i, (1024, 1199 - 1024))
        self.assertEqual(((1, 0, 0), (0.0, -1, 1199)), crs2i)

        image_geom = _TestGridMapping(**self.kwargs(size=(1440, 720), xy_res=0.25))
        crs2i = image_geom.xy_to_ij_transform
        self.assertMatrixPoint((0, 719), crs2i, (-179.875, -89.875))
        self.assertMatrixPoint((720, 360), crs2i, (0.125, -0.125))
        self.assertMatrixPoint((1439, 0), crs2i, (179.875, 89.875))
        self.assertEqual(((4.0, 0.0, 719.5), (0.0, -4.0, 359.5)), crs2i)

        image_geom = _TestGridMapping(
            **self.kwargs(size=(1440, 720), xy_res=0.25, is_j_axis_up=True)
        )
        crs2i = image_geom.xy_to_ij_transform
        self.assertMatrixPoint((0, 0), crs2i, (-179.875, -89.875))
        self.assertMatrixPoint((720, 360), crs2i, (0.125, 0.125))
        self.assertMatrixPoint((1439, 719), crs2i, (179.875, 89.875))
        self.assertEqual(((4.0, -0.0, 719.5), (-0.0, 4.0, 359.5)), crs2i)

    def test_ij_transform_to_and_from(self):
        gm1 = _TestGridMapping(
            **self.kwargs(size=(1440, 720), xy_res=0.25, is_j_axis_up=True)
        )
        gm2 = _TestGridMapping(
            **self.kwargs(
                size=(1000, 1000), xy_min=(10, 50), xy_res=0.025, is_j_axis_up=True
            )
        )
        self.assertEqual(
            ((10.0, 0.0, -7595.0), (0.0, 10.0, -5595.0)), gm1.ij_transform_to(gm2)
        )
        self.assertEqual(
            ((10.0, -0.0, -7595.0), (-0.0, 10.0, -5595.0)), gm2.ij_transform_from(gm1)
        )
        self.assertEqual(
            ((0.1, 0.0, 759.5), (0.0, 0.1, 559.5)), gm2.ij_transform_to(gm1)
        )
        self.assertEqual(
            ((0.1, 0.0, 759.5), (0.0, 0.1, 559.5)), gm1.ij_transform_from(gm2)
        )

    def assertMatrixPoint(self, expected_point, matrix, point):
        affine_matrix = affine.Affine(*matrix[0], *matrix[1])
        actual_point = affine_matrix * point
        self.assertAlmostEqual(expected_point[0], actual_point[0])
        self.assertAlmostEqual(expected_point[1], actual_point[1])
        return actual_point

    def test_derive(self):
        gm = _TestGridMapping(**self.kwargs())
        self.assertEqual((720, 360), gm.size)
        self.assertEqual((360, 180), gm.tile_size)
        self.assertEqual(False, gm.is_j_axis_up)

        # force creating of xy_coords array and save value
        xy_coords = gm.xy_coords

        derived_gm = gm.derive(
            tile_size=270,
            is_j_axis_up=True,
            xy_var_names=("u", "v"),
            xy_dim_names=("i", "j"),
        )

        self.assertIsNot(gm, derived_gm)
        self.assertIsInstance(derived_gm, _TestGridMapping)
        self.assertEqual((720, 360), derived_gm.size)
        self.assertEqual((270, 270), derived_gm.tile_size)
        self.assertEqual(True, derived_gm.is_j_axis_up)
        self.assertEqual(("u", "v"), derived_gm.xy_var_names)
        self.assertEqual(("i", "j"), derived_gm.xy_dim_names)

        derived_xy_coords = derived_gm.xy_coords
        self.assertIsNot(xy_coords, derived_xy_coords)
        self.assertEqual(((2,), (270, 90), (270, 270, 180)), derived_xy_coords.chunks)

    def test_scale(self):
        gm = _TestGridMapping(**self.kwargs())
        self.assertEqual((720, 360), gm.size)
        self.assertEqual((360, 180), gm.tile_size)
        self.assertEqual(False, gm.is_j_axis_up)

        # force creating of xy_coords array and save value
        xy_coords = gm.xy_coords

        scaled_gm = gm.scale((0.25, 0.5))

        self.assertIsNot(gm, scaled_gm)
        self.assertIsInstance(scaled_gm, RegularGridMapping)
        self.assertEqual((180, 180), scaled_gm.size)
        self.assertEqual((180, 180), scaled_gm.tile_size)
        self.assertEqual(False, scaled_gm.is_j_axis_up)
        self.assertEqual(("x", "y"), scaled_gm.xy_var_names)
        self.assertEqual(("x", "y"), scaled_gm.xy_dim_names)

        scaled_xy_coords = scaled_gm.xy_coords
        self.assertIsNot(xy_coords, scaled_xy_coords)
        self.assertEqual(((2,), (180,), (180,)), scaled_xy_coords.chunks)

        scaled_gm = gm.scale((0.25, 0.5), tile_size=(90, 90))

        self.assertIsNot(gm, scaled_gm)
        self.assertIsInstance(scaled_gm, RegularGridMapping)
        self.assertEqual((180, 180), scaled_gm.size)
        self.assertEqual((90, 90), scaled_gm.tile_size)
        self.assertEqual(False, scaled_gm.is_j_axis_up)
        self.assertEqual(("x", "y"), scaled_gm.xy_var_names)
        self.assertEqual(("x", "y"), scaled_gm.xy_dim_names)

        scaled_xy_coords = scaled_gm.xy_coords
        self.assertIsNot(xy_coords, scaled_xy_coords)
        self.assertEqual(((2,), (90, 90), (90, 90)), scaled_xy_coords.chunks)

    def test_transform(self):
        gm = _TestGridMapping(
            **self.kwargs(
                xy_min=(20, 56),
                size=(400, 200),
                tile_size=(400, 200),
                xy_res=(0.01, 0.01),
            )
        )
        transformed_gm = gm.transform("EPSG:32633")

        self.assertIsNot(gm, transformed_gm)
        self.assertIsInstance(transformed_gm, Coords2DGridMapping)
        self.assertEqual(pyproj.CRS.from_string("EPSG:32633"), transformed_gm.crs)
        self.assertEqual((400, 200), transformed_gm.size)
        self.assertEqual((400, 200), transformed_gm.tile_size)
        self.assertEqual(False, transformed_gm.is_j_axis_up)
        self.assertEqual(
            ("transformed_x", "transformed_y"), transformed_gm.xy_var_names
        )
        self.assertEqual(("lon", "lat"), transformed_gm.xy_dim_names)

    def test_transform_xy_res(self):
        gm = _TestGridMapping(
            **self.kwargs(
                xy_min=(20, 56),
                size=(400, 200),
                tile_size=(200, 200),
                xy_res=(0.01, 0.01),
            )
        )
        transformed_gm = gm.transform("EPSG:32633", xy_res=1000)

        self.assertIsNot(gm, transformed_gm)
        self.assertIsInstance(transformed_gm, Coords2DGridMapping)
        self.assertEqual(pyproj.CRS.from_string("EPSG:32633"), transformed_gm.crs)
        self.assertEqual((400, 200), transformed_gm.size)
        self.assertEqual((200, 200), transformed_gm.tile_size)
        self.assertAlmostEqual(605.566, transformed_gm.xy_res[0])
        self.assertAlmostEqual(1109.47, transformed_gm.xy_res[1])
        self.assertEqual(False, transformed_gm.is_j_axis_up)
        self.assertEqual(
            ("transformed_x", "transformed_y"), transformed_gm.xy_var_names
        )
        self.assertEqual(("lon", "lat"), transformed_gm.xy_dim_names)

        transformed_gm_regular = transformed_gm.to_regular()
        self.assertIsNot(gm, transformed_gm_regular)
        self.assertIsInstance(transformed_gm_regular, RegularGridMapping)
        self.assertEqual(
            pyproj.CRS.from_string("EPSG:32633"), transformed_gm_regular.crs
        )
        self.assertEqual((439, 223), transformed_gm_regular.size)
        self.assertEqual((200, 200), transformed_gm_regular.tile_size)
        self.assertAlmostEqual(605.566, transformed_gm_regular.xy_res[0])
        self.assertAlmostEqual(1109.47, transformed_gm_regular.xy_res[1])
        self.assertEqual(False, transformed_gm_regular.is_j_axis_up)
        self.assertEqual(("x", "y"), transformed_gm_regular.xy_var_names)
        self.assertEqual(("x", "y"), transformed_gm_regular.xy_dim_names)

    def test_to_regular(self):
        gm = _TestGridMapping(
            **self.kwargs(
                xy_min=(9.6, 47.6),
                size=(1000, 1000),
                tile_size=(1000, 1000),
                xy_res=(0.0002, 0.0002),
            )
        )
        transformed_gm = gm.transform("EPSG:32633")
        transformed_gm_regular = transformed_gm.to_regular()

        self.assertIsNot(gm, transformed_gm_regular)
        self.assertIsInstance(transformed_gm_regular, RegularGridMapping)
        self.assertEqual(
            pyproj.CRS.from_string("EPSG:32633"), transformed_gm_regular.crs
        )
        self.assertEqual((1102, 1046), transformed_gm_regular.size)
        self.assertEqual((1000, 1000), transformed_gm_regular.tile_size)
        self.assertEqual(False, transformed_gm_regular.is_j_axis_up)
        self.assertEqual(False, transformed_gm_regular.is_lon_360)

    def test_is_close(self):
        gm1 = _TestGridMapping(
            **self.kwargs(xy_min=(0, 0), size=(400, 200), xy_res=(0.01, 0.01))
        )
        gm2 = _TestGridMapping(
            **self.kwargs(xy_min=(0, 0), size=(400, 200), xy_res=(0.01, 0.01))
        )
        self.assertTrue(gm1.is_close(gm1))
        self.assertTrue(gm2.is_close(gm2))
        self.assertTrue(gm1.is_close(gm2))
        self.assertTrue(gm2.is_close(gm1))

        tolerance = 0.001

        gm2 = _TestGridMapping(
            **self.kwargs(
                xy_min=(tolerance / 2, tolerance / 2),
                size=(400, 200),
                xy_res=(0.01, 0.01),
            )
        )
        self.assertTrue(gm1.is_close(gm1, tolerance=tolerance))
        self.assertTrue(gm2.is_close(gm2, tolerance=tolerance))
        self.assertTrue(gm1.is_close(gm2, tolerance=tolerance))
        self.assertTrue(gm2.is_close(gm1, tolerance=tolerance))

        gm2 = _TestGridMapping(
            **self.kwargs(
                xy_min=(tolerance * 2, tolerance * 2),
                size=(400, 200),
                xy_res=(0.01, 0.01),
            )
        )
        self.assertTrue(gm1.is_close(gm1, tolerance=tolerance))
        self.assertTrue(gm2.is_close(gm2, tolerance=tolerance))
        self.assertFalse(gm1.is_close(gm2, tolerance=tolerance))
        self.assertFalse(gm2.is_close(gm1, tolerance=tolerance))

    def test_is_regular_raise_error(self):
        gm = _TestGridMapping(**self.kwargs(is_regular=False))
        with self.assertRaises(ValueError) as cm:
            GridMapping.assert_regular(gm)
        self.assertIn("must be a regular grid mapping", str(cm.exception))

        with self.assertRaises(NotImplementedError) as cm:
            gm._assert_regular()
        self.assertIn(
            "Operation not implemented for non-regular grid mappings", str(cm.exception)
        )

    def test_ij_bbox_from_xy_bbox(self):
        gm = _TestGridMapping(**self.kwargs())

        ij_bbox = gm.ij_bbox_from_xy_bbox((-180, -90, 180, 90))
        self.assertEqual((0, 0, 720, 360), ij_bbox)

        ij_bbox = gm.ij_bbox_from_xy_bbox((-180, -90, 0, 0))
        self.assertEqual((0, 180, 360, 360), ij_bbox)

        ij_bbox = gm.ij_bbox_from_xy_bbox((0, 0, 180, 90))
        self.assertEqual((360, 0, 720, 180), ij_bbox)

        ij_bbox = gm.ij_bbox_from_xy_bbox((-180, -90, 0, 0), ij_border=1)
        self.assertEqual((0, 179, 361, 360), ij_bbox)

        ij_bbox = gm.ij_bbox_from_xy_bbox((0, 0, 180, 90), ij_border=1)
        self.assertEqual((359, 0, 720, 181), ij_bbox)

        ij_bbox = gm.ij_bbox_from_xy_bbox((-190, -100, -170, -80), ij_border=1)
        self.assertEqual((0, 339, 21, 360), ij_bbox)

        ij_bbox = gm.ij_bbox_from_xy_bbox((-190, -100, -180, -90), ij_border=1)
        self.assertEqual((-1, -1, -1, -1), ij_bbox)

    def test_ij_bboxes_from_xy_bboxes(self):
        gm = _TestGridMapping(**self.kwargs())

        ij_bboxes = gm.ij_bboxes_from_xy_bboxes(
            xy_bboxes=np.array(
                [
                    [-180, -90, 180, 90],
                    [-180, -90, 0, 0],
                    [0, 0, 180, 90],
                    [-180, -90, 0, 0],
                    [0, 0, 180, 90],
                    [-190, -100, -170, -80],
                    [-190, -100, -180, -90],
                ],
                dtype=np.float32,
            )
        )

        np.testing.assert_equal(
            ij_bboxes,
            np.array(
                [
                    [0, 0, 720, 360],
                    [0, 180, 360, 360],
                    [360, 0, 720, 180],
                    [0, 180, 360, 360],
                    [360, 0, 720, 180],
                    [0, 340, 20, 360],
                    [-1, -1, -1, -1],
                ],
                dtype=np.int64,
            ),
        )

    def test_repr_markdown(self):
        gm = _TestGridMapping(**self.kwargs())
        md = gm._repr_markdown_()

        # Basic structure
        self.assertIn("class: **_TestGridMapping**", md)
        self.assertIn("* is_regular: True", md)
        self.assertIn("* is_j_axis_up: False", md)
        self.assertIn("* is_lon_360: False", md)
        self.assertIn("* crs: EPSG:4326", md)
        self.assertIn("* xy_res: (0.5, 0.5)", md)
