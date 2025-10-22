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

from xcube_resampling.affine import affine_transform_dataset
from xcube_resampling.gridmapping import CRS_CRS84, CRS_WGS84, GridMapping

from .sampledata import (
    create_2x8x6_dataset_with_regular_coords,
    create_8x6_dataset_with_regular_coords,
)


class AffineTransformDatasetTest(unittest.TestCase):
    def setUp(self):
        self.source_ds = create_8x6_dataset_with_regular_coords()
        self.source_ds_3d = create_2x8x6_dataset_with_regular_coords()

        self.source_gm = GridMapping.from_dataset(self.source_ds)
        self.res = 0.1

    def test_subset(self):
        target_gm = GridMapping.regular(
            (3, 3), (50.0, 10.0), self.res, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            interp_methods=1,
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [1, 0, 2],
                    [0, 3, 0],
                    [4, 0, 1],
                ]
            ),
        )

        target_gm = GridMapping.regular(
            (3, 3), (50.1, 10.1), self.res, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds, target_gm, interp_methods=1
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [4, np.nan, np.nan],
                    [0, 2, 0],
                    [3, 0, 4],
                ]
            ),
        )

        target_gm = GridMapping.regular(
            (3, 3), (50.05, 10.05), self.res, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds, target_gm, interp_methods=1
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [1.25, 1.5, np.nan],
                    [1.0, 1.25, 1.5],
                    [1.75, 1.0, 1.25],
                ]
            ),
        )

        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            source_gm=self.source_gm,
            interp_methods=1,
            prevent_nan_propagations=True,
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [1.25, 1.5, 0.6666667],
                    [1.0, 1.25, 1.5],
                    [1.75, 1.0, 1.25],
                ]
            ),
        )

    def test_subset_3d(self):
        target_gm = GridMapping.regular(
            (3, 3), (50.0, 10.0), self.res, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds_3d,
            target_gm,
            interp_methods=1,
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds_3d.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((2, 3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [
                        [1, 0, 2],
                        [0, 3, 0],
                        [4, 0, 1],
                    ],
                    [
                        [1, 0, 2],
                        [0, 3, 0],
                        [4, 0, 1],
                    ],
                ]
            ),
        )

    def test_subset_with_source_gm(self):
        target_gm = GridMapping.regular(
            (3, 3), (50.0, 10.0), self.res, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            source_gm=self.source_gm,
            interp_methods="bilinear",
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [1, 0, 2],
                    [0, 3, 0],
                    [4, 0, 1],
                ]
            ),
        )

        target_gm = GridMapping.regular(
            (3, 3), (50.1, 10.1), self.res, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            source_gm=self.source_gm,
            interp_methods={"refl": "bilinear"},
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [4, np.nan, np.nan],
                    [0, 2, 0],
                    [3, 0, 4],
                ]
            ),
        )

        target_gm = GridMapping.regular(
            (3, 3), (50.05, 10.05), self.res, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            source_gm=self.source_gm,
            interp_methods={"refl": 1},
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array([[1.25, 1.5, np.nan], [1.0, 1.25, 1.5], [1.75, 1.0, 1.25]]),
        )

    def test_different_geographic_crses(self):
        expected = np.array([[1.25, 1.5, np.nan], [1.0, 1.25, 1.5], [1.75, 1.0, 1.25]])

        target_gm = GridMapping.regular((3, 3), (50.05, 10.05), self.res, CRS_WGS84)
        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            source_gm=self.source_gm,
            interp_methods=1,
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(target_ds.refl.values, expected)

        target_gm = GridMapping.regular((3, 3), (50.05, 10.05), self.res, CRS_CRS84)
        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            source_gm=self.source_gm,
            interp_methods=1,
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((3, 3), target_ds.refl.shape)
        np.testing.assert_almost_equal(target_ds.refl.values, expected)

        target_gm = GridMapping.regular(
            (3, 3), (50.05, 10.05), self.res, pyproj.crs.CRS(3035)
        )
        with self.assertRaises(AssertionError) as cm:
            affine_transform_dataset(
                self.source_ds, target_gm, source_gm=self.source_gm
            )

        self.assertIn(
            "Affine transformation cannot be applied to source CRS 'WGS 84' "
            "and target CRS 'ETRS89-extended / LAEA Europe'",
            str(cm.exception),
        )

    def test_downscale_x2(self):
        target_gm = GridMapping.regular(
            (8, 6), (50, 10), 2 * self.res, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            source_gm=self.source_gm,
            interp_methods=1,
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((6, 8), target_ds.refl.shape)
        print(repr(target_ds.refl.values))
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [0.75, 1.0, 1.75, 1.25, np.nan, np.nan, np.nan, np.nan],
                    [1.25, 1.0, 1.25, 1.75, np.nan, np.nan, np.nan, np.nan],
                    [1.75, 1.25, 0.75, 1.25, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
        )

    def test_downscale_x2_and_shift(self):
        target_gm = GridMapping.regular(
            (8, 6), (49.8, 9.8), 2 * self.res, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            source_gm=self.source_gm,
            interp_methods=1,
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((6, 8), target_ds.refl.shape)
        print(repr(target_ds.refl.values))
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 0.75, 1.0, 1.75, 1.25, np.nan, np.nan, np.nan],
                    [np.nan, 1.25, 1.0, 1.25, 1.75, np.nan, np.nan, np.nan],
                    [np.nan, 1.75, 1.25, 0.75, 1.25, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
        )

    def test_upscale_x2(self):
        target_gm = GridMapping.regular(
            (8, 6), (50, 10), self.res / 2, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            source_gm=self.source_gm,
            interp_methods=1,
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((6, 8), target_ds.refl.shape)
        print(repr(target_ds.refl.values))
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [1.0, 0.5, 0.0, 1.0, 2.0, 1.0, 0.0, 1.5],
                    [0.5, 1.0, 1.5, 1.25, 1.0, 1.5, 2.0, 1.75],
                    [0.0, 1.5, 3.0, 1.5, 0.0, 2.0, 4.0, 2.0],
                    [2.0, 1.75, 1.5, 1.0, 0.5, 1.25, 2.0, 1.5],
                    [4.0, 2.0, 0.0, 0.5, 1.0, 0.5, 0.0, 1.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
        )

    def test_upscale_x2_and_shift(self):
        target_gm = GridMapping.regular(
            (8, 6), (49.9, 9.95), self.res / 2, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            source_gm=self.source_gm,
            interp_methods=1,
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((6, 8), target_ds.refl.shape)
        print(repr(target_ds.refl.values))
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [np.nan, np.nan, 0.5, 1.0, 1.5, 1.25, 1.0, 1.5],
                    [np.nan, np.nan, 0.0, 1.5, 3.0, 1.5, 0.0, 2.0],
                    [np.nan, np.nan, 2.0, 1.75, 1.5, 1.0, 0.5, 1.25],
                    [np.nan, np.nan, 4.0, 2.0, 0.0, 0.5, 1.0, 0.5],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
        )

    def test_shift(self):
        target_gm = GridMapping.regular(
            (8, 6), (50.2, 10.1), self.res, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            source_gm=self.source_gm,
            interp_methods=1,
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((6, 8), target_ds.refl.shape)
        print(repr(target_ds.refl.values))
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [0.0, 2.0, 0.0, 3.0, 0.0, 4.0, np.nan, np.nan],
                    [np.nan, np.nan, 4.0, 0.0, 1.0, 0.0, np.nan, np.nan],
                    [np.nan, np.nan, 0.0, 2.0, 0.0, 3.0, np.nan, np.nan],
                    [2.0, 0.0, 3.0, 0.0, 4.0, 0.0, np.nan, np.nan],
                    [0.0, 4.0, 0.0, 1.0, 0.0, 2.0, np.nan, np.nan],
                ]
            ),
        )

        target_gm = GridMapping.regular(
            (8, 6), (49.8, 9.9), self.res, self.source_gm.crs
        )
        target_ds = affine_transform_dataset(
            self.source_ds,
            target_gm,
            source_gm=self.source_gm,
            interp_methods=1,
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(self.source_ds.variables).union(["spatial_ref"]),
            set(target_ds.variables),
        )
        self.assertEqual((6, 8), target_ds.refl.shape)
        print(repr(target_ds.refl.values))
        np.testing.assert_almost_equal(
            target_ds.refl.values,
            np.array(
                [
                    [np.nan, np.nan, 2.0, 0.0, np.nan, np.nan, 4.0, 0.0],
                    [np.nan, np.nan, 0.0, 4.0, np.nan, np.nan, 0.0, 2.0],
                    [np.nan, np.nan, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
                    [np.nan, np.nan, 0.0, 3.0, 0.0, 4.0, 0.0, 1.0],
                    [np.nan, np.nan, 4.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
        )

    def test_affine_raise_value_error(self):
        target_gm = GridMapping.regular(
            (8, 6), (50.2, 10.1), self.res, self.source_gm.crs
        )
        with self.assertRaises(ValueError) as cm:
            _ = affine_transform_dataset(
                self.source_ds,
                target_gm,
                source_gm=self.source_gm,
                interp_methods=3,
            )
        self.assertIn(
            "interp_methods must be one of 0, 1, 'nearest', 'bilinear'. "
            "Higher order is not supported for 3D arrays in affine transforms, "
            "as it causes unintended blending across the non-spatial (e.g., time) "
            "dimension.",
            str(cm.exception),
        )
