# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np
import xarray as xr

from xcube_resampling.gridmapping import CRS_WGS84, GridMapping
from xcube_resampling.spatial import resample_in_space

from .sampledata import (
    create_2x2_dataset_with_irregular_coords,
    create_4x4_dataset_with_irregular_coords,
    create_5x5_dataset_regular_utm,
    create_8x6_dataset_with_regular_coords,
)

nan = np.nan


# noinspection PyMethodMayBeStatic
class ResampleInSpaceTest(unittest.TestCase):
    def test_affine_transform_dataset(self):
        source_ds = create_8x6_dataset_with_regular_coords()
        source_gm = GridMapping.from_dataset(source_ds)
        target_gm = GridMapping.regular((3, 3), (50.0, 10.0), 0.1, source_gm.crs)
        target_ds = resample_in_space(
            source_ds,
            target_gm=target_gm,
            interp_methods=1,
        )
        self.assertIsInstance(target_ds, xr.Dataset)
        self.assertEqual(
            set(source_ds.variables).union(["spatial_ref"]),
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

    def test_rectify_and_downscale_dataset(self):
        source_ds = create_4x4_dataset_with_irregular_coords()
        target_gm = GridMapping.regular(
            size=(2, 2), xy_min=(-1, 51), xy_res=2, crs=CRS_WGS84
        )
        target_ds = resample_in_space(source_ds, target_gm=target_gm, interp_methods=0)
        np.testing.assert_almost_equal(
            target_ds.rad.values,
            np.array(
                [
                    [5, 2],
                    [14, 8],
                ],
                dtype=target_ds.rad.dtype,
            ),
        )
        target_ds = resample_in_space(source_ds, target_gm=target_gm, interp_methods=1)
        np.testing.assert_almost_equal(
            target_ds.rad.values,
            np.array(
                [
                    [7.5, 4.5],
                    [12.5, 9.5],
                ],
                dtype=target_ds.rad.dtype,
            ),
        )

    def test_rectify_and_upscale_dataset(self):
        source_ds = create_2x2_dataset_with_irregular_coords()
        target_gm = GridMapping.regular(
            size=(4, 4), xy_min=(-1, 49), xy_res=2, crs=CRS_WGS84
        )
        target_ds = resample_in_space(source_ds, target_gm=target_gm, interp_methods=0)
        np.testing.assert_almost_equal(
            target_ds.rad.values,
            np.array(
                [
                    [nan, nan, nan, nan],
                    [nan, 1.0, 2.0, nan],
                    [3.0, 3.0, 2.0, nan],
                    [nan, 4.0, nan, nan],
                ],
                dtype=target_ds.rad.dtype,
            ),
        )

    def test_reproject_dataset(self):
        source_ds = create_5x5_dataset_regular_utm()

        # test projected CRS similar resolution
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(4320080, 3382480), xy_res=80, crs="epsg:3035"
        )
        target_ds = resample_in_space(source_ds, target_gm=target_gm, interp_methods=0)
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [1, 1, 2, 3, 4],
                    [6, 6, 7, 8, 9],
                    [11, 12, 12, 13, 14],
                    [16, 17, 17, 18, 19],
                    [21, 17, 17, 18, 19],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )

        # test projected CRS finer resolution
        # test if subset calculation works as expected
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(4320080, 3382480), xy_res=20, crs="epsg:3035"
        )
        target_ds = resample_in_space(source_ds, target_gm=target_gm, interp_methods=0)
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [15, 16, 16, 16, 16],
                    [15, 16, 16, 16, 16],
                    [15, 16, 16, 16, 16],
                    [20, 21, 21, 21, 21],
                    [20, 21, 21, 21, 21],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )

        # test geographic CRS with similar resolution
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(9.9886, 53.5499), xy_res=0.0006, crs=CRS_WGS84
        )
        target_ds = resample_in_space(source_ds, target_gm=target_gm, interp_methods=0)
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [7, 8, 8, 8, 9],
                    [12, 13, 13, 13, 14],
                    [12, 13, 13, 13, 14],
                    [17, 18, 18, 18, 19],
                    [22, 23, 23, 23, 24],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )

        # test geographic CRS with 1/2 resolution
        # test if subset calculation works as expected
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(9.9886, 53.5499), xy_res=0.0003, crs=CRS_WGS84
        )
        target_ds = resample_in_space(source_ds, target_gm=target_gm, interp_methods=0)
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [12, 12, 12, 13, 13],
                    [17, 17, 17, 18, 18],
                    [17, 17, 17, 18, 18],
                    [22, 17, 17, 18, 18],
                    [22, 22, 22, 23, 23],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )

    def test_resample_in_space_raise_logs(self):
        source_ds = create_5x5_dataset_regular_utm()
        with self.assertLogs("xcube.resampling", level="WARNING") as cm:
            _ = resample_in_space(source_ds)
        self.assertIn(
            "If source grid mapping is regular `target_gm` must be given. "
            "Source dataset is returned.",
            cm.output[0],
        )

    def test_resample_in_space_return_input_dataset(self):
        source_ds = create_5x5_dataset_regular_utm()
        target_gm = GridMapping.from_dataset(source_ds)
        target_ds = resample_in_space(source_ds, target_gm=target_gm)
        xr.testing.assert_equal(target_ds, source_ds)
