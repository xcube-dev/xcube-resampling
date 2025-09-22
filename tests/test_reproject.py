# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np

from xcube_resampling.gridmapping import CRS_WGS84, GridMapping
from xcube_resampling.reproject import reproject_dataset

from .sampledata import (
    create_2x5x5_dataset_regular_utm,
    create_5x5_dataset_regular_utm,
    create_large_dataset_for_reproject,
)


# noinspection PyMethodMayBeStatic
class ReprojectDatasetTest(unittest.TestCase):
    def test_reproject_target_gm(self):
        source_ds = create_5x5_dataset_regular_utm()

        # test projected CRS, similar resolution
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(4320080, 3382480), xy_res=80, crs="epsg:3035"
        )
        target_ds = reproject_dataset(source_ds, target_gm)
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

    def test_reproject_target_gm_3d(self):
        source_ds = create_2x5x5_dataset_regular_utm()

        # test projected CRS, similar resolution
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(4320080, 3382480), xy_res=80, crs="epsg:3035"
        )
        target_ds = reproject_dataset(source_ds, target_gm)
        self.assertEqual(
            set(source_ds.variables),
            set(target_ds.variables),
        )
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [
                        [1, 1, 2, 3, 4],
                        [6, 6, 7, 8, 9],
                        [11, 12, 12, 13, 14],
                        [16, 17, 17, 18, 19],
                        [21, 17, 17, 18, 19],
                    ],
                    [
                        [1, 1, 2, 3, 4],
                        [6, 6, 7, 8, 9],
                        [11, 12, 12, 13, 14],
                        [16, 17, 17, 18, 19],
                        [21, 17, 17, 18, 19],
                    ],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )

    def test_reproject_target_gm_j_axis_up(self):
        source_ds = create_5x5_dataset_regular_utm()
        target_gm = GridMapping.regular(
            size=(5, 5),
            xy_min=(4320080, 3382480),
            xy_res=80,
            crs="epsg:3035",
            is_j_axis_up=True,
        )
        target_ds = reproject_dataset(source_ds, target_gm)
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [21, 17, 17, 18, 19],
                    [16, 17, 17, 18, 19],
                    [11, 12, 12, 13, 14],
                    [6, 6, 7, 8, 9],
                    [1, 1, 2, 3, 4],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )

    def test_reproject_source_gm_j_axis_up(self):
        source_ds = create_5x5_dataset_regular_utm()
        source_ds = source_ds.isel(y=slice(None, None, -1))
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(4320080, 3382480), xy_res=80, crs="epsg:3035"
        )
        target_ds = reproject_dataset(source_ds, target_gm)
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

    def test_reproject_target_gm_finer_res(self):
        source_ds = create_5x5_dataset_regular_utm()
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(4320080, 3382480), xy_res=20, crs="epsg:3035"
        )
        target_ds = reproject_dataset(source_ds, target_gm)
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

    def test_reproject_target_gm_coarser_res(self):
        source_ds = create_5x5_dataset_regular_utm()
        target_gm = GridMapping.regular(
            size=(3, 3), xy_min=(4320050, 3382500), xy_res=120, crs="epsg:3035"
        )
        target_ds = reproject_dataset(source_ds, target_gm)
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [0, 1, 2],
                    [5, 6, 7],
                    [15, 16, 17],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )

    def test_reproject_target_gm_geographic_crs(self):
        source_ds = create_5x5_dataset_regular_utm()
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(9.9886, 53.5499), xy_res=0.0006, crs=CRS_WGS84
        )
        target_ds = reproject_dataset(source_ds, target_gm)
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

    def test_reproject_target_gm_geographic_crs_fine_res(self):
        source_ds = create_5x5_dataset_regular_utm()

        # test geographic CRS with 1/2 resolution
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(9.9886, 53.5499), xy_res=0.0003, crs=CRS_WGS84
        )
        target_ds = reproject_dataset(source_ds, target_gm)
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

    def test_reproject_complex_dask_array(self):
        source_ds = create_large_dataset_for_reproject()
        target_gm = GridMapping.regular(
            size=(10, 10),
            xy_min=(6.0, 48.0),
            xy_res=0.2,
            crs=CRS_WGS84,
            tile_size=(5, 5),
        )

        target_ds = reproject_dataset(source_ds, target_gm, interp_methods="triangular")
        self.assertCountEqual(["temperature", "onedim_data"], list(target_ds.data_vars))
        self.assertAlmostEqual(
            target_ds.temperature.values[0, 0, 0], 6353.582, places=4
        )
        self.assertAlmostEqual(
            target_ds.temperature.values[0, -1, -1], 3007.1228, places=4
        )
        self.assertEqual(
            [2, 5, 5],
            [
                target_ds.temperature.chunksizes["time"][0],
                target_ds.temperature.chunksizes["lat"][0],
                target_ds.temperature.chunksizes["lon"][0],
            ],
        )

        target_ds = reproject_dataset(source_ds, target_gm, interp_methods=1)
        self.assertCountEqual(["temperature", "onedim_data"], list(target_ds.data_vars))
        self.assertAlmostEqual(
            target_ds.temperature.values[0, 0, 0], 6353.5823, places=4
        )
        self.assertAlmostEqual(
            target_ds.temperature.values[0, -1, -1], 3007.1228, places=4
        )
        self.assertEqual(
            [2, 5, 5],
            [
                target_ds.temperature.chunksizes["time"][0],
                target_ds.temperature.chunksizes["lat"][0],
                target_ds.temperature.chunksizes["lon"][0],
            ],
        )

    def test_reproject_raise_not_implemented(self):
        source_ds = create_5x5_dataset_regular_utm()
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(4320080, 3382480), xy_res=20, crs="epsg:3035"
        )
        with self.assertRaises(NotImplementedError) as cm:
            _ = reproject_dataset(source_ds, target_gm, interp_methods="cubic")
        self.assertIn(
            "interp_methods must be one of 0, 1, 'nearest', 'bilinear', 'triangular'",
            str(cm.exception),
        )
