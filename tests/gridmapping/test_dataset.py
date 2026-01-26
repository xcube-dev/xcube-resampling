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

import os.path
import unittest

import numpy as np
import pyproj
import xarray as xr

from xcube_resampling.gridmapping import GridMapping

from ..sampledata import create_s2plus_dataset

GEO_CRS = pyproj.crs.CRS(4326)
NOT_A_GEO_CRS = pyproj.crs.CRS(5243)


# noinspection PyMethodMayBeStatic
class DatasetGridMappingTest(unittest.TestCase):
    def test_from_non_regular_cube(self):
        lon = np.array(
            [[8, 9.3, 10.6, 11.9], [8, 9.2, 10.4, 11.6], [8, 9.1, 10.2, 11.3]],
            dtype=np.float32,
        )
        lat = np.array(
            [[56, 56.1, 56.2, 56.3], [55, 55.2, 55.4, 55.6], [54, 54.3, 54.6, 54.9]],
            dtype=np.float32,
        )
        rad = np.random.random(3 * 4).reshape((3, 4))
        dims = ("y", "x")
        dataset = xr.Dataset(
            dict(
                lon=xr.DataArray(lon, dims=dims),
                lat=xr.DataArray(lat, dims=dims),
                rad=xr.DataArray(rad, dims=dims),
            )
        )
        gm = GridMapping.from_dataset(dataset)
        self.assertEqual((4, 3), gm.size)
        self.assertEqual((4, 3), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(False, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(False, gm.is_j_axis_up)
        self.assertEqual((2, 3, 4), gm.xy_coords.shape)
        self.assertEqual(("coord", "y", "x"), gm.xy_coords.dims)
        self.assertEqual((1.2, 0.85), gm.xy_res)

    def test_crs(self):
        ds = xr.Dataset(
            {
                "var": (("lat", "lon"), np.random.rand(2, 2)),
            },
            coords={
                "lon": ("lon", [0, 1]),
                "lat": ("lat", [0, 1]),
            },
        )

        result = GridMapping.from_dataset(ds, crs="EPSG:4326")
        self.assertTrue(result.is_regular)
        self.assertEqual(result.crs.to_string(), "EPSG:4326")

    def test_from_real_olci(self):
        olci_l2_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "examples",
            "inputdata",
            "S3-OLCI-L2A.zarr.zip",
        )

        dataset = xr.open_zarr(olci_l2_path, consolidated=False)
        gm = GridMapping.from_dataset(dataset)
        self.assertEqual((1189, 1890), gm.size)
        self.assertEqual((512, 512), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual((0.00447132, 0.00255235), gm.xy_res)
        self.assertEqual(False, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(False, gm.is_j_axis_up)
        self.assertEqual((2, 1890, 1189), gm.xy_coords.shape)
        self.assertEqual(("coord", "y", "x"), gm.xy_coords.dims)

        gm = gm.to_regular()
        self.assertEqual((1636, 2132), gm.size)

    def test_from_sentinel_2(self):
        dataset = create_s2plus_dataset()
        tol = 1e-6

        gm = GridMapping.from_dataset(dataset, tolerance=tol)
        # Should pick the projected one which is regular
        self.assertIn("Projected", gm.crs.type_name)
        self.assertEqual(True, gm.is_regular)

        gm = GridMapping.from_dataset(dataset, prefer_is_regular=True, tolerance=tol)
        # Should pick the projected one which is regular
        self.assertIn("Projected", gm.crs.type_name)
        self.assertEqual(True, gm.is_regular)

        gm = GridMapping.from_dataset(dataset, prefer_is_regular=False, tolerance=tol)
        # Should pick the geographic one which is irregular
        self.assertIn("Geographic", gm.crs.type_name)
        self.assertEqual(False, gm.is_regular)

        gm = GridMapping.from_dataset(dataset, prefer_crs=GEO_CRS, tolerance=tol)
        # Should pick the geographic one which is irregular
        self.assertIn("Geographic", gm.crs.type_name)
        self.assertEqual(False, gm.is_regular)

        gm = GridMapping.from_dataset(
            dataset, prefer_crs=GEO_CRS, prefer_is_regular=True, tolerance=tol
        )
        # Should pick the geographic one which is irregular
        self.assertIn("Geographic", gm.crs.type_name)
        self.assertEqual(False, gm.is_regular)

    def test_no_grid_mapping_found(self):
        with self.assertRaises(ValueError) as cm:
            GridMapping.from_dataset(xr.Dataset())
        self.assertEqual("cannot find any grid mapping in dataset", f"{cm.exception}")
