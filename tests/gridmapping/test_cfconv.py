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

import shutil
import unittest
import warnings

import numpy as np
import pyproj
import xarray as xr
import zarr

# noinspection PyProtectedMember
from xcube_resampling.gridmapping.cfconv import (
    GridCoords,
    GridMappingProxy,
    _find_potential_coord_vars,
    _is_potential_coord_var,
    add_spatial_ref,
    get_dataset_grid_mapping_proxies,
)

CRS_WGS84 = pyproj.crs.CRS(4326)
CRS_CRS84 = pyproj.crs.CRS.from_string("urn:ogc:def:crs:OGC:1.3:CRS84")
CRS_UTM_33N = pyproj.crs.CRS(32633)

CRS_ROTATED_POLE = pyproj.crs.CRS.from_cf(
    dict(
        grid_mapping_name="rotated_latitude_longitude",
        grid_north_pole_latitude=32.5,
        grid_north_pole_longitude=170.0,
    )
)


class GetDatasetGridMappingsTest(unittest.TestCase):
    def test_no_crs_lon_lat_common_names(self):
        dataset = xr.Dataset(
            coords=dict(
                lon=xr.DataArray(np.linspace(10, 12, 11), dims="lon"),
                lat=xr.DataArray(np.linspace(50, 52, 11), dims="lat"),
            )
        )
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn(None, grid_mappings)
        grid_mapping = grid_mappings.get(None)
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertEqual(CRS_WGS84, grid_mapping.crs)
        self.assertEqual("latitude_longitude", grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual("lon", grid_mapping.coords.x.name)
        self.assertEqual("lat", grid_mapping.coords.y.name)

    def test_no_crs_lon_lat_standard_names(self):
        dataset = xr.Dataset(
            coords=dict(
                weird_x=xr.DataArray(
                    np.linspace(10, 12, 11),
                    dims="i",
                    attrs=dict(standard_name="longitude"),
                ),
                weird_y=xr.DataArray(
                    np.linspace(50, 52, 11),
                    dims="j",
                    attrs=dict(standard_name="latitude"),
                ),
            )
        )
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn(None, grid_mappings)
        grid_mapping = grid_mappings.get(None)
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertEqual(CRS_WGS84, grid_mapping.crs)
        self.assertEqual("latitude_longitude", grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual("weird_x", grid_mapping.coords.x.name)
        self.assertEqual("weird_y", grid_mapping.coords.y.name)

    def test_crs_x_y_with_common_names(self):
        dataset = xr.Dataset(
            dict(crs=xr.DataArray(0, attrs=CRS_UTM_33N.to_cf())),
            coords=dict(
                x=xr.DataArray(np.linspace(1000, 12000, 11), dims="x"),
                y=xr.DataArray(np.linspace(5000, 52000, 11), dims="y"),
            ),
        )
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn("crs", grid_mappings)
        grid_mapping = grid_mappings.get("crs")
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertEqual(CRS_UTM_33N, grid_mapping.crs)
        self.assertEqual("transverse_mercator", grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual("x", grid_mapping.coords.x.name)
        self.assertEqual("y", grid_mapping.coords.y.name)

    def test_crs_x_y_with_standard_names(self):
        dataset = xr.Dataset(
            dict(crs=xr.DataArray(0, attrs=CRS_UTM_33N.to_cf())),
            coords=dict(
                myx=xr.DataArray(
                    np.linspace(1000, 12000, 11),
                    dims="x",
                    attrs=dict(standard_name="projection_x_coordinate"),
                ),
                myy=xr.DataArray(
                    np.linspace(5000, 52000, 11),
                    dims="y",
                    attrs=dict(standard_name="projection_y_coordinate"),
                ),
            ),
        )
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn("crs", grid_mappings)
        grid_mapping = grid_mappings.get("crs")
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertEqual(CRS_UTM_33N, grid_mapping.crs)
        self.assertEqual("transverse_mercator", grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual("myx", grid_mapping.coords.x.name)
        self.assertEqual("myy", grid_mapping.coords.y.name)

    def test_latitude_longitude_with_x_y(self):
        # This is what we get when opening a CRS-84 GeoTIFF using rioxarray
        dataset = xr.Dataset(
            dict(
                band_1=xr.DataArray(np.zeros((11, 11)), dims=["y", "x"]),
                spatial_ref=xr.DataArray(0, attrs=CRS_CRS84.to_cf()),
            ),
            coords=dict(
                x=xr.DataArray(np.linspace(10, 20, 11), dims="x"),
                y=xr.DataArray(np.linspace(50, 40, 11), dims="y"),
            ),
        )
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn("spatial_ref", grid_mappings)
        grid_mapping = grid_mappings.get("spatial_ref")
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertEqual(CRS_CRS84, grid_mapping.crs)
        self.assertEqual("latitude_longitude", grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual("x", grid_mapping.coords.x.name)
        self.assertEqual("y", grid_mapping.coords.y.name)

    def test_crs_in_attrs(self):
        dataset = xr.Dataset(
            coords=dict(
                lon=xr.DataArray(np.linspace(10, 12, 11), dims="lon"),
                lat=xr.DataArray(np.linspace(50, 52, 11), dims="lat"),
            ),
            attrs={
                "crs_wkt": (
                    'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",'
                    'MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World '
                    'Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 '
                    '(G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER'
                    '["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic '
                    'System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"]'
                    ',MEMBER["World Geodetic System 1984 (G2296)"],ELLIPSOID'
                    '["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],'
                    'ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT'
                    '["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS'
                    '["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT'
                    '["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)"'
                    ',east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],'
                    'USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."]'
                    ',BBOX[-90,-180,90,180]],ID["EPSG",4326]]'
                ),
                "semi_major_axis": 6378137.0,
                "semi_minor_axis": 6356752.314245179,
                "inverse_flattening": 298.257223563,
                "reference_ellipsoid_name": "WGS 84",
                "longitude_of_prime_meridian": 0.0,
                "prime_meridian_name": "Greenwich",
                "geographic_crs_name": "WGS 84",
                "horizontal_datum_name": "World Geodetic System 1984 ensemble",
                "grid_mapping_name": "latitude_longitude",
            },
        )
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn(None, grid_mappings)
        grid_mapping = grid_mappings.get(None)
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertEqual(CRS_WGS84, grid_mapping.crs)
        self.assertEqual("latitude_longitude", grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual("lon", grid_mapping.coords.x.name)
        self.assertEqual("lat", grid_mapping.coords.y.name)

    def test_emit_warning(self):
        dataset = xr.Dataset(
            coords=dict(
                lon=xr.DataArray([10], dims="lon"),
                lat=xr.DataArray([50], dims="lat"),
            ),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = get_dataset_grid_mapping_proxies(dataset, emit_warnings=True)
        self.assertEqual(len(w), 1)
        self.assertIn("missing x- and/or y-coordinates", str(w[0].message))

    def test_rotated_pole_with_common_names(self):
        dataset = xr.Dataset(
            dict(rotated_pole=xr.DataArray(0, attrs=CRS_ROTATED_POLE.to_cf())),
            coords=dict(
                rlon=xr.DataArray(np.linspace(-180, 180, 11), dims="rlon"),
                rlat=xr.DataArray(np.linspace(0, 90, 11), dims="rlat"),
            ),
        )
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn("rotated_pole", grid_mappings)
        grid_mapping = grid_mappings.get("rotated_pole")
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertIn("Geographic", grid_mapping.crs.type_name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual("rlon", grid_mapping.coords.x.name)
        self.assertEqual("rlat", grid_mapping.coords.y.name)

    def test_rotated_pole_with_standard_names(self):
        dataset = xr.Dataset(
            dict(rotated_pole=xr.DataArray(0, attrs=CRS_ROTATED_POLE.to_cf())),
            coords=dict(
                u=xr.DataArray(
                    np.linspace(-180, 180, 11),
                    dims="u",
                    attrs=dict(standard_name="grid_longitude"),
                ),
                v=xr.DataArray(
                    np.linspace(0, 90, 11),
                    dims="v",
                    attrs=dict(standard_name="grid_latitude"),
                ),
            ),
        )
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn("rotated_pole", grid_mappings)
        grid_mapping = grid_mappings.get("rotated_pole")
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertIn("Geographic", grid_mapping.crs.type_name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual("u", grid_mapping.coords.x.name)
        self.assertEqual("v", grid_mapping.coords.y.name)

    def test_bounds_detection(self):
        dataset = xr.Dataset(
            coords={
                "lon": xr.DataArray(np.linspace(0, 10, 5), dims="lon"),
                "lat": xr.DataArray(np.linspace(0, 5, 5), dims="lat"),
                "lon_bnds": xr.DataArray(np.linspace(0, 10, 10), dims="bnds"),
                "lat_bounds": xr.DataArray(np.linspace(0, 5, 10), dims="bnds"),
                "alt": xr.DataArray(np.linspace(0, 100, 5), dims="alt"),
            }
        )
        dataset["lat"].attrs["bounds"] = "lat_bounds"

        potential_vars = _find_potential_coord_vars(dataset)

        # lon_bnds and lat_bounds should be excluded, but lon/lat/alt included
        self.assertIn("lon", potential_vars)
        self.assertIn("lat", potential_vars)
        self.assertIn("alt", potential_vars)
        self.assertNotIn("lon_bnds", potential_vars)
        self.assertNotIn("lat_bounds", potential_vars)

    def test_coordinates_in_attribute(self):
        dataset = xr.Dataset(
            {
                "x": xr.DataArray([0, 1]),
                "y": xr.DataArray([0, 1]),
            },
            attrs={"coordinates": "x y"},
        )

        result = _find_potential_coord_vars(dataset)
        self.assertIn("x", result)
        self.assertIn("y", result)

    def test_var_not_in_dataset(self):
        dataset = xr.Dataset(
            coords={
                "lon": xr.DataArray(np.linspace(0, 10, 5), dims="lon"),
                "lat": xr.DataArray(np.linspace(0, 5, 5), dims="lat"),
            }
        )
        bounds_vars = set()
        self.assertFalse(_is_potential_coord_var(dataset, bounds_vars, "missing_var"))


class XarrayDecodeCfTest(unittest.TestCase):
    """Find out how xarray treats 1D and 2D coordinate variables when decode_cf=True or =False"""

    def test_cf_1d_coords(self):
        self._write_coords(*self._gen_1d())
        self.assertVarNames({"noise", "crs"}, {"lon", "lat"}, decode_cf=True)
        self.assertVarNames({"noise", "crs"}, {"lon", "lat"}, decode_cf=False)

    def test_cf_1d_data_vars(self):
        self._write_data_vars(*self._gen_1d())
        self.assertVarNames({"noise", "crs"}, {"lon", "lat"}, decode_cf=True)
        self.assertVarNames({"noise", "crs"}, {"lon", "lat"}, decode_cf=False)

    def test_cf_2d_coords(self):
        self._write_coords(*self._gen_2d())
        self.assertVarNames({"noise", "crs"}, {"lon", "lat"}, decode_cf=True)
        self.assertVarNames({"noise", "crs", "lon", "lat"}, set(), decode_cf=False)

    def test_cf_2d_data_vars(self):
        self._write_data_vars(*self._gen_2d())
        self.assertVarNames({"noise", "crs", "lon", "lat"}, set(), decode_cf=True)
        self.assertVarNames({"noise", "crs", "lon", "lat"}, set(), decode_cf=False)

    def assertVarNames(
        self, exp_data_vars: set[str], exp_coords: set[str], decode_cf: bool
    ):
        ds1 = xr.open_zarr("noise.zarr", decode_cf=decode_cf)
        self.assertEqual(exp_coords, set(ds1.coords))
        self.assertEqual(exp_data_vars, set(ds1.data_vars))

    @classmethod
    def _write_coords(cls, noise, crs, lon, lat):
        dataset = xr.Dataset(dict(noise=noise, crs=crs), coords=dict(lon=lon, lat=lat))
        dataset.to_zarr("noise.zarr", mode="w")

    @classmethod
    def _write_data_vars(cls, noise, crs, lon, lat):
        dataset = xr.Dataset(dict(noise=noise, crs=crs, lon=lon, lat=lat))
        dataset.to_zarr("noise.zarr", mode="w")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("noise.zarr")

    def _gen_1d(self):
        noise = xr.DataArray(np.random.random((11, 11)), dims=("lat", "lon"))
        crs = xr.DataArray(0, attrs=CRS_CRS84.to_cf())
        lon = xr.DataArray(np.linspace(10, 12, 11), dims="lon")
        lat = xr.DataArray(np.linspace(50, 52, 11), dims="lat")
        noise.attrs["grid_mapping"] = "crs"
        lon.attrs["standard_name"] = "longitude"
        lat.attrs["standard_name"] = "latitude"
        self.assertEqual(("lon",), lon.dims)
        self.assertEqual(("lat",), lat.dims)
        return noise, crs, lon, lat

    def _gen_2d(self):
        noise = xr.DataArray(np.random.random((11, 11)), dims=("y", "x"))
        crs = xr.DataArray(0, attrs=CRS_CRS84.to_cf())
        lon = xr.DataArray(np.linspace(10, 12, 11), dims="x")
        lat = xr.DataArray(np.linspace(50, 52, 11), dims="y")
        lat, lon = xr.broadcast(lat, lon)
        noise.attrs["grid_mapping"] = "crs"
        lon.attrs["standard_name"] = "longitude"
        lat.attrs["standard_name"] = "latitude"
        self.assertEqual(("y", "x"), lon.dims)
        self.assertEqual(("y", "x"), lat.dims)
        return noise, crs, lon, lat


class TestAddSpatialRef(unittest.TestCase):

    def setUp(self):
        # Create an in-memory Zarr group
        self.store = zarr.MemoryStore()
        self.group = zarr.group(store=self.store, overwrite=True)
        # Create a dummy dataset array
        self.group.zeros("data", shape=(3, 3), chunks=(3, 3), dtype=np.float32)
        # Add _ARRAY_DIMENSIONS attribute to simulate xarray
        self.group["data"].attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]

    def test_add_spatial_ref_creates_variable(self):
        crs = pyproj.CRS.from_epsg(4326)
        add_spatial_ref(self.store, crs, crs_var_name="spatial_ref_test")

        # The spatial_ref_test variable should exist
        self.assertIn("spatial_ref_test", self.group)
        spatial_ref = self.group["spatial_ref_test"]
        self.assertEqual(spatial_ref.shape, ())
        self.assertTrue(spatial_ref.attrs)  # CRS attributes added
        self.assertIn("_ARRAY_DIMENSIONS", spatial_ref.attrs)
        self.assertEqual(spatial_ref.attrs["_ARRAY_DIMENSIONS"], [])

    def test_add_grid_mapping_attribute(self):
        crs = pyproj.CRS.from_epsg(4326)
        add_spatial_ref(
            self.store,
            crs,
            crs_var_name="spatial_ref_test",
            xy_dim_names=("x", "y"),
        )

        # Original array should have grid_mapping pointing to spatial_ref_test
        self.assertEqual(
            self.group["data"].attrs.get("grid_mapping"), "spatial_ref_test"
        )
