# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import math
import unittest
import dask
import pyproj
import numpy as np
import dask.array as da
import xarray as xr
from dask.highlevelgraph import HighLevelGraph
from distributed import LocalCluster, Client
from xcube_resampling.grid import Grid
from xcube_resampling.rectifier import Rectifier

# noinspection PyTypeChecker
class RectificationTest(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        #self._client = Client(LocalCluster(n_workers=8, processes=True))
        #self._client = Client(n_workers=1, threads_per_worker=6)
        #dask.config.set(scheduler='synchronous')
        self.path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'

    @classmethod
    def tearDownClass(self) -> None:
        #self._client.close()
        pass

    def test_olci_forward_index(self):
        l1b = xr.open_mfdataset([self.path + '/geo_coordinates.nc'] +
                                [(self.path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (2400, 2400))
        r = Rectifier(l1b['longitude'].data, l1b['latitude'].data, dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        index,_ = r.compute_forward_index()
        print()
        print(r.forward_index)
        print(index)

    def test_olci_covering_grid(self):
        l1b = xr.open_mfdataset([self.path + '/geo_coordinates.nc'] +
                                [(self.path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (0, 0), (0.003, -0.003), (0, 0), (2400, 2400))
        r = Rectifier(l1b['longitude'].data, l1b['latitude'].data, dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        dst_grid = r.determine_covering_dst_grid()
        print()
        print(dst_grid)

    def test_olci_inverse_index(self):
        l1b = xr.open_mfdataset([self.path + '/geo_coordinates.nc'] +
                                [(self.path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (2400, 2400))
        r = Rectifier(l1b['longitude'].data, l1b['latitude'].data, dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        r.prepare_inverse_index()
        print()
        print(r.dask_inverse_index)
        index = r.dask_inverse_index.compute()
        #print(index)
        src_lon = l1b['longitude'].data.compute()
        src_lat = l1b['latitude'].data.compute()
        #
        for dst_pos in [(1000, 1000), (5000, 2600), (3166,2832)]:
            src_pos = index[:,dst_pos[1],dst_pos[0]]
            src_int = (math.floor(src_pos[0]-0.5), math.floor(src_pos[1]-0.5))
            dst_lon = dst_grid.x_min + (dst_pos[0] + 0.5) * dst_grid.x_res
            dst_lat = dst_grid.y_min + (dst_pos[1] + 0.5) * dst_grid.y_res
            print("src frac idx", src_pos)
            print("upper left  ", src_lon[src_int[1], src_int[0]], src_lat[src_int[1], src_int[0]])
            print("upper right ", src_lon[src_int[1], src_int[0]+1], src_lat[src_int[1], src_int[0]+1])
            print("dst point   ", dst_lon, dst_lat)
            print("lower left ", src_lon[src_int[1]+1, src_int[0]], src_lat[src_int[1]+1, src_int[0]])
            print("lower right ", src_lon[src_int[1]+1, src_int[0]+1], src_lat[src_int[1]+1, src_int[0]+1])
            assert dst_lon >= min(src_lon[src_int[1], src_int[0]], src_lon[src_int[1]+1, src_int[0]])
            assert dst_lon <= max(src_lon[src_int[1], src_int[0]+1], src_lon[src_int[1]+1, src_int[0]+1])
            assert dst_lat <= max(src_lat[src_int[1], src_int[0]], src_lat[src_int[1], src_int[0]+1])
            assert dst_lat >= min(src_lat[src_int[1]+1, src_int[0]], src_lat[src_int[1]+1, src_int[0]+1])

    def test_olci_rectify_one_tile(self):
        l1b = xr.open_mfdataset([self.path + '/geo_coordinates.nc'] +
                                [(self.path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (3.539, 45.249), (0.003, -0.003), (1867, 1904), (2400, 2400))
        r = Rectifier(l1b['longitude'].data, l1b['latitude'].data, dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        r.prepare_inverse_index()
        r.prepare_rectification(l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)
        print()
        print(r.dask_rectified)
        result = r.compute_rectification()
        print(result)

    def test_olci_rectify(self):
        l1b = xr.open_mfdataset([self.path + '/geo_coordinates.nc'] +
                                [(self.path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        #dst_grid = Grid(pyproj.CRS("EPSG:4326"), (0, 0), (0.003, -0.003), (0, 0), (2400, 2400))
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (2400, 2400))
        r = Rectifier(l1b['longitude'].data, l1b['latitude'].data, dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        r.prepare_inverse_index()
        r.prepare_rectification(l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)
        print()
        print(r.dask_rectified)
        result = r.compute_rectification()
        print(result)

    def test_olci_rectify_write(self):
        self.path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([self.path + '/geo_coordinates.nc'] +
                                [(self.path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        #dst_grid = Grid(pyproj.CRS("EPSG:4326"), (0, 0), (0.003, -0.003), (0, 0), (2400, 2400))
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (2400, 2400))
        r = Rectifier(l1b['longitude'].data, l1b['latitude'].data, dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        r.prepare_inverse_index()
        result = r.prepare_rectification(l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)
        print()
        #print(dict(r.dask_rectified.dask))
        #result = r.compute_rectification()
        print(result)
        ds = xr.Dataset({"oa08": (["lat", "lon"], result[1]),
                         "oa12": (["lat", "lon"], result[0])},
                        coords={
                            "lat": (["lat"], r.dst_grid.y_axis()),
                            "lon": (["lon"], r.dst_grid.x_axis())
                        })
        ds.to_netcdf("/windows/tmp/eopf/repojected2.nc")

    def test_olci_rectify_write_profiling(self):
        import cProfile
        cProfile.runctx('self.test_olci_rectify_write()', globals(), locals(), None)

    def test_olci_rectify_to_covering_grid_write(self):
        l1b = xr.open_mfdataset([self.path + '/geo_coordinates.nc'] +
                                [(self.path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        result, dst_grid = Rectifier.rectify_to_covering_grid(l1b['longitude'].data,
                                                              l1b['latitude'].data,
                                                              pyproj.CRS("EPSG:4326"),
                                                              (0.003, -0.003),
                                                              (2400, 2400),
                                                              l1b['Oa12_radiance'].data,
                                                              l1b['Oa08_radiance'].data)
        print(result)
        ds = xr.Dataset({"oa08": (["lat", "lon"], result[1]),
                         "oa12": (["lat", "lon"], result[0])},
                        coords={
                            "lat": (["lat"], dst_grid.y_axis()),
                            "lon": (["lon"], dst_grid.x_axis())
                        })
        ds.to_netcdf("/windows/tmp/eopf/repojected3.nc")

    # ===== tests using single chunk full images with numpy instead of dask arrays =====

    def test_olci_numpy_read_inputs(self):
        l1b = xr.open_mfdataset([self.path + '/geo_coordinates.nc'] +
                                [(self.path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        longitude, latitude, oa12, oa08 = dask.compute(l1b["longitude"].data, l1b["latitude"].data, l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)

    def test_olci_numpy_forward_index(self):
        l1b = xr.open_mfdataset([self.path + '/geo_coordinates.nc'] +
                                [(self.path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        longitude, latitude, oa12, oa08 = dask.compute(l1b["longitude"].data, l1b["latitude"].data, l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)
        #dst_grid = Grid(pyproj.CRS("EPSG:4326"), (0, 0), (0.003, -0.003), (0, 0), (2400, 2400))
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (6667, 4304))
        r = Rectifier(longitude, latitude, dst_grid)
        r.forward_index = Rectifier.forward_index_block(longitude, latitude, r.dst_grid)
        bbox_blocks_raw = Rectifier.bboxes_block(
            r.forward_index,
            dst_grid=r.dst_grid,
            src_tile_size=r.src_lat.shape,
            src_size=r.src_lat.shape,
            block_id=(0,0))
        # we compute min and max of all 4 bounds though we need only two of each,
        r.src_bboxes = np.vstack((np.nanmin(bbox_blocks_raw, axis=(3,4)),
                                  np.nanmax(bbox_blocks_raw, axis=(3,4))))

    def test_olci_numpy_inverse_index(self):
        l1b = xr.open_mfdataset([self.path + '/geo_coordinates.nc'] +
                                [(self.path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        longitude, latitude, oa12, oa08 = dask.compute(l1b["longitude"].data, l1b["latitude"].data, l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)
        #dst_grid = Grid(pyproj.CRS("EPSG:4326"), (0, 0), (0.003, -0.003), (0, 0), (2400, 2400))
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (6667, 4304))
        r = Rectifier(longitude, latitude, dst_grid)
        r.forward_index = Rectifier.forward_index_block(longitude, latitude, r.dst_grid)
        bbox_blocks_raw = Rectifier.bboxes_block(
                                        r.forward_index,
                                        dst_grid=r.dst_grid,
                                        src_tile_size=r.src_lat.shape,
                                        src_size=r.src_lat.shape,
            block_id=(0,0))
        # we compute min and max of all 4 bounds though we need only two of each,
        # dask seems not to delay a []
        r.src_bboxes = np.vstack((np.nanmin(bbox_blocks_raw, axis=(3,4)),
                                          np.nanmax(bbox_blocks_raw, axis=(3,4))))
        lon_lat = np.stack((longitude, latitude))
        r.inverse_index = Rectifier.inverse_index_block(
                                 (0, 0),
                                 r.dst_grid,
                                 r.src_bboxes[:, 0, 0],
                                 lon_lat[0,
                                     r.src_bboxes[1, 0, 0]:r.src_bboxes[3, 0, 0],
                                     r.src_bboxes[0, 0, 0]:r.src_bboxes[2, 0, 0]],
                                 lon_lat[1,
                                     r.src_bboxes[1, 0, 0]:r.src_bboxes[3, 0, 0],
                                     r.src_bboxes[0, 0, 0]:r.src_bboxes[2, 0, 0]])

    def test_olci_numpy_rectify_data(self):
        l1b = xr.open_mfdataset([self.path + '/geo_coordinates.nc'] +
                                [(self.path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        longitude, latitude, oa12, oa08 = dask.compute(l1b["longitude"].data, l1b["latitude"].data, l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)
        #dst_grid = Grid(pyproj.CRS("EPSG:4326"), (0, 0), (0.003, -0.003), (0, 0), (2400, 2400))
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (6667, 4304))
        r = Rectifier(longitude, latitude, dst_grid)
        r.forward_index = Rectifier.forward_index_block(longitude, latitude, r.dst_grid)
        bbox_blocks_raw = Rectifier.bboxes_block(
                                        r.forward_index,
                                        dst_grid=r.dst_grid,
                                        src_tile_size=r.src_lat.shape,
                                        src_size=r.src_lat.shape,
                                        block_id=(0,0))
        # we compute min and max of all 4 bounds though we need only two of each,
        # dask seems not to delay a []
        r.src_bboxes = np.vstack((np.nanmin(bbox_blocks_raw, axis=(3,4)),
                                          np.nanmax(bbox_blocks_raw, axis=(3,4))))
        r.inverse_index = Rectifier.inverse_index_block(
                                 (0, 0),
                                 r.dst_grid,
                                 r.src_bboxes[:, 0, 0],
                                 longitude[r.src_bboxes[1, 0, 0]:r.src_bboxes[3, 0, 0],
                                     r.src_bboxes[0, 0, 0]:r.src_bboxes[2, 0, 0]],
                                 latitude[r.src_bboxes[1, 0, 0]:r.src_bboxes[3, 0, 0],
                                     r.src_bboxes[0, 0, 0]:r.src_bboxes[2, 0, 0]])
        r.rectified = Rectifier.rectify_block((0, 0), r.src_bboxes[:, 0, 0], r.inverse_index, oa12, oa08)

    def test_olci_numpy_rectify_write(self):
        l1b = xr.open_mfdataset([self.path + '/geo_coordinates.nc'] +
                                [(self.path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        longitude, latitude, oa12, oa08 = dask.compute(l1b["longitude"].data, l1b["latitude"].data, l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)
        #dst_grid = Grid(pyproj.CRS("EPSG:4326"), (0, 0), (0.003, -0.003), (0, 0), (2400, 2400))
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (6667, 4304))
        r = Rectifier(longitude, latitude, dst_grid)
        r.forward_index = Rectifier.forward_index_block(longitude, latitude, r.dst_grid)
        bbox_blocks_raw = Rectifier.bboxes_block(
                                        r.forward_index,
                                        dst_grid=r.dst_grid,
                                        src_tile_size=r.src_lat.shape,
                                        src_size=r.src_lat.shape,
            block_id=(0,0))
        # we compute min and max of all 4 bounds though we need only two of each,
        # dask seems not to delay a []
        r.src_bboxes = np.vstack((np.nanmin(bbox_blocks_raw, axis=(3,4)),
                                          np.nanmax(bbox_blocks_raw, axis=(3,4))))
        r.inverse_index = Rectifier.inverse_index_block(
            (0, 0),
            r.dst_grid,
            r.src_bboxes[:, 0, 0],
            longitude[r.src_bboxes[1, 0, 0]:r.src_bboxes[3, 0, 0],
            r.src_bboxes[0, 0, 0]:r.src_bboxes[2, 0, 0]],
            latitude[r.src_bboxes[1, 0, 0]:r.src_bboxes[3, 0, 0],
            r.src_bboxes[0, 0, 0]:r.src_bboxes[2, 0, 0]])
        r.rectified = Rectifier.rectify_block((0, 0), r.src_bboxes[:, 0, 0], r.inverse_index, oa12, oa08)

        ds = xr.Dataset({"oa08": (["lat", "lon"], r.rectified[1]),
                         "oa12": (["lat", "lon"], r.rectified[0])},
                        coords={
                            "lat": (["lat"], r.dst_grid.y_axis()),
                            "lon": (["lon"], r.dst_grid.x_axis())
                        })
        ds.to_netcdf("/windows/tmp/eopf/repojected3.nc")
