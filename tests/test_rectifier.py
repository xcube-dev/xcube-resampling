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
        self.src_lat = da.from_array([[53.98, 53.94, 53.90],
                                 [53.88, 53.84, 53.80],
                                 [53.78, 53.74, 53.70],
                                 [53.68, 53.64, 53.60]], chunks=((3,1),(2,1)))
        self.src_lon = da.from_array([[10.35, 10.50, 10.65],
                                 [10.25, 10.40, 10.55],
                                 [10.15, 10.30, 10.45],
                                 [10.05, 10.20, 10.35]], chunks=((3,1),(2,1)))
        self.dst_grid = Grid(pyproj.CRS(4326), (10.0, 54.0), (0.2, -0.125), (3, 3), (2, 2))
        self.payload = da.from_array([[1,2,3],
                                      [4,5,6],
                                      [7,8,9],
                                      [10,11,12]], chunks=((3,1),(2,1)))

    @classmethod
    def tearDownClass(self) -> None:
        #self._client.close()
        pass

    # ===== unit tests for cascading chains of methods =====

    def test_forward_index(self):
        r = Rectifier(self.src_lon, self.src_lat, self.dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        (result, _) = r.compute_forward_index()
        assert np.array_equal(result,
                              [[[1, 2, 3], [1, 2, 2], [0, 1, 2], [0, 0, 1]],
                               [[0, 0, 0], [0, 1, 1], [1, 2, 2], [2, 2, 3]]])
        print(result)
        print("i", result[0])
        print("j", result[1])

    def test_src_bboxes(self):
        r = Rectifier(self.src_lon, self.src_lat, self.dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        (_, result) = r.compute_forward_index()
        print()
        print("min_x ", result[0])
        print("max_x ", result[2])
        print("min_y ", result[1])
        print("max_y ", result[3])
        assert np.array_equal(result,
                              np.array([[[0, 0], [0, 1]],
                                        [[0, 0], [1, 1]],
                                        [[2, 3], [3, 3]],
                                        [[4, 3], [4, 4]]]))

    def test_covering_grid(self):
        dst_grid = Grid(pyproj.CRS(4326), (0, 0), (0.2, -0.125), (0, 0), (2, 2))
        r = Rectifier(self.src_lon, self.src_lat, dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        dst_grid = r.determine_covering_dst_grid()
        print()
        print(dst_grid)
        print(r.forward_index)

    def test_inverse_index(self):
        r = Rectifier(self.src_lon, self.src_lat, self.dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        dask_index = r.prepare_inverse_index()
        index, = dask.compute(dask_index)
        print()
        print(index)
        np.testing.assert_almost_equal(index,
                                       np.array([[[np.nan, np.nan, 1.51315789],
                                                  [np.nan, 1.11842105, 2.17105263],
                                                  [0.72368421, 1.77631579, np.nan]],
                                                 [[np.nan, np.nan, 0.51973684],
                                                  [np.nan, 1.92763158, 1.50657895],
                                                  [3.33552632, 2.91447368, np.nan]]]),
                                       decimal=3)

    def test_rectify(self):
        r = Rectifier(self.src_lon, self.src_lat, self.dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        r.prepare_inverse_index()
        r.prepare_rectification(self.payload)
        result = r.compute_rectification()
        print(result)

    # ===== unit tests for single methods =====

    def test_method_dst_pixels_of_src_block(self):
        src_lat = self.src_lat.blocks[1, 0]
        src_lon = self.src_lon.blocks[1, 0]
        block_result = Rectifier.dst_pixels_of_src_block(src_lon, src_lat, dst_grid=self.dst_grid)
        assert np.array_equal(block_result,
                              [[[0, 0]], [[2, 2]]])
        print()
        print(block_result)

    def test_method_bboxes_of_src_block(self):
        r = Rectifier(self.src_lon, self.src_lat, self.dst_grid)
        dask_forward_index = r.prepare_forward_index()
        forward_index, = dask.compute(dask_forward_index)
        result = r.dst_bboxes_of_src_block(forward_index[:,0:3,2:3],
                                           self.dst_grid,
                                           (3, 2),
                                           (4, 3),
                                           (0, 1))
        result = result.reshape(result.shape[0:3])
        assert np.array_equal(result,
                              np.array([[[3, 1], [3, 1]],
                                        [[4, 0], [4, 1]],
                                        [[-1, 3], [-1, 3]],
                                        [[-1, 3], [-1, 4]]]))

    def test_method_bboxes_map_blocks(self):
        r = Rectifier(self.src_lon, self.src_lat, self.dst_grid)
        r.prepare_forward_index()
        bbox_blocks_raw = da.map_blocks(r.dst_bboxes_of_src_block,
                                        r.dask_forward_index,
                                        dst_grid=r.dst_grid,
                                        src_tile_size=r.src_lat.chunksize,
                                        src_size=r.src_lat.shape,
                                        drop_axis=0,
                                        new_axis=[0,1,2],
                                        meta=np.array([], dtype=np.int32),
                                        name=r.name + "_bboxes")
        print()
        print(bbox_blocks_raw)
        bbox_blocks_raw = bbox_blocks_raw.compute()
        assert np.array_equal(bbox_blocks_raw[:,:,:,0,0],
                              np.array([[[0, 0], [0, 3]],
                                        [[0, 0], [1, 4]],
                                        [[2, 3], [3, -1]],
                                        [[4, 3], [4, -1]]]))
        assert np.array_equal(bbox_blocks_raw[:,:,:,0,1],
                              np.array([[[3, 1], [3, 1]],
                                        [[4, 0], [4, 1]],
                                        [[-1, 3], [-1, 3]],
                                        [[-1, 3], [-1, 4]]]))
        assert np.array_equal(bbox_blocks_raw[:,:,:,1,0],
                              np.array([[[3, 3], [0, 3]],
                                        [[4, 4], [2, 4]],
                                        [[-1, -1], [3, -1]],
                                        [[-1, -1], [4, -1]]]))
        assert np.array_equal(bbox_blocks_raw[:,:,:,1,1],
                              np.array([[[3, 3], [3, 3]],
                                        [[4, 4], [4, 4]],
                                        [[-1, -1], [-1, -1]],
                                        [[-1, -1], [-1, -1]]]))
        bbox_blocks = np.stack((np.min(bbox_blocks_raw[0], axis=(2,3)),
                                np.min(bbox_blocks_raw[1], axis=(2,3)),
                                np.max(bbox_blocks_raw[2], axis=(2,3)),
                                np.max(bbox_blocks_raw[3], axis=(2,3))))
        assert np.array_equal(bbox_blocks,
                              np.array([[[0, 0], [0, 1]],
                                        [[0, 0], [1, 1]],
                                        [[2, 3], [3, 3]],
                                        [[4, 3], [4, 4]]]))
        print(bbox_blocks)

    def test_method_triangles_in_dst_pixel_grid(self):
        r = Rectifier(self.src_lon, self.src_lat, self.dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        src_lon_lat = da.stack((self.src_lon, self.src_lat))
        src_subset_lon_lat = src_lon_lat[:, 0:4, 0:2]
        src_subset_lon_lat = src_subset_lon_lat.compute()
        four_points_i, four_points_j = Rectifier.triangles_in_dst_pixel_grid(src_subset_lon_lat, self.dst_grid, (0, 0))
        # result is an array of the extent of the subset 4 x 2 with i, j for each of the four points
        assert four_points_i.shape == (4, 3, 1)
        assert four_points_j.shape == (4, 3, 1)
        ref = 0
        np.testing.assert_almost_equal(
            [[four_points_i[0,ref,0], four_points_j[0,ref,0]],
             [four_points_i[1,ref,0], four_points_j[1,ref,0]],
             [four_points_i[2,ref,0], four_points_j[2,ref,0]],
             [four_points_i[3,ref,0], four_points_j[3,ref,0]]],
            [[1.75, 0.16], [2.5, 0.48], [1.25, 0.96], [2.0, 1.28]], decimal=3)
        ref = 1
        np.testing.assert_almost_equal(
            [[four_points_i[0,ref,0], four_points_j[0,ref,0]],
             [four_points_i[1,ref,0], four_points_j[1,ref,0]],
             [four_points_i[2,ref,0], four_points_j[2,ref,0]],
             [four_points_i[3,ref,0], four_points_j[3,ref,0]]],
            [[1.25, 0.96], [2.0, 1.28], [0.75, 1.76], [1.5, 2.08]], decimal=3)
        ref = 2
        np.testing.assert_almost_equal(
            [[four_points_i[0,ref,0], four_points_j[0,ref,0]],
             [four_points_i[1,ref,0], four_points_j[1,ref,0]],
             [four_points_i[2,ref,0], four_points_j[2,ref,0]],
             [four_points_i[3,ref,0], four_points_j[3,ref,0]]],
            [[0.75, 1.76], [1.5, 2.08], [0.25, 2.56], [1.0, 2.88]], decimal=3)

    def test_inverse_index_of_dst_block_with_src_tiles(self):
        r = Rectifier(self.src_lon, self.src_lat, self.dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        src_lon_lat = da.stack((self.src_lon, self.src_lat))
        # determine src box that covers dst block plus buffer
        tj = 0
        ti = 0
        src_subset_lon_lat1 = src_lon_lat[:, 0:3, 0:2].compute()
        src_subset_lon_lat2 = src_lon_lat[:, 3:4, 0:2].compute()
        index = Rectifier.inverse_index_block((tj, ti),
                                              self.dst_grid,
                                                                    r.src_bboxes[:, tj, ti],
                                              self.src_lat.chunksize,
                                              src_subset_lon_lat1, src_subset_lon_lat2)
        print()
        print(index)
        np.testing.assert_almost_equal(index, [[[np.nan, np.nan], [np.nan, 1.11842105]],
                                               [[np.nan, np.nan], [np.nan, 1.92763158]]], decimal=3)

    def test_inverse_index_of_dst_block_with_src_tiles2(self):
        r = Rectifier(self.src_lon, self.src_lat, self.dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        src_lon_lat = da.stack((self.src_lon, self.src_lat))
        # determine src box that covers dst block plus buffer
        tj = 0
        ti = 1
        src_subset_lon_lat1 = src_lon_lat[:, 0:3, 0:2].compute()
        src_subset_lon_lat2 = src_lon_lat[:, 0:3, 2:3].compute()
        index = Rectifier.inverse_index_block((tj, ti),
                                              self.dst_grid,
                                                                    r.src_bboxes[:, tj, ti],
                                              self.src_lat.chunksize,
                                              src_subset_lon_lat1, src_subset_lon_lat2)
        np.testing.assert_almost_equal(index, [[[1.513], [2.171]],
                                               [[0.519], [1.506]]], decimal=3)

    # ===== tests with OLCI input =====
        
    def test_olci_forward_index(self):
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
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
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
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
        print(r.forward_index)

    def test_olci_inverse_index(self):
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
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
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
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
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
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
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
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
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
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

    # ===== feasibility tests =====
        
    @staticmethod
    def testfunc(a, i):
        return np.asarray(a) + i

    def test_dask_graph(self):
        input = da.from_array(np.zeros((2,3), dtype=int), chunks=(2,1))
        layer = { ("nodename", 0, 0): input.blocks[0,0],
                  ("nodename", 0, 1): input.blocks[0,1],
                  ("nodename", 0, 2): input.blocks[0,2],
                  ("nodename2", 0, 0): (RectificationTest.testfunc, ("nodename", 0, 0), 3),
                  ("nodename2", 0, 1): (RectificationTest.testfunc, ("nodename", 0, 1), 3),
                  ("nodename2", 0, 2): (RectificationTest.testfunc, ("nodename", 0, 2), 3) }
        graph = HighLevelGraph.from_collections("nodename2", layer, dependencies=[])
        a = da.Array(graph, "nodename2", shape=(2,3), chunks=(2,1), meta=np.ndarray([], dtype=int))
        print()
        print(a)
        print(a.compute())

    @staticmethod
    def block_smooth(a):
        stack = np.stack([np.roll(a, (-1,-1)),
                          np.roll(a, (-1,0)),
                          np.roll(a, (-1,1)),
                          np.roll(a, (0,-1)),
                          a,
                          np.roll(a, (0,1)),
                          np.roll(a, (1,-1)),
                          np.roll(a, (1,0)),
                          np.roll(a, (1,1))])
        result = np.nanmean(stack, axis=0)
        return result

    def test_smooth(self):
        n = np.arange(30).reshape((5,6))
        a = da.from_array(n, chunks=(3,4))
        r = da.map_overlap(self.block_smooth, a, depth=(0,1), boundary=None, align_arrays=False)
        print()
        print(n)
        print(r)
        print(r.compute())

    # ===== tests using single chunk full images with numpy instead of dask arrays =====

    def test_olci_numpy_read_inputs(self):
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        longitude, latitude, oa12, oa08 = dask.compute(l1b["longitude"].data, l1b["latitude"].data, l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)

    def test_olci_numpy_forward_index(self):
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        longitude, latitude, oa12, oa08 = dask.compute(l1b["longitude"].data, l1b["latitude"].data, l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)
        #dst_grid = Grid(pyproj.CRS("EPSG:4326"), (0, 0), (0.003, -0.003), (0, 0), (2400, 2400))
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (6667, 4304))
        r = Rectifier(longitude, latitude, dst_grid)
        r.forward_index = Rectifier.dst_pixels_of_src_block(longitude, latitude, r.dst_grid)
        bbox_blocks_raw = Rectifier.dst_bboxes_of_src_block(
            r.forward_index,
            dst_grid=r.dst_grid,
            src_tile_size=r.src_lat.shape,
            src_size=r.src_lat.shape,
            block_id=(0,0))
        # we compute min and max of all 4 bounds though we need only two of each,
        # dask seems not to delay a []
        r.src_bboxes = np.vstack((np.nanmin(bbox_blocks_raw, axis=(3,4)),
                                  np.nanmax(bbox_blocks_raw, axis=(3,4))))

    def test_olci_numpy_inverse_index(self):
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        longitude, latitude, oa12, oa08 = dask.compute(l1b["longitude"].data, l1b["latitude"].data, l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)
        #dst_grid = Grid(pyproj.CRS("EPSG:4326"), (0, 0), (0.003, -0.003), (0, 0), (2400, 2400))
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (6667, 4304))
        r = Rectifier(longitude, latitude, dst_grid)
        r.forward_index = Rectifier.dst_pixels_of_src_block(longitude, latitude, r.dst_grid)
        bbox_blocks_raw = Rectifier.dst_bboxes_of_src_block(
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
        r.inverse_index = Rectifier.inverse_index_of_dst_block_with_src_subset(
                                 lon_lat[:,
                                     r.src_bboxes[1, 0, 0]:r.src_bboxes[3, 0, 0],
                                     r.src_bboxes[0, 0, 0]:r.src_bboxes[2, 0, 0]],
                                 (r.src_bboxes[0, 0, 0] + 0.5, r.src_bboxes[1, 0, 0] + 0.5),
                                 r.dst_grid,
                                 (0, 0))

    def test_olci_numpy_rectify_data(self):
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        longitude, latitude, oa12, oa08 = dask.compute(l1b["longitude"].data, l1b["latitude"].data, l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)
        #dst_grid = Grid(pyproj.CRS("EPSG:4326"), (0, 0), (0.003, -0.003), (0, 0), (2400, 2400))
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (6667, 4304))
        r = Rectifier(longitude, latitude, dst_grid)
        r.forward_index = Rectifier.dst_pixels_of_src_block(longitude, latitude, r.dst_grid)
        bbox_blocks_raw = Rectifier.dst_bboxes_of_src_block(
                                        r.forward_index,
                                        dst_grid=r.dst_grid,
                                        src_tile_size=r.src_lat.shape,
                                        src_size=r.src_lat.shape,
            block_id=(0,0))
        # we compute min and max of all 4 bounds though we need only two of each,
        # dask seems not to delay a []
        r.src_bboxes = np.vstack((np.nanmin(bbox_blocks_raw, axis=(3,4)),
                                          np.nanmax(bbox_blocks_raw, axis=(3,4))))
        r.inverse_index = Rectifier.inverse_index_of_dst_block_with_src_subset(
                                 np.stack((longitude, latitude))[:,
                                     r.src_bboxes[1, 0,0]:r.src_bboxes[3, 0,0],
                                     r.src_bboxes[0, 0,0]:r.src_bboxes[2, 0,0]],
                                 (r.src_bboxes[0, 0, 0] + 0.5, r.src_bboxes[1, 0, 0] + 0.5),
                                 r.dst_grid,
                                 (0, 0))
        intblock = np.around(r.inverse_index).astype(int)
        intblock[np.isnan(r.inverse_index)] = -1
        block_i = intblock[0]
        block_j = intblock[1]
        r.rectified = Rectifier.rectify_block(
            block_i, block_j, 0, 0, oa12.shape, 1, 2, np.array([0]), oa12, oa08)

    def test_olci_numpy_rectify_write(self):
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        longitude, latitude, oa12, oa08 = dask.compute(l1b["longitude"].data, l1b["latitude"].data, l1b['Oa12_radiance'].data, l1b['Oa08_radiance'].data)
        #dst_grid = Grid(pyproj.CRS("EPSG:4326"), (0, 0), (0.003, -0.003), (0, 0), (2400, 2400))
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (6667, 4304))
        r = Rectifier(longitude, latitude, dst_grid)
        r.forward_index = Rectifier.dst_pixels_of_src_block(longitude, latitude, r.dst_grid)
        bbox_blocks_raw = Rectifier.dst_bboxes_of_src_block(
                                        r.forward_index,
                                        dst_grid=r.dst_grid,
                                        src_tile_size=r.src_lat.shape,
                                        src_size=r.src_lat.shape,
            block_id=(0,0))
        # we compute min and max of all 4 bounds though we need only two of each,
        # dask seems not to delay a []
        r.src_bboxes = np.vstack((np.nanmin(bbox_blocks_raw, axis=(3,4)),
                                          np.nanmax(bbox_blocks_raw, axis=(3,4))))
        r.inverse_index = Rectifier.inverse_index_of_dst_block_with_src_subset(
                                 np.stack((longitude, latitude))[:,
                                     r.src_bboxes[1, 0,0]:r.src_bboxes[3, 0,0],
                                     r.src_bboxes[0, 0,0]:r.src_bboxes[2, 0,0]],
                                 (r.src_bboxes[0, 0, 0] + 0.5, r.src_bboxes[1, 0, 0] + 0.5),
                                 r.dst_grid,
                                 (0, 0))
        intblock = np.around(r.inverse_index).astype(int)
        intblock[np.isnan(r.inverse_index)] = -1
        block_i = intblock[0]
        block_j = intblock[1]
        r.rectified = Rectifier.rectify_block(
            block_i, block_j, 0, 0, oa12.shape, 1, 2, np.array([0]), oa12, oa08)

        ds = xr.Dataset({"oa08": (["lat", "lon"], r.rectified[1]),
                         "oa12": (["lat", "lon"], r.rectified[0])},
                        coords={
                            "lat": (["lat"], r.dst_grid.y_axis()),
                            "lon": (["lon"], r.dst_grid.x_axis())
                        })
        ds.to_netcdf("/windows/tmp/eopf/repojected3.nc")
