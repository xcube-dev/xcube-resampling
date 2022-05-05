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
from xcube_resampling.grid import Grid
from xcube_resampling.rectifier2 import Rectifier
import xarray as xr

# noinspection PyTypeChecker
class RectificationTest(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        #self._client = Client(LocalCluster(n_workers=8, processes=True))
        #self._client = Client(n_workers=1, threads_per_worker=6)
        self.src_lat = da.from_array([[53.98, 53.94, 53.90],
                                 [53.88, 53.84, 53.80],
                                 [53.78, 53.74, 53.70],
                                 [53.68, 53.64, 53.60]], chunks=((3,1),(2,1)))
        self.src_lon = da.from_array([[10.35, 10.50, 10.65],
                                 [10.25, 10.40, 10.55],
                                 [10.15, 10.30, 10.45],
                                 [10.05, 10.20, 10.35]], chunks=((3,1),(2,1)))
        self.dst_grid = Grid(pyproj.CRS(4326), (10.0, 54.0), (0.2, -0.125), (3, 3), (2, 2))

    @classmethod
    def tearDownClass(self) -> None:
        #self._client.close()
        pass

    def test_dst_pixels_of_src_block(self):
        src_lat = self.src_lat.blocks[1, 0]
        src_lon = self.src_lon.blocks[1, 0]
        trafo = pyproj.Transformer.from_crs(pyproj.CRS(4326),
                                            self.dst_grid.crs,
                                            always_xy=True)
        block_result = Rectifier.block_dst_pixels_of_src_block(src_lon, src_lat, trafo=trafo, dst_grid=self.dst_grid)
        assert np.array_equal(block_result,
                              [[[0, 0]], [[2, 2]]])
        print(block_result)

    def test_forward_pixel_index(self):
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

    def test_bbox_block2(self):
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

    def test_bbox_blocks_raw(self):
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

    def test_triangles_in_dst_pixel_grid(self):
        r = Rectifier(self.src_lon, self.src_lat, self.dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        src_lon_lat = da.stack((self.src_lon, self.src_lat))
        src_subset_lon_lat = src_lon_lat[:, 0:4, 0:2]
        src_subset_lon_lat = src_subset_lon_lat.compute()
        four_points_i, four_points_j = Rectifier.triangles_in_dst_pixel_grid(src_subset_lon_lat, self.dst_grid, r.trafo, (0, 0))
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

    def test_inverse_index_of_dst_block_with_src_subset(self):
        r = Rectifier(self.src_lon, self.src_lat, self.dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        src_lon_lat = da.stack((self.src_lon, self.src_lat))
        # determine src box that covers dst block plus buffer
        tj = 0
        ti = 0
        src_subset_lon_lat = src_lon_lat[:,
                                         r.src_bboxes[1, tj, ti]:r.src_bboxes[3, tj, ti],
                                         r.src_bboxes[0, tj, ti]:r.src_bboxes[2, tj, ti]]
        src_subset_lon_lat = src_subset_lon_lat.compute()
        index = Rectifier.inverse_index_of_dst_block_with_src_subset(src_subset_lon_lat,
                                                                     (r.src_bboxes[0, tj, ti] + 0.5, r.src_bboxes[1, tj, ti] + 0.5),
                                                                     self.dst_grid,
                                                                     r.trafo,
                                                                     (tj, ti))
        np.testing.assert_almost_equal(index, [[[np.nan, np.nan], [np.nan, 1.11842105]],
                                               [[np.nan, np.nan], [np.nan, 1.92763158]]], decimal=3)

    def test_inverse_index_of_dst_block_with_src_subset2(self):
        r = Rectifier(self.src_lon, self.src_lat, self.dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        r.compute_forward_index()
        src_lon_lat = da.stack((self.src_lon, self.src_lat))
        # determine src box that covers dst block plus buffer
        tj = 0
        ti = 1
        src_subset_lon_lat = src_lon_lat[:,
                                         r.src_bboxes[1, tj, ti]:r.src_bboxes[3, tj, ti],
                                         r.src_bboxes[0, tj, ti]:r.src_bboxes[2, tj, ti]]
        src_subset_lon_lat = src_subset_lon_lat.compute()
        index = Rectifier.inverse_index_of_dst_block_with_src_subset(src_subset_lon_lat,
                                                                     (r.src_bboxes[0, tj, ti] + 0.5, r.src_bboxes[1, tj, ti] + 0.5),
                                                                     self.dst_grid,
                                                                     r.trafo,
                                                                     (tj, ti))
        np.testing.assert_almost_equal(index, [[[1.513], [2.171]],
                                               [[0.519], [1.506]]], decimal=3)

    def test_olci_forward_index(self):
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        # dst_grid = Grid(pyproj.CRS("EPSG:4326"), (0, 0), (0.003, -0.003), (0, 0), (2400, 2400))
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (2400, 2400))
        r = Rectifier(l1b['longitude'].data, l1b['latitude'].data, dst_grid)
        r.prepare_forward_index()
        r.prepare_src_bboxes()
        index,_ = r.compute_forward_index()
        print()
        print(r.forward_index)
        print(index)

    def test_olci_covering_grid_old(self):
        path = '/windows/tmp/eopf/S3A_OL_1_EFR____20210801T102426_20210801T102726_20210802T141313_0179_074_336_2160_LN1_O_NT_002.SEN3'
        l1b = xr.open_mfdataset([path + '/geo_coordinates.nc'] +
                                [(path + '/Oa{:02d}_radiance.nc'.format(x)) for x in range(1, 22)],
                                engine="netcdf4",
                                chunks=2048)
        dst_grid = Grid(pyproj.CRS("EPSG:4326"), (-10.863, 52.449), (0.003, -0.003), (6667, 4304), (2400, 2400))
        r = Rectifier(l1b['longitude'].data, l1b['latitude'].data, dst_grid)
        r.create_forward_pixel_index()
        print()
        print(r.forward_index)
        print(r.forward_index.compute())
        dst_grid = r.determine_covering_dst_grid()
        print()
        print(dst_grid)
        print(r.forward_index)
        print(r.forward_index.compute())

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
        for dst_pos in [(1000, 1000), (5000, 2600)]:
            src_pos = index[:,dst_pos[1],dst_pos[0]]
            src_int = (math.floor(src_pos[0]-0.5), math.floor(src_pos[1]-0.5))
            dst_lon = dst_grid.x_min + (dst_pos[0] + 0.5) * dst_grid.x_res
            dst_lat = dst_grid.y_min + (dst_pos[1] + 0.5) * dst_grid.y_res
            print("src frac idx", src_pos)
            print("upper left  ", src_lon[src_int[1], src_int[0]], src_lat[src_int[1], src_int[0]])
            print("dst point   ", dst_lon, dst_lat)
            print("lower right ", src_lon[src_int[1]+1, src_int[0]+1], src_lat[src_int[1]+1, src_int[0]+1])
            assert dst_lon >= src_lon[src_int[1], src_int[0]] and dst_lon <= src_lon[src_int[1]+1, src_int[0]+1]
            assert dst_lat <= src_lat[src_int[1], src_int[0]] and dst_lat >= src_lat[src_int[1]+1, src_int[0]+1]

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
