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

import unittest
import pyproj
import numpy as np
import dask.array as da
from xcube_resampling.grid import Grid
from xcube_resampling.reprojector import Reprojector

# noinspection PyTypeChecker
class ResamplingTest(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        #self._client = Client(LocalCluster(n_workers=8, processes=True))
        #self._client = Client(n_workers=1, threads_per_worker=6)
        pass

    @classmethod
    def tearDownClass(self) -> None:
        #self._client.close()
        pass

    def test_corners(self):
        src_crs = pyproj.CRS(4326)
        target_crs = pyproj.CRS(3035)
        trafo = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
        x = [10.0,11.0]
        y = [53.6,53.0]
        result = trafo.transform(x, y)
        assert int(result[0][0]) == 4321000
        assert int(result[1][0]) == 3388043
        print(result)

    def test_target_grid(self):
        src_crs = pyproj.CRS(4326)
        target_crs = pyproj.CRS(3035)
        src_i_1d = np.array([10.1,10.3,10.5,10.7,10.9,11.1])
        src_j_1d = np.array([53.5,53.3,53.1])
        src_grid = Grid.from_coords(src_crs, (src_i_1d, src_j_1d))
        r = Reprojector(src_grid)
        r.create_transformer(target_crs)
        r.create_covering_target_grid((10000.0, -10000.0), (20000.0, -20000.0))
        assert r.target_grid.x_min == 4320000
        assert r.target_grid.x_res == 10000
        assert r.target_grid.width == 8
        assert r.target_grid.y_min == 3390000
        assert r.target_grid.y_res == -10000
        assert r.target_grid.height == 7
        print(r.target_grid)

    def test_source_pixels_of(self):
        target_x = np.array([[4325000, 4335000]])
        target_y = np.array([[3385000], [3375000]])
        src_crs = pyproj.CRS(4326)
        target_crs = pyproj.CRS(3035)
        trafo = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
        src_i_1d = np.array([10.1, 10.3, 10.5, 10.7, 10.9, 11.1])
        src_j_1d = np.array([53.5, 53.3, 53.1])
        src_grid = Grid.from_coords(src_crs, (src_i_1d, src_j_1d), (2, 2))
        index = Reprojector.source_pixels_of(target_x,
                                             target_y,
                                             trafo=trafo,
                                             src_grid=src_grid)
        assert index.shape[0] == 2  # two layers j and i
        assert index[0][0][0] == 0
        assert index[0][0][1] == 1
        assert index[0][1][0] == 0
        assert index[0][1][1] == 1
        assert index[1][0][0] == 0
        assert index[1][0][1] == 0
        assert index[1][1][0] == 0
        assert index[1][1][1] == 0
        print(index)

    def test_inverse_index(self):
        src_crs = pyproj.CRS(4326)
        target_crs = pyproj.CRS(3035)
        src_i_1d = np.array([10.1, 10.3, 10.5, 10.7, 10.9, 11.1])
        src_j_1d = np.array([53.5, 53.3, 53.1])
        src_grid = Grid.from_coords(src_crs, (src_i_1d, src_j_1d), (2, 2))
        r = Reprojector(src_grid)
        r.create_transformer(target_crs)
        r.create_covering_target_grid((10000.0, -10000.0), (20000.0, -20000.0))
        print(r.target_grid)
        result = r.create_inverse_pixel_index().compute()
        print(result)
        assert result.shape[0] == 2
        assert result[1][0][0] == 0  # j
        assert result[0][0][0] == 0  # i
        assert result[1][-1][-1] == 2
        assert result[0][-1][-1] == 5
        assert result[0][0][0] == 0
        assert result[0][0][1] == 1
        assert result[0][0][2] == 1
        assert result[0][0][3] == 2
        assert result[0][0][4] == 3
        assert result[0][0][5] == 4
        assert result[0][0][6] == 4
        assert result[0][0][7] == 5

    def test_reproject_tiles(self):
        measurements = da.from_array([[[[1.0,2.0],[7.0,8.0]]]], chunks=(1,1,2,2))
        #measurements = np.array([[[[1.0,2.0],[7.0,8.0]]]])
        tiles = np.array([0])
        num_tiles_i = 2
        src_grid = Grid(pyproj.CRS(4326), (10.0, 53.6), (0.2, -0.2), (6, 3), (2, 2))
        result = Reprojector.reproject_tiles_to_block(np.array([[0,1],[0,1]]),
                                                      np.array([[0,0],[0,0]]),
                                                      src_grid,
                                                      num_tiles_i,
                                                      tiles,
                                                      *measurements)
        assert result[0][0][0] == 1,0
        assert result[0][0][1] == 2,0
        assert result[0][1][0] == 1.0
        assert result[0][1][1] == 2.0
        print(result)

    def test_reproject_with_index(self):
        src_crs = pyproj.CRS(4326)
        target_crs = pyproj.CRS(3035)
        src_i_1d = np.array([10.1, 10.3, 10.5, 10.7, 10.9, 11.1])
        src_j_1d = np.array([53.5, 53.3, 53.1])
        src_grid = Grid.from_coords(src_crs, (src_i_1d, src_j_1d), (2, 2))
        r = Reprojector(src_grid)
        r.create_transformer(target_crs)
        r.create_covering_target_grid((10000.0, -10000.0), (20000.0, -20000.0))
        r.create_inverse_pixel_index()
        measurements = [da.from_array([[1.0,2.0,3.0,4.0,5.0,6.0],
                                       [7.0,8.0,9.0,10.0,11.0,12.0],
                                       [13.0,14.0,15.0,16.0,17.0,18.0]], chunks=(2,2))]
        repro = r.reproject(*measurements)
        result = repro.compute()
        print(result)

    def test_reproject(self):
        measurements = [da.from_array([[1.0,2.0,3.0,4.0,5.0,6.0],
                                       [7.0,8.0,9.0,10.0,11.0,12.0],
                                       [13.0,14.0,15.0,16.0,17.0,18.0]], chunks=(2,2))]
        repro = Reprojector.reproject_to_covering_grid(
            np.array([53.5, 53.3, 53.1]),
            np.array([10.1, 10.3, 10.5, 10.7, 10.9, 11.1]),
            pyproj.CRS(4326),
            (2, 2),
            pyproj.CRS(3035),
            (10000.0, -10000.0),
            (20000.0, -20000.0),
            *measurements)
        result = repro.compute()
        print(result)



    def test_eye(self):
        """simple example of high level graphs and array construction"""
        from dask.highlevelgraph import HighLevelGraph
        n=4
        blocksize=2
        chunks = ((blocksize,) * (n // blocksize),
                  (blocksize,) * (n // blocksize))
        name = 'eye'  # unique identifier
        layer = {(name, i, j): (np.eye, blocksize)
                           if i == j else
                           (np.zeros, (blocksize, blocksize))
             for i in range(n // blocksize)
             for j in range(n // blocksize)}
        clayer = {k:layer[k][0](*layer[k][1:]) for k in layer}
        dsk = HighLevelGraph.from_collections(name, layer, dependencies=())
        dtype = np.eye(0).dtype  # take dtype default from numpy
        result = da.Array(dsk, name, chunks, dtype)
        print(result.compute())
