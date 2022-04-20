# -*- coding: utf-8 -*-

# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__author__ = "Martin Böttcher, Brockmann Consult GmbH"
__copyright__ = "Copyright (c) 2022 by the xcube development team and contributors"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "info@brockmann-consult.de"
__status__ = "Development"

# changes in 1.1
# ...

import math
import pyproj
import numpy as np
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
from .grid import Grid
from typing import Tuple, Union, Optional

class Rectifier:
    """
    Reprojects measurement layers from a source image with pixel geo-coding to a destination grid
    with CRS and a similar resolution, with nearest neighbour resampling as default method.
    Source image must be given, dst grid may be created such that it covers the source image area.
    Pre-calculates inverse pixel index that contains the source pixel coordinates for each destination pixel.
    Generates and returns a dask array with a dask graph for the delayed computation. Just the inverse 
    index is pre-calculated as it determines how the graph is constructed. Pre-calculation is done by dask,
    too.

    If source grid and destination grid are given the calling sequence is::

        src_lon = da.Array([[...],[...],...], chunksize=(...,...))
        src_lat = da.Array([[...],[...],...], chunksize=(...,...))
        dst_grid = Grid(pyproj.CRS(3035), (4320000, 3390000), (10000, -10000), (8, 7), (2, 2))
        rectifier = Rectifier(src_lon, src_lat, dst_grid)
        rectifier.create_forward_pixel_index()
        rectifier.create_inverse_pixel_index()
        rectified_measurements = rectifier.rectify_nearest_neighbour(*measurements)

    If only the source image and the CRS and resolution of the dst grid is given the calling sequence is::

        src_lon = da.Array([[...],[...],...], chunksize=(...,...))
        src_lat = da.Array([[...],[...],...], chunksize=(...,...))
        preliminary_dst_grid = Grid(pyproj.CRS(3035), (0, 0), (10000, -10000), (0, 0), (2, 2))
        rectifier = Rectifier(src_lon, src_lat, preliminary_dst_grid)
        rectifier.create_forward_pixel_index()
        rectifier.determine_covering_dst_grid()
        rectifier.create_inverse_pixel_index()
        rectified_measurements = rectifier.rectify_nearest_neighbour(*measurements)
        
    Variable naming convention:
    - src lat, lon are the geo-positions of the source image, provided
    - src row, col are pixel positions of the source image
    - dst x, y are CRS coordinates of the destination image
    - dst i, j are pixel positions of the destination image
    """
    def __init__(self,
                 src_lon: da.Array,
                 src_lat: da.Array,
                 dst_grid: Grid = None,
                 name: str = "rectify"):
        """
        Constructor
        :param src_lon: source x as 2-D array in geographic coordinates
        :param src_lon: source y as 2-D array in geographic coordinates
        :param dst_grid: dst grid, optional, can be determined with create_covering_dst_grid
        :param name: unique name of dask graph node, default "rectify",
                     must be set if several Rectifiers may be in the graph
        """
        self.src_lon = src_lon
        self.src_lat = src_lat
        self.dst_grid = dst_grid
        self.name = name

    @staticmethod
    def block_dst_pixels_of_src_block(src_lon:np.ndarray,
                                      src_lat:np.ndarray,
                                      trafo:pyproj.Transformer=None,
                                      dst_grid:Grid=None) -> np.ndarray:
        """
        Calculates for one source block the integer dst image coordinates j and i
        for each source pixel. map_blocks function used in create_forward_pixel_index,
        must be called with new_axis=0 .
        :param src_lon: one block of src x coordinates at src pixel centres
        :param src_lat: one block of src y coordinates at src pixel centres
        :param trafo: transformation from src to dst
        :param dst_grid: defines a grid with the origin at the upper left corner of the start pixel
        :return: stack of i and j arrays for the block
        """
        # transform into source coordinates
        dst_x, dst_y = trafo.transform(src_lon, src_lat)
        # convert to pixel positions
        # "floor" in fact rounds because transformed pixel centers are related to upper left corner
        dst_i = np.floor((dst_x - dst_grid.x_min) / dst_grid.x_res).astype(dtype=int)
        dst_j = np.floor((dst_y - dst_grid.y_min) / dst_grid.y_res).astype(dtype=int)
        # we stack to return a single np array for the block
        result = np.stack((dst_i, dst_j))
        return result

    def create_forward_pixel_index(self):
        """
        Creates dst pixel index (integer) of each source pixel. 
        """
        trafo = pyproj.Transformer.from_crs(pyproj.CRS(4326),
                                            self.dst_grid.crs,
                                            always_xy=True)
        self.forward_index = da.map_blocks(Rectifier.block_dst_pixels_of_src_block,
                                           self.src_lon,
                                           self.src_lat,
                                           trafo=trafo,
                                           dst_grid=self.dst_grid,
                                           new_axis=0,
                                           dtype=np.int32,
                                           #meta=np.array((), dtype=np.int32),
                                           name=self.name + "_forward")
        return self.forward_index

    @staticmethod
    def dst_bbox_of_src_block(forward_index_block: np.ndarray):
        """
        Determines the bounding box of a source image block in dst pixel coordinates.
        map_blocks function called by determine_covering_dst_grid,
        must be called with new_axis=0 .
        :param forward_index_block: dst pixel position (integer) of each source pixel of the source block,
                                    array of shape (2, height, width), sequence i, j
        :return: numpy array of shape (4,1,1) with imin, jmin, imax, jmax
        """
        i_min = np.nanmin(forward_index_block[0])
        j_min = np.nanmin(forward_index_block[1])
        i_max = np.nanmax(forward_index_block[0])
        j_max = np.nanmax(forward_index_block[1])
        return np.array([i_min, j_min, i_max, j_max]).reshape((4,1,1))

    def determine_covering_dst_grid(self) -> Grid:
        """
        Determines the dst grid extent covering a source image in a different projection.
        Uses forward pixel index to source-blockwise determine bboxes in dst grid,
        merge them into one bbox, align with dst grid origin,
        update dst grid origin and size, shift forward pixel index
        :return: dst grid
        """
        if not self.forward_index:
            raise ValueError("missing forward index. Call create_forward_pixel_index() first.")
        bboxes = da.map_blocks(self.dst_bbox_of_src_block, 
                               self.forward_index, 
                               new_axis=0, 
                               name=self.name + "_bbox").compute()
        i_min = np.min(bboxes[0])
        j_min = np.min(bboxes[1])
        i_max = np.max(bboxes[2])
        j_max = np.max(bboxes[3])
        # adjust origin and size of dst grid
        self.dst_grid.x_min = i_min * self.dst_grid.x_res
        self.dst_grid.y_min = j_min * self.dst_grid.y_res
        self.dst_grid.width = i_max - i_min + 1
        self.dst_grid.height = j_max - j_min + 1
        # adjust pixel coordinates of forward mapping
        self.forward_index -= np.array([[i_min], [j_min]])
        return self.dst_grid

    @staticmethod
    def dst_bboxes_of_src_block(forward_index_block: np.ndarray,
                                dst_grid: Grid = None,
                                src_tile_size: Tuple[int, int] = None,
                                src_size: Tuple[int, int] = None,
                                block_id: Tuple[int, int] = None):
        """
        Determines for one source block the source bounding box for each dst block
        if source block and dst block intersect.
        map_blocks function called by creata_inverse_pixel_index
        :param forward_index_block: source block of pixel coordinates of source pixels in dst grid
                                    shape (2, src_tile_height, src_tile_width), sequence i, j
        :param dst_grid: dst number of blocks and their sizes
        :param src_tile_size: source tile size, to determine source offset of the block, sequence lat, lon
        :param block_id source block row and block column
        :return: numpy array of shape (4, num_dst_tiles_y, num_dst_tiles_x, 1, 1) with imin, jmin, imax, jmax
        """
        # offset of source block this call is done for
        src_block_offset_col = block_id[-1] * src_tile_size[1]
        src_block_offset_row = block_id[-2] * src_tile_size[0]
        # identity array of src rows and cols of the block, starting from 0
        num_rows = forward_index_block.shape[-2]
        num_cols = forward_index_block.shape[-1]
        src_local_cols = np.tile(np.arange(num_cols), (num_rows, 1))
        src_local_rows = np.tile(np.arange(num_rows), (num_cols, 1)).transpose()
        # vectors of all dst block borders in pixel coordinates
        dst_left = np.arange(0, dst_grid.width, dst_grid.tile_width)
        dst_right = np.hstack([dst_left[1:], np.array([dst_grid.width])])
        dst_down = np.arange(0, dst_grid.height, dst_grid.tile_height)
        dst_up = np.hstack([dst_down[1:], np.array([dst_grid.height])])
        # number of dst blocks
        dst_num_blocks_i = math.ceil(dst_grid.width / dst_grid.tile_width)
        dst_num_blocks_j = math.ceil(dst_grid.height / dst_grid.tile_height)
        result_boxes = np.empty((4, dst_num_blocks_j, dst_num_blocks_i, 1, 1), dtype=np.int32)
        result_boxes[:] = -1
        # TODO replace the two loops by dimensions of the inside arrays and dst arrays
        for dst_block_j in range(dst_num_blocks_j):
            # filter condition that forward index is between dst block borders
            # array of dimension of the source block
            inside_block_j = (forward_index_block[1] >= dst_down[dst_block_j]) & \
                             (forward_index_block[1] < dst_up[dst_block_j])
            for dst_block_i in range(dst_num_blocks_i):
                # filter condition that forward index is between dst block borders
                # array of dimension of the source block
                inside_block_i = (forward_index_block[0] >= dst_left[dst_block_i]) & \
                                 (forward_index_block[0] < dst_right[dst_block_i])
                # mask src row and col by those inside dst block
                src_block_rows = src_local_rows[inside_block_j & inside_block_i] + src_block_offset_row
                src_block_cols = src_local_cols[inside_block_j & inside_block_i] + src_block_offset_col
                # determine min and max of src block row and col inside dst block
                result_boxes[0, dst_block_j, dst_block_i, 0, 0] = np.min(src_block_cols) if len(src_block_cols) > 0 else src_size[1]
                result_boxes[1, dst_block_j, dst_block_i, 0, 0] = np.min(src_block_rows) if len(src_block_rows) > 0 else src_size[0]
                result_boxes[2, dst_block_j, dst_block_i, 0, 0] = np.max(src_block_cols) + 1 if len(src_block_cols) > 0 else -1
                result_boxes[3, dst_block_j, dst_block_i, 0, 0] = np.max(src_block_rows) + 1 if len(src_block_rows) > 0 else -1
        return result_boxes

    @staticmethod
    def triangles_in_dst_pixel_grid(src_subset_lon_lat: np.ndarray,
                                    dst_grid: Grid,
                                    block_id: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stacks the four points P0, P1, P2, P3 that form two triangles P0-P1-P2 and P3-P1-P2. 
        Coordinates of P are fractional pixel coordinates in the dst grid relative to the dst block.
        :param src_subset_lon_lat: lon and lat coordinates of source pixels of a subset 
                                   covering the dst block
        :param dst_grid: destination grid
        :param block_id destination block coordinates
        :return: pair i, j of stacks of the four points with the extent of the source subset
        """
        # determine fractional pixel positions of subset in dst grid
        # TODO apply trafo from 4326 to dst CRS
        # subset_i and _j have the extent of the source subset.
        # subset_i and _j contain fractional pixel coordinates in the dst grid's coordinate system
        # subset_i and _j are fractions related to the upper left corner of the dst pixel
        # i.e. in case of 0.5 the source pixel matches a dst position
        subset_i = (src_subset_lon_lat[0] - dst_grid.x_min - block_id[0] * dst_grid.tile_width) / dst_grid.x_res
        subset_j = (src_subset_lon_lat[1] - dst_grid.y_min - block_id[1] * dst_grid.tile_height) / dst_grid.y_res
        # extend subset by one column and row, duplicate last column and row
        # stacked subset stack the fractional pixel coordinates of the four points P0 .. P3
        # stacked_subset_i and _j are in the extent of the source subset.
        # stacked_subset_i and _j contain shifted fractional pixel coordinates in the dst grid.
        extended_subset_i = np.hstack([subset_i, subset_i[:,-1:]])
        extended_subset_i = np.vstack([extended_subset_i, extended_subset_i[-1:,:]])
        extended_subset_j = np.hstack([subset_j, subset_j[:,-1:]])
        extended_subset_j = np.vstack([extended_subset_j, extended_subset_j[-1:,:]])
        stacked_subset_i = np.stack((extended_subset_i[:-1,:-1],
                                    extended_subset_i[:-1,1:],
                                    extended_subset_i[1:,:-1],
                                    extended_subset_i[1:,1:]))
        stacked_subset_j = np.stack((extended_subset_j[:-1,:-1],
                                    extended_subset_j[:-1,1:],
                                    extended_subset_j[1:,:-1],
                                    extended_subset_j[1:,1:]))
        return (stacked_subset_i, stacked_subset_j) 
        
    @staticmethod
    def bboxes_of_triangles(four_points_i: np.ndarray, four_points_j: np.ndarray, dst_grid: Grid):
        is_inside_dst_block = (four_points_i[0] >= 0.0) & (four_points_j[0] >= 0.0) & (four_points_i[0] <= dst_grid.tile_width) & (four_points_j[0] <= dst_grid.tile_height)
        bboxes_min_i = np.floor(np.min(four_points_i, axis=0)).astype(int)
        bboxes_min_j = np.floor(np.min(four_points_j, axis=0)).astype(int)
        bboxes_max_i = np.floor(np.max(four_points_i, axis=0)).astype(int)
        bboxes_max_j = np.floor(np.max(four_points_j, axis=0)).astype(int)
        bboxes_width = bboxes_max_i - bboxes_min_i + 1
        bboxes_height = bboxes_max_j - bboxes_min_j + 1
        # determine maximum bbox size of all pairs of triangles
        bboxes_max_width = np.max(bboxes_width[is_inside_dst_block]).astype(int)
        bboxes_max_height = np.max(bboxes_height[is_inside_dst_block]).astype(int)
        return (bboxes_min_i, bboxes_min_j, bboxes_max_width, bboxes_max_height)

    @staticmethod
    def inverse_index_of_dst_block_with_src_subset(src_subset_lon_lat: np.ndarray,
                                                   src_offset: Tuple[int, int],
                                                   dst_grid: Grid,
                                                   block_id: Tuple[int, int]) -> np.array:
        """
        Determines inverse index col, row of fractional source image pixel coordinates for 
        each dst pixel of a dst block. Uses painter algorithm to transform src triangles to 
        dst grid and mark dst pixels inside triangles.
        High level graph function used in create_inverse_pixel_index.
        :param src_subset_lon_lat: lon and lat coordinates of source pixels of a subset 
                                   covering the dst block
        :param src_offset: offset of src_subset_lon_lat in src pixel coordinates to be 
                           added to inverse index
        :param dst_grid: destination grid
        :param block_id destination block coordinates
        :return: inverse pixel index with fractional source pixel index for each dst block pixel
        """
        # generate four points with two triangles for the src subset in dst pixel fractional coordinates
        four_points_i, four_points_j = Rectifier.triangles_in_dst_pixel_grid(src_subset_lon_lat, dst_grid, block_id)
        # create small bboxes for the four points
        bboxes_min_i, bboxes_min_j, bboxes_max_width, bboxes_max_height = Rectifier.bboxes_of_triangles(four_points_i, four_points_j, dst_grid)
        # create source subset identity vector for rows and columns
        src_id_col = np.arange(src_subset_lon_lat.shape[1])
        src_id_row = np.arange(src_subset_lon_lat.shape[0])
        # result has the extent of the dst grid, initialised with nan
        result_col = np.empty((dst_grid.tile_height, dst_grid.tile_width))
        result_row = np.empty((dst_grid.tile_height, dst_grid.tile_width))
        result_col[:,:] = np.nan
        result_row[:,:] = np.nan
        # det_a and _b have the extent of the source subset.
        # _fdet = (px0 - px1) * (py0 - py2) - (px0 - px2) * (py0 - py1)
        det_a = (four_points_i[0] - four_points_i[1]) * (four_points_j[0] - four_points_j[2]) \
              - (four_points_i[0] - four_points_i[2]) * (four_points_j[0] - four_points_j[1])
        det_b = (four_points_i[3] - four_points_i[2]) * (four_points_j[3] - four_points_j[1]) \
              - (four_points_i[3] - four_points_i[1]) * (four_points_j[3] - four_points_j[2])
        # numerical accuracy parameters
        uv_delta = 0.001
        u_min = v_min = -uv_delta
        uv_max = 1.0 + 2 * uv_delta
        # loops over bboxes max size
        for j_offset in range(bboxes_max_height):
            dst_j = bboxes_min_j + j_offset
            for i_offset in range(bboxes_max_width):
                dst_i = bboxes_min_i + i_offset
                # dst_j and dst_i have the extent of the source subset
                # dst_j and dst_i contain integer pixel coordinates of the considered point in the dst grid
                # u and v have the extent of the source subset
                # _fu = (px0 - px) * (py0 - py2) - (py0 - py) * (px0 - px2)
                ua = ((four_points_i[0] - dst_i) * (four_points_j[0] - four_points_j[2]) - \
                      (four_points_j[0] - dst_j) * (four_points_i[0] - four_points_i[2])) / det_a \
                # _fv = (py0 - py) * (px0 - px1) - (px0 - px) * (py0 - py1)
                va = ((four_points_j[0] - dst_j) * (four_points_i[0] - four_points_i[1]) - \
                      (four_points_i[0] - dst_i) * (four_points_j[0] - four_points_j[1])) / det_a
                is_inside_triangle_a = (ua >= u_min) & (va >= v_min) & (ua + va <= uv_max) & \
                                       (dst_i >= 0) & (dst_i < dst_grid.tile_width) & \
                                       (dst_j >= 0) & (dst_j < dst_grid.tile_height)
                # insert pixel with this offset into result if inside
                result_col[dst_j[is_inside_triangle_a], dst_i[is_inside_triangle_a]] = \
                    src_id_col[is_inside_triangle_a] + src_offset[0] + ua[is_inside_triangle_a]
                result_row[dst_j[is_inside_triangle_a], dst_i[is_inside_triangle_a]] = \
                    src_id_row[is_inside_triangle_a] + src_offset[1] + va[is_inside_triangle_a]
                # do the same for triangle b ...
                # _fu = (px0 - px) * (py0 - py2) - (py0 - py) * (px0 - px2)
                ub = ((four_points_i[3] - dst_i) * (four_points_j[3] - four_points_j[1]) - \
                      (four_points_j[3] - dst_j) * (four_points_i[3] - four_points_i[1])) / det_b \
                # _fv = (py0 - py) * (px0 - px1) - (px0 - px) * (py0 - py1)
                vb = ((four_points_j[3] - dst_j) * (four_points_i[3] - four_points_i[2]) - \
                      (four_points_i[3] - dst_i) * (four_points_j[3] - four_points_j[2])) / det_b
                is_inside_triangle_b = (ub >= u_min) & (vb >= v_min) & (ub + vb <= uv_max) & \
                                       (dst_i >= 0) & (dst_i < dst_grid.tile_width) & \
                                       (dst_j >= 0) & (dst_j < dst_grid.tile_height)
                # insert pixel with this offset into result if inside
                result_col[dst_j[is_inside_triangle_b], dst_i[is_inside_triangle_b]] = \
                    src_id_col[is_inside_triangle_b] + src_offset[0] + ub[is_inside_triangle_b]
                result_row[dst_j[is_inside_triangle_b], dst_i[is_inside_triangle_b]] = \
                    src_id_row[is_inside_triangle_b] + src_offset[1] + vb[is_inside_triangle_b]
        result = np.stack((result_col, result_row))
        return result

    def create_inverse_pixel_index(self) -> da.Array:
        # determine borders x dst_j x dst_i for src_row x src_col
        # merge borders into one per dst block
        bbox_blocks_raw = da.map_blocks(self.dst_bboxes_of_src_block,
                                        self.forward_index,
                                        dst_grid=self.dst_grid,
                                        src_tile_size=self.src_lat.chunksize,
                                        src_size=self.src_lat.shape,
                                        drop_axis=0,
                                        new_axis=[0,1,2],
                                        meta=np.array([], dtype=np.int32),
                                        name=self.name + "_bboxes").compute()
        bbox_blocks = np.stack((np.nanmin(bbox_blocks_raw[0], axis=(2,3)),
                                np.nanmin(bbox_blocks_raw[1], axis=(2,3)),
                                np.nanmax(bbox_blocks_raw[2], axis=(2,3)),
                                np.nanmax(bbox_blocks_raw[3], axis=(2,3))))
        src_lon_lat = da.stack((self.src_lon, self.src_lat))
        # create graph with one call per dst block
        layer = dict()
        dependencies = []
        # dst row blocks loop and col blocks loop
        num_blocks_j = math.ceil(self.dst_grid.height / self.dst_grid.tile_height)
        num_blocks_i = math.ceil(self.dst_grid.width / self.dst_grid.tile_width)
        for tj in range(num_blocks_j):
            for ti in range(num_blocks_i):
                # determine src box that covers dst block plus buffer
                src_offset_i = bbox_blocks[0, tj, ti]
                src_offset_j = bbox_blocks[1, tj, ti]
                src_subset_lon_lat = src_lon_lat[:,
                                                 src_offset_j:bbox_blocks[3, tj, ti],
                                                 src_offset_i:bbox_blocks[2, tj, ti]]
                # compose call for blockwise inverse index
                src_id = (self.name+'_src', 0, tj, ti)
                inv_id = (self.name+'_inverse', 0, tj, ti)
                layer[src_id] = src_subset_lon_lat
                layer[inv_id] = (self.inverse_index_of_dst_block_with_src_subset,
                                 src_id,
                                 (src_offset_i, src_offset_j),
                                 self.dst_grid,
                                 (ti, tj))
        # compose dask array of reprojected results
        graph = HighLevelGraph.from_collections(name,
                                                layer,
                                                dependencies=[])
        result = da.Array(graph,
                          name,
                          shape=(2, dst_grid.rows.count, dst_grid.cols.count),
                          chunks=(2, dst_grid.row_chunksize, dst_grid.col_chunksize),
                          meta=np.ndarray([], dtype=np.float32))
        return result


    @staticmethod
    def rectify_tiles_to_block(block_i: np.ndarray,
                               block_j: np.ndarray,
                               src_grid: Grid,
                               num_blocks_i: int,
                               tiles: np.ndarray,
                               *measurements: da.Array) -> np.array:
        """
        High level graph block function used in rectify,
        uses inverse tile index of block and masks measurements of each tile for the block.
        :param block_i: inverse index for the block
        :param block_j: inverse index for the block
        :param src_grid: chunk sizes to calculate tile-local positions
        :param num_blocks_i: to split tile numbers into tile row and tile column
        :param tiles: list of tile numbers
        :param measurements: list of tile measurement stacks, same length as tiles
        :return: stack of measurements of the block
        """
        dst_data = np.empty((measurements[0].shape[0], *block_j.shape), dtype=np.float32)
        dst_data[:] = np.nan
        # loop over source tiles overlapping with dst block
        for tile, tile_measurements in zip(tiles, measurements):
            # split tile index into tile row and tile column
            tile_j = tile // num_blocks_i
            tile_i = tile % num_blocks_i
            # reduce indices (at dst block positions) to upper left corner of source tile
            shifted_j = block_j - tile_j * src_grid.tile_height
            shifted_i = block_i - tile_i * src_grid.tile_width
            # mask dst block positions that shall be filled by this source tile
            tile_mask = (block_j // src_grid.tile_height == tile_j) & (block_i // src_grid.tile_width == tile_i)
            # get values, TODO is compute required here?
            m = tile_measurements.compute()  # m is bands x j x i
            # set dst positions by values from source measurements
            dst_data[:,tile_mask] = m[:,shifted_j[tile_mask],shifted_i[tile_mask]]
        return dst_data


    def tile_measurements_of(self, tiles: np.ndarray, *measurements: da.Array) -> np.ndarray:
        """
        Utility function that collects measurement stacks of the source tiles
        :param tiles: list of tile indices
        :param measurements: stack of source measurements
        :return: list of source measurement stacks, one per tile in tiles
        """
        num_tiles_i = math.ceil(self.src_grid.width / self.src_grid.tile_width)
        tile_measurements = []
        for tile in tiles:
            # split tile index into tile row and tile column
            tile_i = tile % num_tiles_i
            tile_j = tile // num_tiles_i
            # stack source measurements of the tile and accumulate the stack
            tile_m_stack = da.stack([m.blocks[tile_j, tile_i] for m in measurements])
            # TODO test without stacking, then collect dask node names from list of list elements
            tile_m_rechunked = tile_m_stack.rechunk(tile_m_stack.shape)
            tile_measurements.append(tile_m_rechunked)
        return tile_measurements


    def rectify_nearest_neighbour(self,
                                  *measurements: da.Array) -> da.Array:
        """
        Rectifies stack of measurements from source image to dst grid using pre-computed inverse index
        :param measurements: stack of source measurements
        :return: dask array with stack of rectified measurements on dst grid
        """
        if not self.dst_grid:
            raise ValueError("missing dst grid. Call create_covering_dst_grid() first.")
        if self.inverse_index is None:
            raise ValueError("missing inverse index. Call create_inverse_pixel_index() first.")
        num_blocks_x = math.ceil(self.dst_grid.width / self.dst_grid.tile_width)
        num_blocks_y = math.ceil(self.dst_grid.height / self.dst_grid.tile_height)
        num_tiles_col = math.ceil(self.src_grid.width / self.src_grid.tile_width)
        num_tiles_row = math.ceil(self.src_grid.height / self.src_grid.tile_height)
        # create graph with one call per dst block
        layer = dict()
        dependencies = []
        # dst rows loop and cols loop
        for ty in range(num_blocks_y):
            for tx in range(num_blocks_x):
                # ty and tx are the dst block numbers 0..num_blocks
                # determine inverse index values for dst block
                block = self.inverse_index.blocks[:, ty, tx]
                block = block.compute()
                block_i = block[0]
                block_j = block[1]
                # determine source tile indices for the dst block, still on dst pixel resolution
                # mask areas outside inverse index range
                block_tile = (block_j // self.src_grid.tile_height) * num_tiles_col + (block_i // self.src_grid.tile_width)
                block_tile[(block_j == -1) | (block_i == -1)] = -1
                # determine set of the few src tile indices that occur in this dst block using some numpy indexing
                # initialise array with possible set of tiles + one extra at the end for out-of-index, all masked -1
                # set the few positions (many times all at once) where there are tiles set in dst block
                # slice away tile indices not set and extra element, tile_set is a small array of source tiles
                all_tile_numbers = np.arange(num_tiles_row * num_tiles_col + 1, dtype=int)
                tile_array = np.empty(shape=(num_tiles_row * num_tiles_col + 1), dtype=int)
                tile_array[:] = -1
                tile_array[block_tile] = all_tile_numbers[block_tile]
                tile_set = tile_array[(tile_array != -1) & (tile_array != all_tile_numbers[-1])]
                # collect the measurement stacks of the source tiles of this block
                tile_measurements = self.tile_measurements_of(tile_set, *measurements)
                # get the dependencies for the dask graph complete
                tile_measurement_ids = []
                for measurement in tile_measurements:
                    id = (measurement.name, 0, 0, 0)
                    if id not in layer:
                        layer[id] = measurement
                        dependencies.append(measurement)
                    tile_measurement_ids.append(id)
                # add dask graph entry to reproject source tiles to block
                # additional first index is 0 for the single chunk of the complete measurement stack
                layer[(self.name, 0, ty, tx)] = (
                    Rectifier.rectify_tiles_to_block, block_i, block_j, self.src_grid, num_tiles_col, tile_set, *tile_measurement_ids
                )
        # compose dask array of reprojected results
        graph = HighLevelGraph.from_collections(self.name,
                                                layer,
                                                dependencies=dependencies)
        meta = np.ndarray(shape=(len(measurements), num_blocks_y, num_blocks_x), dtype=np.float32)
        result = da.Array(graph,
                          self.name,
                          shape=(len(measurements), self.dst_grid.height, self.dst_grid.width),
                          chunks=(len(measurements), self.dst_grid.tile_height, self.dst_grid.tile_width),
                          meta=meta)
        return result


    @staticmethod
    def rectify_to_covering_grid(src_lon: da.Array,
                                 src_lat: da.Array,
                                 dst_crs: pyproj.CRS,
                                 dst_res: Tuple[float, float],
                                 dst_tilesize: Tuple[float, float],
                                 *measurements: da.Array,
                                 name: str = "rectify") -> da.Array:
        """
        Convenience function that instruments rectifier and rectifies measurements
        :param src_lon: source image longitudes
        :param src_lat: source image latitudes
        :param dst_crs: CRS for destination grid
        :param dst_step: resolution of destination grid in CRS units
        :param dst_blocksize: size of one block of destination grid
        :param measurements: dask array with stack of source measurements
        :param name: unique name of array in dask graph, optional, required if this is part of a larger graph
        :return: dask array with stack of rectified measurements on dst grid
        """
        preliminary_dst_grid = Grid(dst_crs, (0, 0), dst_res, (0, 0), dst_tilesize)
        rectifier = Rectifier(src_lon, src_lat, preliminary_dst_grid)
        rectifier.create_forward_pixel_index()
        rectifier.determine_covering_dst_grid()
        rectifier.create_inverse_pixel_index()
        rectified_measurements = rectifier.rectify_nearest_neighbour(*measurements)
        return rectified_measurements
