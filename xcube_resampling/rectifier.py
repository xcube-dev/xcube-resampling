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
import dask
import dask.array as da
from .grid import Grid
from typing import Tuple


class Rectifier:
    """
    TODO update!

    Reprojects measurement layers from a source image with pixel geo-coding to a destination grid
    with CRS and a similar resolution, with nearest neighbour resampling as default method.
    Source image must be given, dst grid is given or can be created such that it covers the source image area.
    Pre-calculates inverse pixel index that contains the source pixel coordinates for each destination pixel.
    Generates and returns a dask array with a dask graph for the delayed computation. Just the forward
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
    Parameter sequence convention:
    - Arrays are structured row-column, with col being the inner loop.
    - Stacks of coordinate arrays are x-y, lon-lat. Same for stacks of index arrays i-j, col-row.
    - block_id parameters are (j, i), i.e. in the sequence of array dimensions, as in map_blocks.
    - All other parameters are x-first, i.e. i, j etc.
    """
    def __init__(self,
                 src_lon: da.Array,
                 src_lat: da.Array,
                 dst_grid: Grid = None,
                 name: str = "rectify"):
        """
        Constructor
        :param src_lon: source x as 2-D array in geographic coordinates, row-column order
        :param src_lon: source y as 2-D array in geographic coordinates, row-column order
        :param dst_grid: dst grid, optional, can be determined with create_covering_dst_grid
        :param name: unique name of dask graph node, default "rectify",
                     must be set if several Rectifiers may be in the graph
        """
        self.src_lon = src_lon
        self.src_lat = src_lat
        self.dst_grid = dst_grid
        self.name = name

    def prepare_forward_index(self) -> da.Array:
        self.dask_forward_index = da.map_blocks(Rectifier.forward_index_block,
                                                self.src_lon,
                                                self.src_lat,
                                                dst_grid=self.dst_grid,
                                                new_axis=0,
                                                #dtype=np.int32,
                                                meta=np.array((), dtype=np.int32))
        return self.dask_forward_index

    def prepare_src_bboxes(self) -> da.Array:
        bbox_blocks_raw = da.map_blocks(self.bboxes_block,
                                        self.dask_forward_index,
                                        dst_grid=self.dst_grid,
                                        src_tile_size=self.src_lat.chunksize,
                                        src_size=self.src_lat.shape,
                                        drop_axis=0,
                                        new_axis=[0,1,2],
                                        meta=np.array([], dtype=np.int32))
        # we compute min and max of all 4 bounds though we need only two of each,
        # dask seems not to delay a []
        self.dask_src_bboxes = da.vstack((da.nanmin(bbox_blocks_raw, axis=(3,4)),
                                          da.nanmax(bbox_blocks_raw, axis=(3,4))))
        return self.dask_src_bboxes

    def compute_forward_index(self) -> Tuple[np.ndarray, np.ndarray]:
        self.forward_index, src_bboxes = dask.compute(self.dask_forward_index, self.dask_src_bboxes)
        # we select min and max of two bounds each
        self.src_bboxes = np.stack((src_bboxes[0], src_bboxes[1], src_bboxes[6], src_bboxes[7]))
        return (self.forward_index, self.src_bboxes)

    def determine_covering_dst_grid(self) -> Grid:
        """
        Determines the dst grid extent covering a source image in a different projection.
        Uses forward pixel index to source-blockwise determine bboxes in dst grid,
        merge them into one bbox, align with dst grid origin,
        update dst grid origin and size, shift forward pixel index
        :return: dst grid
        """
        if self.forward_index is None:
            raise ValueError("missing forward index. Call create_forward_pixel_index() first.")
        bboxes = da.map_blocks(self.dst_bbox_of_src_block,
                               self.forward_index,
                               #new_axis=0,
                               name=self.name + "_bbox",
                               meta=np.array([], dtype=int)).compute()
        i_min = np.min(bboxes[0])
        j_min = np.min(bboxes[1])
        i_max = np.max(bboxes[2])
        j_max = np.max(bboxes[3])
        # adjust origin and size of dst grid
        self.dst_grid.x_min = i_min * self.dst_grid.x_res
        self.dst_grid.y_min = j_min * self.dst_grid.y_res
        self.dst_grid.width = i_max - i_min + 1
        self.dst_grid.height = j_max - j_min + 1
        self.dst_grid = Grid(self.dst_grid.crs,
                             (i_min * self.dst_grid.x_res, j_min * self.dst_grid.y_res),
                             (self.dst_grid.x_res, self.dst_grid.y_res),
                             (i_max - i_min + 1, j_max - j_min + 1),
                             (self.dst_grid.tile_width, self.dst_grid.tile_height))
        # adjust pixel coordinates of forward mapping
        self.forward_index -= np.array([[[i_min]], [[j_min]]])
        return self.dst_grid

    def prepare_inverse_index(self):
        num_blocks_j = math.ceil(self.dst_grid.height / self.dst_grid.tile_height)
        num_blocks_i = math.ceil(self.dst_grid.width / self.dst_grid.tile_width)
        num_complete_tile_cols = num_blocks_i - 1
        num_complete_tile_rows = num_blocks_j - 1
        block_rows = []
        for tj in range(num_blocks_j):
            block_cols = []
            for ti in range(num_blocks_i):
                # result has the extent of the dst grid tile, initialised with nan
                dst_width = self.dst_grid.tile_width \
                    if ti < num_complete_tile_cols \
                    else self.dst_grid.width - num_complete_tile_cols * self.dst_grid.tile_width
                dst_height = self.dst_grid.tile_height \
                    if tj < num_complete_tile_rows \
                    else self.dst_grid.height - num_complete_tile_rows * self.dst_grid.tile_height
                # dst rows loop and cols loop
                # determine src box that covers dst block plus buffer
                lon_lat_blocks = []
                for tile_row in range(self.src_bboxes[1, tj, ti] // self.src_lon.chunksize[0],
                                      (self.src_bboxes[3, tj, ti] - 1) // self.src_lon.chunksize[0] + 1):
                    for tile_col in range(self.src_bboxes[0, tj, ti] // self.src_lon.chunksize[1],
                                          (self.src_bboxes[2, tj, ti] - 1) // self.src_lon.chunksize[1] + 1):
                        lon_lat_blocks.append(self.src_lon.blocks[tile_row, tile_col])
                        lon_lat_blocks.append(self.src_lat.blocks[tile_row, tile_col])
                delayed_block = dask.delayed(self.inverse_index_block)((tj, ti),
                                                                       self.dst_grid,
                                                                       self.src_bboxes[:, tj, ti],
                                                                       self.src_lon.chunksize,
                                                                       *lon_lat_blocks)
                da_block = da.from_delayed(delayed_block, shape=(2, dst_height, dst_width), dtype=float)
                block_cols.append(da_block)
            block_row = da.concatenate(block_cols, axis=2)
            block_rows.append(block_row)
        self.dask_inverse_index = da.concatenate(block_rows, axis=1)
        return self.dask_inverse_index

    def prepare_rectification(self, *measurements: da.Array) -> da.Array:
        """
        Rectifies stack of measurements from source image to dst grid using pre-computed inverse index
        :param measurements: stack of source measurements
        :return: dask array with stack of rectified measurements on dst grid
        """
        if not self.dst_grid:
            raise ValueError("missing dst grid. Call create_covering_dst_grid() first.")
        if self.dask_inverse_index is None:
            raise ValueError("missing inverse index. Call create_inverse_pixel_index() first.")
        num_blocks_j = math.ceil(self.dst_grid.height / self.dst_grid.tile_height)
        num_blocks_i = math.ceil(self.dst_grid.width / self.dst_grid.tile_width)
        num_complete_tile_cols = num_blocks_i - 1
        num_complete_tile_rows = num_blocks_j - 1
        block_rows = []
        # dst rows loop and cols loop
        for tj in range(num_blocks_j):
            block_cols = []
            for ti in range(num_blocks_i):
                # result has the extent of the dst grid tile, initialised with nan
                dst_width = self.dst_grid.tile_width \
                    if ti < num_complete_tile_cols \
                    else self.dst_grid.width - num_complete_tile_cols * self.dst_grid.tile_width
                dst_height = self.dst_grid.tile_height \
                    if tj < num_complete_tile_rows \
                    else self.dst_grid.height - num_complete_tile_rows * self.dst_grid.tile_height
                measurement_blocks = []
                for tile_row in range(self.src_bboxes[1, tj, ti] // self.src_lon.chunksize[0],
                                      (self.src_bboxes[3, tj, ti] - 1) // self.src_lon.chunksize[0] + 1):
                    for tile_col in range(self.src_bboxes[0, tj, ti] // self.src_lon.chunksize[1],
                                          (self.src_bboxes[2, tj, ti] - 1) // self.src_lon.chunksize[1] + 1):
                        for k in range(len(measurements)):
                            block = measurements[k].blocks[tile_row, tile_col]
                            measurement_blocks.append(block)
                inverse_index_block = self.dask_inverse_index.blocks[:, tj, ti]
                delayed_block = dask.delayed(Rectifier.rectify_block)((tj, ti),
                                                                      self.src_bboxes[:, tj, ti],
                                                                      self.src_lon.chunksize,
                                                                      len(measurements),
                                                                      inverse_index_block,
                                                                      *measurement_blocks)
                da_block = da.from_delayed(delayed_block,
                                           shape=(len(measurements), dst_height, dst_width),
                                           dtype=float)  # TODO distinguish band types
                block_cols.append(da_block)
            block_row = da.concatenate(block_cols, axis=2)
            block_rows.append(block_row)
        self.dask_rectified = da.concatenate(block_rows, axis=1)
        return self.dask_rectified

    def compute_rectification(self) -> np.ndarray:
        return self.dask_rectified.compute()



    @staticmethod
    def forward_index_block(src_lon:np.ndarray,
                            src_lat:np.ndarray,
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
        if dst_grid.crs.is_geographic:
            dst_x, dst_y = src_lon, src_lat
        else:
            trafo = pyproj.Transformer.from_crs(pyproj.CRS(4326), dst_grid.crs, always_xy=True)
            dst_x, dst_y = trafo.transform(src_lon, src_lat)
        # convert to pixel positions
        # "floor" because dst_xy contain pixel centre coordinates
        dst_i = np.floor((dst_x - dst_grid.x_min) / dst_grid.x_res).astype(dtype=int)
        dst_j = np.floor((dst_y - dst_grid.y_min) / dst_grid.y_res).astype(dtype=int)
        # we stack to return a single np array for the block
        result = np.stack((dst_i, dst_j))
        return result

    @staticmethod
    def bboxes_block(forward_index_block: np.ndarray,
                     dst_grid: Grid = None,
                     src_tile_size: Tuple[int, int] = None,
                     src_size: Tuple[int, int] = None,
                     block_id: Tuple[int, int] = None):
        """
        Determines for one source block the source bounding box for each dst block
        if source block and dst block intersect.
        map_blocks function called by create_inverse_pixel_index
        :param forward_index_block: source block of pixel coordinates of source pixels in dst grid
                                    shape (2, src_tile_height, src_tile_width), sequence i, j
        :param dst_grid: dst number of blocks and their sizes
        :param src_tile_size: source tile size, to determine source offset of the block, sequence lat, lon
        :param block_id source block row and block column, in this sequence
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
                # add 1 src pixel margin in each direction
                result_boxes[0, dst_block_j, dst_block_i, 0, 0] = max(np.min(src_block_cols) - 1, 0) if len(src_block_cols) > 0 else src_size[1]
                result_boxes[1, dst_block_j, dst_block_i, 0, 0] = max(np.min(src_block_rows) - 1, 0) if len(src_block_rows) > 0 else src_size[0]
                result_boxes[2, dst_block_j, dst_block_i, 0, 0] = min(np.max(src_block_cols) + 2, src_size[1]) if len(src_block_cols) > 0 else -1
                result_boxes[3, dst_block_j, dst_block_i, 0, 0] = min(np.max(src_block_rows) + 2, src_size[0]) if len(src_block_rows) > 0 else -1
        return result_boxes

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

    @staticmethod
    def inverse_index_block(block_id: Tuple[int, int],
                            dst_grid: Grid,
                            src_bbox: np.ndarray,
                            src_chunksize: Tuple[int, int],
                            *lon_lat_tiles: np.ndarray) -> np.ndarray:
        """
        TODO update
        Determines inverse index col, row of fractional source image pixel coordinates for
        each dst pixel of a dst block. Uses painter algorithm to transform src triangles to
        dst grid and mark dst pixels inside triangles.
        High level graph function used in create_inverse_pixel_index.
        :param src_subset_lon_lat: lon and lat coordinates of source pixels of a subset
                                   covering the dst block
        :param src_offset: offset of src_subset_lon_lat in fractional src pixel coordinates to be
                           added to inverse index, pixel position of the origin of src_subset_lon_lat
                           TODO this is an exact "0.5 pixel" position, is this intended?
        :param dst_grid: destination grid
        :param block_id destination block coordinates, j and i in this sequence
        :return: inverse pixel index with fractional source pixel index for each dst block pixel
        """
        # mosaic tiles and subset them to src bbox
        src_subset_lon_lat = \
            Rectifier.mosaic_src_blocks(lon_lat_tiles, src_bbox, src_chunksize, 2)
        # generate four points with two triangles for the src subset in dst pixel fractional coordinates
        # TODO check whether generation of fractional forward index is an alternative to avoid duplicate reprojection
        four_points_i, four_points_j = \
            Rectifier.triangles_in_dst_pixel_grid(src_subset_lon_lat, dst_grid, block_id)
        # create small bboxes for the four points
        bboxes_min_i, bboxes_min_j, bboxes_max_width, bboxes_max_height = \
            Rectifier.bboxes_of_triangles(four_points_i, four_points_j, dst_grid)
        # create source subset offset and identity vector for rows and columns
        src_offset = (src_bbox[0] + 0.5, src_bbox[1] + 0.5)
        src_width = src_subset_lon_lat.shape[1] - 1
        src_height = src_subset_lon_lat.shape[2] - 1
        src_id_col = np.tile(np.arange(src_height), (src_width, 1))
        src_id_row = np.tile(np.arange(src_width), (src_height, 1)).transpose()
        # result has the extent of the dst grid tile, initialised with nan
        num_complete_tile_cols = math.ceil(dst_grid.width / dst_grid.tile_width) - 1
        num_complete_tile_rows = math.ceil(dst_grid.height / dst_grid.tile_height) - 1
        dst_width = dst_grid.tile_width \
            if block_id[1] < num_complete_tile_cols \
            else dst_grid.width - num_complete_tile_cols * dst_grid.tile_width
        dst_height = dst_grid.tile_height \
            if block_id[0] < num_complete_tile_rows \
            else dst_grid.height - num_complete_tile_rows * dst_grid.tile_height
        result_col = np.empty((dst_height, dst_width))
        result_row = np.empty((dst_height, dst_width))
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
        # loops over bboxes max size, shift the dst pixel for each triangle
        for j_offset in range(bboxes_max_height):
            dst_j = bboxes_min_j + j_offset
            for i_offset in range(bboxes_max_width):
                dst_i = bboxes_min_i + i_offset
                # dst_j and dst_i have the extent of the source subset
                # dst_j and dst_i contain integer pixel coordinates of the considered point in the dst grid
                # u and v have the extent of the source subset
                # _fu = (px0 - px) * (py0 - py2) - (py0 - py) * (px0 - px2)
                ua = ((four_points_i[0] - dst_i - 0.5) * (four_points_j[0] - four_points_j[2]) - \
                      (four_points_j[0] - dst_j - 0.5) * (four_points_i[0] - four_points_i[2])) / det_a
                # _fv = (py0 - py) * (px0 - px1) - (px0 - px) * (py0 - py1)
                va = ((four_points_j[0] - dst_j - 0.5) * (four_points_i[0] - four_points_i[1]) - \
                      (four_points_i[0] - dst_i - 0.5) * (four_points_j[0] - four_points_j[1])) / det_a
                is_inside_triangle_a = (ua >= u_min) & (va >= v_min) & (ua + va <= uv_max) & \
                                       (dst_i >= 0) & (dst_i < dst_width) & \
                                       (dst_j >= 0) & (dst_j < dst_height)
                # insert pixel with this offset into result if inside
                if is_inside_triangle_a.any():
                    result_col[dst_j[is_inside_triangle_a], dst_i[is_inside_triangle_a]] = \
                        src_id_col[is_inside_triangle_a] + src_offset[0] + ua[is_inside_triangle_a]
                    result_row[dst_j[is_inside_triangle_a], dst_i[is_inside_triangle_a]] = \
                        src_id_row[is_inside_triangle_a] + src_offset[1] + va[is_inside_triangle_a]
                # do the same for triangle b ...
                # _fu = (px0 - px) * (py0 - py2) - (py0 - py) * (px0 - px2)
                ub = ((four_points_i[3] - dst_i - 0.5) * (four_points_j[3] - four_points_j[1]) - \
                      (four_points_j[3] - dst_j - 0.5) * (four_points_i[3] - four_points_i[1])) / det_b
                # _fv = (py0 - py) * (px0 - px1) - (px0 - px) * (py0 - py1)
                vb = ((four_points_j[3] - dst_j - 0.5) * (four_points_i[3] - four_points_i[2]) - \
                      (four_points_i[3] - dst_i - 0.5) * (four_points_j[3] - four_points_j[2])) / det_b
                is_inside_triangle_b = (ub >= u_min) & (vb >= v_min) & (ub + vb <= uv_max) & \
                                       (dst_i >= 0) & (dst_i < dst_width) & \
                                       (dst_j >= 0) & (dst_j < dst_height)
                # insert pixel with this offset into result if inside
                if is_inside_triangle_b.any():
                    result_col[dst_j[is_inside_triangle_b], dst_i[is_inside_triangle_b]] = \
                        src_id_col[is_inside_triangle_b] + src_offset[0] + 1.0 - ub[is_inside_triangle_b]
                    result_row[dst_j[is_inside_triangle_b], dst_i[is_inside_triangle_b]] = \
                        src_id_row[is_inside_triangle_b] + src_offset[1] + 1.0 - vb[is_inside_triangle_b]
        result = np.stack((result_col, result_row))
        return result

    @staticmethod
    def mosaic_src_blocks(lon_lat_tiles, src_bbox, src_chunksize, num_measurements):
        subset_i0 = src_bbox[0]
        subset_j0 = src_bbox[1]
        src_subset_lon_lat = np.empty((num_measurements,
                                       src_bbox[3] - src_bbox[1],
                                       src_bbox[2] - src_bbox[0]),
                                      dtype=lon_lat_tiles[0].dtype)
        i = 0
        for tile_j in range(src_bbox[1] // src_chunksize[0], (src_bbox[3] - 1) // src_chunksize[0] + 1):
            tile_j0 = tile_j * src_chunksize[0]
            tile_j1 = tile_j0 + src_chunksize[0]
            j0 = max(tile_j0, src_bbox[1])
            j1 = min(tile_j1, src_bbox[3])
            for tile_i in range(src_bbox[0] // src_chunksize[1], (src_bbox[2] - 1) // src_chunksize[1] + 1):
                tile_i0 = tile_i * src_chunksize[1]
                tile_i1 = tile_i0 + src_chunksize[1]
                i0 = max(tile_i0, src_bbox[0])
                i1 = min(tile_i1, src_bbox[2])
                # print(*block_id, j0, j1, i0, i1, tile_j0, tile_j1)
                for k in range(num_measurements):
                    src_subset_lon_lat[k, j0 - subset_j0:j1 - subset_j0, i0 - subset_i0:i1 - subset_i0] = \
                        lon_lat_tiles[i][j0 - tile_j0:j1 - tile_j0, i0 - tile_i0:i1 - tile_i0]
                    i += 1
        return src_subset_lon_lat

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
        :param block_id destination block coordinates, j and i in this sequence
        :return: pair i, j of stacks of the four points
                 with the extent of the source subset minus one in each direction
                 as fractional dst coordinates
        """
        if dst_grid.crs.is_geographic:
            src_x, src_y = src_subset_lon_lat[0], src_subset_lon_lat[1]
        else:
            trafo = pyproj.Transformer.from_crs(pyproj.CRS(4326), dst_grid.crs, always_xy=True)
            src_x, src_y = trafo.transform(src_subset_lon_lat[0], src_subset_lon_lat[1])
        # determine fractional pixel positions of subset in dst grid
        # subset_i and _j have the extent of the source subset.
        # subset_i and _j contain fractional pixel coordinates in the dst grid's coordinate system
        # subset_i and _j are fractions related to the upper left corner of the dst pixel
        # i.e. in case of 0.5 the source pixel matches a dst position
        subset_i = (src_x - dst_grid.x_min) / dst_grid.x_res - block_id[1] * dst_grid.tile_width
        subset_j = (src_y - dst_grid.y_min) / dst_grid.y_res - block_id[0] * dst_grid.tile_height
        stacked_subset_i = np.stack((subset_i[ :-1,  :-1],
                                     subset_i[ :-1, 1:  ],
                                     subset_i[1:  ,  :-1],
                                     subset_i[1:  , 1:  ]))
        stacked_subset_j = np.stack((subset_j[ :-1,  :-1],
                                     subset_j[ :-1, 1:  ],
                                     subset_j[1:  ,  :-1],
                                     subset_j[1:  , 1:  ]))
        return (stacked_subset_i, stacked_subset_j)

    @staticmethod
    def bboxes_of_triangles(four_points_i: np.ndarray, four_points_j: np.ndarray, dst_grid: Grid):
        is_inside_dst_block = (four_points_i[0] >= 0.0) \
                              & (four_points_j[0] >= 0.0) \
                              & (four_points_i[0] <= dst_grid.tile_width) \
                              & (four_points_j[0] <= dst_grid.tile_height)
        bboxes_min_i = np.floor(np.min(four_points_i, axis=0)).astype(int)
        bboxes_min_j = np.floor(np.min(four_points_j, axis=0)).astype(int)
        bboxes_max_i = np.ceil(np.max(four_points_i, axis=0)).astype(int)
        bboxes_max_j = np.ceil(np.max(four_points_j, axis=0)).astype(int)
        bboxes_width = bboxes_max_i - bboxes_min_i
        bboxes_height = bboxes_max_j - bboxes_min_j
        # determine maximum bbox size of all pairs of triangles
        if np.any(is_inside_dst_block):
            bboxes_max_width = np.max(bboxes_width[is_inside_dst_block]).astype(int)
            bboxes_max_height = np.max(bboxes_height[is_inside_dst_block]).astype(int)
        else:
            bboxes_max_width = 0
            bboxes_max_height = 0
        return (bboxes_min_i, bboxes_min_j, bboxes_max_width, bboxes_max_height)

    @staticmethod
    def rectify_block(block_id: Tuple[int, int],
                      src_bbox: np.ndarray,
                      src_chunksize: Tuple[int, int],
                      num_measurements: int,
                      inverse_index: np.ndarray,
                      *measurements: np.ndarray) -> np.array:
        """
        TODO
        """
        # convert inverse index to nearest neighbour int
        # floor because inverse index 0.5 is pixel centre
        inverse_index_int = np.floor(inverse_index).astype(int)
        inverse_index_int[np.isnan(inverse_index)] = -1
        shifted_i = inverse_index_int[0] - src_bbox[0]
        shifted_j = inverse_index_int[1] - src_bbox[1]
        # mosaic tiles and subset them to src bbox
        src_subset_measurements = Rectifier.mosaic_src_blocks(measurements, src_bbox, src_chunksize, num_measurements)
        num_src_rows = src_subset_measurements.shape[1]
        num_src_cols = src_subset_measurements.shape[2]
        # prepare and fill dst block
        dst_data = np.empty((num_measurements, *shifted_j.shape), dtype=np.float32)
        dst_data[:] = np.nan
        tile_mask = (shifted_j >= 0) \
                        & (shifted_j < num_src_rows) \
                        & (shifted_i >= 0) \
                        & (shifted_i < num_src_cols)
        # set dst positions by values from source measurements
        dst_data[:,tile_mask] = src_subset_measurements[:,shifted_j[tile_mask],shifted_i[tile_mask]]
        return dst_data


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
        rectifier = Rectifier(src_lon, src_lat, preliminary_dst_grid, name=name)
        rectifier.prepare_forward_index()
        rectifier.prepare_src_bboxes()
        rectifier.compute_forward_index()
        rectifier.determine_covering_dst_grid()
        # bboxex have to be computed after dst grid is available
        rectifier.compute_forward_index()
        rectifier.prepare_inverse_index()
        rectifier.prepare_rectification(*measurements)
        return rectifier.dask_rectified, rectifier.dst_grid
