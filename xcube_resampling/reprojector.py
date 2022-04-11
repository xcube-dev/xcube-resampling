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

class Reprojector:
    """
    Reprojects measurement layers from a source grid to a target grid with nearest neighbour resampling.
    Source grid must be given, target grid may be created such that it covers the source grid area.
    Pre-calculates inverse pixel index that contains the source pixel coordinates for each target pixel.
    Generates and returns a dask array with a dask graph for the delayed computation. Just the inverse 
    index is pre-calculated as it determines how the graph is constructed. Pre-calculation is done by dask,
    too.

    If source grid and target grid are given the calling sequence is::

        src_grid = Grid(pyprj.CRS(4326), (10.0, 53.6), (0.2, -0.2), (6, 3), (2, 2))
        target_grid = Grid(pyproj.CRS(3035), (4320000, 3390000), (10000, -10000), (8, 7), (2, 2))
        reprojector = Reprojector(src_grid, target_grid)
        reprojector.create_transformer()
        reprojector.create_inverse_pixel_index()
        repro_measurements = reprojector.reproject(*measurements)

    If only the source grid is given the calling sequence is::

        src_grid = Grid(pyprj.CRS(4326), (10.0, 53.6), (0.2, -0.2), (6, 3), (2, 2))
        target_crs = pyproj.CRS(3035)
        reprojector = Reprojector(src_grid)
        reprojector.create_transformer(target_crs)
        reprojector.create_target_grid(-10000.0, 10000.0, -20000.0, 20000.0)
        reprojector.create_inverse_pixel_index()
        repro_measurements = reprojector.reproject(*measurements)
    """
    def __init__(self,
                 src_grid: Grid,
                 target_grid: Grid = None,
                 trafo: pyproj.Transformer = None,
                 inverse_index: Tuple[np.ndarray, np.ndarray] = None,
                 name: str = "reproject"):
        """
        Constructor
        :param src_grid: source grid
        :param target_grid: target grid, optional, can be determined with create_covering_target_grid
        :param trafo: transformer from source to target CRS, optional, can be determined with create_transformer
        :param inverse_index: inverse pixel index with the extent of the target grid, can be determined with create_inverse_pixel_index
        :param name: unique name of dask graph node, default "reproject", must be set if several Reprojectors may be in the graph
        """
        if not src_grid:
            raise ValueError("missing source grid. src_grid must not be Null")
        self.src_grid = src_grid
        self.target_grid = target_grid
        self.trafo = trafo
        self.inverse_index = inverse_index
        self.name = name


    def create_transformer(self, target_crs: pyproj.CRS = None) -> pyproj.Transformer:
        """
        Creates Transformer from source CRS to target CRS
        :param target_crs: target CRS, optional, must be set if target_grid has not been set in constructor
        """
        if target_crs:
            if self.target_grid:
                raise ValueError(f"duplicate provision of target CRS: {target_crs}")
            self.target_grid = Grid(target_crs, None, None, None)
        elif not self.target_grid:
            raise ValueError("missing target grid")
        self.trafo = pyproj.Transformer.from_crs(self.src_grid.crs,
                                                 self.target_grid.crs,
                                                 always_xy=True)
        return self.trafo


    def create_covering_target_grid(self,
                                    target_res: Union[Tuple[np.float64, np.float64], np.float64],
                                    target_blocksize: Optional[Union[Tuple[np.float64, np.float64], np.float64]] = None,
                                    densify_pts: int = 21) -> Grid:
        """
        Determines a target grid covering a source grid in a different projection.
           Transforms source border, determines bounding box, and creates target grid in x and y.
           Aligns target grid to have a pixel border at 0/0.
        :param target_res: target x and y resolution in units of target CRS
        :param target_blocksize: target x and y tile size in units of target CRS, multiple of target_res
        :param densify_points: passed to Transformer.transform_bounds, defaults to 21
        :return: Grid of target grid. Tile size may be adjusted
        """
        if not self.trafo:
            raise ValueError("missing transformer. Call create_transformer() first.")
        if isinstance(target_res, (int, float)):
            target_res_x, target_res_y = (target_res, target_res)
        else:
            target_res_x, target_res_y = target_res
        if target_blocksize is None or isinstance(target_blocksize, (int, float)):
            target_blocksize_x, target_blocksize_y = (target_blocksize, target_blocksize)
        else:
            target_blocksize_x, target_blocksize_y = target_blocksize
        # determine src bbox of pixel centres and src pixel size
        src_bottom = self.src_grid.y_min + self.src_grid.y_res * self.src_grid.height
        src_right = self.src_grid.x_min + self.src_grid.x_res * self.src_grid.width
        if self.src_grid.crs.is_geographic and src_right <= self.src_grid.x_min:
            src_right += 360.0
        # transform bbox of pixel borders
        left, bottom, right, top = self.trafo.transform_bounds(self.src_grid.x_min,
                                                               src_bottom,
                                                               src_right,
                                                               self.src_grid.y_min,
                                                               densify_pts=densify_pts)
        # snap to target origin at 0/0, "round" to have pixel centre inside source bbox
        left = round(left / target_res_x) * target_res_x
        right = round(right / target_res_x) * target_res_x
        bottom = - round(- bottom / target_res_y) * target_res_y
        top = - round(- top / target_res_y) * target_res_y
        if self.target_grid.crs.is_geographic and right <= left:
            right += 360.0
        rows = round((bottom - top) / target_res_y)
        cols = round((right - left) / target_res_x)
        # construct target grid in y and x
        self.target_grid = Grid(self.target_grid.crs,
                                (left, top),
                                (target_res_x, target_res_y),
                                (cols, rows),
                                (round(target_blocksize_y / target_res_y) if target_blocksize_y else rows,
                                 round(target_blocksize_x / target_res_x) if target_blocksize_x else cols))
        return self.target_grid


    @staticmethod
    def source_pixels_of(target_x:np.ndarray,
                         target_y:np.ndarray,
                         trafo:pyproj.Transformer=None,
                         src_grid:Grid=None) -> np.ndarray:
        """
        A map_blocks function used in inverse_pixel_index,
           calculates for one target block the source image coordinates j and i for each target pixel.
        :param target_x: flat array with one block row of target x coordinates at target pixel centres
        :param target_y: narrow array with one block column of target y coordinates at target pixel centres
        :param trafo: transformation from src to target, applied inverse in this function
        :param src_grid: defines a grid with the origin at the upper left corner of the start pixel
        :return: stack of i and j arrays for the block
        """
        num_block_rows = target_y.shape[0]
        num_block_cols = target_x.shape[1]
        # expand to full block
        x_block = np.tile(target_x, (num_block_rows, 1))
        y_block = np.tile(target_y, (1, num_block_cols))
        # transform into source coordinates
        i_coords, j_coords = trafo.transform(x_block, y_block, direction=pyproj.enums.TransformDirection.INVERSE)
        # convert to pixel positions
        # "floor" in fact rounds because transformed pixel centers are related to upper left corner
        i_index = np.floor((i_coords - src_grid.x_min) / src_grid.x_res).astype(dtype=int)
        j_index = np.floor((j_coords - src_grid.y_min) / src_grid.y_res).astype(dtype=int)
        # we must stack to return a single np array for the block
        result = np.stack((i_index, j_index))
        # mask area outside of source image
        is_valid = (j_index < 0) | (j_index >= src_grid.height) | (i_index < 0) | (i_index >= src_grid.width)
        result[:, is_valid] = -1
        return result


    def create_inverse_pixel_index(self) -> da.Array:
        """
        Transforms target to source and memorizes pixel index of source for each target pixel
        :return: dask array with two layers i and j. Needs compute()
        """
        if not self.trafo:
            raise ValueError("missing transformer. Call create_transformer() first.")
        if not self.target_grid:
            raise ValueError("missing target grid. Call create_covering_target_grid() first.")
        # construct lightweight blocks of target 2d coordinates, i.e.
        # generate row coordinates, duplicate for each x block, transpose to get one column per x block, chunk into y blocks
        # generate col coordinates, duplicate for each y block to get one row per y block, chunk into x blocks
        num_x_blocks = math.ceil(self.target_grid.width / self.target_grid.tile_width)
        num_y_blocks = math.ceil(self.target_grid.height / self.target_grid.tile_height)
        target_x_2d = da.tile(self.target_grid.x_axis(),
                              (num_y_blocks, 1)).rechunk((1, self.target_grid.tile_width))
        target_y_2d = da.tile(self.target_grid.y_axis(),
                              (num_x_blocks, 1)).transpose().rechunk((self.target_grid.tile_height, 1))
        # transform them into dask arrays of source indices
        # add new first axis for i and j
        self.inverse_index = da.map_blocks(Reprojector.source_pixels_of,
                                           target_x_2d,
                                           target_y_2d,
                                           trafo=self.trafo,
                                           src_grid=self.src_grid,
                                           new_axis=0,
                                           dtype=np.int32,
                                           meta=np.array((), dtype=np.int32))
        return self.inverse_index


    @staticmethod
    def reproject_tiles_to_block(block_i: np.ndarray,
                                 block_j: np.ndarray,
                                 src_grid: Grid,
                                 num_blocks_i: int,
                                 tiles: np.ndarray,
                                 *measurements: da.Array) -> np.array:
        """
        High level graph block function used in reproject,
        uses inverse tile index of block and masks measurements of each tile for the block, transfers them.
        :param block_i: inverse index for the block
        :param block_j: inverse index for the block
        :param src_grid: chunk sizes to calculate tile-local positions
        :param num_blocks_i: to split tile numbers into tile row and tile column
        :param tiles: list of tile numbers
        :param measurements: list of tile measurement stacks, same length as tiles
        :return: stack of measurements of the block
        """
        target_data = np.empty((measurements[0].shape[0], *block_j.shape), dtype=np.float32)
        target_data[:] = np.nan
        # loop over source tiles overlapping with target block
        for tile, tile_measurements in zip(tiles, measurements):
            # split tile index into tile row and tile column
            tile_j = tile // num_blocks_i
            tile_i = tile % num_blocks_i
            # reduce indices (at target block positions) to upper left corner of source tile
            shifted_j = block_j - tile_j * src_grid.tile_height
            shifted_i = block_i - tile_i * src_grid.tile_width
            # mask target block positions that shall be filled by this source tile
            tile_mask = (block_j // src_grid.tile_height == tile_j) & (block_i // src_grid.tile_width == tile_i)
            # get values, TODO is compute required here?
            m = tile_measurements.compute()  # m is bands x j x i
            # set target positions by values from source measurements
            target_data[:,tile_mask] = m[:,shifted_j[tile_mask],shifted_i[tile_mask]]
        return target_data


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


    def reproject(self,
                  *measurements: da.Array) -> da.Array:
        """
        Reprojects stack of measurements from source grid to target grid using pre-computed inverse index
        :param measurements: stack of source measurements
        :return: dask array with stack of reprojected measurements on target grid
        """
        if not self.target_grid:
            raise ValueError("missing target grid. Call create_covering_target_grid() first.")
        if self.inverse_index is None:
            raise ValueError("missing inverse index. Call create_inverse_pixel_index() first.")
        num_blocks_x = math.ceil(self.target_grid.width / self.target_grid.tile_width)
        num_blocks_y = math.ceil(self.target_grid.height / self.target_grid.tile_height)
        num_tiles_i = math.ceil(self.src_grid.width / self.src_grid.tile_width)
        num_tiles_j = math.ceil(self.src_grid.height / self.src_grid.tile_height)
        # create graph with one call per target block
        layer = dict()
        dependencies = []
        # target rows loop and cols loop
        for ty in range(num_blocks_y):
            for tx in range(num_blocks_x):
                # ty and tx are the target block numbers 0..num_blocks
                # determine inverse index values for target block
                block = self.inverse_index.blocks[:, ty, tx]
                block = block.compute()
                block_i = block[0]
                block_j = block[1]
                # determine source tile indices for the target block, still on target pixel resolution
                # mask areas outside inverse index range
                block_tile = (block_j // self.src_grid.tile_height) * num_tiles_i + (block_i // self.src_grid.tile_width)
                block_tile[(block_j == -1) | (block_i == -1)] = -1
                # determine set of the few src tile indices that occur in this target block using some numpy indexing
                # initialise array with possible set of tiles + one extra at the end for out-of-index, all masked -1
                # set the few positions (many times all at once) where there are tiles set in target block
                # slice away tile indices not set and extra element, tile_set is a small array of source tiles
                all_tile_numbers = np.arange(num_tiles_j * num_tiles_i + 1, dtype=int)
                tile_array = np.empty(num_tiles_j * num_tiles_i + 1, dtype=int)
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
                    Reprojector.reproject_tiles_to_block, block_i, block_j, self.src_grid, num_tiles_i, tile_set, *tile_measurement_ids
                )
        # compose dask array of reprojected results
        graph = HighLevelGraph.from_collections(self.name,
                                                layer,
                                                dependencies=dependencies)
        meta = np.ndarray(shape=(len(measurements), num_blocks_y, num_blocks_x), dtype=np.float32)
        result = da.Array(graph,
                          self.name,
                          shape=(len(measurements), self.target_grid.height, self.target_grid.width),
                          chunks=(len(measurements), self.target_grid.tile_height, self.target_grid.tile_width),
                          meta=meta)
        return result


    @staticmethod
    def reproject_to_covering_grid(src_j_1d: np.ndarray,
                                   src_i_1d: np.ndarray,
                                   src_crs: pyproj.CRS,
                                   src_chunksize: Tuple[int, int],
                                   target_crs: pyproj.CRS,
                                   target_step: Tuple[float, float],
                                   target_blocksize: Tuple[float, float],
                                   *measurements: da.Array,
                                   name: str = "reproject") -> da.Array:
        """
        Convenience function that instruments reprojector and reprojects measurements
        :param src_grid: source grid (crs, count and chunk size used)
        :param target_grid: target grid (crs, count and chunk size used)
        :param measurements: dask array with stack of source measurements
        :param name: unique name of array in dask graph, optional, required if this is part of a larger graph
        :param trafo: transformer from source to target grid, optional, created if not provided
        :return: dask array with stack of reprojected measurements on target grid
        """
        src_grid = Grid.from_coords(src_crs, (src_i_1d, src_j_1d), src_chunksize)
        reprojector = Reprojector(src_grid, name=name)
        reprojector.create_transformer(target_crs=target_crs)
        reprojector.create_covering_target_grid(target_step, target_blocksize)
        reprojector.create_inverse_pixel_index()
        return reprojector.reproject(*measurements)

