## User Guide for xcube-resampling

**xcube-resampling** provides algorithms for representing a dataset in a different
grid mapping. It supports:

- Simple resampling via affine transformation  
- Reprojection between coordinate reference systems (CRS)  
- Rectification of non-regular grids to regular grids

All resampling methods are built around the
[`GridMapping`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.gridmapping.GridMapping)
class, which represents a spatial grid mapping and contains all necessary information
for resampling. We therefore introduce the `GridMapping` class in a bit more detail 
before we will explain the three different resampling algorithms. 

### `GridMapping` - the grid-mapping object

A `GridMapping` object contains information like, the coordinate reference system (CRS),
spatial resolution, bounding bbox, spatial size, spatial coordinates, tile_size 
(if chunked dataset) etc. There are three ways to initialize a `GridMapping` object:

#### 1. Create a regular grid-mapping

To create a regular grid mapping with just a few parameters, one can use the method 
[`GridMapping.regular`](https://xcube-dev.github.io/xcube-resampling/#xcube_resampling.gridmapping.GridMapping.regular)
as follows:

```python
from xcube_resampling.gridmapping import GridMapping

gm = GridMapping.regular(size, xy_min, xy_res, crs)
```
where the parameters can be looked up [here](https://xcube-dev.github.io/xcube-resampling/#xcube_resampling.gridmapping.GridMapping.regular).



#### 2. Create a grid-mapping from an existing dataset

To create a grid mapping from an exsisting dataset, one can use the method 
[`GridMapping.from_dataset`](https://xcube-dev.github.io/xcube-resampling/#xcube_resampling.gridmapping.GridMapping.from_dataset)
as follows:

```python
from xcube_resampling.gridmapping import GridMapping

gm = GridMapping.from_dataset(ds)
```
where `ds` is a `xarray.Dataset`. Further optional parameters can be looked up [here](https://xcube-dev.github.io/xcube-resampling/#xcube_resampling.gridmapping.GridMapping.from_dataset).
Note that this method is used in the resampling functions, if no grid mapping is given
associated with the input dataset. 

#### 3. Create a grid-mapping from coordinates 
To create a grid mapping from coordinates, one can use the method 
[`GridMapping.from_coords`](https://xcube-dev.github.io/xcube-resampling/#xcube_resampling.gridmapping.GridMapping.from_coords)
as follows:

```python
from xcube_resampling.gridmapping import GridMapping

gm = GridMapping.from_coords(x_coords, y_coords, crs)
```
where `x_coords` and `y_coords` are `xarray.Array`, and `crs` is the coordinate 
reference system (CRS). Further details of the parameters can be looked up [here](https://xcube-dev.github.io/xcube-resampling/#xcube_resampling.gridmapping.GridMapping.from_dataset).

#### Operations to derive new `GridMapping` instances from existing ones

To derive new `GridMapping` instances from existing `GridMapping` instances, once 
can use one of the following methods: 

- [derive](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.gridmapping.GridMapping.derive): 
  Derive a new grid mapping with some properties changed.
- [scale](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.gridmapping.GridMapping.scale): 
  Derive a scaled version of this regular grid mapping.
- [to_regular](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.gridmapping.GridMapping.derive):
  Transform an irregular grid mapping into a regular grid mapping.
- [transform](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.gridmapping.GridMapping.transform):
  Transforms the coordinate reference system (CRS) of the grid mapping. Note it
  transforms a regular grid mapping into a irregular grid mapping with 2d coordinates. 

### Resampling algorithms

xcube-resampling provides three resampling algorithms, which are integrated into the function
[`resample_in_space`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.spatial.resample_in_space),
which integrates all three algorithms and **automatically selects** the appropriate one
based on the criteria below.

| Algorithm             | Function                                                                                                                      | Selection Criteria                                                                                   |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Affine Transformation** | [`affine_transform_dataset`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.affine.affine_transform_dataset) | Source and target grids are both regular and share the same CRS.                                    |
| **Reprojection**      | [`reproject_dataset`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.reproject.reproject_dataset)         | Source and target grids are both regular but have different CRS.                                    |
| **Rectification**     | [`rectify_dataset`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.rectify.rectify_dataset)               | Source grid is irregular and contains 2D coordinates.                                               |

With `resample_in_space`, users do **not** need to worry about selecting the right
algorithm—the function determines and applies it automatically.

#### Common parameters applicable of all resampling algorithms

`variables`: A single variable name or iterable of variable names to be
    resampled. If None, all data variables will be processed.
`spline_orders`: Spline orders used for upsampling spatial data variables.
    Can be a single spline order (0=nearest, 1=linear, 2=bilinear, 3=cubic),
    or a dictionary mapping variable names or data dtypes to spline orders.
    Defaults to 3 for floating-point data and 0 for integers.
`agg_methods`: Aggregation methods for downsampling spatial data variables.
    Can be a single method (e.g., `np.mean`) or a dictionary mapping variable
    names or dtypes to aggregation methods. These are passed to
    `dask.array.coarsen`.
`recover_nans`: Whether to apply a special algorithm to recover values that
    would otherwise become NaN during resampling. This can be a single boolean
    or a dictionary mapping variable names or dtypes to booleans. Defaults to
    False.
`fill_values`: Fill values to use for areas where data is unavailable.

The three resampling algorithms are explained below in detail for better understanding. 

#### 1. Affine transformation

An affine transformation can be applied, if the source and target grid mapping are both
regular and both have the same CRS. The function [`affine_transform_dataset`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.affine.affine_transform_dataset)
requires the input dataset and the target grid mapping. For any data array in the 
dataset, which have the two spatial dimensions on the last two dimension, an affine
transformation is performed using [`dask_image.ndinterp.affine_transform`](https://image.dask.org/en/latest/dask_image.ndinterp.html).
The new dataset contains resampled data arrays aligned to the target grid mapping. Data 
variables without spatial dimensions are copied to the output. Data variables with 
only one spatial dimension are ignored.
