## User Guide for xcube-resampling

**xcube-resampling** provides algorithms for representing a dataset in a different
grid mapping. It supports:

- Simple resampling via affine transformation  
- Reprojection between coordinate reference systems (CRS)  
- Rectification of non-regular grids to regular grids

All resampling methods are built around the
[`GridMapping`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.gridmapping.GridMapping)
class, which represents a spatial grid mapping and contains all necessary information
for resampling.  
We first introduce the `GridMapping` class in more detail before explaining the three
resampling algorithms.

---

### `GridMapping` – the grid mapping object

A `GridMapping` object contains metadata such as the coordinate reference system (CRS),
spatial resolution, bounding box, spatial size, coordinates, and tile size (if the
dataset is chunked).  

There are three main ways to create a `GridMapping` object:

#### 1. Create a regular grid mapping

Use the [`GridMapping.regular`](https://xcube-dev.github.io/xcube-resampling/#xcube_resampling.gridmapping.GridMapping.regular)
method to create a regular grid mapping with just a few parameters:

```python
from xcube_resampling.gridmapping import GridMapping

gm = GridMapping.regular(size, xy_min, xy_res, crs)
```
Parameter descriptions can be found [here](https://xcube-dev.github.io/xcube-resampling/#xcube_resampling.gridmapping.GridMapping.regular).



#### 2. Create a grid mapping from an existing dataset

Use the [`GridMapping.from_dataset`](https://xcube-dev.github.io/xcube-resampling/#xcube_resampling.gridmapping.GridMapping.from_dataset)
method:

```python
from xcube_resampling.gridmapping import GridMapping

gm = GridMapping.from_dataset(ds)
```
Here, `ds` is a `xarray.Dataset`. Further optional parameters can be found [here](https://xcube-dev.github.io/xcube-resampling/#xcube_resampling.gridmapping.GridMapping.from_dataset).
> **Note:** If no grid mapping is provided for the input dataset, the resampling functions 
> use this method to derive one.

#### 3. Create a grid mapping from coordinates 
Use the [`GridMapping.from_coords`](https://xcube-dev.github.io/xcube-resampling/#xcube_resampling.gridmapping.GridMapping.from_coords)
method:

```python
from xcube_resampling.gridmapping import GridMapping

gm = GridMapping.from_coords(x_coords, y_coords, crs)
```
Here, `x_coords` and `y_coords` are `xarray.Array` instances, and `crs` is the 
coordinate reference system (CRS). Further details of the parameters can be found
[here](https://xcube-dev.github.io/xcube-resampling/#xcube_resampling.gridmapping.GridMapping.from_dataset).

#### Derive new `GridMapping` instances

You can create new grid mappings from existing ones using:

- [derive](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.gridmapping.GridMapping.derive): 
  change selected properties.
- [scale](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.gridmapping.GridMapping.scale): 
  create a scaled version of a regular grid mapping.
- [to_regular](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.gridmapping.GridMapping.derive):
  convert an irregular grid mapping to a regular one.
- [transform](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.gridmapping.GridMapping.transform):
  change the CRS of a grid mapping (regular → irregular with 2D coordinates).

### Resampling Algorithms

The function [`resample_in_space`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.spatial.resample_in_space)
integrates all three resampling algorithms and automatically selects the most
appropriate one:

| Algorithm             | Function                                                                                                                      | Selection Criteria                                                                                   |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Affine Transformation** | [`affine_transform_dataset`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.affine.affine_transform_dataset) | Source and target grids are both regular and share the same CRS.                                    |
| **Reprojection**      | [`reproject_dataset`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.reproject.reproject_dataset)         | Source and target grids are both regular but have different CRS.                                  |
| **Rectification**     | [`rectify_dataset`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.rectify.rectify_dataset)               | 	Source grid is irregular with 2D coordinates.                                            |

With `resample_in_space`, users do **not** need to worry about selecting the right
algorithm—the function determines and applies it automatically.

#### Common parameters for all resampling algorithms

| Parameter       | Type / Accepted Values | Description | Default                                                                     |
|-----------------|------------------------|-------------|-----------------------------------------------------------------------------|
| `variables`     | `str` or iterable of `str` | Name(s) of variables to resample. If `None`, all data variables are processed. | `None`                                                                      |
| `interp_methods`| `int`, `str`, or `dict` mapping var/dtype to method. Supported:<br>• `0` — nearest neighbor<br>• `1` — linear / bilinear<br>• `"nearest"`<br>• `"triangular"`<br>• `"bilinear"` | Interpolation method for upsampling spatial data variables. Can be a single value or per-variable/dtype mapping. | `0` for integer arrays, else `1`                                            |
| `agg_methods`   | `str` or `dict` mapping var/dtype to method. Supported:<br>`"center"`, `"count"`, `"first"`, `"last"`, `"max"`, `"mean"`, `"median"`, `"mode"`, `"min"`, `"prod"`, `"std"`, `"sum"`, `"var"` | Aggregation method for downsampling spatial data variables. | `"center"` for integer arrays, else `"mean"`                                |
| `recover_nans`  | `bool` or `dict` mapping var/dtype to `bool` | Enable NaN recovery during upsampling (only applies when interpolation method is not nearest). | `False`                                                                     |
| `fill_values`   | scalar or `dict` mapping var/dtype to value. | Fill value(s) for areas outside input coverage. | <br>• float — NaN<br>• uint8 — 255<br>• uint16 — 65535<br>• other ints — -1 |


---

#### 1. Affine Transformation

An affine transformation can be applied when both the source and target grid mappings are regular and share the same CRS. The function [`affine_transform_dataset`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.affine.affine_transform_dataset) requires the input dataset and the target grid mapping.  

For any data array in the dataset with two spatial dimensions as the last two axes, an affine transformation is performed using [`dask_image.ndinterp.affine_transform`](https://image.dask.org/en/latest/dask_image.ndinterp.html). The resulting dataset contains resampled data arrays aligned to the target grid mapping. Data variables without spatial dimensions are copied to the output, while variables with only one spatial dimension are ignored.

> **Note:** The `interp_methods` parameter corresponds to the `order` parameter in [`dask_image.ndinterp.affine_transform`](https://image.dask.org/en/latest/dask_image.ndinterp.html). Only spline orders `[0, 1]` are supported to avoid unintended blending across non-spatial dimensions (e.g., time) in 3D arrays.

#### 2. Reprojection

Reprojection can be applied when both source and target grid mappings are regular but use different CRSs. The function [`reproject_dataset`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.reproject.reproject_dataset) requires the input dataset and the target grid mapping.  

During reprojection, the coordinates of the target grid mapping are transformed to the source CRS, producing 2D irregular coordinates. For each transformed irregular pixel location, neighboring pixels are identified, and their values are used to perform the selected interpolation. Supported interpolation methods are described in [Section Interpolation Methods](#interpolation-methods).

#### 3. Rectification

Rectification is used when the source dataset has an irregular grid. The function [`rectify_dataset`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.rectify.rectify_dataset) requires only the input dataset. If no target grid mapping is provided, the source grid mapping is converted to a regular grid, and interpolation is performed so that the new dataset is defined on this regular grid.  

For each regular target grid point, neighboring pixels from the irregular source grid are located, and their values are used for the selected interpolation method. Supported interpolation methods are described in [Section Interpolation Methods](#interpolation-methods).


add stuff from xcube doc. 


### Interpolation Methods
 add from xcube docs. 

#### nearest 

#### triangular

#### bilinear
