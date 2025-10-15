# xcube-resampling

**xcube-resampling** provides efficient algorithms for transforming datasets into 
different spatial grids and temporal scales. It is designed for geospatial workflows that need 
flexible resampling and reprojection. This library provides up and downsampling for both 
spatial and temporal domains.

### ✨ Features
- #### Spatial Resampling
    - **Affine resampling** – simple resampling using affine transformations  
    - **Reprojection** – convert datasets between different coordinate reference systems (CRS)  
    - **Rectification** – transform irregular grids into regular, well-structured grids  

- #### Temporal Resampling

All methods work seamlessly with chunked (lazily loaded) [xarray.Datasets](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) and are powered by [Dask](https://www.dask.org/) for scalable, out-of-core computation.

### ⚡ Lightweight & Independent
The package is independent of the core *xcube* framework and has minimal dependencies:
`affine, dask, dask-image, numba, numpy, pyproj, xarray, zarr`.


## Overview

The spatial resampling methods are built around the
[`GridMapping`](api.md/#xcube_resampling.gridmapping.GridMapping)
class, which represents a spatial grid mapping and contains all necessary information
for resampling.  

A `GridMapping` is required for affine transformation and reprojection, and optional
for rectification. If omitted for rectification, a simple rectification will be
performed while staying in the same CRS.

The temporal resampling method is built around the [`Resampler`](https://docs.xarray.dev/en/stable/generated/xarray.groupers.Resampler.html#xarray.groupers.Resampler)
class, which provides methods to up or downsample a temporal dataset
into the requested frequency.
---

### `resample_in_space` — the gateway to Spatial Resampling of Gridded datasets

The **central function** for spatial up and down resampling
[`resample_in_space`](api.md/#xcube_resampling.spatial.resample_in_space), integrates all three algorithms and **automatically selects** the appropriate one
based on the criteria below.

| Algorithm             | Function                                                                               | Selection Criteria                                                                                   |
|-----------------------|----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Affine Transformation** | [`affine_transform_dataset`](api.md/#xcube_resampling.affine.affine_transform_dataset) | Source and target grids are both regular and share the same CRS.                                    |
| **Reprojection**      | [`reproject_dataset`](api.md/#xcube_resampling.reproject.reproject_dataset)            | Source and target grids are both regular but have different CRS.                                    |
| **Rectification**     | [`rectify_dataset`](api.md/#xcube_resampling.rectify.rectify_dataset)                  | Source grid is irregular and contains 2D coordinates.                                               |

With `resample_in_space`, users do **not** need to worry about selecting the right
algorithm—the function determines and applies it automatically.

### `resample_in_time` — the gateway to Temporal Resampling of Gridded datasets

The **central function** for resampling in the temporal domain is
`resample_in_time` which handles both up and downsampling of a dataset in the 
`time` dimension.

👉 For usage examples and details, see the [User Guide](guide.md).


## License

`xcube-resampling` is open source made available under the terms and conditions of the 
[MIT license](https://opensource.org/license/mit).

`