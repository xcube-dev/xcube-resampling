# xcube-resampling

**xcube-resampling** provides algorithms for representing a dataset in a different
grid mapping. It supports:

- Simple resampling via affine transformation  
- Reprojection between coordinate reference systems (CRS)  
- Rectification of non-regular grids to regular grids  

All algorithms work with chunked (lazily loaded) [xarray.Datasets](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html),
as they are powered by [Dask](https://www.dask.org/).

## Overview

The resampling methods are built around the
[`GridMapping`](https://xcube-dev.github.io/xcube-resampling/api/#xcube_resampling.gridmapping.GridMapping)
class, which represents a spatial grid mapping and contains all necessary information
for resampling.  

A `GridMapping` is required for affine transformation and reprojection, and optional
for rectification. If omitted for rectification, a simple rectification will be
performed while staying in the same CRS.

---

### `resample_in_space` — the gateway to xcube-resampling

The **central function** in this package is
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

👉 For usage examples and details, see the [User Guide](guide.md).


## License

`xcube-resampling` is open source made available under the terms and conditions of the 
[MIT license](https://opensource.org/license/mit).

