## Changes in 0.2.3 (under development)

- In `xcube_resampling.resample_in_time` the method is only added if the parameters
  `agg_methods` or `interp_methods` are a mapping or if multiple resampling  methods
  are applied to the data variable.


## Changes in 0.2.2

- Added support for spatial rectification of datasets in
  `xcube_resampling.rectify_dataset` even when spatial coordinates are not stored in
  the dataset’s coordinate variables.


## Changes in 0.2.1

- Added the `output_indices_names` parameter to `xcube_resampling.rectify_dataset` to
  allow storing the source pixel indices.


## Changes in 0.2.0

- Enhanced the function `bbox_overlap` so that it can handle bounding boxes crossing
  the antimeridian
- Added new class method `GridMapping.regular_from_bbox`, which allows creating a
  regular grid mapping directly from a bounding box, spatial resolution, and CRS.
- Added new function `utils.resolution_meters_to_degrees` which converts spatial
  resolution from meters to degrees in latitude and longitude at a given geographic
  latitude.
    - 1 degree of latitude ≈ 111,320 meters (constant approximation).
    - 1 degree of longitude ≈ 111,320 * cos(latitude) meters.
- Bug fix: fixed grid mapping creation for irregular grids with decreasing longitude
  along axis 1.
- Added `xcube_resampling.resample_in_time` which allows to resample a dataset along
  the time axis. It supports up- and down-sampling.
- Renamed the parameter `recover_nans` to `prevent_nan_propagations`, which can be
  a boolean or a mapping (from variable name or dtype to boolean) that, when `True`,
  prevents NaN propagation during upsampling or interpolation.


## Changes in 0.1.1

- Improved `xcube_resampling.utils.clip_dataset_by_bbox` to support datasets with 
  2D coordinates. This function is also used internally by 
  `xcube_resampling.rectify.rectify_dataset`.  
- Added `reproject_bbox`, a utility to reproject a bounding box from one CRS to another.  
- Added `bbox_overlap`, a utility that computes the fraction of the source bounding box
  overlapped by the target bounding box.

## Changes in 0.1.0

- Added algorithm for **affine transformation**. (#4)
- Added algorithm for **rectification of non-regular grids**. (#4)
- Added algorithm for **reprojection to a different coordinate reference system (CRS)**.
  (#4)
- Introduced main function `resample_in_space`, which dynamically selects the 
  appropriate resampling algorithm based on the input dataset. (#4)
- Added initial **unit tests** to verify core functionality. (#4)
- Introduced a new unified keyword argument `interp_method` that supports values `0`, 
  `1`, `"nearest"`, `"triangular"`, and `"bilinear"`. This argument applies 
  consistently across all three resampling algorithms, simplifying usage and 
  improving API consistency. (#8)
- Documentation added, which is available at 
  https://xcube-dev.github.io/xcube-resampling/. (#10) 


