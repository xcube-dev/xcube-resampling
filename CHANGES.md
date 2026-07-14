## Changes in 0.3.5 (in development)

- Fixed a bug in `rectify_dataset` where the conversion from block-space indices to 
  global-space indices was incorrect when `output_indices_names` was provided.

## Changes in 0.3.4

- Added `utils.transform_resolution` to convert spatial resolution between coordinate 
  reference systems (CRS).


## Changes in 0.3.3

- Added function `utils.resolution_degrees_to_meters` to convert spatial resolution
  from degrees to meters at a given geographic latitude.


## Changes in 0.3.2

- Change inconsistent license classifier in `pyproject.toml` 
- Removed the dependency on `zarr`. As part of this change, 
  `xcube_resampling.grid_mapping.cfconv.add_spatial_ref` has been removed, as it 
  only loosely fit the scope of the **xcube-resampling** repository.
- Added `xcube_resampling.utils.get_utm_crs` to return the UTM CRS for a given 
  longitude and latitude pair.
- In `xcube_resampling.reproject_dataset`, the dataset is now clipped when the target
  grid-mapping covers less than 80% of the source grid-mapping, improving performance
  for reprojected cutouts.
- In `xcube_resampling.affine_transform_dataset`, resampling is now performed per 
  spatial slice, ensuring that interpolation is applied only along the spatial axes 
  and no longer across non-spatial dimensions (e.g., time).
- When `prevent_nan_propagation=True`, computation is now fully lazy. Previously, the 
  implementation checked for NaN values in the source array, which triggered loading
  of the entire dataset into memory.
- Enhanced `GridMapping` API documentation.


## Changes in 0.3.1

- Fixed a bug in `xcube_resampling.rectify_dataset` that occurred when processing  
  datasets with decreasing x-coordinates.
- Boolean (`bool`) data arrays now default to nearest-neighbor interpolation.  
  For spatial aggregation, values are centered, and `0` is used as the default fill 
  value—matching the behavior of integer arrays to ensure consistent and valid results.


## Changes in 0.3.0

- Fixed an issue when retrieving coordinate names from regular grid mappings.
- Improved documentation for the `frequency` and `tolerance` arguments of
  `xcube_resampling.resample_in_time`.
- Fixed incorrect time coverage metadata in dataset attributes after calling
  `xcube_resampling.resample_in_time`.
- `xcube_resampling.rectify_dataset` is now fully lazy and scalable.
- Removed methods `test_ij_bbox_from_xy_bbox` and `test_ij_bboxes_from_xy_bboxes` from
  `xcube_resampling.gridmapping.GridMapping`. These methods were only used by
  `xcube_resampling.rectify_dataset`; their updated implementations now live in
  `xcube_resampling.rectify._compute_source_tile_indexing`.
- Added support for rectifying datasets with decreasing x-coordinate in 
  `xcube_resampling.rectify_dataset`.


## Changes in 0.2.4

- Improved performance when creating `GridMapping` from 2D coordinates.
  Parameter estimation now uses only corner points instead of the full set of
  coordinates, significantly reducing runtime for large datasets.
- Bug fix in affine transformations.
  Fixed a half-pixel shift that occurred in `xcube_resampling.affine_transform_dataset`
  and `xcube_resampling.resample_in_space` when the source and target grid mappings
  shared the same CRS and were regular.
 

## Changes in 0.2.3

- In `xcube_resampling.resample_in_time`, the method suffix in the output data variable
  names is now added only when the `agg_methods` or `interp_methods` parameters are
  provided as mappings, or when multiple resampling methods are applied to the data
  variable.


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


