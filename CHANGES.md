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


