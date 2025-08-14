## Changes in 0.1.0 (in development)

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


