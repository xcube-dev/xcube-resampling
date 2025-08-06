# xcube-resampling

xcube-resampling contains alogrithms to represent a dataset in another grid-mapping. It 
compromises simple resampling via affine tranformation, reprojection and recification 
of nonregular grids to regular grids. All these algorithms can be applied to chunked
(lazily loaded) datasets, since they are backed by dask. 


## License

`xcube-resampling` is open source made available under the terms and conditions of the 
[MIT license](https://opensource.org/license/mit).

