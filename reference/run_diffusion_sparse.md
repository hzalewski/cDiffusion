# Run sparse diffusion map

Performs non-linear dimensionality reduction using a memory-efficient
sparse k-NN method. It is better suited for larger datasets than the
exact dense method, as its memory usage scales linearly rather than
quadratically. Uses an adaptive Gaussian kernel (local sigma based on
the distance to the k-th nearest neighbor).

## Usage

``` r
run_diffusion_sparse(
  data,
  dims = 2,
  k_neighbors = 20,
  oversampling = 10,
  n_iter = 300
)
```

## Arguments

- data:

  Numeric matrix or data.frame. Missing values (NA) and infinite values
  are not supported and will cause an error.

- dims:

  Integer. Number of diffusion dimensions to return (default 2).

- k_neighbors:

  Integer. Number of nearest neighbors for sparse method (default 20).

- oversampling:

  Integer. Dimensions added for stability in randomized SVD algorithm
  (default 10). Higher values may negatively impact results of thr
  algorithm.

- n_iter:

  Integer. Number of subspace iterations in randomized SVD algorithm
  (default 300).

## Value

A diffmap object

## Examples

``` r
data <- matrix(runif(5000), nrow = 1000, ncol = 5)
model <- run_diffusion_sparse(data, k_neighbors = 20, oversampling = 10, n_iter = 200)
```
