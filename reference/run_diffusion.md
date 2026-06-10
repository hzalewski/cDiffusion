# Run diffusion map

Performs non-linear dimensionality reduction using the exact dense
method. Best suited for smaller datasets where calculating the full N x
N distance matrix fits safely into memory.

## Usage

``` r
run_diffusion(data, sigma = 1, dims = 2, oversampling = 10, n_iter = 10)
```

## Arguments

- data:

  Numeric matrix or data.frame. Missing values (NA) and infinite values
  are not supported and will cause an error.

- sigma:

  Numeric. Bandwidth for Gaussian kernel (default 1.0).

- dims:

  Integer. Number of diffusion dimensions to return (default 2).

- oversampling:

  Integer. Dimensions added for stability in randomized SVD algorithm
  (default 10). Higher values may negatively impact results of thr
  algorithm.

- n_iter:

  Integer. Number of subspace iterations in randomized SVD algorithm
  (default 10).

## Value

A diffmap object

## Examples

``` r
data <- matrix(runif(1000), nrow = 50, ncol = 20)
optimal_sigma <- estimate_sigma(data)
model <- run_diffusion(data, sigma = optimal_sigma)
```
