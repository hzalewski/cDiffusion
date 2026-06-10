# Estimate optimal sigma for Gaussian kernel

Uses the median heuristic to automatically estimate the optimal
bandwidth (sigma) parameter for the Gaussian kernel in diffusion maps.

## Usage

``` r
estimate_sigma(data, sample_size = 500)
```

## Arguments

- data:

  Numeric matrix or data.frame.

- sample_size:

  Integer. Number of points to sample for estimation (default 500).

## Value

A numeric scalar representing the estimated sigma.

## Examples

``` r

data <- matrix(rnorm(100), nrow = 20, ncol = 5)

optimal_sigma <- estimate_sigma(data)
```
