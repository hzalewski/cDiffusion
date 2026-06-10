# Spectral Clustering using Diffusion Maps with Eigengap Heuristic

Applies K-means clustering on the diffusion coordinates. If the number
of clusters is not provided, it automatically estimates the optimal
number of clusters (k) using the Eigengap heuristic (finding the largest
drop between consecutive eigenvalues).

## Usage

``` r
cdiff_kmeans(x, centers = "auto", max_k = 10, nstart = 25)
```

## Arguments

- x:

  A 'diffmap' object returned by \`run_diffusion()\` or
  \`run_diffusion_sparse()\`.

- centers:

  Integer or "auto". The number of clusters. If "auto" (default),
  optimal k is estimated.

- max_k:

  Integer. Maximum number of clusters to consider when \`centers =
  "auto"\` (default 10).

- nstart:

  Integer. Number of random sets to choose for K-means (default 25 for
  stability).

## Value

A standard 'kmeans' list object containing the clustering results.
