# cDiffusion

`cDiffusion` is a high-performance R package for non-linear
dimensionality reduction and manifold learning via Diffusion Maps.
Written using R C API, it provides a fast and memory-efficient way to
compute diffusion maps for moderately large datasets. This package
offers both a classic dense method for small-scale analytics and a
highly optimized sparse implementation designed to scale with larger
data.

## Documentation

For API reference, performance benchmarks, and package overview, please
visit the official package website:

**[cDiffusion Documentation
Website](https://hzalewski.github.io/cDiffusion/)**

## Package Features

- **Two Approaches:** Provides a dense matrix solver for exact
  computation on smaller datasets, and a sparse solver for heavier
  workloads.
- **Optimized Memory Usage:** The sparse implementation eliminates the
  need to allocate full $`N \times N`$ distance matrices. It computes an
  exact k-nearest neighbors graph on the fly using a max-heap structure
  and stores the transition graph in Compressed Sparse Row (CSR) format,
  reducing memory complexity to $`\mathcal{O}(N \cdot k\_neighbors)`$.
- **rSVD solver:** Eigen-decompositions on matrices are done via custom
  Randomized Singular Value Decomposition (rSVD) solver. Most heavy
  computations are multithreaded with the use of OpenMP. Only then
  LAPACK finds eigenvectors for a small $`m \times m`$ matrix where
  $`m`$ is the requested number of dimensions.
- **Straightforwards Design:** Zero external dependencies. The codebase
  relies strictly on the native R C API (including LAPACK/BLAS), OpenMP,
  and the base R `stats` package, ensuring straightforward compilation.

## Installation

Install the development version directly from GitHub:

``` r

install.packages("remotes")
remotes::install_github("hzalewski/cDiffusion")
```

## Usage

`cDiffusion` is designed to be easy in use. The API consists of 4 core
functions:

1.  **[`run_diffusion()`](https://hzalewski.github.io/cDiffusion/reference/run_diffusion.md)**:
    The exact, dense solver. Perfect for capturing global topologies on
    small to medium datasets.
2.  **[`run_diffusion_sparse()`](https://hzalewski.github.io/cDiffusion/reference/run_diffusion_sparse.md)**:
    The sparse solver. Bypasses the need to allocate dense
    $`N \times N`$ euclidean distance matrix by taking only k - nearest
    neighbours for each point and converting those distances into
    Compressed Sparse Row (CSR) format.
3.  **[`estimate_sigma()`](https://hzalewski.github.io/cDiffusion/reference/estimate_sigma.md)**:
    Computes the global Gaussian kernel width ($`\sigma`$) using the
    Median Heuristic by subsampling of the entry data.
4.  **[`cdiff_kmeans()`](https://hzalewski.github.io/cDiffusion/reference/cdiff_kmeans.md)**:
    Spectral Clustering function. It applies K-means to the diffusion
    coordinates and features an Eigengap Heuristic for automatic
    detection of the optimal number of clusters.

### 1. Dense Method

``` r

library(cDiffusion)

data_matrix <- matrix(rnorm(6000), ncol = 3)

opt_sigma <- estimate_sigma(data_matrix, sample_size = 500)

model_dense <- run_diffusion(
  data_matrix, 
  dims = 3,
  sigma = opt_sigma, 
  n_iter = 50)

plot(model_dense$coordinates[, 1], model_dense$coordinates[, 2],
    pch = 16, col = "blue",
    main = "Diffusion map result",
    xlab = "Diffusion Coord 1", ylab = "Diffusion Coord 2")
```

### 2. Sparse Method

``` r

library(cDiffusion)

data_matrix <- matrix(rnorm(30000 * 100), nrow = 30000, ncol = 100)

# Here we don't need to be concerned for the sigma parameter 
# since sparse method uses local sigma

model_sparse <- run_diffusion_sparse(
  data_matrix, 
  dims = 3,
  k_neighbors = 35,   
  oversampling = 10,   
  n_iter = 1000
)
```

### 3. Spectral Clustering (diffusion map + kmeans)

``` r

# Provide the diffmap object directly
# centers = "auto" will automatically find the best 'k'
clusters <- cdiff_kmeans(model_sparse, centers = "auto")
#> Eigengap heuristic: selected k = 4 clusters.

# The result is a standard kmeans object
table(clusters$cluster)
```

## Output

`cDiffusion` returns a standard R list with the `S3` class `diffmap`.
This allows for easy extraction of the results for tasks like
visualization or passing it to
[`cdiff_kmeans()`](https://hzalewski.github.io/cDiffusion/reference/cdiff_kmeans.md)
function.

Here is the exact structure of the output depending on the chosen
method:

**Dense Method Output ($`N = 2000`$):**

``` r

str(model_dense)
#> List of 5
#>  $ coordinates: num [1:2000, 1:2] -0.0691 -0.1416 0.0883 0.2579 -0.
#>  $ eigenvalues: num [1:2] 0.38 0.37
#>  $ sigma      : num 1.57
#>  $ data_dim   : int [1:2] 2000 3
#>  $ method     : chr "dense"
#>  - attr(*, "class")= chr "diffmap"
```

**Sparse Method Output ($`N = 30000`$):**

``` r

str(model_sparse)
#> List of 4
#>  $ coordinates: num [1:30000, 1:2] -0.00457 0.01001 -0.02034 0.02162 
#>  $ eigenvalues: num [1:2] 0.65 0.52
#>  $ data_dim   : int [1:2] 30000 100
#>  $ method     : chr "sparse"
#>  - attr(*, "class")= chr "diffmap"
```
