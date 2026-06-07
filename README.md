# cDiffusion

[![R-CMD-check](https://github.com/hzalewski/cDiffusion/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/hzalewski/cDiffusion/actions/workflows/R-CMD-check.yaml)

`cDiffusion` is a high-performance R package for non-linear dimensionality reduction and manifold learning via Diffusion Maps. Written using R C API, it provides a fast and memory-efficient way to compute diffusion maps for moderately large datasets.
This package offers both a classic dense method for small-scale analytics and a highly optimized sparse implementation designed to scale with larger data.

## Key Features

* **Two Approaches:** Provides a dense matrix solver for exact computation on smaller datasets, and a sparse solver for heavier workloads.
* **Optimized Memory Usage:** The sparse implementation eliminates the need to allocate full $N \times N$ distance matrices. It computes an exact k-nearest neighbors graph on the fly using a max-heap structure and stores the transition graph in Compressed Sparse Row (CSR) format, reducing memory complexity to $\mathcal{O}(N \cdot k)$.
* **rSVD solver:** Eigen-decompositions on matrices are done via custom Randomized Singular Value Decomposition (rSVD) solver. Most heavy computations are multithreaded with the use of OpenMP. Then LAPACK finds eigenvectors for a small `m x m` matrix where m is the requested number of dimensions.
* **Straightforwards Design:** Zero external dependencies. The codebase relies strictly on the native R C API (including LAPACK/BLAS), OpenMP, and the base R `stats` package, ensuring straightforward compilation.

## Installation

Install the development version directly from GitHub:

```R
install.packages("remotes")
remotes::install_github("hzalewski/cDiffusion")
```


## Usage

`cDiffusion` is designed to be easy in use. The API is built around three core functions:

1. **`estimate_sigma()`**: Computes the global Gaussian kernel width ($\sigma$) using the Median Heuristic by subsampling of the entry data.
2. **`run_diffusion()`**: The exact, dense solver. Perfect for capturing global topologies on small to medium datasets.
3. **`run_diffusion_sparse()`**: The sparse solver. Bypasses the need to allocate dense NxN euclidean distance matrix by taking only k - nearest neighbours for each point and converting those distances into Compressed Sparse Row (CSR) format. 

### 1. Dense Method


```r
library(cDiffusion)

data_matrix <- matrix(rnorm(6000), ncol = 3)

opt_sigma <- estimate_sigma(data_matrix, sample_size = 500)

model_dense <- run_diffusion(data_matrix, sigma = opt_sigma, n_iter = 50)

plot(model_dense$coordinates[, 1], model_dense$coordinates[, 2],
    pch = 16, col = "blue",
    main = "Diffusion map result",
    xlab = "Diffusion Coord 1", ylab = "Diffusion Coord 2")
```

### 2. Sparse Method

```R
library(cDiffusion)

data_matrix <- matrix(rnorm(30000 * 100), nrow = 30000, ncol = 100)

# Here we don't need to be concerned for the sigma parameter 
# since sparse method uses local sigma

model_sparse <- run_diffusion_sparse(
  data_matrix, 
  k_neighbors = 35,   
  oversampling = 10,   
  n_iter = 1000
)
```

## Output

 `cDiffusion` returns a standard R list with the `S3` class `diffmap`. This allows for immediate extraction of the results for tasks like clustering (e.g., `kmeans()`) or visualization.

Here is the exact structure of the output depending on the chosen method:

**Dense Method Output ($N = 2000$):**
```r
str(model_dense)
#> List of 5
#>  $ coordinates: num [1:2000, 1:2] -0.0691 -0.1416 0.0883 0.2579 -0.
#>  $ eigenvalues: num [1:2] 0.38 0.37
#>  $ sigma      : num 1.57
#>  $ data_dim   : int [1:2] 2000 3
#>  $ method     : chr "dense"
#>  - attr(*, "class")= chr "diffmap"
```
**Sparse Method Output ($N = 30000$):**
```R
str(model_sparse)
#> List of 4
#>  $ coordinates: num [1:30000, 1:2] -0.00457 0.01001 -0.02034 0.02162 
#>  $ eigenvalues: num [1:2] 1.65 1.65
#>  $ data_dim   : int [1:2] 30000 100
#>  $ method     : chr "sparse"
#>  - attr(*, "class")= chr "diffmap"
```

## Which Method to Choose?

Choosing between the dense and sparse engines is not only about memory limitations. Because they use different mathematical approaches (global Euclidean distances vs. local k-NN graphs), they interpret your data's geometry differently.

Use `run_diffusion()` (Dense / Global) when:
* **Dataset is small to medium $N < 10,000$**: Your system's memory permits allocating the full distance matrix.

* **Global Distances Matter**: You must preserve the exact relative distances between distant clusters (e.g., preserving the information that an extreme outlier is vastly separated from the rest of the data).

* **Risk of Disconnection**: You have highly separated, distinct clusters. A sparse k-NN graph might fracture them into completely disconnected islands, whereas the dense matrix connects everything (even with small weights), preserving the overall global topology.

* **Uniform Density**: Your dataset has a relatively consistent sampling density, making a single, global kernel width ($\sigma$) appropriate.

Use `run_diffusion_sparse()` (Sparse / Local) when:
* **Memory Efficiency** is required ($N \ge 10,000$). Dense distance matrices scale quadratically. For $N=30,000$, a dense solver allocates over 6.7 GB of RAM just to store pairwise distances. The sparse engine stores only the k-nearest neighbors in a CSR format, scaling linearly.

* **Varying Densities**: Your data contains a mix of tightly packed clusters and sparse, spread-out points. The sparse method uses an adaptive, local $\sigma$, preventing dense clusters from collapsing into a single point.

* **Complex Branching or Manifolds**: You are unrolling highly non-linear, continuous structures (like different trajectories or famous "Swiss Roll"). The k-NN graph perfectly follows the local curvature without jumping across empty Euclidean space.