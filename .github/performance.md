# Performance Benchmarks

## Benchmark Methodology & Specifications

To ensure a fair and rigorous comparison, the benchmarks were conducted under the following conditions:

### Hardware Specifications
* **CPU:** Intel Core i5-12400F (6 Cores / 12 Threads)
* **RAM:** 16 GB
* **Multithreading:** OpenMP was allowed to utilize all available CPU cores.

### Algorithm Parameters
We generated different datasets of varying observations ($N$) and input dimensions ($P$). We requested $d = 3$ diffusion dimensions for all methods. Data was generated from
standard normal distribution.

* **`cDiffusion`:**
  * Sparse method: `n_iter = 500` (for $N \le 5000$) or `1000` (for $N \ge 10000$).
  * Dense method: `n_iter = 50`.
  * `k_neighbors` scaled dynamically with $N$: $15$ ($N=2k$), $20$ ($N=5k$), $25$ ($N=10k$), and $30$ ($N=30k$).
* **`destiny`:**
  * To ensure a fair comparison with a sparse method, we forced `destiny` to use the exact same number of neighbors (`k = K_NEIGHBORS`).
  * These parameters where used to measure pure computational speed - `sigma = 10.0`, `density_norm = FALSE`.
* **`diffusionMap` (CRAN):**
  * **Note:** This package requires a pre-computed distance matrix. To give it an advantage, we pre-calculated `dist(data)` before starting the timer. The benchmark only measures its eigendecomposition time.

Each test was repeated 3 to 5 times using the `microbenchmark` package. The tables below show the **Median** execution time in seconds.

---

## 1. Small to Medium Datasets 

For smaller datasets, we compared both our Dense and Sparse methods against different R packages that implement diffusion maps.

### Test A: $N = 2000$
| Package / Method | $P = 3$ Dimensions | $P = 10$ Dimensions |
| :--- | :--- | :--- |
| **`cDiffusion` (Sparse)** | **0.100 s** | **0.099 s** |
| `destiny` (Bioconductor) | 0.115 s | 0.407 s |
| `cDiffusion` (Dense) | 0.242 s | 0.239 s |
| `diffusionMap` * | 0.695 s | 0.770 s |

### Test B: $N = 5000$
| Package / Method | $P = 3$ Dimensions | $P = 10$ Dimensions |
| :--- | :--- | :--- |
| **`cDiffusion` (Sparse)** | **0.216 s** | **0.253 s** |
| `destiny` (Bioconductor) | 0.449 s | 2.878 s |
| `cDiffusion` (Dense) | 1.435 s | 1.460 s |
| `diffusionMap` * | 6.761 s | 6.951 s |

*\* `diffusionMap` execution time does not include the time required to compute the $O(N^2)$ distance matrix.*

---

## 2. Larger Datasets 

For $N \ge 10000$, dense matrices begin to cause severe memory issues (a dense distance matrix for 30,000 points requires ~7.2 GB of RAM). Therefore, Dense methods were excluded, leaving a direct head-to-head comparison between the two Sparse implementations.

### Test C: $N = 10000$
| Package / Method | $P = 3$ Dimensions | $P = 10$ Dimensions |
| :--- | :--- | :--- |
| **`cDiffusion` (Sparse)** | **0.956 s** | **1.074 s** |
| `destiny` (Bioconductor) | 1.257 s | 10.933 s |

### Test D: $N = 30000$ 
| Package / Method | $P = 3$ Dimensions | $P = 10$ Dimensions |
| :--- | :--- | :--- |
| **`cDiffusion` (Sparse)** | **4.146 s** | **5.383 s** |
| `destiny` (Bioconductor) | 5.714 s | 104.625 s |

---

