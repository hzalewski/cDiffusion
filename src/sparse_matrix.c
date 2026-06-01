#include "sparse_matrix.h"
#include "diffusion_core.h"
#include "knn.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>

double sq_dist(double *data, int n, int dim, int m1, int m2)
{
    double dist = 0.0;
    for (int d = 0; d < dim; d++)
    {
        double diff = data[d + m1 * dim] - data[d + m2 * dim];
        dist += diff * diff;
    }
    return dist;
}

// Builds a sparse k-NN matrix in CSR (Compressed Sparse Row) format.
// Uses max-heap for finding k-NN and transforms the matrix to be symmetric (maybe some other way to do it?).
void sparse(double *data, int n, int dim, int k, double sigma, double **out_data, int **out_indices, int **out_indptr)
{
    double denom = 2.0 * sigma * sigma;

    // initial k-NN results
    int *tmp_col = (int *)malloc(n * k * sizeof(int));
    double *tmp_val = (double *)malloc(n * k * sizeof(double));

    // k-NN for each point and applying kernel
#pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < n; j++)
    {
        knn_node *heap = (knn_node *)malloc(k * sizeof(knn_node));
        int i = 0, filled = 0;

        while (i < n && filled < k)  // First k elements are inserted automatically
        {
            heap[filled].idx = i;
            heap[filled].dist = sq_dist(data, n, dim, j, i);
            filled++;
            i++;
        }
        build_heap(heap, k); 

        for (; i < n; i++)    // max distance in heap gets replaced if a closer point gets found
        {
            double d_sq = sq_dist(data, n, dim, j, i);
            if (d_sq < heap[0].dist)
            {
                heap[0].idx = i;
                heap[0].dist = d_sq;
                down_heap(heap, 0, k);
            }
        }

        for (int p = 0; p < k; p++)
        {
            tmp_col[j * k + p] = heap[p].idx;
            tmp_val[j * k + p] = exp(-heap[p].dist / denom);
        }
        free(heap);
    }

    // Count only mutual edges (i->j and j->i) to calculate row pointers
    *out_indptr = (int *)calloc((n + 1), sizeof(int));

#pragma omp parallel for
    for (int j = 0; j < n; j++)
    {
        int valid_edges = 0;
        for (int p = 0; p < k; p++)
        {
            int neighbor = tmp_col[j * k + p];

            int is_mutual = 0;
            for (int m = 0; m < k; m++)
            {
                if (tmp_col[neighbor * k + m] == j)
                {
                    is_mutual = 1;
                    break;
                }
            }

            if (is_mutual)
                valid_edges++;
        }
        (*out_indptr)[j + 1] = valid_edges;
    }
    // Prefix sum - start of each row in CSR
    for (int i = 0; i < n; i++)
    {
        (*out_indptr)[i + 1] += (*out_indptr)[i];
    }

    int total_edges = (*out_indptr)[n];
    *out_data = (double *)malloc(total_edges * sizeof(double));
    *out_indices = (int *)malloc(total_edges * sizeof(int));


    // Allocating final arrays
#pragma omp parallel for
    for (int j = 0; j < n; j++)
    {
        int pos = (*out_indptr)[j];

        for (int p = 0; p < k; p++)
        {
            int neighbor = tmp_col[j * k + p];

            int is_mutual = 0;
            for (int m = 0; m < k; m++)
            {
                if (tmp_col[neighbor * k + m] == j)
                {
                    is_mutual = 1;
                    break;
                }
            }

            if (is_mutual)
            {
                (*out_indices)[pos] = neighbor;
                (*out_data)[pos] = tmp_val[j * k + p];
                pos++;
            }
        }
    }

    free(tmp_col);
    free(tmp_val);
}

void sparse_multiplication(int n, int m, double *val, int *col, int *row_ptr, double *X, double *Y)
{
#pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < n; row++)
    {
        int start = row_ptr[row];
        int end = row_ptr[row + 1];

        for (int j = 0; j < m; j++)
        {
            double sum = 0.0;
            for (int p = start; p < end; p++)
            {
                sum += val[p] * X[col[p] + j * n];
            }
            Y[row + j * n] = sum;
        }
    }
}

//W_ij = W_ij / (q_i * q_j), then D_i = sqrt(sum_j W_ij), W_ij = W_ij / (D_i * D_j)
void sparse_normalization(double *val, int *col, int *row_ptr, double *D_sqrt, int n)
{
    double *q_inv = (double *)malloc(n * sizeof(double));
    double *D_inv = (double *)malloc(n * sizeof(double));

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
            sum += val[p];
        q_inv[i] = 1.0 / sum;
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        double qi = q_inv[i];
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
        {
            val[p] *= (qi * q_inv[col[p]]);
        }
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
            sum += val[p];
        D_sqrt[i] = sqrt(sum);
        D_inv[i] = 1.0 / D_sqrt[i];
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        double di = D_inv[i];
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
        {
            val[p] *= (di * D_inv[col[p]]);
        }
    }
    free(q_inv);
    free(D_inv);
}

void sparse_rsvd(double *val, int *col, int *row_ptr, double *X, double *eigenvectors, double *eigenvalues, int n, int m, int n_iter)
{
    double *Y = (double *)malloc(n * m * sizeof(double));
    double *Y_mem = Y;

    for (int iter = 0; iter < n_iter; iter++)
    {
        sparse_multiplication(n, m, val, col, row_ptr, X, Y);

#pragma omp parallel for
        for (int i = 0; i < n * m; i++)
            Y[i] += X[i];

        ortogonalize(Y, n, m);

        double *temp = X;
        X = Y;
        Y = temp;
    }

    sparse_multiplication(n, m, val, col, row_ptr, X, Y);

#pragma omp parallel for
    for (int i = 0; i < n * m; i++)
        Y[i] += X[i];

    find_eigenvectors(X, Y, eigenvectors, eigenvalues, n, m);

    free(Y_mem);
}