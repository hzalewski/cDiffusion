#include "cDiffusion.h"

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

void sparse(double *data, int n, int dim, int k_neighbours, double **out_csr_data, int **out_csr_indices, int **out_csr_indptr)
{

   
    double *tmp_dist_sq = malloc(n * k_neighbours * sizeof(double)); 
    double *local_sigmas = malloc(n * sizeof(double));
    int *tmp_col = malloc(n * k_neighbours * sizeof(int));

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < n; j++) {
        knn_node *heap = (knn_node *)malloc(k_neighbours * sizeof(knn_node));
        int i = 0, filled = 0;

        while (i < n && filled < k_neighbours) {
            heap[filled].idx = i;
            heap[filled].dist = sq_dist(data, n, dim, j, i);
            filled++; i++;
        }
        build_heap(heap, k_neighbours); 

        for (; i < n; i++) {
            double d_sq = sq_dist(data, n, dim, j, i);
            if (d_sq < heap[0].dist) {
                heap[0].idx = i;
                heap[0].dist = d_sq;
                down_heap(heap, 0, k_neighbours);
            }
        }
        // adaptive kernel - local sigma is the distance to k-th nearest neighbour 
        local_sigmas[j] = sqrt(heap[0].dist); 

        for (int p = 0; p < k_neighbours; p++) {
            tmp_col[j * k_neighbours + p] = heap[p].idx;
            tmp_dist_sq[j * k_neighbours + p] = heap[p].dist; 
        }
        free(heap);
    }

    *out_csr_indptr = calloc((n + 1), sizeof(int));

    // To create symmetry we add edges in both directions
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k_neighbours; p++) {
            int neighbor = tmp_col[j * k_neighbours + p];

            (*out_csr_indptr)[j + 1]++; // Edge fron j to neighbour

            int is_mutual = 0;
            for (int m = 0; m < k_neighbours; m++) {
                if (tmp_col[neighbor * k_neighbours + m] == j) {
                    is_mutual = 1; break;
                }
            }
            // Space for connection in other direction
            if (!is_mutual) {
                (*out_csr_indptr)[neighbor + 1]++;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        (*out_csr_indptr)[i + 1] += (*out_csr_indptr)[i];
    }

    // Final CSR arrays 
    int total_edges = (*out_csr_indptr)[n];
    *out_csr_data = malloc(total_edges * sizeof(double));
    *out_csr_indices = malloc(total_edges * sizeof(int));

    // current insertion position for each row
    int *current_pos = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) current_pos[i] = (*out_csr_indptr)[i];

    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k_neighbours; p++) {
            int neighbor = tmp_col[j * k_neighbours + p];
            double d_sq = tmp_dist_sq[j * k_neighbours + p]; 
            double weight = exp(-d_sq / (local_sigmas[j] * local_sigmas[neighbor]));

            int pos1;
            // insert edge
            pos1 = current_pos[j];
            current_pos[j]++; 
            
            (*out_csr_indices)[pos1] = neighbor;
            (*out_csr_data)[pos1] = weight;

            // if not mutual insert reverse edge
            int is_mutual = 0;
            for (int m = 0; m < k_neighbours; m++) {
                if (tmp_col[neighbor * k_neighbours + m] == j) {
                    is_mutual = 1; break;
                }
            }

            if (!is_mutual) {
                int pos2;

                pos2 = current_pos[neighbor]; 
                current_pos[neighbor]++; 
                
                (*out_csr_indices)[pos2] = j;
                (*out_csr_data)[pos2] = weight;
            }
        }
    }

    free(tmp_col);
    free(tmp_dist_sq);
    free(local_sigmas);
    free(current_pos);
}

void sparse_multiplication(int n, int m, double *csr_data, int *csr_indices, int *csr_indptr, double *X, double *Y)
{
#pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < n; row++)
    {
        int start = csr_indptr[row];
        int end = csr_indptr[row + 1];

        for (int j = 0; j < m; j++)
        {
            double sum = 0.0;
            for (int p = start; p < end; p++)
            {
                sum += csr_data[p] * X[csr_indices[p] + j * n]; 
            }
            Y[row + j * n] = sum;
        }
    }
}
// same logic as in regular matrix (diffusion_core.c)
void sparse_normalization(double *csr_data, int *csr_indices, int *csr_indptr, double *D_sqrt, int n)
{
    double *q_inv = malloc(n * sizeof(double));
    double *D_inv = malloc(n * sizeof(double));



#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for (int p = csr_indptr[i]; p < csr_indptr[i + 1]; p++)
            sum += csr_data[p];
        q_inv[i] = 1.0 / sum;
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        double qi = q_inv[i];
        for (int p = csr_indptr[i]; p < csr_indptr[i + 1]; p++)
        {
            csr_data[p] *= (qi * q_inv[csr_indices[p]]);
        }
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for (int p = csr_indptr[i]; p < csr_indptr[i + 1]; p++)
            sum += csr_data[p];
        D_sqrt[i] = sqrt(sum);
        D_inv[i] = 1.0 / D_sqrt[i];
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        double di = D_inv[i];
        for (int p = csr_indptr[i]; p < csr_indptr[i + 1]; p++)
        {
            csr_data[p] *= (di * D_inv[csr_indices[p]]);
        }
    }
    free(q_inv);
    free(D_inv);
}

// RSVD for CSR, same logic as for dense matrix
void sparse_rsvd(double *csr_data, int *csr_indices, int *csr_indptr, double *X, double *eigenvectors, double *eigenvalues, int n, int m, int n_iter)
{
    double *Y = malloc(n * m * sizeof(double));
    double *Y_mem = Y;

    for (int iter = 0; iter < n_iter; iter++)
    {
        sparse_multiplication(n, m, csr_data, csr_indices, csr_indptr, X, Y);

        #pragma omp parallel for
        for (int i = 0; i < n * m; i++)
            Y[i] += X[i];

        ortogonalize(Y, n, m);

        double *temp = X;
        X = Y;
        Y = temp;
    }

    sparse_multiplication(n, m, csr_data, csr_indices, csr_indptr, X, Y);

    #pragma omp parallel for
    for (int i = 0; i < n * m; i++)
        Y[i] += X[i];

    find_eigenvectors(X, Y, eigenvectors, eigenvalues, n, m);

    free(Y_mem);
}