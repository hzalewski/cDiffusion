#ifndef KNN_H
#define KNN_H


typedef struct {
    double dist;
    int idx;
} knn_node;


void down_heap(knn_node* heap, int i, int k_neighbors);
void build_heap(knn_node* heap, int k_neighbors);
void insert(knn_node* heap, int k_neighbors, int new_idx, double new_dist);

#endif 