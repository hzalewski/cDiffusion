#include "cDiffusion.h"


// Maintaining max-heap, largest distance at the root
void down_heap(knn_node* heap, int i, int k_neighbors) {
    knn_node v = heap[i];
    
  
    while (2 * i + 1 < k_neighbors) {
        int child = 2 * i + 1; 
       
        if (child + 1 < k_neighbors && heap[child + 1].dist > heap[child].dist) {
            child++;
        }
        
       
        if (v.dist >= heap[child].dist) {
            break;
        }
        
        
        heap[i] = heap[child];
        i = child;
    }
    
    heap[i] = v;
}


void build_heap(knn_node* heap, int k_neighbors) {
 
    for (int i = (k_neighbors / 2) - 1; i >= 0; i--) {
        down_heap(heap, i, k_neighbors);
    }
}

// inserting and restoring property
void insert(knn_node* heap, int k_neighbors, int new_idx, double new_dist) {
    if (new_dist < heap[0].dist) {
       
        heap[0].idx = new_idx;
        heap[0].dist = new_dist;
        
        down_heap(heap, 0, k_neighbors);
    }
}