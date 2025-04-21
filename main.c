#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "quadtree.h"
#include "aabb.h"

#define NUM_POINTS 100000

typedef struct {
    float x, y;
} Point;

int point_in_range(void *ptr, aabb *range) {
    Point *p = (Point *)ptr;
    return aabb_contains(range, p->x, p->y);
}

// Funções de mutex com OpenMP
void *omp_new_mutex() {
    omp_lock_t *lock = (omp_lock_t *)malloc(sizeof(omp_lock_t));
    omp_init_lock(lock);
    return (void *)lock;
}

int omp_lock(void *lock) {
    omp_set_lock((omp_lock_t *)lock);
    return 0;
}

int omp_unlock(void *lock) {
    omp_unset_lock((omp_lock_t *)lock);
    return 0;
}

int omp_free_mutex(void *lock) {
    omp_destroy_lock((omp_lock_t *)lock);
    free(lock);
    return 0;
}

int main() {
    printf("Iniciando inserção de pontos na Quadtree com múltiplas threads...\n");

    // Gerar pontos aleatórios
    Point *points = malloc(sizeof(Point) * NUM_POINTS);
    for (int i = 0; i < NUM_POINTS; i++) {
        points[i].x = (float)(rand() % 1000);
        points[i].y = (float)(rand() % 1000);
    }

    int num_threads = omp_get_max_threads();
    qtree *thread_trees = malloc(sizeof(qtree) * num_threads);

    for (int i = 0; i < num_threads; i++) {
        thread_trees[i] = qtree_new(500, 500, 1000, 1000, point_in_range);
        qtree_setMaxNodeCnt(thread_trees[i], 16);
        qtree_set_mutex(thread_trees[i], omp_new_mutex, omp_lock, omp_unlock, omp_free_mutex);
    }

    double startTime = omp_get_wtime();

    // Inserção paralela em árvores independentes
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int chunk = NUM_POINTS / num_threads;
        int start = tid * chunk;
        int end = (tid == num_threads - 1) ? NUM_POINTS : start + chunk;

        for (int i = start; i < end; i++) {
            qtree_insert(thread_trees[tid], &points[i]);
        }
    }

    // Mescla para árvore final
    qtree final_tree = qtree_new(500, 500, 1000, 1000, point_in_range);
    qtree_setMaxNodeCnt(final_tree, 16);
    qtree_set_mutex(final_tree, omp_new_mutex, omp_lock, omp_unlock, omp_free_mutex);

    for (int t = 0; t < num_threads; t++) {
        uint32_t count;
        void **elems = qtree_findInArea(thread_trees[t], 0, 0, 1000, 1000, &count);

        for (uint32_t i = 0; i < count; i++) {
            qtree_insert(final_tree, elems[i]);
        }

        free(elems);
        qtree_free(thread_trees[t]);
    }
    free(thread_trees);

    double end = omp_get_wtime();
    printf("Tempo total com paralelismo e merge: %.6f segundos\n", end - startTime);

    qtree_free(final_tree);
    free(points);
    return 0;
}
