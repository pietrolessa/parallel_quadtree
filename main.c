#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>              // Para medir tempo e usar OpenMP
#include "quadtree.h"
#include "aabb.h"

//extern double totalSubdivisionTime;

// Número de pontos a inserir na árvore
#define NUM_POINTS 100000

// Struct simples de ponto com coordenadas
typedef struct {
    float x, y;
} Point;

// Função de comparação usada pela quadtree
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
    printf("Iniciando inserção de pontos na Quadtree...\n");

    // Cria a árvore cobrindo de (0,0) até (1000,1000)
    qtree tree = qtree_new(500, 500, 1000, 1000, point_in_range);

    // Ativa suporte a múltiplas threads com OpenMP
    qtree_set_mutex(tree, omp_new_mutex, omp_lock, omp_unlock, omp_free_mutex);

    // Número máximo de pontos por nó antes de subdividir
    qtree_setMaxNodeCnt(tree, 4);

    // Gerar pontos aleatórios
    Point *points = malloc(sizeof(Point) * NUM_POINTS);
    for (int i = 0; i < NUM_POINTS; i++) {
        points[i].x = (float)(rand() % 1000);
        points[i].y = (float)(rand() % 1000);
    }

    // Medir tempo com omp_get_wtime()
    double start = omp_get_wtime();

    #pragma omp parallel for shared(tree, points)
    for (int i = 0; i < NUM_POINTS; i++) {
        qtree_insert(tree, &points[i]);
    }

    double end = omp_get_wtime();
    printf("Tempo de inserção: %.6f segundos\n", end - start);
   // printf("Tempo total apenas da subdivisão paralela: %.6f segundos\n", totalSubdivisionTime);

    // Limpeza
    qtree_free(tree);
    free(points);

    return 0;
}
