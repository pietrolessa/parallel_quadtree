#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>              // Para medir tempo e usar OpenMP
#include "quadtree.h"
#include "aabb.h"

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

int main() {
    printf("Iniciando inserção de pontos na Quadtree...\n");

    // Cria a árvore cobrindo de (0,0) até (1000,1000)
    qtree tree = qtree_new(500, 500, 1000, 1000, point_in_range);
    qtree_setMaxNodeCnt(tree, 1); // Número máximo de pontos por nó

    // Gerar pontos aleatórios
    Point *points = malloc(sizeof(Point) * NUM_POINTS);
    for (int i = 0; i < NUM_POINTS; i++) {
        points[i].x = (float)(rand() % 1000);
        points[i].y = (float)(rand() % 1000);
    }

    // Medir tempo com omp_get_wtime()
    double start = omp_get_wtime();

    for (int i = 0; i < NUM_POINTS; i++) {
        qtree_insert(tree, &points[i]);
    }

    double end = omp_get_wtime();
    printf("Tempo de inserção: %.6f segundos\n", end - start);

    // Limpeza
    qtree_free(tree);
    free(points);

    return 0;
}
