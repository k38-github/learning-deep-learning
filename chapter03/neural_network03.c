#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"
#include "forward.h"

int main() {

    double *X, *Y;
    int n = 1;
    int m = 2;
    int l = 3;

    X = (double *)malloc(sizeof(double)*n*m);
    Y = (double *)malloc(sizeof(double)*n*m);

    // 1x2
    X[0] = 1.0; X[1] = 0.5;
    printf("X\n");
    print_matrix(X, n, m, "f");
    printf("\n");

    forward(X, Y);

    // 1x2
    printf("Y\n");
    print_matrix(Y, n, m, "f");
    printf("\n");

    free(X);
    free(Y);

    return 0;

}
