#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main() {
    double *X, *W, *Y;
    int i;
    int j;
    int n = 1;
    int m = 2;
    int l = 3;

    X = (double *)malloc(sizeof(double)*n*l);
    W = (double *)malloc(sizeof(double)*l*m);
    Y = (double *)malloc(sizeof(double)*n*m);

    X[0] = 1.0; X[1] = 2.0; X[2] = 3.0;
    print_matrix(X, n, l, "f");
    printf("\n");

    W[0] = 1.0; W[1] = 2.0;
    W[2] = 3.0; W[3] = 4.0;
    W[4] = 5.0; W[5] = 6.0;
    print_matrix(W, l, m, "f");
    printf("\n");

    dot_function(&Y, X, W, n, l, m);
    print_matrix(Y, n, m, "f");
    printf("\n");

    free(X);
    free(W);
    free(Y);

    return 0;

}
