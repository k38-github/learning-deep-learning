#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main(int argc, char *argv[]) {

    double *a, *b, *c;
    int i;
    int j;
    int n = 2;
    int m = 3;

    a = (double *)malloc(sizeof(double)*n*n);
    b = (double *)malloc(sizeof(double)*n*m);
    c = (double *)malloc(sizeof(double)*n*m);

    a[0] = 1.0; a[1] = 2.0;
    a[2] = 3.0; a[3] = 4.0;

    b[0] = 5.0; b[1] = 6.0; b[2] = 7.0;
    b[3] = 8.0; b[4] = 9.0; b[5] = 10.0;

    dot_function(&c, a, b, n, n, m);
    print_matrix(a, n, n, "f");
    printf("\n");
    print_matrix(b, n, m, "f");
    printf("\n");
    print_matrix(c, n, m, "f");

    free(a);
    free(b);
    free(c);

    return 0 ;
}
