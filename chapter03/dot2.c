#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main(int argc, char *argv[]) {

    double a[20] = {1.0, 2.0,
                    2.0, 2.0,
                    3.0, 2.0,
                    1.0, 2.0,
                    2.0, 2.0,
                    3.0, 4.0,
                    1.0, 4.0,
                    2.0, 4.0,
                    3.0, 4.0,
                    1.0, 4.0};

    double b[6] = {1.0, 2.0, 3.0,
                   1.0, 2.0, 3.0};
    double *c;

    int i;
    int j;
    int n = 10;
    int m = 2;
    int l = 3;

    c = (double *)malloc(sizeof(double)*n*l);

    dot_function(&c, a, b, n, m, l);
    print_matrix(a, n, m, "f");
    printf("\n");
    print_matrix(b, m, l, "f");
    printf("\n");
    print_matrix(c, n, l, "f");

    free(c);

    return 0 ;
}
