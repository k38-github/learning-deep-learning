#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int sum_function(double *y, double *sum_a, int element) {
    int i;

    for (i=0;i<element;i++) {
        *sum_a += y[i];
    }

    return 0;
}

int cross_entropy_error(double *y, double *t, double *E , int element) {
    int i;
    double delta = pow(10, -7.0);
    double *sum;
    double sum_a;

    sum = (double *)malloc(sizeof(double)*element);

    for (i=0;i<element;i++) {
        sum[i] = t[i] * log(y[i] + delta);
    }

    sum_function(sum, &sum_a, element);

    *E = -1 * sum_a;

    free(sum);

    return 0;
}
