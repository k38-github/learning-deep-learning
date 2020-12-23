#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main(void) {
    double min = 0.0;
    double max = 20.0;
    double step = 0.1;
    double *x;
    double *y;

    int element = (fabs(min)+fabs(max))/step;

    x = malloc(sizeof(double) * element);
    y = malloc(sizeof(double) * element);

    array_range(min, max, step, x);

    int i;
    for (i=0;i<element;i++) {
        y[i] = function_1(x[i]);
    }

    plot_graph(x, y, element);

    double dx;
    numerical_diff(function_1, 5.0, &dx);
    printf("%12.8f\n", dx);

    for (i=0;i<element;i++) {
        numerical_diff(function_1, x[i], &y[i]);
    }
    plot_graph(x, y, element);

    numerical_diff(function_1, 10.0, &dx);
    printf("%12.8f\n", dx);

    free(x);
    free(y);

    return 0;
}
