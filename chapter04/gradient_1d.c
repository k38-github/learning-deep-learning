#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main(void) {
    FILE *gp;
    gp = popen("gnuplot -persist", "w");
    fprintf(gp, "set multiplot\n");
    fprintf(gp, "set grid\n");
    fprintf(gp, "set xrange [0.0:20.0]\n");
    fprintf(gp, "set yrange [-1.0:6.0]\n");

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
    plot_graph_f(&gp, x, y, element);

    double dx;
    numerical_diff(function_1, 5.0, &dx);
    printf("%12.8f\n", dx);

    double *t;
    double *u;
    t = malloc(sizeof(double) * element);
    u = malloc(sizeof(double) * element);

    array_range(min, max, step, u);

    for (i=0;i<element;i++) {
        tangent_line(function_1, 5.0, u[i], &t[i]);
    }
    plot_graph_f(&gp, u, t, element);

    for (i=0;i<element;i++) {
        tangent_line(function_1, 10.0, u[i], &t[i]);
    }
    plot_graph_f(&gp, u, t, element);

    fprintf(gp, "set nomultiplot\n");
    fprintf(gp, "exit\n");

    free(x);
    free(y);
    free(t);
    free(u);
    pclose(gp);

    return 0;
}
