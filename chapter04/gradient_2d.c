#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main(void) {

    double min = -2.0;
    double max = 2.5;
    double step = 0.25;
    double *x0;
    double *x1;

    int element = (fabs(min)+fabs(max))/step;

    x0 = malloc(sizeof(double) * element);
    x1 = malloc(sizeof(double) * element);

    array_range(min, max, step, x0);
    array_range(min, max, step, x1);

    double *X;
    double *Y;

    X = malloc(sizeof(double) * element * element);
    Y = malloc(sizeof(double) * element * element);

    meshgrid(x0, element, x1, element, X, Y);

    double array[2] = {0};
    int array_size = sizeof(array)/sizeof(double);
    double grad[2] = {0};

    double *X1;
    double *Y1;

    X1 = malloc(sizeof(double) * element * element);
    Y1 = malloc(sizeof(double) * element * element);

    int i;
    for (i=0;i<element*element;i++) {
        array[0] = X[i];
        array[1] = Y[i];

        numerical_gradient(function_2, array, array_size, grad);
        X1[i] = grad[0];
        Y1[i] = grad[1];
    }

    for (i=0;i<element*element;i++) {
        printf("%f %f %f %f\n", X[i], Y[i], -1*X1[i], -1*Y1[i]);
    }

    FILE *gp;
    gp = popen("gnuplot -persist", "w");
    fprintf(gp, "set grid\n");
    fprintf(gp, "set xrange [-2.0:2.0]\n");
    fprintf(gp, "set yrange [-2.0:2.0]\n");

    fprintf(gp, "plot '-' with vectors\n");

    double normalize = 0.0;
    for (i=0;i<element*element;i++) {
        normalize = sqrt(pow(X1[i], 2.0) + pow(Y1[i], 2.0));
        fprintf(gp, "%f %f %f %f\n", X[i], Y[i], -1*X1[i]/(normalize*5.0), -1*Y1[i]/(normalize*5.0));
    }

    fprintf(gp, "e\n");
    fprintf(gp, "exit\n");
    pclose(gp);

    free(x0);
    free(x1);
    free(X);
    free(Y);
    free(X1);
    free(Y1);

    return 0;
}
