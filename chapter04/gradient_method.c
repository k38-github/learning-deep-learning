#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main(void) {

    double init_x[2] = {-3.0, 4.0};
    int element = sizeof(init_x)/sizeof(double);
    double lr = 0.1;
    int step_num = 100;
    double ret[2] = {0};

    double *X;
    double *Y;

    X = malloc(sizeof(double) * step_num);
    Y = malloc(sizeof(double) * step_num);

    int i;
    for (i=0;i<step_num;i++) {
      gradient_descent(function_2, init_x, element, lr, i+1, ret);
      printf("%e %e\n", ret[0], ret[1]);
      X[i] = ret[0];
      Y[i] = ret[1];
    }

    FILE *gp;
    gp = popen("gnuplot -persist", "w");
    fprintf(gp, "set grid\n");
    fprintf(gp, "set xrange [-5.0:5.0]\n");
    fprintf(gp, "set yrange [-5.0:5.0]\n");

    fprintf(gp, "plot '-' with points pointtype 7\n");

    for (i=0;i<step_num;i++) {
        fprintf(gp, "%f %f\n", X[i], Y[i]);
    }

    fprintf(gp, "e\n");
    fprintf(gp, "exit\n");
    pclose(gp);

    free(X);
    free(Y);

    return 0;
}
