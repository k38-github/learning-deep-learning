#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../common/function.h"
#include "../common/optimizer/SGD.h"
#include "../common/optimizer/Momentum.h"
//#include "../common/optimizer/Nesterov.h"
#include "../common/optimizer/AdaGrad.h"
//#include "../common/optimizer/RMSprop.h"
#include "../common/optimizer/Adam.h"
#include "../dataset/mnist.h"

int plot_contour(double *, double *, char *);

int f(double *ret, double x, double y) {
    *ret =  pow(x, 2.0) / 20.0 + pow(y, 2.0);

    return 0;
}

int df(double *x_ret, double *y_ret, double x, double y) {
    *x_ret = x / 10.0;
    *y_ret = 2.0 * y;

    return 0;
}

int main(void) {
    double init_pos_x = -7.0;
    double init_pos_y = 2.0;

    double param_x[1] = {init_pos_x};
    double param_y[1] = {init_pos_y};

    double grad_x[1] = {0};
    double grad_y[1] = {0};

    double x_history[30] = {0};
    double y_history[30] = {0};

    // SGD
    double learning_rate = 0.95;
    SGD sgd_optimizer;
    sgd_init(&sgd_optimizer, learning_rate);

    int i;
    for (i=0;i<30;i++) {
        //printf("x:%f y:%f\n", param_x[0], param_y[0]);
        x_history[i] = param_x[0];
        y_history[i] = param_y[0];

        df(&grad_x[0], &grad_y[0], param_x[0], param_y[0]);
        sgd_update(&sgd_optimizer, param_x, grad_x, 1);
        sgd_update(&sgd_optimizer, param_y, grad_y, 1);
    }

    plot_contour(x_history, y_history, "SGD");

    // Momentum
    param_x[0] = init_pos_x;
    param_y[0] = init_pos_y;
    grad_x[0] = 0.0;
    grad_y[0] = 0.0;

    learning_rate = 0.1;
    double momentum = 0.9;
    double v[2] = {0};
    Momentum momentum_optimizer;
    momentum_init(&momentum_optimizer, learning_rate, momentum);

    for (i=0;i<30;i++) {
        //printf("x:%f y:%f\n", param_x[0], param_y[0]);
        x_history[i] = param_x[0];
        y_history[i] = param_y[0];

        df(&grad_x[0], &grad_y[0], param_x[0], param_y[0]);
        momentum_update(&momentum_optimizer, param_x, grad_x, &v[0], 1);
        momentum_update(&momentum_optimizer, param_y, grad_y, &v[1], 1);
    }

    plot_contour(x_history, y_history, "Momentum");

    // Adagrad
    param_x[0] = init_pos_x;
    param_y[0] = init_pos_y;
    grad_x[0] = 0.0;
    grad_y[0] = 0.0;

    learning_rate = 1.5;
    double h[2] = {0};
    AdaGrad adagrad_optimizer;
    adagrad_init(&adagrad_optimizer, learning_rate);

    for (i=0;i<30;i++) {
        //printf("x:%f y:%f\n", param_x[0], param_y[0]);
        x_history[i] = param_x[0];
        y_history[i] = param_y[0];

        df(&grad_x[0], &grad_y[0], param_x[0], param_y[0]);
        adagrad_update(&adagrad_optimizer, param_x, grad_x, &h[0], 1);
        adagrad_update(&adagrad_optimizer, param_y, grad_y, &h[1], 1);
    }

    plot_contour(x_history, y_history, "AraGrad");

    // Adam
    param_x[0] = init_pos_x;
    param_y[0] = init_pos_y;
    grad_x[0] = 0.0;
    grad_y[0] = 0.0;

    learning_rate = 0.3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double m[2] = {0};
    v[0] = 0.0; v[1] = 0.0;
    Adam adam_optimizer;
    adam_init(&adam_optimizer, learning_rate, beta1, beta2);

    double params[2] = {0};
    double grads[2] = {0};

    for (i=0;i<30;i++) {
        //printf("x:%f y:%f\n", param_x[0], param_y[0]);
        x_history[i] = param_x[0];
        y_history[i] = param_y[0];

        df(&grad_x[0], &grad_y[0], param_x[0], param_y[0]);

        params[0] = param_x[0]; params[1] = param_y[0];
        grads[0] = grad_x[0]; grads[1] = grad_y[0];

        adam_update(&adam_optimizer, params, grads, m, v, 2);
        //adam_update(&adam_optimizer, param_x, grad_x, &m[0], &v[0], 1);
        //adam_update(&adam_optimizer, param_y, grad_y, &m[1], &v[1], 1);

        param_x[0] = params[0]; param_y[0] = params[1];
    }

    plot_contour(x_history, y_history, "Adam");

    return 0;
}

int plot_contour(double *x_history, double *y_history, char *name) {

    double x_min = -10.0;
    double x_max = 10.0;
    double x_step = 0.01;
    int    x_size = (fabs(x_min)+fabs(x_max))/x_step;
    double *x = (double *)malloc(sizeof(double) * x_size);

    double y_min = -5.0;
    double y_max = 5.0;
    double y_step = 0.01;
    int    y_size = (fabs(y_min)+fabs(y_max))/y_step;
    double *y = (double *)malloc(sizeof(double) * y_size);

    array_range(x_min, x_max, x_step, x);
    array_range(y_min, y_max, y_step, y);

    double *X = (double *)malloc(sizeof(double) * x_size * y_size);
    double *Y = (double *)malloc(sizeof(double) * x_size * y_size);
    double *Z = (double *)malloc(sizeof(double) * x_size * y_size);

    meshgrid(x, x_size, y, y_size, X, Y);

    int i;
    for (i=0;i<x_size*y_size;i++) {
        f(&Z[i], X[i], Y[i]);

        if (Z[i] > 7) {
            Z[i] = 0;
        }
    }


    FILE *file;

    file = fopen("out.txt", "w");
    if (file == NULL) {
        printf("can't open\n");
        exit(1);
    }

    for (i=0;i<x_size*y_size;i++) {
        if (i != 0 && i%x_size == 0) {
            fprintf(file, "\n");
        }

        fprintf(file, "%f %f %f\n", X[i], Y[i], Z[i]);
    }

    fclose(file);


    FILE *gp;
    gp = popen("gnuplot -persist", "w");
    fprintf(gp, "set title \"%s\"\n", name);
    fprintf(gp, "set multiplot\n");
    fprintf(gp, "set view 0,0\n");
    fprintf(gp, "set contour\n");
    fprintf(gp, "set cntrparam levels incremental 0,1,100\n");
    fprintf(gp, "unset surface\n");
    fprintf(gp, "set xrange [-10.0:10.0]\n");
    fprintf(gp, "set yrange [-10.0:10.0]\n");

    fprintf(gp, "splot 'out.txt' using 1:2:3 with lines\n");

    fprintf(gp, "unset contour\n");
    fprintf(gp, "set surface\n");
    fprintf(gp, "splot '-' with lines linetype 1 lc rgb \"red\"\n");
    for (i=0;i<30;i++) {
        fprintf(gp, "%f %f %f\n", x_history[i], y_history[i], 0.0);
    }
    fprintf(gp, "e\n");

    fprintf(gp, "splot '-' with point pointtype 1\n");
    fprintf(gp, "%f %f %f\n", 0.0, 0.0, 0.0);
    fprintf(gp, "e\n");

    fprintf(gp, "exit\n");
    pclose(gp);

    free(x);
    free(y);
    free(X);
    free(Y);
    free(Z);


    return 0;
}
