#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../common/function.h"
#include "../common/optimizer/SGD.h"
//#include "../common/optimizer/Momentum.h"
//#include "../common/optimizer/Nesterov.h"
//#include "../common/optimizer/AdaGrad.h"
//#include "../common/optimizer/RMSprop.h"
//#include "../common/optimizer/Adam.h"
#include "../dataset/mnist.h"

int f(double *ret, double x, double y) {
    *ret =  pow(x, 2.0) / 20.0 +pow(y, 2.0);

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
    double learning_rate = 0.1;
    SGD optimizer;
    sgd_init(&optimizer, learning_rate);

    int i;
    for (i=0;i<30;i++) {
        x_history[i] = param_x[0];
        y_history[i] = param_y[0];

        df(&grad_x[0], &grad_y[0], param_x[0], param_y[0]);
        sgd_update(&optimizer, param_x, grad_x, 1);
        sgd_update(&optimizer, param_y, grad_y, 1);

        printf("x:%f y:%f\n", param_x[0], param_y[0]);
    }

    return 0;
}
