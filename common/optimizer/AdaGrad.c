#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "AdaGrad.h"

int adagrad_init(AdaGrad *this, double lr) {
    this->lr = lr;

    return 0;
}

int adagrad_update(AdaGrad *this, double *param, double *grad, double *h, int size) {

    int i;
    for (i=0;i<size;i++) {
        h[i] += grad[i] * grad[i];
        param[i] -= this->lr * grad[i] / (sqrt(h[i]) + pow(10, -7.0));
    }

    return 0;
}
