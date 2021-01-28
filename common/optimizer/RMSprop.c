#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "RMSprop.h"

int rmsprop_init(RMSprop *this, double lr, double decay_rate) {
    this->lr = lr;
    this->decay_rate = decay_rate;

    return 0;
}

int rmsprop_update(RMSprop *this, double *param, double *grad, double *h, int size) {

    int i;
    for (i=0;i<size;i++) {
        h[i] *= this->decay_rate;
        h[i] += (1 - this->decay_rate) * grad[i] * grad[i];
        param[i] -= this->lr * grad[i] / (sqrt(h[i]) + pow(10, -7.0));
    }

    return 0;
}
