#include <stdio.h>
#include <stdlib.h>
#include "Momentum.h"

int momentum_init(Momentum *this, double lr, double momentum) {
    this->lr = lr;
    this->momentum = momentum;

    return 0;
}

int momentum_update(Momentum *this, double *param, double *grad, double *v, int size) {

    int i;
    for (i=0;i<size;i++) {
        v[i] = this->momentum * v[i] - this->lr * grad[i];
        param[i] += v[i];
    }

    return 0;
}
