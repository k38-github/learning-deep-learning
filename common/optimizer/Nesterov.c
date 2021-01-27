#include <stdio.h>
#include <stdlib.h>
#include "Nesterov.h"

int nesterov_init(Nesterov *this, double lr, double momentum) {
    this->lr = lr;
    this->momentum = momentum;

    return 0;
}

int nesterov_update(Nesterov *this, double *param, double *grad, double *v, int size) {

    int i;
    for (i=0;i<size;i++) {
        param[i] += this->momentum * this->momentum * v[i];
        param[i] -= (1 + this->momentum) * this->lr * grad[i];
        v[i] *= this->momentum;
        v[i] -= this->lr * grad[i];
    }

    return 0;
}
