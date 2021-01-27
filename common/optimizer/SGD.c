#include <stdio.h>
#include <stdlib.h>
#include "SGD.h"

int sgd_init(SGD *this, double lr) {
    this->lr = lr;

    return 0;
}

int sgd_update(SGD *this, double *param, double *grad, int size) {

    int i;
    for (i=0;i<size;i++) {
        param[i] -= this->lr * grad[i];
    }

    return 0;
}
