#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Adam.h"

int adam_init(Adam *this, double lr, double beta1, double beta2) {
    this->lr = lr;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->iter = 0;

    return 0;
}

int adam_update(Adam *this, double *param, double *grad, double *m, double *v, int size) {

    double lr_t = 0.0;

    this->iter += 1;
    lr_t = this->lr * sqrt(1.0 - pow(this->beta2, this->iter)) / (1.0 - pow(this->beta2, this->iter));

    int i;
    for (i=0;i<size;i++) {
        m[i] = this->beta1 * m[i] + (1 - this->beta1) * grad[i];
        v[i] = this->beta2 * v[i] + (1 - this->beta2) * pow(grad[i], 2.0);

        param[i] -= lr_t * m[i] / (sqrt(v[i]) + pow(10, -7.0));
    }

    return 0;
}
