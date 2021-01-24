#include <stdio.h>
#include <stdlib.h>
#include "ReluLayer.h"

int relulayer_init(ReluLayer *this, int size) {
    this->size = size;
    this->mask = (int *)malloc(sizeof(int) * this->size);

    return 0;
}

int relulayer_free(ReluLayer *this) {

    free(this->mask);

    return 0;
}

int relulayer_forward(ReluLayer *this, double *out, double *x) {
    int i;
    for (i=0;i<this->size;i++) {
        if (x[i] <= 0) {
            this->mask[i] = 0;
            out[i] = 0;
        } else {
            this->mask[i] = 1;
            out[i] = x[i];
        }
    }

    return 0;
}

int relulayer_backward(ReluLayer *this, double *dx, double *dout) {
    int i;
    for (i=0;i<this->size;i++) {
        if (this->mask[i] == 0) {
            dx[i] = 0.0;
        } else {
            dx[i] = dout[i] * 1.0;
        }
    }

    return 0;
}
