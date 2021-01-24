#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "SigmoidLayer.h"
#include "../function.h"

int sigmoidlayer_init(SigmoidLayer *this, int size) {
    this->size = size;
    this->out = (double *)malloc(sizeof(double) * this->size);

    return 0;
}

int sigmoidlayer_free(SigmoidLayer *this) {

    free(this->out);

    return 0;
}

int sigmoidlayer_forward(SigmoidLayer *this, double *out, double *x) {
    sigmoid_function(x, out, this->size);
    memcpy(this->out, out, sizeof(double) * this->size);

    return 0;
}

int sigmoidlayer_backward(SigmoidLayer *this, double *dx, double *dout) {
    int i;
    for (i=0;i<this->size;i++) {
        dx[i] = dout[i] * (1.0 - this->out[i]) * this->out[i];
    }

    return 0;
}
