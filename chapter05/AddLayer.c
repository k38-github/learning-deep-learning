#include <stdio.h>
#include <stdlib.h>
#include "AddLayer.h"

int addlayer_init(AddLayer *this) {

    return 0;
}

int addlayer_forward(AddLayer *this, double *out, double x, double y) {
    *out = x + y;

    return 0;
}

int addlayer_backward(AddLayer *this, double *dx, double *dy, double dout) {
    *dx = dout * 1;
    *dy = dout * 1;

    return 0;
}
