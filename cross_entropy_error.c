#include <stdio.h>
#include <stdlib.h>
#include "function.h"

int main(void) {
    double t[10] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    double y[10] = {0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0};
    double E;
    int element = 10;

    cross_entropy_error(y, t, &E, element);
    printf("%20.18f\n", E);

    double y2[10] = {0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0};
    cross_entropy_error(y2, t, &E, element);

    printf("%20.18f\n", E);

    return 0;
}
