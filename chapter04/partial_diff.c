#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main(void) {

    double dx;
    numerical_diff(function_tmp1, 3.0, &dx);
    printf("%12.8f\n", dx);

    numerical_diff(function_tmp2, 4.0, &dx);
    printf("%12.8f\n", dx);

    return 0;
}
