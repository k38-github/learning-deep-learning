#include <stdio.h>
#include <stdlib.h>
#include "../common/function.h"

int main() {

    double *arr;
    int x = 3;
    int y = 3;

    arr = (double *)malloc(sizeof(double) * x * y);

    random_randn(arr, x, y);
    print_matrix(arr, x, y, "f");
    printf("\n");

    free(arr);

    return 0;
}
