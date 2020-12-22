#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"

int main() {
    char *x_train;
    char *t_train;
    char *x_test;
    char *t_test;
    int size[4];

    load_mnist(&x_train, &t_train, &x_test, &t_test, size);
    printf("size: %d\n", size[0]);
    printf("size: %d\n", size[1]);
    printf("size: %d\n", size[2]);
    printf("size: %d\n", size[3]);

    view_train(x_train, 3);

    double *X;
    X = (double *)malloc(sizeof(double) * size[0]);

    normalize(x_train, X, size[0]);

    int i, j;
    for (i=0;i<3;i++) {
        for (j=784*i;j<784*(i+1);j++) {
            if (j%28 == 0) {
                printf("\n");
            }
            printf("%f", X[j]);
        }
    }
    printf("\n");

    view_label(t_train, 3);

    int *T;
    T = (int *)malloc(sizeof(int) * size[1] * 10);

    one_hot(t_train, T, size[1]);

    for (i=0;i<3;i++) {
        for (j=10*i;j<10*(i+1);j++) {
            if (j%10 == 0) {
                printf("\n");
            }
            printf("%2d", T[j]);
        }
    }
    printf("\n");

    char *file = "out.pgm";
    open_pgm_image_file(file, 1, x_train);


    free(x_train);
    free(t_train);
    free(x_test);
    free(t_test);
    free(X);
    free(T);

    return 0;
}
