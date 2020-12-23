#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../common/function.h"
#include "../dataset/mnist.h"

int main(void) {
    char *x_train;
    char *t_train;
    char *x_test;
    char *t_test;
    int size[4];

    load_mnist(&x_train, &t_train, &x_test, &t_test, size);

    int train_size = size[0];
    int row_size = 784;
    int batch_size = 10;
    int *index;

    index = (int *)malloc(sizeof(int) * batch_size);
    random_choice(train_size, row_size, batch_size, index);

    int i;
    for (i=0;i<batch_size;i++) {
        printf("%d\n", index[i]);
        view_label(t_train, index[i], 1);
        view_train(x_train, index[i], 1);
    }

    free(index);

    return 0;
}
