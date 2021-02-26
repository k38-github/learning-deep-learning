#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Dropout.h"
#include "../function.h"

int dropout_init(Dropout *this, double dropout_ratio, int col_size, int row_size) {
    this->dropout_ratio = dropout_ratio;
    this->mask = (int *)malloc(sizeof(int) * this->col_size * this->row_size);

    return 0;
}

int dropout_free(Dropout *this) {

    free(this->mask);

    return 0;
}

int dropout_forward(Dropout *this, double *out, double *x, char *train_flg) {
    int *tmp;

    if ((tmp = (int *)realloc(this->mask, sizeof(int) * this->col_size * this->row_size)) == NULL) {
        printf("Unable to allocate memory during realloc\n");
        exit(EXIT_FAILURE);
    } else {
        this->mask = tmp;
    }

    int i;
    double *tmp_mask;

    tmp_mask = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

    if (strcmp(train_flg, "true") == 0) {
        random_randn(tmp_mask, this->col_size, this->row_size);

        for (i=0;i<this->col_size*this->row_size;i++) {
            if (tmp_mask[i] < this->dropout_ratio) {
                this->mask[i] = 1;
            } else {
                this->mask[i] = 0;
            }
        }

        for (i=0;i<this->col_size*this->row_size;i++) {
            out[i] = x[i] * this->mask[i];
        }
    } else {
        for (i=0;i<this->col_size*this->row_size;i++) {
            out[i] = x[i] * (1.0 - this->dropout_ratio);
        }
    }

    free(tmp_mask);

    return 0;
}

int dropout_backward(Dropout *this, double *dx, double *dout) {

    int i;
    for (i=0;i<this->col_size*this->row_size;i++) {
        dx[i] = dout[i] * this->mask[i];
    }

    return 0;
}
