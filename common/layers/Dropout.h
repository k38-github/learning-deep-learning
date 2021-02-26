#ifndef _DROPOUT
#define _DROPOUT

typedef struct Dropout {
    int *mask;
    double dropout_ratio;
    int col_size;
    int row_size;
}Dropout;

int dropout_init(Dropout *, double, int, int);
int dropout_free(Dropout *);
int dropout_forward(Dropout *, double *, double *, char *);
int dropout_backward(Dropout *, double *, double *);

#endif
