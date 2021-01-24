#ifndef _AFFINELAYER
#define _AFFINELAYER

typedef struct AffineLayer {
    double *W;
    double *b;
    double *x;
    int w_col_size;
    int w_row_size;
    int x_col_size;
    int x_row_size;
    double *dW;
    double *db;
}AffineLayer;

int affinelayer_init(AffineLayer *, double *, double *, int, int);
int affinelayer_free(AffineLayer *);
int affinelayer_forward(AffineLayer *, double *, double *, int, int);
int affinelayer_backward(AffineLayer *, double *, double *);

#endif
