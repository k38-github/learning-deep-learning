#ifndef _SOFTMAXWITHLOSSLAYER
#define _SOFTMAXWITHLOSSLAYER

typedef struct SoftmaxWithLossLayer {
    double *y;
    double *t;
    double *loss;
    int col_size;
    int row_size;
}SoftmaxWithLossLayer;

int softmaxwithlosslayer_init(SoftmaxWithLossLayer *, double *, int, int);
int softmaxwithlosslayer_forward(SoftmaxWithLossLayer *, double *, double *, int, int);
int softmaxwithlosslayer_backward(SoftmaxWithLossLayer *, double *, double *);

#endif
