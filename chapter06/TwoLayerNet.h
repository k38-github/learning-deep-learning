#ifndef _TWOLAYERNET
#define _TWOLAYERNET

#include "../common/layers/AffineLayer.h"
#include "../common/layers/ReluLayer.h"
#include "../common/layers/SoftmaxWithLossLayer.h"

typedef struct Layers {
    AffineLayer Affine1;
    ReluLayer   Relu1;
    AffineLayer Affine2;
    SoftmaxWithLossLayer SoftmaxWithLoss;
}Layers;

typedef struct TwoLayerNet {
    double *W1;
    double *b1;
    double *W2;
    double *b2;
    int input_size;
    int hidden_size;
    int output_size;
    int batch_size;
    double weight_init_std;
    double *gW1;
    double *gb1;
    double *gW2;
    double *gb2;
    double *x_batch;
    double *t_batch;
    Layers layers;
}TwoLayerNet;

int init(TwoLayerNet *, int, int, int, int, double);
int layers_free(TwoLayerNet *);
int predict(TwoLayerNet *, double *, double *);
int loss(TwoLayerNet *, double *, double *, double *);
int accuracy(TwoLayerNet *, double *, double *, int *);
int gradient(TwoLayerNet *, double *, double *);

#endif
