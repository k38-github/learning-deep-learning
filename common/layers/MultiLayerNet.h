#ifndef _MULTILAYERNET
#define _MULTILAYERNET

#include "AffineLayer.h"
#include "ReluLayer.h"
#include "SigmoidLayer.h"
#include "SoftmaxWithLossLayer.h"

typedef struct Layers {
    AffineLayer *Affine;
    ReluLayer *Relu;
    SigmoidLayer *Sigmoid;
    SoftmaxWithLossLayer SoftmaxWithLoss;
}Layers;

typedef struct MultiLayerNet {
    double **W;
    double **b;
    int *all_size_list;
    int input_size;
    int *hidden_size_list;
    int hidden_layer_num;
    int output_size;
    int batch_size;
    double activation;
    double weight_init_std;
    double weight_decay_lambda;
    double **gW;
    double **gb;
    double *x_batch;
    double *t_batch;
    Layers layers;
}MultiLayerNet;

int multilayer_init(MultiLayerNet *, int, int *, int, int, int, char *, char *, double);
int multilayer_init_weight(MultiLayerNet *, char *);
//int layers_free(MultiLayerNet *);
int predict(MultiLayerNet *, double *, double *);
int loss(MultiLayerNet *, double *, double *, double *);
//int accuracy(MultiLayerNet *, double *, double *, int *);
int gradient(MultiLayerNet *, double *, double *);

#endif
