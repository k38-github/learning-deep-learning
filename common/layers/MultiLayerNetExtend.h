#ifndef _MULTILAYERNET
#define _MULTILAYERNET

#include "AffineLayer.h"
#include "BatchNormalization.h"
#include "ReluLayer.h"
#include "SigmoidLayer.h"
#include "Dropout.h"
#include "SoftmaxWithLossLayer.h"

typedef struct Layers {
    AffineLayer *Affine;
    BatchNormalization *BatchNormalization;
    ReluLayer *Relu;
    SigmoidLayer *Sigmoid;
    Dropout *Dropout;
    SoftmaxWithLossLayer SoftmaxWithLoss;
}Layers;

typedef struct MultiLayerNetExtend {
    double **W;
    double **b;
    int *all_size_list;
    int input_size;
    int *hidden_size_list;
    int hidden_layer_num;
    int output_size;
    int batch_size;
    char *activation;
    double weight_init_std;
    double weight_decay_lambda;
    char *use_dropout;
    char *use_batchnorm;
    double **gW;
    double **gb;
    double *x_batch;
    double *t_batch;
    Layers layers;
}MultiLayerNetExtend;

int multilayerextend_init(MultiLayerNetExtend *, int, int *, int, int, int, char *, char *, double, char *, double, char *);
int multilayerextend_init_weight(MultiLayerNetExtend *, char *);
int multilayerextend_free(MultiLayerNetExtend *);
int predict(MultiLayerNetExtend *, double *, double *, char *);
int loss(MultiLayerNetExtend *, double *, double *, double *, char *);
//int accuracy(MultiLayerNet *, double *, double *, int *);
int gradient(MultiLayerNetExtend *, double *, double *);

#endif
