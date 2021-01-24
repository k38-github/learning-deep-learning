#ifndef _SIGMOIDLAYER
#define _SIGMOIDLAYER

typedef struct SigmoidLayer {
    double *out;
    int size;
}SigmoidLayer;

int sigmoidlayer_init(SigmoidLayer *, int);
int sigmoidlayer_free(SigmoidLayer *);
int sigmoidlayer_forward(SigmoidLayer *, double *, double *);
int sigmoidlayer_backward(SigmoidLayer *, double *, double *);

#endif
