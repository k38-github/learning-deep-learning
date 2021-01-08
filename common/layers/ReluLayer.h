#ifndef _RELULAYER
#define _RELULAYER

typedef struct ReluLayer {
    int *mask;
    int size;
}ReluLayer;

int relulayer_init(ReluLayer *, int);
int relulayer_forward(ReluLayer *, double *, double *);
int relulayer_backward(ReluLayer *, double *, double *);

#endif
