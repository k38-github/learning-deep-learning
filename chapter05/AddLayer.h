#ifndef _ADDLAYER
#define _ADDLAYER

typedef struct AddLayer {
    double x;
    double y;
}AddLayer;

int addlayer_init(AddLayer *);
int addlayer_forward(AddLayer *, double *, double, double);
int addlayer_backward(AddLayer *, double *, double *, double);

#endif
