#ifndef _SIMPLENET
#define _SIMPLENET

typedef struct simpleNet {
    double *W;
    int n;
    int m;
}simpleNet;

int init(simpleNet *);
int predict(simpleNet *, double *, double *, int);
//int loss(simpleNet *, double *, double *);

#endif
