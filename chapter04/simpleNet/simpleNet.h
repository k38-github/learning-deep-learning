#ifndef _SIMPLENET
#define _SIMPLENET

typedef struct simpleNet {
    double *W;
    double *x;
    int x_col;
    int x_row;
    double *t;
    int t_col;
    int t_row;
}simpleNet;

int init(simpleNet *, double *, int, int, double *, int, int);
int predict(simpleNet *, double *, double *);
int loss(simpleNet *, double *, double *, int);

#endif
