#ifndef _NESTEROV
#define _NESTEROV

// Nesterov's Accelerated Gradient
typedef struct Nesterov {
    double lr;
    double momentum;
}Nesterov;

int nesterov_init(Nesterov *, double, double);
int nesterov_update(Nesterov *, double *, double *, double *, int);

#endif
