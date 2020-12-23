#ifndef _MNIST
#define _MNIST

int load_mnist(char **, char **, char **, char **, int *);
int read_mnist(char *, char **, int *);
int normalize(char *, double *, int);
int one_hot(char *, int *, int);
int view_train(char *, int, int);
int view_label(char *, int, int);
int open_pgm_image_file(char *, int, char *);

#endif
