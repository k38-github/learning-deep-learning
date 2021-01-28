all: SGD.o Momentum.o Nesterov.o AdaGrad.o RMSprop.o Adam.o

SGD.o:
	gcc -g -c SGD.c SGD.h

Momentum.o:
	gcc -g -c Momentum.c Momentum.h

Nesterov.o:
	gcc -g -c Nesterov.c Nesterov.h
AdaGrad.o:
	gcc -g -c AdaGrad.c AdaGrad.h
RMSprop.o:
	gcc -g -c RMSprop.c RMSprop.h
Adam.o:
	gcc -g -c Adam.c Adam.h

clean:
	rm -rf SGD.o
	rm -rf Momentum.o
	rm -rf Nesterov.o
	rm -rf AdaGrad.o
	rm -rf RMSprop.o
	rm -rf Adam.o
