all: SGD.o Momentum.o Nesterov.o

SGD.o:
	gcc -g -c SGD.c SGD.h

Momentum.o:
	gcc -g -c Momentum.c Momentum.h

Nesterov.o:
	gcc -g -c Nesterov.c Nesterov.h

clean:
	rm -rf SGD.o
	rm -rf Momentum.o
	rm -rf Nesterov.o
