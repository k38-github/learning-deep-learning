all: main

main: TwoLayerNet.o
	gcc -g -o main main.c TwoLayerNet.o ../../common/function.o ../../dataset/mnist.o -lm
TwoLayerNet.o:
	gcc -g -c TwoLayerNet.c TwoLayerNet.h

run:
	./main

clean:
	rm -rf main
	rm -rf TwoLayerNet.o
