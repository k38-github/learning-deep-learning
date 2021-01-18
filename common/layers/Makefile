all: SoftmaxWithLossLayer.o ReluLayer.o SigmoidLayer.o AffineLayer.o

SoftmaxWithLossLayer.o:
	gcc -g -c SoftmaxWithLossLayer.c SoftmaxWithLossLayer.h
ReluLayer.o:
	gcc -g -c ReluLayer.c ReluLayer.h
SigmoidLayer.o:
	gcc -g -c SigmoidLayer.c SigmoidLayer.h
AffineLayer.o:
	gcc -g -c AffineLayer.c AffineLayer.h

clean:
	rm -rf SoftmaxWithLossLayer.o
	rm -rf ReluLayer.o
	rm -rf SigmoidLayer.o
	rm -rf AffineLayer.o
