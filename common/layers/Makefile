all: SoftmaxWithLossLayer.o ReluLayer.o SigmoidLayer.o AffineLayer.o MultiLayerNet.o Dropout.o BatchNormalization.o MultiLayerNetExtend.o

SoftmaxWithLossLayer.o:
	gcc -g -c SoftmaxWithLossLayer.c SoftmaxWithLossLayer.h
ReluLayer.o:
	gcc -g -c ReluLayer.c ReluLayer.h
SigmoidLayer.o:
	gcc -g -c SigmoidLayer.c SigmoidLayer.h
AffineLayer.o:
	gcc -g -c AffineLayer.c AffineLayer.h
MultiLayerNet.o:
	gcc -g -c MultiLayerNet.c MultiLayerNet.h
Dropout.o:
	gcc -g -c Dropout.c Dropout.h
BatchNormalization.o:
	gcc -g -c BatchNormalization.c BatchNormalization.h
MultiLayerNetExtend.o:
	gcc -g -c MultiLayerNetExtend.c MultiLayerNetExtend.h

clean:
	rm -rf SoftmaxWithLossLayer.o
	rm -rf ReluLayer.o
	rm -rf SigmoidLayer.o
	rm -rf AffineLayer.o
	rm -rf MultiLayerNet.o
	rm -rf Dropout.o
	rm -rf BatchNormalization.o
	rm -rf MultiLayerNetExtend.o
