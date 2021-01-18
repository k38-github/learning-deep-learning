all: train_neuralnet TwoLayerNet.o MulLayer.o AddLayer.o buy_apple buy_apple_orange

buy_apple: MulLayer.o AddLayer.o
	gcc -g -o buy_apple buy_apple.c MulLayer.o AddLayer.o
buy_apple_orange: MulLayer.o AddLayer.o
	gcc -g -o buy_apple_orange buy_apple_orange.c MulLayer.o AddLayer.o
MulLayer.o:
	gcc -g -c MulLayer.c MulLayer.h
AddLayer.o:
	gcc -g -c AddLayer.c AddLayer.h
TwoLayerNet.o:
	gcc -g -c TwoLayerNet.c TwoLayerNet.h
train_neuralnet: TwoLayerNet.o
	gcc -g -o train_neuralnet train_neuralnet.c TwoLayerNet.o ../common/function.o ../common/layers/AffineLayer.o ../common/layers/SigmoidLayer.o ../common/layers/ReluLayer.o ../common/layers/SoftmaxWithLossLayer.o ../dataset/mnist.o -lm

arun:
	./buy_apple
aorun:
	./buy_apple_orange
run:
	./train_neuralnet

clean:
	rm -rf train_neuralnet
	rm -rf TwoLayerNet.o
	rm -rf MulLayer.o
	rm -rf AddLayer.o
	rm -rf buy_apple
	rm -rf buy_apple_orange
