all: train_neuralnet TwoLayerNet.o

TwoLayerNet.o:
	gcc -g -c TwoLayerNet.c TwoLayerNet.h
train_neuralnet: TwoLayerNet.o
	gcc -g -o train_neuralnet train_neuralnet.c TwoLayerNet.o ../common/function.o ../common/layers/AffineLayer.o ../common/layers/SigmoidLayer.o ../common/layers/ReluLayer.o ../common/layers/SoftmaxWithLossLayer.o ../dataset/mnist.o ../common/optimizer/SGD.o ../common/optimizer/Momentum.o ../common/optimizer/Nesterov.o ../common/optimizer/AdaGrad.o ../common/optimizer/RMSprop.o ../common/optimizer/Adam.o -lm

run:
	./train_neuralnet

clean:
	rm -rf train_neuralnet
	rm -rf TwoLayerNet.o
