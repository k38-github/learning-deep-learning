all: MulLayer.o AddLayer.o buy_apple buy_apple_orange

buy_apple: MulLayer.o AddLayer.o
	gcc -g -o buy_apple buy_apple.c MulLayer.o AddLayer.o
buy_apple_orange: MulLayer.o AddLayer.o
	gcc -g -o buy_apple_orange buy_apple_orange.c MulLayer.o AddLayer.o
MulLayer.o:
	gcc -g -c MulLayer.c MulLayer.h
AddLayer.o:
	gcc -g -c AddLayer.c AddLayer.h

arun:
	./buy_apple
aorun:
	./buy_apple_orange

clean:
	rm -rf MulLayer.o
	rm -rf AddLayer.o
	rm -rf buy_apple
	rm -rf buy_apple_orange
