all: main

main: simpleNet.o
	gcc -g -o main main.c simpleNet.o ../../common/function.o -lm
simpleNet.o:
	gcc -g -c simpleNet.c simpleNet.h

run:
	./main

clean:
	rm -rf main
	rm -rf simpleNet.o
