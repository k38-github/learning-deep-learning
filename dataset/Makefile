all: main 

main: mnist.o
	gcc -g -o main main.c mnist.o ../common/function.o -lm

mnist.o:
	gcc -g -c mnist.c mnist.h
run:
	./main

clean:
	rm -rf main
	rm -rf mnist.o
