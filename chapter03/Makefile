all: main dot neural neural02 neural03 softmax_test

main:
	gcc -g -o main main.c ../common/function.o -lm

dot:
	gcc -g -o dot dot.c ../common/function.o -lm

neural:
	gcc -g -o neural_network01 neural_network01.c ../common/function.o -lm

neural02:
	gcc -g -o neural_network02 neural_network02.c ../common/function.o -lm

neural03: forward.o
	gcc -g -o neural_network03 neural_network02.c ../common/function.o forward.o -lm

softmax_test:
	gcc -g -o softmax_test softmax_test.c ../common/function.o -lm

forward.o:
	gcc -g -c forward.c forward.h

clean:
	rm -rf main
	rm -rf dot
	rm -rf neural_network01
	rm -rf neural_network02
	rm -rf neural_network03
	rm -rf softmax_test
	rm -rf mean_squared_error
	rm -rf forward.h.gch
	rm -rf forward.o
