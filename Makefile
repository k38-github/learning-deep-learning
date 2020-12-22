all: cross_entroypy_error

cross_entroypy_error: function.o
	gcc -g -o cross_entropy_error cross_entropy_error.c function.o -lm
function.o:
	gcc -g -c function.c function.h -lm
run:
	./cross_entropy_error
clean:
	rm -rf function.o
	rm -rf cross_entropy_error
