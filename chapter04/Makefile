all: cross_entropy_error

mean_squared_error:
	gcc -g -o mean_squared_error mean_squared_error.c ../common/function.o -lm
cross_entropy_error:
	gcc -g -o cross_entropy_error cross_entropy_error.c ../common/function.o -lm
meanrun:
	./mean_squared_error
crossrun:
	./cross_entropy_error
clean:
	rm -rf mean_squared_error
	rm -rf cross_entropy_error
