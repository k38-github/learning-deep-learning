all: mean_squared_error cross_entropy_error random_choice differential_test partial_diff numerical_gradient_test

mean_squared_error:
	gcc -g -o mean_squared_error mean_squared_error.c ../common/function.o -lm
cross_entropy_error:
	gcc -g -o cross_entropy_error cross_entropy_error.c ../common/function.o -lm
random_choice:
	gcc -g -o random_choice random_choice.c ../common/function.o ../dataset/mnist.o -lm
differential_test:
	gcc -g -o differential_test differential_test.c ../common/function.o -lm
partial_diff:
	gcc -g -o partial_diff partial_diff.c ../common/function.o -lm
numerical_gradient_test:
	gcc -g -o numerical_gradient_test numerical_gradient_test.c ../common/function.o -lm
meanrun:
	./mean_squared_error
crossrun:
	./cross_entropy_error
choicerun:
	./random_choice
drun:
	./differential_test
prun:
	./partial_diff
nrun:
	./numerical_gradient_test

clean:
	rm -rf mean_squared_error
	rm -rf cross_entropy_error
	rm -rf random_choice
	rm -rf differential_test
	rm -rf partial_diff
	rm -rf numerical_gradient_test
