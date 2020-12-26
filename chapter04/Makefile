all: mean_squared_error cross_entropy_error random_choice gradient_1d partial_diff numerical_gradient_test gradient_2d gradient_test gradient_method random_randn

mean_squared_error:
	gcc -g -o mean_squared_error mean_squared_error.c ../common/function.o -lm
cross_entropy_error:
	gcc -g -o cross_entropy_error cross_entropy_error.c ../common/function.o -lm
random_choice:
	gcc -g -o random_choice random_choice.c ../common/function.o ../dataset/mnist.o -lm
gradient_1d:
	gcc -g -o gradient_1d gradient_1d.c ../common/function.o -lm
partial_diff:
	gcc -g -o partial_diff partial_diff.c ../common/function.o -lm
numerical_gradient_test:
	gcc -g -o numerical_gradient_test numerical_gradient_test.c ../common/function.o -lm
gradient_2d:
	gcc -g -o gradient_2d gradient_2d.c ../common/function.o -lm
gradient_test:
	gcc -g -o gradient_test gradient_test.c ../common/function.o -lm
gradient_method:
	gcc -g -o gradient_method gradient_method.c ../common/function.o -lm
random_randn:
	gcc -g -o random_randn random_randn.c ../common/function.o -lm

meanrun:
	./mean_squared_error
crossrun:
	./cross_entropy_error
choicerun:
	./random_choice
drun:
	./gradient_1d
prun:
	./partial_diff
nrun:
	./numerical_gradient_test
2drun:
	./gradient_2d
trun:
	./gradient_test
mrun:
	./gradient_method
rrun:
	./random_randn

clean:
	rm -rf mean_squared_error
	rm -rf cross_entropy_error
	rm -rf random_choice
	rm -rf gradient_1d
	rm -rf partial_diff
	rm -rf numerical_gradient_test
	rm -rf gradient_2d
	rm -rf gradient_test
	rm -rf gradient_method
	rm -rf random_randn
