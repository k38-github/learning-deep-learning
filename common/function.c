#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/**
 * 各軸の要素数に応じたグリッドを生成する(2次元)
 * x: 入力配列
 * x_size: 要素のサイズ
 * y: 入力配列
 * y_size: 取得したindex
 * X: 生成した配列
 * Y: 生成した配列
 **/
int meshgrid(double *x, int x_size, double *y, int y_size, double *X, double *Y) {

    int i, j, k;

    k = 0;
    for (i=0;i<y_size;i++) {
        for (j=0;j<x_size;j++) {
            X[k] = x[j];
            k++;
        }
    }

    k = 0;
    for (i=0;i<x_size;i++) {
        for (j=0;j<y_size;j++) {
            Y[k] = y[i];
            k++;
        }
    }

    return 0;
}

int random_randn(double *c, int row, int col) {

    double z = 0.0;
    double uniform = 0.0;
    double mu = 0.0;
    double sigma = 1.0;

    srand((unsigned int) time(NULL));

    int i, j;
    for (i=0;i<row;i++) {
        for (j=0;j<col;j++) {
            uniform = ((double)rand()+1.0)/((double)RAND_MAX+2.0);
            z = sqrt(-2.0*log(uniform)) * sin(2.0*M_PI*uniform);

            c[i*col+j] = mu + sigma * z;
        }
    }

    return 0;
}

/**
 * 指定された個数indexを取得する
 * data_size: 入力配列のサイズ
 * element_size: 要素のサイズ
 * batch_size: 必要なindexの個数
 * index: 取得したindex
 **/
int random_choice(int data_size, int element_size, int batch_size, int *index) {
    int i;
    int index_range = data_size / element_size;

    srand((unsigned int) time(NULL));

    for (i=0;i<batch_size;i++) {
        index[i] = rand() % index_range;
    }

    return 0;
}

/**
 * 行列の要素の最小値
 * x: 入力配列
 * y: 行列の要素の最小値
 * element: 要素数
 **/
int min_function(double *x, double *y, int element) {
    int i = 0;

    *y = x[i];
    for(i=0;i<element;i++) {
        if (x[i] < *y) {
            *y = x[i];
        }
    }

    return 0;
}

/**
 * 行列の要素の最大値
 * x: 入力配列
 * y: 行列の要素の最大値
 * element: 要素数
 **/
int max_function(double *x, double *y, int element) {
    int i;

    *y = x[i];
    for(i=0;i<element;i++) {
        if (*y < x[i]) {
            *y = x[i];
        }
    }

    return 0;
}

/**
 * 行列の要素の和
 * x: 入力配列
 * y: 行列の要素の和
 * element: 要素数
 **/
int sum_function(double *x, double *y, int element) {
    int i;

    for(i=0;i<element;i++) {
        *y += x[i];
    }

    return 0;
}

/**
 * 行列の和
 * a: 入力行列
 * b: 入力行列
 * c: 出力行列
 * row: 出力行のサイズ
 * col: 出力列のサイズ
 **/
int matrix_sum(double **c, double *a, double *b, int row, int col) {
    int i, j;

    for(i=0;i<row;i++) {
        for(j=0;j<col;j++) {
            (*c)[i*col+j] = 0.0;

            (*c)[i*col+j] = a[i*col+j]+b[i*col+j];
        }
    }

    return 0;
}

/**
 * 行列のドット積
 * a: 入力行列
 * b: 入力行列
 * c: 出力行列
 * row: 出力行のサイズ
 * mid: 中間のサイズ
 * col: 出力列のサイズ
 **/
int dot_function(double **c, double *a, double *b, int row, int mid, int col) {
    int i, j, k;

    for(i=0;i<row;i++) {
        for(j=0;j<col;j++) {
            (*c)[i*col+j] = 0.0;

            if (mid < col) {
                for(k=0;k<col;k++) {
                    (*c)[i*col+j] += a[i*row+k]*b[k*col+j];
                }
            } else {
                for(k=0;k<mid;k++) {
                    (*c)[i*col+j] += a[i*row+k]*b[k*col+j];
                }
            }
        }
    }

    return 0;
}

/**
 * 行列の中身を表示する
 * a: 入力行列
 * row: 行のサイズ
 * col: 列のサイズ
 * type: f(実数を出力する)/ e(実数を指数表示で出力する)
 **/
int print_matrix(double *a, int row, int col, char *type) {
    int i, j;

    for(i=0;i<row;i++) {
        for(j=0;j<col;j++) {
            if (strcmp(type, "e") == 0) {
                printf("%e ", a[i*col+j]);
            } else {
                printf("%11.8f ", a[i*col+j]);
            }
        }
        printf("\n") ;
    }

    return 0;
}

/**
 * step関数
 * x: step関数の入力配列
 * y: step関数の出力配列
 * element: 要素数
 **/
int step_function(double *x, double *y, int element) {
    int i;

    for (i=0;i<element;i++) {
        if (x[i] > 0) {
            y[i] = 1.0;
        } else {
            y[i] = 0;
        }
    }

    return 0;
}

/**
 * sigmoid関数
 * x: sigmoid関数の入力配列
 * y: sigmoid関数の出力配列
 * element: 要素数
 **/
int sigmoid_function(double *x, double *y, int element) {
    int i;

    for (i=0;i<element;i++) {
        y[i] = 1.0 / (1.0 + exp(-1 * x[i]));
        // printf("%f %f\n", x[i], y[i]);
    }

    return 0;
}

/**
 * relu関数
 * x: relu関数の入力配列
 * y: relu関数の出力配列
 * element: 要素数
 **/
int relu_function(double *x, double *y, int element) {
    int i;

    for (i=0;i<element;i++) {
        if (x[i] > 0) {
            y[i] = x[i];
        } else {
            y[i] = 0;
        }
    }

    return 0;
}

/**
 * 恒等関数
 * x: 恒等関数の入力配列
 * y: 恒等関数の出力配列
 * element: 要素数
 **/
int identity_function(double *x, double *y, int element) {
    int i;

    for (i=0;i<element;i++) {
        y[i] = x[i];
    }

    return 0;
}

/**
 * softmax関数 
 * x: softmax関数の入力配列
 * y: softmax関数の出力配列
 * element: 要素数
 **/
int softmax_function(double *x, double *y, int element) {
    int i;
    double *e;
    double sum_exp_a = 0.0;

    e = (double *)malloc(sizeof(double)*element);

    for (i=0;i<element;i++) {
        e[i] = exp(x[i]);
    }

    for (i=0;i<element;i++) {
        sum_exp_a += e[i];
    }

    for (i=0;i<element;i++) {
        y[i] = e[i] / sum_exp_a;
    }

    free(e);

    return 0;
}

/**
 * softmax関数のオーバーフロー対策版
 * x: softmax関数の入力配列
 * y: softmax関数の出力配列
 * element: 要素数
 **/
int softmax_measures_function(double *x, double *y, int element) {
    int i;
    double *e;
    double c = 0.0;
    double sum_exp_a = 0.0;


    max_function(x, &c, element);
    e = (double *)malloc(sizeof(double)*element);

    for (i=0;i<element;i++) {
        e[i] = exp(x[i] - c);
    }

    sum_function(e, &sum_exp_a, element);

    for (i=0;i<element;i++) {
        y[i] = e[i] / sum_exp_a;
    }

    free(e);

    return 0;
}

/**
 * 2乗和誤差を求める
 * y: ニューラルネットワークの出力配列
 * t: 教師データ配列
 * E: 2乗和誤差
 * element: 配列の要素数
 **/
int mean_squared_error(double *y, double *t, double *E, int element) {
    int i;
    double *sum;
    double sum_a = 0.0;

    sum = (double *)malloc(sizeof(double)*element);

    for (i=0;i<element;i++) {
        sum[i] = pow(y[i] - t[i], 2.0);
    }

    sum_function(sum, &sum_a, element);

    *E = 0.5 * sum_a;

    free(sum);

    return 0;
}

/**
 * 交差エントロピー誤差を求める
 * y: ニューラルネットワークの出力配列
 * t: 教師データ配列
 * E: 交差エントロピー誤差
 * element: 配列の要素数
 **/
int cross_entropy_error(double *y, double *t, double *E , int element) {
    int i;
    double delta = pow(10, -7.0);
    double *sum;
    double sum_a = 0.0;

    sum = (double *)malloc(sizeof(double)*element);

    for (i=0;i<element;i++) {
        sum[i] = t[i] * log(y[i] + delta);
    }

    sum_function(sum, &sum_a, element);

    *E = -1 * sum_a;

    free(sum);

    return 0;
}

/**
 * 微分する
 * 引数:double, 返り値:doubleの関数ポインタ
 * x: 変数
 * ret: 計算結果
 **/
int numerical_diff(double (*func)(double f), double x, double *ret) {
    double h = pow(10, -4);
    *ret = (func(x+h) - func(x-h)) / (2*h);
    return 0;
}

/**
 * 関数1
 * x: 変数
 **/
double function_1(double x) {
    return 0.01*pow(x, 2.0) + 0.1*x;
}

/**
 * 関数2
 * x: 変数
 **/
double function_tmp1(double x) {
    return x*x + pow(4.0, 2.0);
}

/**
 * 関数3
 * x: 変数
 **/
double function_tmp2(double x) {
    return pow(3.0, 2.0) + x*x;
}

/**
 * 接線の方程式
 * y - f(a) = f'(a)(x -a)
 * y = f'(a)x + (f(a) - f'(a)a)
 * 引数:double, 返り値:doubleの関数ポインタ
 * a: 関数f(a)の引数 
 * x: 接線の方程式の引数
 * ret: 計算結果
 **/
int tangent_line(double (*func)(double f), double a, double x, double *ret) {

    double dx = 0.0;

    numerical_diff(func, a, &dx);
    *ret = dx*x + (func(a) - dx*a);

    return 0;
}

/**
 * 偏微分用の関数
 * x: 変数
 * element: 要素数
 **/
double function_2(double *x, int element) {
    int i;
    double sum = 0.0;

    for (i=0;i<element;i++) {
        sum += pow(x[i], 2.0);
    }

    return sum;
}

/**
 * 勾配を求める
 * 引数:double, int, 返り値:doubleの関数ポインタ
 * x: 変数
 * element: 要素数
 * ret: 計算結果
 **/
int numerical_gradient(double (*func)(double *f, int e), double *x, int element, double *ret) {
    double h = pow(10, -4);
    double *grad;

    grad = (double *)calloc(element, sizeof(double));

    int idx = 0;
    double tmp_val = 0.0;
    double fxh1 = 0.0;
    double fxh2 = 0.0;

    for (idx=0;idx<element;idx++) {
        tmp_val = x[idx];
        x[idx] = tmp_val + h;
        fxh1 = func(x, element);
        //printf("fxh1: %f\n", fxh1);

        x[idx] = tmp_val - h;
        fxh2 = func(x, element);
        //printf("fxh2: %f\n", fxh2);

        grad[idx] = (fxh1 - fxh2) / (2*h);
        //printf("grad: %f\n", grad[idx]);

        x[idx] = tmp_val;
    }

    for (idx=0;idx<element;idx++) {
        ret[idx] = grad[idx];
    }

    free(grad);

    return 0;
}

/**
 * 勾配法
 * 引数:double, int, 返り値:doubleの関数ポインタ
 * init_x: 変数
 * element: 要素数
 * lr: 学習率
 * step_num: ステップ数
 * ret: 計算結果
 **/
int gradient_descent(double (*func)(double *f, int e), double *init_x, int element, double lr, int step_num, double *ret) {
    double *x;
    x = (double *)malloc(sizeof(double) * element);

    int i;
    for (i=0;i<element;i++) {
        x[i] = init_x[i];
    }

    double grad[2] = {0};
    for (i=0;i<step_num;i++) {
        numerical_gradient(func, x, element, grad);
        x[0] -= lr * grad[0];
        x[1] -= lr * grad[1];
        //printf("%03d: %e %e\n", i+1, x[0], x[1]);
    }

    for (i=0;i<element;i++) {
        ret[i] = x[i];
    }

    free(x);

    return 0;
}

/**
 * 指定された範囲の配列を生成
 * min: 表示する配列の最小値
 * max: 表示する配列の最大値
 * step: 最小値から最大値までの刻み幅
 * array: 活性化関数へ入力するx座標の配列
 **/
int array_range(double min, double max, double step, double *array) {

    int i = 0;
    double x = min;

    printf("%f %f %f\n", min, max, step);

    while (x <= max) {
    array[i] = x;
        x = x + step;
        i++;
    }

    return 0;
}

/**
 * グラフをプロットする
 * x: 表示するx座標の配列
 * y: 表示するy座標の配列
 * element: 要素数
 **/
int plot_graph(double *x, double *y, int element) {

    int i;
    FILE *gp;

    gp = popen("gnuplot -persist", "w");

    fprintf(gp, "plot '-' with lines linetype 1 title \"\"\n");

    for (i=0;i<element;i++) {
        fprintf(gp, "%f %f\n", x[i], y[i]);
    }

    fprintf(gp,"e\n");
    pclose(gp);

    return 0;
}

/**
 * ファイルポインタを受け取ってグラフをプロットする
 * 呼び出す側で色々設定する必要がある
   * FILE *gp;
   * gp = popen("gnuplot -persist", "w");
   * fprintf(gp, "set multiplot\n");
   * fprintf(gp, "set xrange [0.0:20.0]\n");
   * fprintf(gp, "set yrange [-1.0:6.0]\n");
   * ...
   * fprintf(gp, "set nomultiplot\n");
   * fprintf(gp, "exit\n");
   * pclose(gp);
 * gp: ファイルポインタ
 * x: 表示するx座標の配列
 * y: 表示するy座標の配列
 * element: 要素数
 **/

int plot_graph_f(FILE **gp, double *x, double *y, int element) {

    fprintf(*gp, "plot '-' with lines linetype 1 title \"\"\n");

    int i;
    for (i=0;i<element;i++) {
        fprintf(*gp, "%f %f\n", x[i], y[i]);
    }

    fprintf(*gp, "e\n");

    return 0;
}
