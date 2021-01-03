#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

//#define DEBUG_ON

#define TRAIN_IMAGE "/tmp/data/train-images-idx3-ubyte"
#define TRAIN_LABEL "/tmp/data/train-labels-idx1-ubyte"
#define TEST_IMAGE  "/tmp/data/t10k-images-idx3-ubyte"
#define TEST_LABEL  "/tmp/data/t10k-labels-idx1-ubyte"


/**
 * INING SET LABEL FILE (train-labels-idx1-ubyte):
 * [offset] [type]          [value]          [description]
 * 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
 * 0004     32 bit integer  60000            number of items
 * 0008     unsigned byte   ??               label
 * 0009     unsigned byte   ??               label
 * ........
 * xxxx     unsigned byte   ??               label
 * The labels values are 0 to 9.
 **/

/**
 * TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
 * [offset] [type]          [value]          [description]
 * 0000     32 bit integer  0x00000803(2051) magic number
 * 0004     32 bit integer  60000            number of images
 * 0008     32 bit integer  28               number of rows
 * 0012     32 bit integer  28               number of columns
 * 0016     unsigned byte   ??               pixel
 * 0017     unsigned byte   ??               pixel
 * ........
 * xxxx     unsigned byte   ??               pixel
 * pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
 **/

void printb(unsigned int v) {
      unsigned int mask = (int)1 << (sizeof(v) * CHAR_BIT - 1);
        do putchar(mask & v ? '1' : '0');
          while (mask >>= 1);
}

void putb(unsigned int v) {
      putchar('0'), putchar('b'), printb(v), putchar('\n');
}

/**
 * mnistデータを読み込む
 * filePath: mnistデータファイルのパス
 * data: mnistデータから読み込んだデータ
 * return_size: 読み込んだデータのサイズ
 **/
int read_mnist(char *filePath, char **data, int *return_size) {
    FILE *fp;

    if ((fp = fopen(filePath, "rb")) == NULL) {
        fprintf(stderr, "ERROR: Cannot open %s\n", filePath);
        return 1;
    } else {
        printf("OK: File open %s\n", filePath);
    }

    // read magic number 0x00000803: image file
    // read magic number 0x00000801: label file
    char magic_number[4];
    int size = fread(magic_number, 1, 4, fp);
    if (size != 4) {
        fprintf(stderr, "ERROR: Read error %s\n", filePath);
        return 1;
    }
    if ((magic_number[0] != 0x00 || magic_number[1] != 0x00 || magic_number[2] != 0x08) &&
            (magic_number[3] != 0x01 || magic_number[3] != 0x03)) {
        fprintf(stderr, "ERROR: File format error %s\n", filePath);
        return 1;
    }
    printf("INFO: magic number=0x000000%x0%x\n", magic_number[2], magic_number[3]);
    printf("INFO: datatype=%d\n", (unsigned char)magic_number[2]);
    printf("INFO: diminsion=%d\n", (unsigned char)magic_number[3]);

    int dim = (unsigned char)magic_number[3];

    //read size in dimensions
    int i, j;
    char dims[4];
    int N[3] = {0};

    for (i=0;i<dim;i++) {
        size = fread(dims, 1, 4, fp);
        if (size != 4) {
            fprintf(stderr, "ERROR: Read error %s\n", filePath);
            return 1;
        }

        N[i] = 0;
        for (j=0;j<4;j++) {
#ifdef DEBUG_ON
            printf("before: %x %x\n", N[i], dims[j]);
            putb(N[i]);
            putb((unsigned char)dims[j]);
#endif
            N[i] = (N[i] << 8) | (dims[j] & 0xff);

#ifdef DEBUG_ON
            printf("after: %x %x\n", N[i], dims[j]);
            putb(N[i]);
            putb((unsigned char)dims[j]);
#endif
        }
        printf("INFO: dimension %d length=%d\n", i, N[i]);
    }

    // read data
    int totalsize = 1;
    for (i=0;i<dim;i++) {
        totalsize *= N[i];
    }

    *data = (char *)malloc(sizeof(char) * totalsize);
    if(*data == NULL) {
        fprintf(stderr, "ERROR: malloc\n");
        return 1;
    }

    size = fread(*data, 1, totalsize, fp);
    if (size != totalsize) {
        fprintf(stderr, "ERROR: Read error %s\n", filePath);
        return 1;
    }

    *return_size = size;

    //free(data);
    fclose(fp);

    printf("INFO: success!!\n");

    return 0;
}

/**
 * 読み込んだmnistデータを返却する
 * x_train: 訓練画像
 * t_train: 訓練ラベル
 * x_test: テスト画像
 * t_test: テストラベル
 * size: 読み込んだデータのサイズの配列
 *       size[0]:訓練画像サイズ
 *       size[1]:訓練ラベルサイズ
 *       size[2]:テスト画像サイズ
 *       size[3]:テストラベルサイズ
 **/
int load_mnist(char **x_train, char **t_train, char **x_test, char **t_test, int *size) {
    char *train_image = TRAIN_IMAGE;
    char *train_label = TRAIN_LABEL;
    char *test_image = TEST_IMAGE;
    char *test_label = TEST_LABEL;

    if (read_mnist(train_image, &*x_train, &size[0]) != 0) {
        fprintf(stderr, "ERROR: Cannot read train_image\n");
        return 1;
    }

    if (read_mnist(train_label, &*t_train, &size[1]) != 0) {
        fprintf(stderr, "ERROR: Cannot read train_label\n");
        return 1;
    }

    if (read_mnist(test_image, &*x_test, &size[2]) != 0) {
        fprintf(stderr, "ERROR: Cannot read test_image\n");
        return 1;
    }

    if (read_mnist(test_label, &*t_test, &size[3]) != 0) {
        fprintf(stderr, "ERROR: Cannot read test_label\n");
        return 1;
    }

    return 0;
}

/**
 * 画像データを0.0-1.0の値に正規化する
 * data: 画像データ
 * normalize: 正規化された画像データ
 * size: 画像データの大きさ
 **/
int normalize(char *data, double *normalize, int size) {

    int i;
    for (i=0;i<size;i++) {
        normalize[i] = (double)((unsigned char)data[i])/255;
    }

    return 0;
}

/**
 * ラベルデータをone_hot表現にする
 * data: ラベルデータ
 * one_hot: one_hot表現にしたラベルデータ
 * size: ラベルデータの大きさ
 **/
int one_hot(char *data, int *one_hot, int size) {

    int i, j;
    int k = 0;
    for (i=0;i<size;i++) {
        for (j=0;j<10;j++) {
            if (data[i] == j) {
                one_hot[k] = 1;
                k++;
            } else {
                one_hot[k] = 0;
                k++;
            }
        }
    }

    return 0;
}

/**
 * 読み込んだ画像データの中身を表示する
 * (28x28に整形して数値を表示する)
 * data: 画像データ
 * index: 表示するデータの開始位置
 * element: 表示するデータ数
 **/
int view_train(char *data, int index, int element) {
    int i, j;

    for (i=0;i<element;i++) {
        for (j=(index*784)+784*i;j<(index*784)+784*(i+1);j++) {
            if (j%28 == 0) {
                printf("\n");
            }
            printf("%4d", (unsigned char)data[j]);
        }
    }
    printf("\n");


    return 0;
}

/**
 * 読み込んだラベルデータの中身を表示する
 * data: ラベルデータ
 * index: 表示するデータの開始位置
 * element: 表示するデータ数
 **/
int view_label(char *data, int index, int element) {
    int i;

    for (i=index;i<index+element;i++) {
        printf("%2d", (unsigned char)data[i]);
    }
    printf("\n");


    return 0;
}

int open_pgm_image_file(char *fileNameOut, int images, char *data) {

    // open PGM image file
    int i, j;
    FILE *fpw = fopen(fileNameOut, "wb");
    if (fpw == NULL) {
        fprintf(stderr, "ERROR: Cannot open %s\n", fileNameOut);
        return 1;
    }

    // output pgm
    int size = 28;
    int imY, imX, imIdx, pos, pix;
    fprintf(fpw, "P2\n");
    fprintf(fpw, "%d %d\n", size * size, size * size);
    fprintf(fpw, "255\n");
    for (imY=0;imY<size;imY++) {
        for (i=0;i<size;i++) {
            for (imX=0;imX < size; imX++) {
                for (j=0;j<size;j++) {
                    imIdx = imY * size + imX;
                    pos = imIdx * size * size + i * size + j;
                    pix = 255 - data[pos];
                    fprintf(fpw, "%4d", pix);
                }
            }
        }
    }

    fclose(fpw);

    printf("INFO: output PGM file %s success!!\n", fileNameOut);

    return 0;
}
