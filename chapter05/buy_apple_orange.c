#include <stdio.h>
#include "MulLayer.h"
#include "AddLayer.h"

int main(void) {
    double apple = 100.0;
    double apple_num = 2.0;
    double orange = 150.0;
    double orange_num = 3;
    double tax = 1.1;

    // layer
    MulLayer mul_apple_layer;
    MulLayer mul_orange_layer;
    AddLayer add_apple_orange_layer;
    MulLayer mul_tax_layer;

    // forward
    double apple_price = 0.0;

    mullayer_init(&mul_apple_layer);
    mullayer_forward(&mul_apple_layer, &apple_price, apple, apple_num);

    double orange_price = 0.0;

    mullayer_init(&mul_orange_layer);
    mullayer_forward(&mul_orange_layer, &orange_price, orange, orange_num);

    double all_price = 0.0;

    addlayer_init(&add_apple_orange_layer);
    addlayer_forward(&add_apple_orange_layer, &all_price, apple_price, orange_price);

    double price = 0.0;

    mullayer_init(&mul_tax_layer);
    mullayer_forward(&mul_tax_layer, &price, all_price, tax);

    // backward
    double dprice = 1.0;
    double dall_price = 0.0;
    double dtax = 0.0;

    mullayer_backward(&mul_tax_layer, &dall_price, &dtax, dprice);

    double dapple_price = 0.0;
    double dorange_price = 0.0;

    addlayer_backward(&add_apple_orange_layer, &dapple_price, &dorange_price, dall_price);

    double dorange = 0.0;
    double dorange_num = 0.0;

    mullayer_backward(&mul_orange_layer, &dorange, &dorange_num, dorange_price);

    double dapple = 0.0;
    double dapple_num = 0.0;

    mullayer_backward(&mul_apple_layer, &dapple, &dapple_num, dapple_price);

    printf("price: %d\n", (int)price);
    printf("dApple: %f\n", dapple);
    printf("dApple_num: %d\n", (int)dapple_num);
    printf("dOrange: %f\n", dorange);
    printf("dOrange_num: %d\n", (int)dorange_num);
    printf("dTax: %f\n", dtax);

    return 0;
}
