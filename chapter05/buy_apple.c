#include <stdio.h>
#include "MulLayer.h"

int main(void) {
    double apple = 100.0;
    double apple_num = 2.0;
    double tax = 1.1;

    MulLayer mul_apple_layer;
    MulLayer mul_tax_layer;

    // forward
    double apple_price = 0.0;

    mullayer_init(&mul_apple_layer);
    mullayer_forward(&mul_apple_layer, &apple_price, apple, apple_num);

    double price = 0.0;

    mullayer_init(&mul_tax_layer);
    mullayer_forward(&mul_tax_layer, &price, apple_price, tax);

    // backward
    double dprice = 1.0;
    double dapple_price = 0.0;
    double dtax = 0.0;

    mullayer_backward(&mul_tax_layer, &dapple_price, &dtax, dprice);

    double dapple = 0.0;
    double dapple_num = 0.0;

    mullayer_backward(&mul_apple_layer, &dapple, &dapple_num, dapple_price);

    printf("price: %d\n", (int)price);
    printf("dApple: %f\n", dapple);
    printf("dApple_num: %d\n", (int)dapple_num);
    printf("dTax: %f\n", dtax);

    return 0;
}
