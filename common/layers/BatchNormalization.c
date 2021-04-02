#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "BatchNormalization.h"
#include "../function.h"

int batchnormalization_init(BatchNormalization *this, double *gamma, double *beta, double momentum, double *running_mean, double *running_var, int col_size, int row_size) {

    this->gamma = (double *)malloc(sizeof(double) * row_size);
    memcpy(this->gamma, gamma, sizeof(double) * row_size);

    this->beta = (double *)malloc(sizeof(double) * row_size);
    memcpy(this->beta, beta, sizeof(double) * row_size);

    this->momentum = momentum;

    this->running_mean = (double *)malloc(sizeof(double) * row_size);
    memcpy(this->running_mean, running_mean, sizeof(double) * row_size);
    this->running_var = (double *)malloc(sizeof(double) * row_size);
    memcpy(this->running_var, running_var, sizeof(double) * row_size);

    this->xc = (double *)malloc(sizeof(double) * col_size * row_size);
    this->xn = (double *)malloc(sizeof(double) * col_size * row_size);
    this->std = (double *)malloc(sizeof(double) * row_size);

    this->dbeta = (double *)malloc(sizeof(double) * row_size);
    this->dgamma = (double *)malloc(sizeof(double) * row_size);

    this->col_size = col_size;
    this->row_size = row_size;

    return 0;
}

int batchnormalization_free(BatchNormalization *this) {
    return 0;
}

int batchnormalization_forward(BatchNormalization *this, double *out, double *x, char *train_flg) {
    double *xc;
    double *xn;
    double *broadcast_running_mean;
    double *broadcast_running_var;

    xc = (double *)malloc(sizeof(double) * this->col_size * this->row_size);
    xn = (double *)malloc(sizeof(double) * this->col_size * this->row_size);
    broadcast_running_mean = (double *)malloc(sizeof(double) * this->col_size * this->row_size);
    broadcast_running_var = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

    int i, j;
    for (i=0;i<this->col_size;i++) {
        for (j=0;j<this->row_size;j++) {
            broadcast_running_mean[j+(i*this->row_size)] = this->running_mean[j];
        }
    }

    for (i=0;i<this->col_size;i++) {
        for (j=0;j<this->row_size;j++) {
            broadcast_running_var[j+(i*this->row_size)] = this->running_var[j];
        }
    }

    if (strcmp(train_flg, "true") == 0) {
        double *x_trans;
        x_trans = (double *)malloc(sizeof(double) * this->row_size * this->col_size);
        trans_function(x_trans, x, this->col_size, this->row_size);

        double *tmp_x;
        tmp_x = (double *)malloc(sizeof(double) * this->col_size);

        double *mu;
        mu = (double *)malloc(sizeof(double) * this->row_size);

        for (i=0;i<this->row_size;i++) {
            for (j=0;j<this->col_size;j++) {
                tmp_x[i] = x_trans[j+(this->col_size*i)];
            }
            mean_function(tmp_x, &mu[i], this->col_size);
        }

        double *broadcast_mu;
        broadcast_mu = (double *)malloc(sizeof(double) * this->col_size * this->row_size);
        for (i=0;i<this->col_size;i++) {
            for (j=0;j<this->row_size;j++) {
                broadcast_mu[j+(i*this->row_size)] = mu[j];
            }
        }

        for (i=0;i<this->col_size*this->row_size;i++) {
            xc[i] = x[i] - broadcast_mu[i];
        }

        double *tmp_xc;
        tmp_xc = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

        for (i=0;i<this->col_size*this->row_size;i++) {
            tmp_xc[i] = pow(xc[i], 2.0);
           //  printf("tmp_xc[%f] xc[%f] : ", tmp_xc[i], xc[i]);
        }

        double *var;
        var = (double *)malloc(sizeof(double) * this->row_size);

        double *tmp_var;
        tmp_var = (double *)malloc(sizeof(double) * this->col_size);

        double *tmp_xc_trans;
        tmp_xc_trans = (double *)malloc(sizeof(double) * this->row_size * this->col_size);
        trans_function(tmp_xc_trans, tmp_xc, this->col_size, this->row_size);

        for (i=0;i<this->row_size;i++) {
            for (j=0;j<this->col_size;j++) {
                tmp_var[i] = tmp_xc_trans[j+(this->col_size*i)];
            }
            mean_function(tmp_var, &var[i], this->col_size);
        }

        for (i=0;i<this->row_size;i++) {
            this->std[i] = sqrt(var[i] + pow(10.0, -7.0));
        }

        double *broadcast_std;
        broadcast_std = (double *)malloc(sizeof(double) * this->col_size * this->row_size);
        for (i=0;i<this->col_size;i++) {
            for (j=0;j<this->row_size;j++) {
                broadcast_std[j+(i*this->row_size)] = this->std[j];
            }
        }

        for (i=0;i<this->col_size*this->row_size;i++) {
            xn[i] = xc[i] / broadcast_std[i];
        }

        for (i=0;i<this->col_size*this->row_size;i++) {
            this->xc[i] = xc[i];
        }

        for (i=0;i<this->col_size*this->row_size;i++) {
            this->xn[i] = xn[i];
        }

        for (i=0;i<this->row_size;i++) {
            this->running_mean[i] = this->momentum * this->running_mean[i] + (1 - this->momentum) * mu[i];
        }

        for (i=0;i<this->row_size;i++) {
            this->running_var[i] = this->momentum * this->running_var[i] + (1 - this->momentum) * var[i];
        }

        free(x_trans);
        free(tmp_x);
        free(mu);
        free(broadcast_mu);
        free(tmp_xc);
        free(var);
        free(tmp_var);
        free(tmp_xc_trans);
        free(broadcast_std);

    } else {
        for (i=0;i<this->col_size*this->row_size;i++) {
            xc[i] = x[i] - broadcast_running_mean[i];
        }

        for (i=0;i<this->col_size*this->row_size;i++) {
            xn[i] = xc[i] / sqrt(broadcast_running_var[i] + pow(10.0, -7.0));
        }
    }

    double *broadcast_gamma;
    double *broadcast_beta;
    broadcast_gamma = (double *)malloc(sizeof(double) * this->col_size * this->row_size);
    broadcast_beta = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

    for (i=0;i<this->col_size;i++) {
        for (j=0;j<this->row_size;j++) {
            broadcast_gamma[j+(i*this->row_size)] = this->gamma[j];
        }
    }

    for (i=0;i<this->col_size;i++) {
        for (j=0;j<this->row_size;j++) {
            broadcast_beta[j+(i*this->row_size)] = this->beta[j];
        }
    }

    for (i=0;i<this->col_size*this->row_size;i++) {
        out[i] = broadcast_gamma[i] * xn[i] + this->beta[i];
    }

    free(xc);
    free(xn);
    free(broadcast_running_mean);
    free(broadcast_running_var);
    free(broadcast_gamma);
    free(broadcast_beta);

    return 0;
}

int batchnormalization_backward(BatchNormalization *this, double *dx, double *dout) {
    double *dbeta;
    double *dgamma;
    double tmp_dbeta = 0.0;
    double tmp_dgamma = 0.0;

    dbeta = (double *)malloc(sizeof(double) * this->row_size);
    dgamma = (double *)malloc(sizeof(double) * this->row_size);

    double *dout_trans;
    dout_trans = (double *)malloc(sizeof(double) * this->row_size * this->col_size);
    trans_function(dout_trans, dout, this->col_size, this->row_size);

    double *tmp_dout;
    tmp_dout = (double *)malloc(sizeof(double) * this->col_size);

    int i, j;
    for (i=0;i<this->row_size;i++) {
        for (j=0;j<this->col_size;j++) {
            tmp_dout[j] = dout_trans[j+(this->col_size*i)];
        }

        sum_function(tmp_dout, &tmp_dbeta, this->col_size);

        dbeta[i] = tmp_dbeta;
    }

    double *tmp_xn;
    tmp_xn = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

    for (i=0;i<this->col_size*this->row_size;i++) {
        tmp_xn[i] = this->xn[i] * dout[i];
    }

    double *tmp_xn_trans;
    tmp_xn_trans = (double *)malloc(sizeof(double) * this->row_size * this->col_size);
    trans_function(tmp_xn_trans, tmp_xn, this->col_size, this->row_size);

    double *tmp_xn_sum;
    tmp_xn_sum = (double *)malloc(sizeof(double) * this->col_size);

    for (i=0;i<this->row_size;i++) {
        for (j=0;j<this->col_size;j++) {
            tmp_xn_sum[j] = tmp_xn_trans[j+(this->col_size*i)];
        }

        sum_function(tmp_xn_sum, &tmp_dgamma, this->col_size);

        dgamma[i] = tmp_dgamma;
    }

    double *dxn;
    dxn = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

    double *broadcast_gamma;
    broadcast_gamma = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

    for (i=0;i<this->col_size;i++) {
        for (j=0;j<this->row_size;j++) {
            broadcast_gamma[j+(i*this->row_size)] = this->gamma[j];
        }
    }

    for (i=0;i<this->col_size*this->row_size;i++) {
        dxn[i] = broadcast_gamma[i] * dout[i];
    }

    double *dxc;
    dxc = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

    double *broadcast_std;
    broadcast_std = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

    for (i=0;i<this->col_size;i++) {
        for (j=0;j<this->row_size;j++) {
            broadcast_std[j+(i*this->row_size)] = this->std[j];
        }
    }

    for (i=0;i<this->col_size*this->row_size;i++) {
        dxc[i] = dxn[i] / broadcast_std[i];
    }

    double *dstd;
    dstd = (double *)malloc(sizeof(double) * this->row_size);

    double *tmp_dxn;
    tmp_dxn = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

    for (i=0;i<this->col_size*this->row_size;i++) {
        tmp_dxn[i] = (dxn[i] * this->xc[i]) / (broadcast_std[i] * broadcast_std[i]);
    }

    double *tmp_dxn_trans;
    tmp_dxn_trans = (double *)malloc(sizeof(double) * this->row_size * this->col_size);
    trans_function(tmp_dxn_trans, tmp_dxn, this->col_size, this->row_size);

    double *tmp_dxn_sum;
    tmp_dxn_sum = (double *)malloc(sizeof(double) * this->col_size);

    for (i=0;i<this->row_size;i++) {
        for (j=0;j<this->col_size;j++) {
            tmp_dxn_sum[j] = tmp_dxn_trans[j+(this->col_size*i)];
        }

        sum_function(tmp_dxn_sum, &dstd[i], this->col_size);
        dstd[i] = -1.0 * dstd[i];
    }

    double *dvar;
    dvar = (double *)malloc(sizeof(double) * this->row_size);

    for (i=0;i<this->row_size;i++) {
        dvar[i] = 0.5 * dstd[i] / this->std[i];
    }

    double *broadcast_dvar;
    broadcast_dvar = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

    for (i=0;i<this->col_size;i++) {
        for (j=0;j<this->row_size;j++) {
            broadcast_dvar[j+(i*this->row_size)] = dvar[j];
        }
    }

    for (i=0;i<this->col_size*this->row_size;i++) {
        dxc[i] = dxc[i] + (2.0 / this->col_size) * this->xc[i] * broadcast_dvar[i];
    }

    double *dmu;
    dmu = (double *)malloc(sizeof(double) * this->row_size);

    double *dxc_trans;
    dxc_trans = (double *)malloc(sizeof(double) * this->row_size * this->col_size);
    trans_function(dxc_trans, dxc, this->col_size, this->row_size);

    double *tmp_dxc;
    tmp_dxc = (double *)malloc(sizeof(double) * this->col_size);

    for (i=0;i<this->row_size;i++) {
        for (j=0;j<this->col_size;j++) {
            tmp_dxc[j] = dxc_trans[j+(this->col_size*i)];
        }

        sum_function(tmp_dxc, &dmu[i], this->col_size);
    }

    double *broadcast_dmu;
    broadcast_dmu = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

    for (i=0;i<this->col_size;i++) {
        for (j=0;j<this->row_size;j++) {
            broadcast_dmu[j+(i*this->row_size)] = dmu[j];
        }
    }

    for (i=0;i<this->col_size*this->row_size;i++) {
        dx[i] = dxc[i] - broadcast_dmu[i] / this->col_size;
    }

    for (i=0;i<this->col_size*this->row_size;i++) {
        this->dgamma[i] = dgamma[i];
    }

    for (i=0;i<this->col_size*this->row_size;i++) {
        this->dbeta[i] = dbeta[i];
    }

    free(dbeta);
    free(dgamma);
    free(dout_trans);
    free(tmp_dout);
    free(tmp_xn);
    free(tmp_xn_trans);
    free(tmp_xn_sum);
    free(dvar);
    free(broadcast_dvar);
    free(dmu);
    free(dxc_trans);
    free(tmp_dxc);
    free(broadcast_dmu);

    return 0;
}
