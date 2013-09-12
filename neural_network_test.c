#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>

#define D 7
#define N 200
#define M 30
#define K 1
#define layer 2
#define eta 0.1
#define alpha 0.9

#define rando() (rand() * 1.0 / RAND_MAX)

double w[layer][M+1][M+1];
double x[N][D+1];
double a[M+1], z[M+1];
double y[N][K+1];
double t[N][K+1];
double err[layer][M+1];
double d[layer][M+1][M+1];

double tanh(double a){
    return (exp(a) - exp(-a)) / (exp(a) + exp(-a));
}

double equal(double a){
    return a;
}

double sigmoid(double a){
    return 1.0 / (1.0 + exp(-a)); 
}

double feed_forward(double *in, int in_cnt, double *out, int out_cnt, double param[M+1][M+1], double (*trans_method)(double)){
    int i, j;
    double a[M+1];
    for(i = 1; i <= out_cnt; i++){
        a[i] = param[i][0];
        for(j = 1; j <= in_cnt; j++){
            a[i] += param[i][j] * in[j];
        }
        out[i] = trans_method(a[i]);
    }
}

void predict(double in[D+1], double out[K+1]){
    double hidden_unit[M];
    feed_forward(in, D, hidden_unit, M, w[0], sigmoid);
    feed_forward(hidden_unit, M, out, K, w[1], sigmoid);
}

double get_total_error(int n){
    double ret = 0.0;
    int i, k;
    for(k = 0; k <= K; k++){
        ret += 0.5 * (y[n][k] - t[n][k]) * (y[n][k] - t[n][k]);
    }
    return ret;
}

int main(){
    int i, j, k, n; 
    char res[5];
    int d1, d2, d3, d4, d7;
    double d5, d6;
    int loop;
    double total_error;
    double init_weight[] = {0.5, 1.0 / 15, 1.0 / 200, 1.0 / 100, 1.0 / 50, 1.0 / 40, 1.0 / 100, 1.0 / 100};

    double test[K+1];
    double in1[8] = {1, 6, 148, 72, 35, 33.6, 0.627, 50}; //Yes
    double in2[8] = {1, 1, 85, 66, 29, 26.6, 0.351, 31}; //No

    freopen("pima.tr", "r", stdin);

    srand((int)time(0));

    for(i = 0; i < N; i++){
        scanf("%d%d%d%d%lf%lf%d%s", &d1, &d2, &d3, &d4, &d5, &d6, &d7, res);
        x[i][0] = 1.0;
        x[i][1] = 1.0 * d1;
        x[i][2] = 1.0 * d2;
        x[i][3] = 1.0 * d3;
        x[i][4] = 1.0 * d4;
        x[i][5] = 1.0 * d5;
        x[i][6] = 1.0 * d6;
        x[i][7] = 1.0 * d7;
        if(strcmp(res, "Yes") != -1){
            t[i][1] = 1.0;
        }else{
            t[i][1] = 0.0;
        }
    }

    for(k = 0; k < layer; k++)
        for(i = 0; i <= M; i++)
            for(j = 0; j <= M; j++){
                d[k][i][j] = 0;
                if(k == 0){
                    w[k][i][j] = 2.0 * (rando() - 0.5) * init_weight[j];
                }else{
                    w[k][i][j] = 2.0 * (rando() - 0.5) * 0.5;
                }
            }

    for(loop = 0; loop < 100; loop++){
        total_error = 0;
        for(n = 0; n < N; n++){
            feed_forward(x[n], D, z, M, w[0], sigmoid);
            feed_forward(z, M, y[n], K, w[1], sigmoid);
            total_error += get_total_error(n); 
            for(i = 1; i <= K; i++){
                err[1][i] = (t[n][i] - y[n][i]) * y[n][i] * (1.0 -y[n][i]);
            }
            for(i = 1; i <= M; i++){
                err[0][i] = 0;
                for(j = 0; j <= K; j++){
                    err[0][i] += w[1][j][i] * err[1][j]; 
                }
                err[0][i] *= z[i] * (1.0 - z[i]);
            }
            for(j = 1; j <= M; j++){
                d[0][j][0] = alpha * d[0][j][0] + eta * err[0][j];
                w[0][j][0] += d[0][j][0];
                for(i = 0; i <= D; i++){
                    d[0][j][i] = alpha * d[0][j][i] + eta * err[0][j] * x[n][i];
                    w[0][j][i] += d[0][j][i]; 
                }
            }
            for(k = 1; k <= K; k++){
                d[1][k][0] = alpha * d[1][k][0] + eta * err[1][k];
                w[1][k][0] += d[1][k][0];
                for(j = 1; j <= M; j++){
                    d[1][k][j] = alpha * d[1][k][j] + eta * err[1][k] * z[j];
                    w[1][k][j] += d[1][k][j];
                }
            }
        }
    }
    predict(in1, test);
    printf("%lf\n", test[1]);
    predict(in2, test);
    printf("%lf\n", test[1]);
    return 0;
}
