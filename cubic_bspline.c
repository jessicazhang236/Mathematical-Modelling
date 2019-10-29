/* Open Cubic B-spline Curve Fitting*/
#include <stdio.h>
#include <stdlib.h>
#define alpha -0.2679492
#define p 10
int main() {
    float temp, res, x[20], a[5000], b[5000];
    int n;
    printf("Input N:\n"); // number of given points
    scanf("%d", &n);
    x[0] = 1.0;
    for (int i = 1; i <= p; ++i)
        x[i] = alpha * x[i-1];
    for (int i = 1; i <= n; ++i) {
        a[i] = rand() % 1000;
        b[i] = a[i];
    }
    for (int i = 1; i <= n; ++i) 
        a[i] *= 6.0;
    for (int i = 2; i <= n; ++i) 
        a[i] += alpha * a[i-1];
    a[n] *= -alpha;
    for (int i = n - 1; i >= 1; --i)
        a[i] = alpha * (a[i+1] - a[i]);
    temp = a[1];
    for (int i = 1; i <= p; ++i)
        a[i] += x[i] * temp;
    temp = a[n] / (alpha - 1);
    for (int i = 1; i <= p; ++i)
        a[n+1-i] -= x[i] * temp;
    res = abs(5.0 * a[1] + a[2] - 6.0 * b[1]);
    for (int i = 2; i <= n - 1; ++i) {
        temp = a[i-1] + 4.0 * a[i] + a[i+1] - 6.0 * b[i];
        if (temp < 0) temp *= -1.0;
        if (res < temp) res = temp;
    }
    temp = abs(a[n-1] + 5.0 * a[n] - 6.0 * b[n]);
    if (res < temp) res = temp;
    printf("The residual is %10.7f\n", res);
}
