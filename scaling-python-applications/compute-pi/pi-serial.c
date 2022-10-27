#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double drandom() {
    return (double) rand() / RAND_MAX;
}

double estimate_pi(unsigned long n) {
    unsigned long m = 0;
    double x, y;
    for(int i = 0; i < n; ++i) {
        x = drandom();
        y = drandom();
        if (x*x + y*y < 1.0) m++;
    }
    return (double) m * 4.0 / (double) n;
}

int main() {
    
    unsigned long n = 20000000;

    clock_t start = clock();
    double result = estimate_pi(n);
    clock_t end = clock();
    double elapsed_time = (double) (end - start) / CLOCKS_PER_SEC;
    printf("%f\n", result);
    printf("%f\n", elapsed_time);
    return 0;
}
