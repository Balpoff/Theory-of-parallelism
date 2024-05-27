#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

int thread_limit = 0;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

double func(double x)
{
    return exp(-x * x);
}

double integrate(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));

    sum *= h;

    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    #pragma omp parallel
    {
        double local_sum = 0.0;
        #pragma omp for
        for (int i = 0; i < n; i++)
            local_sum += func(a + h * (i + 0.5));
        #pragma omp atomic
        sum += local_sum;
    }
    sum *= h;

    return sum;
}
typedef double (*FunctionFunc)(double);

double run_tests(double (*integrate_func)(FunctionFunc, double, double, int))
{
    double t = cpuSecond();
    double res = integrate_func(func, a, b, nsteps);
    t = cpuSecond() - t;
    printf("Result: %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}
int main(int argc, char **argv)
{
    int threads_n[] = { 2, 4, 7, 8, 16, 20, 40 };
    printf("Integration f(x) on [%.12f, %.12f], nsteps = %d\n", a, b, nsteps);
    double tserial = run_tests(integrate);
    printf("serial test\n");
    printf("Execution time: %.6f\n", tserial);
    printf("parallel tests\n");
    for (int i = 0; i < 7; i++) {
        double tparallel = run_tests(integrate_omp);
        printf("Execution time (parallel %d threads): %.6f\n", threads_n[i], tparallel);
        printf("Speedup: %.2f\n", tserial / tparallel);
        thread_limit = threads_n[i];
    }
    return 0;
}