#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

double epsilon = 0.00001;
double iteration_step = 0.00001;

int threads_number = 0;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void parallel_dot(double* A, double* b, double* c, int n, int m) {
    #pragma omp parallel num_threads(threads_number)
    {
    #pragma omp for
    for (int i = 0; i < n; i++) {
        c[i] = 0;
        for (int j = 0; j < m; j++)
            c[i] += A[i * n + j] * b[j];
    }
    } 
}

void serial_dot(double* A, double* b, double* c, int n, int m) {
    for (int i = 0; i < n; i++) {
        c[i] = 0;
        for (int j = 0; j < m; j++)
            c[i] += A[i * n + j] * b[j];
    }
}

double vector_length(double* vec, int n) {
    double vec_length_sum = 0;
    for (int i = 0; i < n; i++)
        vec_length_sum += vec[i] * vec[i];
    return sqrt(vec_length_sum);
}

void simple_iteration_method_first_realization_serial(double* A, double* x, double* b, int n) {
    double* xn = (double*)malloc(sizeof(double) * n);
    double* x_offset = (double*)malloc(sizeof(double) * n);
    double convergence_coeff = 1;
    double b_length = vector_length(b, n);
    double t = cpuSecond();
    while (convergence_coeff > epsilon) {
        serial_dot(A, x, xn, n, n);
        double x_offset_length = 0;
        for (int i = 0; i < n; i++) {
            x_offset[i] = (xn[i] - b[i]);
            x[i] = x[i] - iteration_step * x_offset[i];
            x_offset_length += x_offset[i] * x_offset[i];
        }
        convergence_coeff = sqrt(x_offset_length) / b_length;
    }
    t = cpuSecond() - t;
    printf("%.12f\n", t);
    free(xn);
    free(x_offset);
}

void simple_iteration_method_first_realization_parallel(double* A, double* x, double* b, int n) {
    double* xn = (double*)malloc(sizeof(double) * n);
    double* x_offset = (double*)malloc(sizeof(double) * n);
    double convergence_coeff = 1;
    int iteration_count = 0;
    double b_length = vector_length(b, n);
    double t = cpuSecond();
    while (convergence_coeff > epsilon) {
        parallel_dot(A, x, xn, n, n);
        double x_offset_length = 0;
        for (int i = 0; i < n; i++) {
            x_offset[i] = (xn[i] - b[i]);
            x[i] = x[i] - iteration_step * x_offset[i];
            x_offset_length += x_offset[i] * x_offset[i];
        }
        convergence_coeff = sqrt(x_offset_length) / b_length;
    }
    t = cpuSecond() - t;
    printf("%.12f\n", t);
    free(xn);
    free(x_offset);
}

void simple_iteration_method_second_realization(double* A, double* x, double* b, int n) {
    double* xn = (double*)malloc(sizeof(double) * n); // Ax
    double* x_offset = (double*)malloc(sizeof(double) * n); // Ax - b
    double* minimize_vector = (double*)malloc(sizeof(double) * n); // (Ax - b) * tau
    double x_offset_length = 0;
    double convergence_coeff = 1;
    double b_length = vector_length(b, n);
    char stop = 0;
    double t = cpuSecond();
    #pragma omp parallel num_threads(threads_number) shared(stop)
    {
    while ((convergence_coeff > epsilon) && !stop) {
        #pragma omp single
        x_offset_length = 0;
        #pragma omp for schedule(guided, 40)
        for (int i = 0; i < n; i++) {
            xn[i] = 0;
            for (int j = 0; j < n; j++)
                xn[i] += A[i * n + j] * x[j];
            x_offset[i] = (xn[i] - b[i]);
            minimize_vector[i] = iteration_step * x_offset[i];
            #pragma omp atomic
            x_offset_length += x_offset[i] * x_offset[i];
            
        }
        #pragma omp barrier

        #pragma omp for 
        for (int i = 0; i < n ; i++)
            x[i] = x[i] - minimize_vector[i];
        #pragma omp single
        convergence_coeff = sqrt(x_offset_length) / b_length;
    }
    stop = 1;
    }
    t = cpuSecond() - t;
    printf("%.12f\n", t);
    free(xn);
    free(x_offset);
    free(minimize_vector);
}

int main() {
    int n = 7000;
    double* A = (double*)malloc(sizeof(double) * n * n);
    double* b = (double*)malloc(sizeof(double) * n);
    double* x = (double*)malloc(sizeof(double) * n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j ++) {
            if (i == j) A[i * n + j] = 2.0;
            else A[i * n + j] = 1.0;
        }
        x[i] = 0;
    }
    for (int i = 0; i < n; i++)
        b[i] = n + 1;
    
    threads_number = 10;
    printf("first realization serial time:\n");
    simple_iteration_method_first_realization_serial(A, x, b, n);
    for (int i = 0; i < n; i++) {
        x[i] = 0;
    }
    int constant_num_threads_size = 7;
    int constant_num_threads[] = { 1, 2, 4, 8, 16, 20, 40 };
    for (int j = 0; j < constant_num_threads_size; j++) {
        threads_number = constant_num_threads[j];
        printf("thread_number = %d\n", threads_number);
        for (int i = 0; i < n; i++) {
            x[i] = 0;
        }
        printf("first realization parallel time:\n");
        simple_iteration_method_first_realization_parallel(A, x, b, n);
        for (int i = 0; i < n; i++) {
            x[i] = 0;
        }
        printf("second realization parallel time:\n");
        simple_iteration_method_second_realization(A, x, b, n);
    }
    return 0;
}