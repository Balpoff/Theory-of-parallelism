#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>

int thread_number = 0;

// Функция для вычисления произведения матрицы на вектор (в линейном режиме)
void matrix_vector_product(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c)
{
    // Перебираем строки матрицы
    for (int i = 0; i < c.size(); i++) {
        c[i] = 0.0; // Инициализируем элемент результирующего вектора нулем
        // Перебираем столбцы матрицы
        for (int j = 0; j < b.size(); j++)
            c[i] += a[i * b.size() + j] * b[j];
    }   
}
// Функция для вычисления произведения матрицы на вектор (в параллельном режиме)
void matrix_vector_product_omp(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c)
{
    // Определение параллельной секции с указанием числа потоков
    #pragma omp parallel num_threads(thread_number)
    {
        // Директива для распараллеливания цикла
        #pragma omp for
        for (int i = 0; i < c.size(); i++) {
            c[i] = 0.0;
            for (int j = 0; j < b.size(); j++)
                c[i] += a[i * b.size() + j] * b[j];
        }
    }
}

double run_tests(int m, int n, void (*func)(std::vector<double>&, std::vector<double>&, std::vector<double>&))
{
    std::vector<double> a, b, c;
    a = std::vector<double>(m * n); // матрица размером m x n
    b = std::vector<double>(n); // вектор размером n
    c = std::vector<double>(m); // результирующий вектор размером m

    // Инициализация матрицы a
    #pragma omp parallel num_threads(40)
    {
        #pragma omp for
        for (int i = 0; i < c.size(); i++) {
            for (int j = 0; j < b.size(); j++)
                a[i * b.size() + j] = i + j;
        }
    }
    // Инициализация матрицы b
    for (int j = 0; j < n; j++)
        b[j] = j;
        
    const auto start{std::chrono::steady_clock::now()};
    func(a, b, c);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    return elapsed_seconds.count();
}


int main(int argc, char* argv[]) 
{
    int m, n;
    if (argc != 4) {
        double serial_time;
        int threads_list[] = { 2, 4, 7, 8, 16, 20, 40 };
        std::cout << "Test for 20k x 20k array:\n\n";
        m = 20000;
        n = m;
        std::cout << "Program was started without parameters. Start for testing...";
        std::cout << "Matrix-vector product (c[m] = a[m, n] * b[n]; m = " << m << ", n = " << n << ")\n";
        std::cout << "Memory used: " << int64_t(((m * n + m + n) * sizeof(double)) >> 20) << " MiB\n";
        std::cout << "Linear product test:\n";
        serial_time = run_tests(m, n, matrix_vector_product);
        std::cout << "Elapsed time(serial realization): " << serial_time << "sec.\n";
        
        for (int i = 0; i < 7; i++) {
            std::cout << "Parallel product test (with " << threads_list[i] << " threads):\n";
            thread_number = threads_list[i];
            double parallel_time = run_tests(m, n, matrix_vector_product_omp);
            std::cout << "Elapsed time(serial realization): " << parallel_time << "sec.\n";
            std::cout << "Speedup: " << serial_time / parallel_time << "\n";
        }
        std::cout << "Test for 40k x 40k array:\n\n";
        m = 40000;
        n = m;
        std::cout << "Program was started without parameters. Start for testing...";
        std::cout << "Matrix-vector product (c[m] = a[m, n] * b[n]; m = " << m << ", n = " << n << ")\n";
        std::cout << "Memory used: " << int64_t(((m * n + m + n) * sizeof(double)) >> 20) << " MiB\n";
        std::cout << "Linear product test:\n";
        serial_time = run_tests(m, n, matrix_vector_product);
        std::cout << "Elapsed time(serial realization): " << serial_time << "sec.\n";
        for (int i = 0; i < 7; i++) {
            std::cout << "Parallel product test (with " << threads_list[i] << " threads):\n";
            thread_number = threads_list[i];
            double parallel_time = run_tests(m, n, matrix_vector_product_omp);
            std::cout << "Elapsed time(serial realization): " << parallel_time << "sec.\n";
            std::cout << "Speedup: " << serial_time / parallel_time << "\n";
        }
    }
    else {
        m = std::stoi(argv[1]);
        n = std::stoi(argv[2]);
        thread_number = std::stoi(argv[3]);
        std::cout << "Matrix-vector product (c[m] = a[m, n] * b[n]; m = " << m << ", n = " << n << ")\n";
        std::cout << "Memory used: " << int64_t(((m * n + m + n) * sizeof(double)) >> 20) << " MiB\n";
        std::cout << "Linear product test:\n";
        run_tests(m, n, matrix_vector_product);
        std::cout << "Parallel product test:\n";
        run_tests(m, n, matrix_vector_product_omp);
    }
    return 0;
}
