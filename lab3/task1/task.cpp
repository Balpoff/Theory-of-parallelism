#include <thread>
#include <iostream>
#include <vector>

void matrix_vector_product(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, int lb, int rb)
{
    for (int i = lb; i < rb; i++) {
        c[i] = 0.0;
        for (int j = 0; j < b.size(); j++)
            c[i] += a[i * b.size() + j] * b[j];
    }
}

void matrix_vector_init(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, int lb, int rb) {
    for (int i = lb; i < rb; i++) {
        for (int j = 0; j < b.size(); j++) {
            a[i * b.size() + j] = i + j;
        }
    }
    // b.size == c.size, so we can use static parallelization with bounds for c array
    for (int j = lb; j < rb; j++)
        b[j] = j;
}

void parallelize_task(void (*func)(std::vector<double>&, std::vector<double>&, std::vector<double>&, int, int), std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, int n, int m, int num_of_threads) {
    std::vector<std::thread> thread_pool;
    int last_lb = 0;
    // separate tasks using static parallelization and create threadpool array
    for (int i = 0; i < num_of_threads; i++) {
        int tasks_count = n / num_of_threads + ((n % num_of_threads) - i > 0);
        std::thread thread(func, std::ref(a), std::ref(b), std::ref(c), std::move(last_lb), std::move(last_lb + tasks_count));
        thread_pool.push_back(std::move(thread));
        last_lb += tasks_count;
    }
    // wait threads
    for (auto& thread : thread_pool) {
        thread.join();
    }
}

void doSomething(int id) {
    std::cout << id << "\n";
}

int main() {
    for (int i = 1; i <= 2; i++) {
        int n = 20000 * i;
        int m = 20000 * i;
        std::cout << n << "x" << m << " test\n";
        std::vector<int> threads_list = { 2, 4, 7, 8, 16, 20, 40 };
        std::vector<double> a, b, c;
        a.resize(n * m);
        b.resize(n);
        c.resize(n);
        std::cout << "Serial test\n";
        const auto start{std::chrono::steady_clock::now()};
        matrix_vector_init(a, b, c, 0, n);
        matrix_vector_product(a, b, c, 0, n);
        const auto end{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{end - start};
        double serial_time = elapsed_seconds.count();
        std::cout << "Elapsed time: " << serial_time << "\n";
        std::cout << "Parallel test\n";
        for (auto thread_num : threads_list) {
            std::cout << thread_num << " threads\n";
            std::vector<double> a, b, c;
            a.resize(n * m);
            b.resize(n);
            c.resize(n);
            const auto start{std::chrono::steady_clock::now()};
            parallelize_task(matrix_vector_init, a, b, c, n, m, thread_num);
            parallelize_task(matrix_vector_product, a, b, c, n, m, thread_num);
            const auto end{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> elapsed_seconds{end - start};
            double parallel_time = elapsed_seconds.count();
            std::cout << "Elapsed time: " << parallel_time << "\n";
            std::cout << "Speed-up: " << serial_time / parallel_time << "\n";
        }
    }
    return 0;
}