#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <string>
#include <chrono>
#include <omp.h>
#include <boost/program_options.hpp>
#include <nvtx3/nvToolsExt.h>
#include "device_vector.h"

#define OFFSET(x, y, m) (((x)*(m)) + (y)) // Макрос для вычисления смещения в матрице

// Функция для инициализации теплового поля
void initialize_field(device_vector<double>& matrix, std::vector<std::tuple<int, double>> heat_points) {
    for (auto heat_point : heat_points) {
        int index = std::get<0>(heat_point); // Индекс точки нагрева
        double temp = std::get<1>(heat_point); // Температура в этой точке
        matrix[index] = temp; // Устанавливаем температуру в матрице
    }
    matrix.update_device(0, matrix.size()); // Обновляем данные на устройстве
}

// Функция для вывода теплового поля
void draw_field(device_vector<double> matrix, int size) {
    for (int i = 1; i < size - 1; i++) { // Итерация по строкам матрицы
        for (int j = 1; j < size - 1; j++) // Итерация по столбцам матрицы
            std::cout << matrix[OFFSET(i, j, size)] << " "; // Вывод значения элемента
        std::cout << "\n";
    }
}

// Функция для вычисления одного шага теплового поля
double calculate_step(device_vector<double>& matrix, device_vector<double>& matrix_out, int size){
    double err = 0;
    // Параллельный цикл с редукцией ошибки
    #pragma acc parallel loop reduction(max:err) collapse(2) async
    for (int i = 1; i < size - 1; i++) {
        for (int j = 1; j < size - 1; j++) {
            double current_point = matrix[OFFSET(i, j, size)]; // Текущая точка
            int num_of_points = 1;
            num_of_points += i > 1 ? 1 : 0; // Увеличение количества точек при наличии соседей
            num_of_points += j > 1 ? 1 : 0;
            num_of_points += i < size-2 ? 1 : 0;
            num_of_points += j < size-2 ? 1 : 0;
            // Пятиточечный шаблон для расчета новой температуры
            matrix_out[OFFSET(i, j, size)] = (matrix[OFFSET(i - 1, j, size)] + 
                                             matrix[OFFSET(i, j - 1, size)] + 
                                             matrix[OFFSET(i + 1, j, size)] +
                                             matrix[OFFSET(i, j + 1, size)] +
                                             matrix[OFFSET(i, j, size)]) * (1 / (float)num_of_points); 
            err = fmax(err, fabs(matrix_out[OFFSET(i, j, size)] - matrix[OFFSET(i, j, size)])); // Вычисляем ошибку
        }
    }
    // Обновление данных на хосте
    #pragma acc update self(matrix_out._A[:size * size]) async
    // Ожидание завершения вычислений
    #pragma acc wait
    return err;
}

// Функция для копирования матрицы
void copy_matrix(device_vector<double>& matrix, device_vector<double>& matrix_out, int size) {
    // Параллельный цикл
    #pragma acc parallel loop
    for(int i = 0; i < size; i++) {
        // Вложенный параллельный цикл
        #pragma acc loop
        for(int j = 0; j < size; j++ )
            matrix[OFFSET(i, j, size)] = matrix_out[OFFSET(i, j, size)]; // Копирование элемента
    }
}

// Функция для вычисления теплового поля до заданной точности или максимального количества итераций
void calculate_heatfield(device_vector<double>& matrix, device_vector<double>& matrix_out, int size, double max_error, int max_iterrations) {
    double error = 1;
    int it = 0;
    nvtxRangePushA("while"); // Начало профилирования блока while
    while (error > max_error && it < max_iterrations) { // Условие завершения цикла
        nvtxRangePushA("calc"); // Начало профилирования блока calc
        error = calculate_step(matrix, matrix_out, size); // Вычисление одного шага
        nvtxRangePop(); // Завершение профилирования блока calc

        nvtxRangePushA("copy"); // Начало профилирования блока copy
        copy_matrix(matrix, matrix_out, size); // Копирование матрицы
        nvtxRangePop(); // Завершение профилирования блока copy
        it++;
    }
    
    std::cout << "num of iterations: " << it << "\n";
    std::cout << "error: " << error << "\n";
}

namespace po = boost::program_options; // Пространство имен для Boost Program Options

int main(int argc, char** argv) {
    int size = 10;
    double max_error = 1e-6;
    int max_iterations = 1000000;

    // Инициализация парсера аргументов командной строки
    po::options_description desc("Allowed options");
    desc.add_options()
                    ("help,h", "Show help")
                    ("size,s", po::value<int>(&size), "field size")
                    ("max_error,me", po::value<double>(&max_error), "max error of calculation")
                    ("max_iterations,mit", po::value<int>(&max_iterations), "max iteration count of calculation")
                    ("draw_output,do", "Draw output matrix");
    po::variables_map vm;
    po::store(po::command_line_parser (argc, argv).options(desc).allow_unregistered().run(), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << "-s - size\n-me - max error of calculation\n-mit - max iteration count of calculation\n";
        return 0;
    }

    // Создание и инициализация матриц
    device_vector<double> matrix = device_vector<double>((size+2) * (size+2));
    device_vector<double> matrix_out = device_vector<double>((size+2) * (size+2));
    size += 2; // Добавление отступов

    // Определение начальных точек нагрева
    std::vector<std::tuple<int, double>> heat_points({std::make_tuple(size + 1, 10), 
                                                    std::make_tuple(2 * (size) - 2 , 20), 
                                                    std::make_tuple(size * size - 2 * (size) + 1, 30), 
                                                    std::make_tuple(size * size - size - 2, 40)
                                                    }); // Инициализация точек с теплом
    
    nvtxRangePushA("init"); // Начало профилирования блока init
    initialize_field(matrix, heat_points); // Инициализация теплового поля
    nvtxRangePop(); // Завершение профилирования блока init
    
    std::cout << "size: " << size-2 << "x" << size-2 << '\n';

    // Начало измерения времени
    const auto start{ std::chrono::steady_clock::now() };
    calculate_heatfield(matrix, matrix_out, size, max_error, max_iterations);
    const auto end{ std::chrono::steady_clock::now() };
    const std::chrono::duration<double> elapsed_seconds{ end - start };
    std::cout << elapsed_seconds.count() << " s\n";

    // Если флаг вывода установлен, выводим конечную матрицу
    if (vm.count("draw_output")) {
        draw_field(matrix_out, size);
    }
    return 0;
}
