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

#define OFFSET(x, y, m) (((x)*(m)) + (y)) // пластина должна быть натянута на сферу?

void initialize_field(device_vector<double>& matrix, std::vector<std::tuple<int, double>> heat_points) {
    for (auto heat_point : heat_points) {
        int index = std::get<0>(heat_point);
        double temp = std::get<1>(heat_point);
        matrix[index] = temp;
    }
    matrix.update_device(0, matrix.size());
}

void draw_field(device_vector<double> matrix, int size) {
    for (int i = 1; i < size - 1; i++) {
        for (int j = 1; j < size - 1; j++)
            std::cout << matrix[OFFSET(i, j, size)] << " ";
        std::cout << "\n";
    }
}

double calculate_step(device_vector<double>& matrix, device_vector<double>& matrix_out, int size){
    double err = 0;
    #pragma acc parallel loop reduction(max:err) collapse(2) async
    for (int i = 1; i < size - 1; i++) {
        for (int j = 1; j < size - 1; j++) {
            double current_point = matrix[OFFSET(i, j, size)];
            int num_of_points = 1;
            num_of_points += i > 1 ? 1 : 0;
            num_of_points += j > 1 ? 1 : 0;
            num_of_points += i < size-2 ? 1 : 0;
            num_of_points += j < size-2 ? 1 : 0;
            // пятиточечный шаблон
            matrix_out[OFFSET(i, j, size)] = (matrix[OFFSET(i - 1, j, size)] + 
                                             matrix[OFFSET(i, j - 1, size)] + 
                                             matrix[OFFSET(i + 1, j, size)] +
                                             matrix[OFFSET(i, j + 1, size)] +
                                             matrix[OFFSET(i, j, size)]) * (1 / (float)num_of_points); 
            err = fmax(err, fabs(matrix_out[OFFSET(i, j, size)] - matrix[OFFSET(i, j, size)]));
        }
    }
    #pragma acc update self(matrix_out._A[:size * size]) async
    #pragma acc wait
    return err;
}

void copy_matrix(device_vector<double>& matrix, device_vector<double>& matrix_out, int size) {
    #pragma acc parallel loop
    for(int i = 0; i < size; i++) {
        #pragma acc loop
        for(int j = 0; j < size; j++ )
            matrix[OFFSET(i, j, size)] = matrix_out[OFFSET(i, j, size)];    
    }
}

void calculate_heatfield(device_vector<double>& matrix, device_vector<double>& matrix_out, int size, double max_error, int max_iterrations) {
    double error = 1;
    int it = 0;
    nvtxRangePushA("while");
    while (error > max_error && it < max_iterrations) {
        nvtxRangePushA("calc");
        error = calculate_step(matrix, matrix_out, size);
        nvtxRangePop();

        nvtxRangePushA("copy");
        copy_matrix(matrix, matrix_out, size);
        nvtxRangePop();
        it++;
    }
    
    std::cout << "num of iterations: " << it << "\n";
    std::cout << "error: " << error << "\n";
}

namespace po = boost::program_options;

int main(int argc, char** argv) {
    //default values init
    int size = 10;
    double max_error = 1e-6;
    int max_iterations = 1000000;

    // arg parser init
    po::options_description desc("Allowed options");
    desc.add_options()
                    ("help,h", "Show help")//"-s - size\n-me - max error of calculation\n-mit - max iteration count of calculation\n")
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


    device_vector<double> matrix = device_vector<double>((size+2) * (size+2));

    device_vector<double> matrix_out = device_vector<double>((size+2) * (size+2));
    size += 2; // add padding
    std::vector<std::tuple<int, double>> heat_points({std::make_tuple(size + 1, 10), 
                                                    std::make_tuple(2 * (size) - 2 , 20), 
                                                    std::make_tuple(size * size - 2 * (size) + 1, 30), 
                                                    std::make_tuple(size * size - size - 2, 40)
                                                    });
    nvtxRangePushA("init");
    initialize_field(matrix, heat_points);
    nvtxRangePop();
    std::cout << "size: " << size-2 << "x" << size-2 << '\n';
    const auto start{ std::chrono::steady_clock::now() };
    calculate_heatfield(matrix, matrix_out, size, max_error, max_iterations);
    const auto end{ std::chrono::steady_clock::now() };
    const std::chrono::duration<double> elapsed_seconds{ end - start };
    std::cout << elapsed_seconds.count() << " s\n";
    if (vm.count("draw_output")) {
        draw_field(matrix_out, size);
    }
    return 0;
}