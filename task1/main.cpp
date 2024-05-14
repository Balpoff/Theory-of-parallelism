#include <iostream>
#include <cmath>
#include <vector>

constexpr long long mult = 10000000;

#ifdef DATA_TYPE_DOUBLE
typedef double DataType;
#else
typedef float DataType;
#endif

int main() {
    std::vector<DataType> sins(mult);
    DataType sum = 0;
    for (long long i = 0; i < mult; ++i) {
        sins[i] = std::sin(static_cast<DataType>(i) * 2 * M_PI / mult);
        sum += sins[i];
    }
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
