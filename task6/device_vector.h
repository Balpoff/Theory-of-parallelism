#include <cstddef>

// clang-format off
template <typename T> class device_vector {
public:
  device_vector() = default;

  explicit device_vector(size_t size) {
    _size = size;
    _A = new T[_size];
    memset(_A, 0, size * sizeof(T));
    #pragma acc enter data copyin(this) create(_A[0:_size])
  }

  explicit device_vector(size_t size, const T &value) {
    _size = size;
    _A = new T[_size];
    memset(_A, 0, size * sizeof(T));
    for (int i = 0; i < _size; ++i) {
      _A[i] = value;
    }
    #pragma acc enter data copyin(this, _A[0:_size])
  }

  ~device_vector() {
    #pragma acc exit data delete (this, _A[0:_size])
    //delete[] _A;
    _A = nullptr;
    _size = 0;
  }

  void update_host(int start, int end) {
    #pragma acc update self(_A[start:end])
  }

  void update_host_async(int start, int end, int block) {
    #pragma acc update self(_A[start:end]) async(block)
  } 

  void update_device(int start, int end) {
    #pragma acc update device(_A[start:end])
  }

  #pragma acc routine seq
  inline T &operator[](size_t idx) { return _A[idx]; }

  #pragma acc routine seq
  inline T &operator[](size_t idx) const { return _A[idx]; }

  #pragma acc routine seq
  size_t size() const { return _size; }

public:
  T *_A{nullptr};
  size_t _size{0};
};