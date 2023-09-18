
#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

#define LEN(x) (sizeof(x) / sizeof(*x))
// template<typename T>
// constexpr uint32_t LEN(T x) { return sizeof(x) / sizeof(*x); }

// template<typename T>
// constexpr uint32_t LEN(Tuple x) { return sizeof(x) / sizeof(*x); }


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



template <typename T>
std::ostream &operator <<(std::ostream &os, const std::vector<T> &v) {
    int N = 16;
    os << "[" << std::endl;
    int i = 0;
    for (auto x : v) {
        os << x << "\t";
        i += 1;
        i %= N;
        if (i == 0) {
            os << std::endl;
        }
    }
    if (i != 0) {
        os << std::endl;
    }
    os << "]" << std::endl;
    return os;
}
