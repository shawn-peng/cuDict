
#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#define LEN(x) (sizeof(x) / sizeof(*x))
// template<typename T>
// constexpr uint32_t LEN(T x) { return sizeof(x) / sizeof(*x); }

// template<typename T>
// constexpr uint32_t LEN(Tuple x) { return sizeof(x) / sizeof(*x); }

#define MEASURE_TIME(desc, ...) do { \
    auto start_time = std::chrono::high_resolution_clock::now(); \
    __VA_ARGS__ \
    auto end_time = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double> elapsed_time = end_time - start_time; \
    std::cout << "Elapsed time: " << elapsed_time.count() << " seconds for " << desc << std::endl; \
} while (0)


template <uint32_t K, typename... T>
struct GetTypeK;

template <typename T, typename... Ts>
struct GetTypeK<0, T, Ts...>
{
    using type = T;
};

template <uint32_t K, typename T, typename... Ts>
struct GetTypeK<K, T, Ts...> : GetTypeK<K - 1, Ts...>
{
};

template <size_t K, typename... Ts>
using TypeK = typename GetTypeK<K, Ts...>::type;


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
bool Expect_Arr_Eq(const std::vector<T> &arr_res, const std::vector<T> &arr_exp)
{
    assert(arr_res.size() == arr_exp.size());
    for (int i = 0; i < arr_exp.size(); i++)
    {
        assert(arr_res[i] == arr_exp[i]);
        return false;
    }
    return true;
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
