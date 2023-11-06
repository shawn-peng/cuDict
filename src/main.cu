#include "cuda_tuple.cuh"
#include "cuda_dict.cuh"
#include <cstdint>
#include <iostream>
#include <vector>

const uint32_t Sentinel = -1;

typedef Tuple<uint32_t> Tuple1;
typedef Tuple<uint32_t, uint32_t> Tuple2;
typedef Tuple<uint32_t, uint32_t, uint16_t> Tuple3;
typedef Tuple<uint32_t, uint32_t, uint32_t, uint32_t> Tuple4;

int main() {
    // Tuple<int, uint16_t> t1 = {-2, 10};
    // std::cout << "Tuple: (" << t1.at<0>() << ", " << t1.at<1>() << ")" << std::endl;

    // Tuple<Tuple<uint32_t>, uint32_t> t2 = {2, 10};
    // std::cout << "Tuple: (" << t2.at<0>() << ", " << t2.at<1>() << ")" << std::endl;

    // auto [x, y] = t2;
    // using xx = decltype(y)::

    auto dict_data = std::vector<Tuple<Tuple3, uint32_t>>{
        {{1, 2, 4}, 3},
        {{2, 3, 4}, 4},
        {{3, 4, 5}, 5},
        {{4, 5, 6}, 6},
        {{5, 5, 7}, 7},
        {{6, 9, 8}, 8}};
    auto d = CUDA_Static_Dict(dict_data, {Sentinel, Sentinel, (uint16_t)Sentinel}, Sentinel);
    std::cout << d;

    auto flags = d.contains_items({
        {1, 2, 3},
        {2, 3, 4},
        {3, 4, 5},
        {4, 5, 6},
    });
    std::cout << flags << std::endl;

    auto vals = d.get_items({
        {1, 2, 4},
        {2, 4, 4},
        {3, 4, 5},
        {4, 5, 6},
    });
    std::cout << vals << std::endl;

    return 0;
}
