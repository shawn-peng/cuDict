#include "cuda_tuple.cuh"
#include "cuda_dict.cuh"
#include <cstdint>
#include <iostream>
#include <vector>

const uint32_t Sentinel = -1;

int main() {
    Tuple<int, uint16_t> t1 = {-2, 10};
    std::cout << "Tuple: (" << t1.at<0>() << ", " << t1.at<1>() << ")" << std::endl;

    Tuple<Tuple<uint32_t>, uint32_t> t2 = {2, 10};
    std::cout << "Tuple: (" << t2.at<0>() << ", " << t2.at<1>() << ")" << std::endl;

    auto dict_data = std::vector<Tuple<Tuple<uint32_t, uint32_t, uint32_t>, uint32_t>>{
        {{1, 2, 4}, 3},
        {{2, 3, 4}, 4},
        {{3, 4, 5}, 5},
        {{4, 5, 6}, 6},
        {{5, 5, 7}, 7},
        {{6, 9, 8}, 8}};
    auto d = CUDA_Dict(dict_data, {Sentinel, Sentinel, Sentinel}, Sentinel);
    std::cout << d;


    return 0;
}
