#include "cuda_tuple.cuh"
#include "cuda_dict.cuh"
#include <cstdint>
#include <iostream>
#include <vector>


int main() {
    Tuple<int, uint16_t> t1 = {-2, 10};
    std::cout << "Tuple: (" << t1.at<0>() << ", " << t1.at<1>() << ")" << std::endl;

    Tuple<Tuple<uint32_t>, uint32_t> t2 = {2, 10};
    std::cout << "Tuple: (" << t2.at<0>() << ", " << t2.at<1>() << ")" << std::endl;

    auto d = CUDA_Dict<Tuple<uint32_t>, uint32_t>({{1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}, {6, 8}});
    std::cout << d;


    return 0;
}
