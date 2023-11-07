
#pragma once

#include <cuda/std/functional>

#include "utils.cuh"

#pragma pack(push)
#pragma pack(2)

template<typename T>
struct atomic { // atomic device view of element
    constexpr uint16_t size = sizeof(T);

    T &data;

    __device__ atomic(T &x) : data(x) {
    }

    __device__ T load();
    bool compare_exchange_strong(expect, newval, memory_order);
};

#pragma pack(pop)

