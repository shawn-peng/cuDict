
#include "cuda_dict.cuh"

__device__ extern uint32_t prime_factors[] = {31, 37, 41, 43, 47};

typedef Tuple<uint32_t> Tuple1;

template
struct CUDA_Dict<Tuple1, int32_t>;

template
std::ostream &operator << <Tuple1, int32_t>(std::ostream &os, const CUDA_Dict<Tuple1, int32_t> &d);
