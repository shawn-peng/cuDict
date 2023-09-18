
#include "cuda_dict.h"

__device__ extern uint32_t prime_factors[] = {31, 37, 41, 43, 47};


template
struct CUDA_Dict<Tuple1, int32_t>;

template
std::ostream &operator << <Tuple1, int32_t>(std::ostream &os, const CUDA_Dict<Tuple1, int32_t> &d);
