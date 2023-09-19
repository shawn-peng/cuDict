# cuDict

## A Dictionary Library Runs on GPU

Use case example

```CUDA
const uint32_t Sentinel = -1;

typedef Tuple<uint32_t> Tuple1;

int main() {
    auto dict_data = std::vector<Tuple<Tuple1, uint32_t>>{
        {{1}, 3},
        {{2}, 4},
        {{3}, 5},
        {{4}, 6},
        {{5}, 7},
        {{6}, 8}};
    auto d = CUDA_Dict(dict_data, {Sentinel}, Sentinel);
    std::cout << d;

    return 0;
}
```
