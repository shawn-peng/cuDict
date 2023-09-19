# cuDict

## A Dictionary Library Runs on GPU

Use case example

```CUDA
const uint32_t Sentinel = -1;

typedef Tuple<uint32_t> Tuple1;

int main() {
    /// Saving data to GPU
    auto dict_data = std::vector<Tuple<Tuple1, uint32_t>>{
        {{1}, 3},
        {{2}, 4},
        {{3}, 5},
        {{4}, 6},
        {{5}, 7},
        {{6}, 8}};
    auto d = CUDA_Dict(dict_data, {Sentinel}, Sentinel);
    std::cout << d;

    /// Checking whether items in dictionary
    auto flags = d.contains_items({
        {1, 2, 3},
        {2, 3, 4},
        {3, 4, 5},
        {4, 5, 6},
    });
    std::cout << flags << std::endl;

    /// Get values associated with keys
    auto vals = d.get_items({
        {1, 2, 4},
        {2, 4, 4},
        {3, 4, 5},
        {4, 5, 6},
    });
    std::cout << vals << std::endl;

    return 0;
}
```
