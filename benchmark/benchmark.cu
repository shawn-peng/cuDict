#include <vector>
#include <map>
#include <cstdlib>
#include <iostream>
#include "utils.cuh"
#include "cuda_tuple.cuh"
#include "cuda_dict.cuh"


// template <typename TBM, typename TSUBJ>
// void run_benchmark(TBM benchmark) {
//     auto obj = TSUBJ(benchmark.init_data);
// }


typedef Tuple<uint32_t> Tuple1;
typedef Tuple<uint32_t, uint32_t> Tuple2;

template <typename TK, typename TV>
static auto generate_items(int size)
{
    auto res = std::vector<Tuple<TK, TV>>();
    for (int i = 0; i < size; i++)
    {
        res.push_back({(uint32_t)std::rand(), std::rand()});
    }
    // std::cout << res << std::endl;
    return res;
}

template <typename TK, typename TV>
static std::vector<TK> extract_keys(std::map<TK, TV> const &input_map)
{
    std::vector<TK> retval;
    for (auto const &element : input_map)
    {
        retval.push_back(element.first);
    }
    return retval;
}

template <typename TK, typename TV>
static std::vector<TV> extract_values(std::map<TK, TV> const &input_map)
{
    std::vector<TV> retval;
    for (auto const &element : input_map)
    {
        retval.push_back(element.second);
    }
    return retval;
}

template <uint32_t I, typename... T>
static auto extract_elem(std::vector<Tuple<T...>> const &input_container)
{
    using TV = TypeK<I, T...>;
    std::vector<TV> retval;
    for (auto const &element : input_container)
    {
        retval.push_back(get<I>(element));
    }
    return retval;
}

void benchmark_cuda_dict(const std::vector<Tuple<Tuple1, int32_t>> &items)
{
    // auto items = generate_items<Tuple1, int32_t>(size);
    // create a dict
    CUDA_Static_Dict<Tuple1, int32_t> *d;
    MEASURE_TIME("CUDA_Dict init", 
        d = new CUDA_Static_Dict<Tuple1, int32_t>(items);
    );
    // take sample items, fetch by keys, compare with values
    auto keys = extract_elem<0>(items);
    auto vals = extract_elem<1>(items);
    // std::cout << d << std::endl;
    // std::cout << keys << std::endl;
    // std::cout << vals << std::endl;

    std::vector<int32_t> outvals;
    MEASURE_TIME("CUDA_Dict get items", 
        outvals = d->get_items(keys);
    );
    Expect_Arr_Eq(outvals, vals);
    delete d;
}


void benchmark_stl(const std::vector<Tuple<Tuple1, int32_t>> &items)
{
    auto d = std::map<Tuple1, int32_t>();
    MEASURE_TIME("STL init", 
        for (const auto& [k, v] : items) {
            d[k] = v;
        }
    );
    auto keys = extract_keys(d);
    MEASURE_TIME("STL get items", 
        auto vals = std::vector<int32_t>();
        for (auto &k : keys) {
            vals.push_back(d[k]);
        }
    );
    // MEASURE_TIME(
    //     // Code to be measured
    //     for (int i = 0; i < 1000000; ++i) {
    //         // Perform some computation
    //     }
    // , "test for-loop");
}

float run_benchmark(int size)
{
    auto items = generate_items<Tuple1, int32_t>(size);
    benchmark_stl(items);
    benchmark_cuda_dict(items);
}

int main() {
    run_benchmark(1000000);
    return 0;
}

