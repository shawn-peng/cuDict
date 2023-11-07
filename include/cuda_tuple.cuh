#include <functional>
#include <cstdint>
#include <iostream>

#pragma once

#pragma pack(push)
#pragma pack(1)

template<uint32_t N, typename ... T>
struct GetHelper { };

template<typename ... T>
class Tuple {
public:
    // template<uint32_t N, typename TRet>
    // __host__ __device__ constexpr TRet &at();

    friend std::ostream &operator <<(std::ostream &os, const Tuple &t) {
        return os;
    }
};

template<typename T, typename ... Rest>
struct GetHelper<0, T, Rest...> {
    __host__ __device__ constexpr static T &get(Tuple<T, Rest...> &x) { return x.first; }
    __host__ __device__ constexpr static const T &get(const Tuple<T, Rest...> &x) { return x.first; }
};

template<uint32_t N, typename T, typename ... Rest>
struct GetHelper<N, T, Rest...> {
    __host__ __device__ constexpr static auto &get(Tuple<T, Rest...> &x) { return GetHelper<N-1, Rest...>::get(x.rest); }
    __host__ __device__ constexpr static const auto &get(const Tuple<T, Rest...> &x) { return GetHelper<N-1, Rest...>::get(x.rest); }
};

template<uint32_t N, typename ... T>
__host__ __device__ constexpr auto &get(Tuple<T...> &x) {
    return GetHelper<N, T...>().get(x);
}

template<uint32_t N, typename ... T>
__host__ __device__ constexpr const auto &get(const Tuple<T...> &x) {
    return GetHelper<N, T...>().get(x);
}



template<typename T>
class Tuple<T>
{
public:
    __host__ __device__ Tuple(const T& first) : first(first) {}
    __host__ __device__ Tuple(const Tuple<T> &other) = default;
    __host__ __device__ Tuple() {}

    T first;

    // template<uint32_t N>
    // __host__ __device__ constexpr T &at() {
    //     static_assert(false);
    //     return first;
    // }

    // template<>
    // __host__ __device__ constexpr T &at<0>() {
    //     return first;
    // }

    __host__ __device__ constexpr auto operator *() { return first; }

    __host__ __device__ constexpr auto operator ==(const Tuple<T> &other) const {
        return first == other.first;
    }

    __host__ __device__ operator T() const { return first; }

    friend std::ostream &operator <<(std::ostream &os, const Tuple<T> &t) {
        os << t.first;
        return os;
    }
};

template<typename T, typename ... Rest>
class Tuple<T, Rest...>
{
public:
    __host__ __device__ Tuple(const T& one)
    {
        Tuple(one, Rest(one) ...);
    }
    __host__ __device__ Tuple(const T& first, const Rest& ... rest)
        : first(first)
        , rest(rest...)
    {}
    __host__ __device__ Tuple(const Tuple<T, Rest...> &other) = default;
    __host__ __device__ Tuple() {}

    T first;
    Tuple<Rest...> rest;

    // template<uint32_t N>
    // __host__ __device__ constexpr auto &at() {
    //     return rest.at<N-1>();
    // }

    // template<>
    // __host__ __device__ constexpr auto &at<0>() {
    //     return first;
    // }

    __host__ __device__ constexpr auto operator *() { return first; }

    __host__ __device__ constexpr auto operator ==(const Tuple<T, Rest...> &other) const {
        return first == other.first && rest == other.rest;
    }

    friend std::ostream &operator <<(std::ostream &os, const Tuple<T, Rest...> &t) {
        os << t.first << ", " << t.rest;
        return os;
    }
};

#pragma pack(pop)


// template<uint32_t N, typename TFirst, typename ... TRest>
// constexpr auto &TupleAt(Tuple<TFirst, TRest...> &tuple) {
//     // if (N == 0) return tuple.first;
//     return TupleAt<N - 1, TRest...>(tuple.rest);

// }

// template<>
// constexpr auto &TupleAt<0, TTuple>(TTuple &tuple) {
//     return tuple.first;
// }



