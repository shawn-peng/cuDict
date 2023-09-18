#include <functional>
#include <cstdint>
#include <iostream>

#pragma once

#pragma pack(push)
#pragma pack(1)
template<typename ... T>
class Tuple {
public:
    friend std::ostream &operator <<(std::ostream &os, const Tuple &t) {
        return os;
    }
};

template<typename T>
class Tuple<T>
{
public:
    __host__ __device__ Tuple(const T& first) : first(first) {}
    __host__ __device__ Tuple(const Tuple<T> &other) = default;
    __host__ __device__ Tuple() {}

    T first;

    template<uint32_t N>
    __host__ __device__ constexpr auto &at() {
        static_assert(false);
        return first;
    }

    template<>
    __host__ __device__ constexpr auto &at<0>() {
        return first;
    }

    __host__ __device__ constexpr auto operator *() { return at<0>(); }

    __host__ __device__ constexpr auto operator ==(const Tuple<T> &other) const {
        return first == other.first;
    }

    friend std::ostream &operator <<(std::ostream &os, const Tuple<T> &t) {
        os << t.first;
        return os;
    }
};

template<typename T, typename ... Rest>
class Tuple<T, Rest...>
{
public:
    __host__ __device__ Tuple(const T& first, const Rest& ... rest)
        : first(first)
        , rest(rest...)
    {}
    __host__ __device__ Tuple(const Tuple<T, Rest...> &other) = default;
    __host__ __device__ Tuple() {}

    T first;
    Tuple<Rest...> rest;

    template<uint32_t N>
    __host__ __device__ constexpr auto &at() {
        return rest.at<N-1>();
    }

    template<>
    __host__ __device__ constexpr auto &at<0>() {
        return first;
    }

    __host__ __device__ constexpr auto operator *() { return at<0>(); }

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



