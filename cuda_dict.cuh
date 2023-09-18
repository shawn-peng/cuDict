#ifndef __CUDA_DICT_H__
#define __CUDA_DICT_H__

#include <vector>
#include <cstdint>
#include <iostream>
// #include "cuda_dict"
#include <cuda/std/atomic>

#include "utils.cuh"
// #include "cuda_array.cuh"
#include "cuda_tuple.cuh"


template <typename TKey, typename TVal>
struct CUDA_Dict;

template<typename TKey, typename TVal> // declaration
std::ostream& operator<<(std::ostream&, const CUDA_Dict<TKey, TVal>&);


__device__ extern uint32_t prime_factors[5];
const uint8_t MPRIME = LEN(prime_factors);

template<typename T>
__device__ uint32_t hash(T x) {
	return x % 999983;
}

// template <class T>
// struct show_type;

// template<typename TTuple, typename = enable_if_t::val>
// __device__ uint32_t tuple_hash(TTuple x, uint32_t MOD) {
// 	uint32_t hash_value;
// 	// for(uint8_t j = 0; j < LEN(x); j++) {
// 	// 	// decltype(x.at<0>())::something_made_up X;
// 	// 	hash_value = (hash_value + (hash(x.at<j>()) * prime_factors[j % MPRIME]) % MOD) % MOD;
// 	// }
// }

template<typename TFirst>
__device__ uint32_t tuple_hash(Tuple<TFirst> x, uint32_t MOD, uint8_t j = 0) {
	uint32_t hash_value;
	hash_value = (hash(x.first) * prime_factors[j % MPRIME]) % MOD;
	return hash_value;
}

template<typename TFirst, typename ... TRest>
__device__ uint32_t tuple_hash(Tuple<TFirst, TRest...> x, uint32_t MOD, uint8_t j = 0) {
	uint32_t hash_value;
	hash_value = (hash(x.first) * prime_factors[j % MPRIME]) % MOD;
	hash_value = (hash_value + tuple_hash(x.rest, MOD, j + 1)) % MOD;
	return hash_value;
}

template <typename T>
struct StrongType {
	T x;
	StrongType(const T &x) : x(x) {}
	__host__ __device__ constexpr operator T() const { return x; }
};

template <typename T>
struct EmptyKey : public StrongType<T> {
	EmptyKey(T x) : StrongType<T>(x) {}
};

template <typename T>
struct EmptyVal : public StrongType<T> {
	EmptyVal(T x) : StrongType<T>(x) {}
};


template <typename TKey, typename TVal>
struct HashTable { // On device view
	using TItem = Tuple<TKey, TVal>;
	using TBuk = cuda::std::atomic<TItem>;

	uint32_t capacity;
	TBuk *buckets;
	TItem sentinel;

	// HashTable(uint32_t capacit, TItem *buckets) : capacity(capacity) {}
	HashTable(uint32_t capacity, TItem *buckets, EmptyKey<TKey> sentinel_key, EmptyVal<TVal> sentinel_value)
	: capacity(capacity), buckets((TBuk *)buckets), sentinel{sentinel_key, sentinel_value}
	{}

	__device__ bool update_item(TItem item) { /// Idea from cuColllecitons
		auto &[k, v] = item;
		// get initial probing position from the hash value of the key
		auto i = tuple_hash(k, capacity);
		while (true) {
			// load the content of the bucket at the current probe position
			auto old_item = buckets[i].load(cuda::std::memory_order_relaxed);
			auto [old_k, old_v] = old_item;
			// if the bucket is empty we can attempt to insert the pair
			if (old_item == sentinel) {
				// try to atomically replace the current content of the bucket with the input pair
				bool success = buckets[i].compare_exchange_strong(
								old_item, item, cuda::std::memory_order_relaxed);
				if (success) {
					// store was successful
					return true;
				}
				// failed in racing, try next bucket
			} else if (old_k == k) {
				// input key is already present in the map
				return false;
			}
			// if the bucket was already occupied move to the next (linear) probing position
			// using the modulo operator to wrap back around to the beginning if we
			// go beyond the capacity
			i = ++i % capacity;
		}
	}

};

template <
	uint32_t BlockSize,
	typename TKey,
	typename TVal>
__global__ void initialize_ht(HashTable<TKey, TVal> ht) {
	auto i = BlockSize * blockIdx.x + threadIdx.x;
	if (i >= ht.capacity) return;
	ht.buckets[i] = ht.sentinel;
}

template <
	uint32_t BlockSize,
	typename TKey,
	typename TVal>
__global__ void update_items(HashTable<TKey, TVal> ht, uint32_t n, Tuple<TKey, TVal> *new_data) {
	auto i = BlockSize * blockIdx.x + threadIdx.x;
	if (i >= n) return;
	auto item = new_data[i];
	ht.update_item(item);
}

const uint16_t BLOCK_SIZE = 256;

template <typename TKey, typename TVal>
struct CUDA_Dict {
	using TItem = Tuple<TKey, TVal>;

	uint32_t size;
	uint32_t capacity;
	TItem *buckets;
	// HashTable<TKey, TVal> *ht;
	
	CUDA_Dict(const std::vector<TItem> &data) {
		auto n = data.size();
		this->size = n;
		this->capacity = 2 * n;
		cudaMalloc(&this->buckets, this->capacity * sizeof(TItem));

		auto const grid_size = (capacity + BLOCK_SIZE - 1/*Ceiling*/) / BLOCK_SIZE;
		initialize_ht<BLOCK_SIZE>
			<<<1, BLOCK_SIZE>>>(hashtable_view());

		this->update(data);
	}

	auto hashtable_view() {
		// return HashTable(2 * this->size, buckets, TKey(-1), TVal(-1));
		return HashTable(2 * this->size, buckets, EmptyKey(TKey(-1)), EmptyVal(TVal(-1)));
	}

	void update(const std::vector<TItem> &data) {
		auto n = data.size();
		TItem *temp_data;
		cudaMalloc(&temp_data, n * sizeof(TItem));
		cudaMemcpy(temp_data, data.data(), n * sizeof(TItem), cudaMemcpyHostToDevice);
		cudaFree(temp_data);
		auto const grid_size = (capacity + BLOCK_SIZE - 1/*Ceiling*/) / BLOCK_SIZE;
		update_items<BLOCK_SIZE>
			<<<grid_size, BLOCK_SIZE>>>(hashtable_view(), n, temp_data);
	}

	void get_data(std::vector<TItem> &data) const {
		data.resize(2 * this->size);
		cudaMemcpy(data.data(), this->buckets, 2 * this->size * sizeof(TItem), cudaMemcpyDeviceToHost);
	}

	friend std::ostream& operator<< <> (std::ostream&, const CUDA_Dict&);

};

template <typename TKey, typename TVal>
std::ostream &operator << (std::ostream &os, const CUDA_Dict<TKey, TVal> &d) {
	std::vector<Tuple<TKey, TVal>> dict_data;
	d.get_data(dict_data);

	os << "{ CUDA_Dict" << std::endl;
	for (int i = 0; i < dict_data.size(); i++) {
		// os << "  " << dict_data[i].at<0>() << ": " << dict_data[i].at<1>() << std::endl;
		auto elem = dict_data[i];
		// std::remove_reference<decltype(elem.at<0>())>::something x;
		auto [k, v] = elem;
		// auto k = elem.at<0>();
		// auto v = elem.at<1>();
		os << "  " << k << ": " << v << std::endl;
	}
	os << "}" << std::endl;
	return os;
}

#endif  // __CUDA_DICT_H__
