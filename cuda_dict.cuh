#ifndef __CUDA_DICT_H__
#define __CUDA_DICT_H__

#include <vector>
#include <cstdint>
#include <iostream>
// #include "cuda_dict"
#include <cuda/std/atomic>
#include <cooperative_groups.h>

#include "utils.cuh"
#include "cuda_tuple.cuh"


#pragma pack(push)
#pragma pack(1)

namespace cg = cooperative_groups;


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

static cudaError_t cuda_ret;

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
	
	enum class insert_result {
		HT_CONTINUE,	///< Insert did not succeed, continue trying to insert
		HT_SUCCESS,		///< New pair inserted successfully
		HT_DUPLICATE	///< Insert did not succeed, key is already present
	};

	uint32_t capacity;
	TBuk *buckets;
	// TItem *buckets;
	TItem sentinel;

	// HashTable(uint32_t capacity, TBuk *buckets, EmptyKey<TKey> sentinel_key, EmptyVal<TVal> sentinel_value)
	HashTable(uint32_t capacity, TBuk *buckets, EmptyKey<TKey> sentinel_key, EmptyVal<TVal> sentinel_value)
	: capacity(capacity), buckets(buckets), sentinel{sentinel_key, sentinel_value}
	{}

	template <uint16_t GroupSize>
	__device__ bool update_item(cg::thread_block_tile<GroupSize> group, TItem item) { /// Idea from cuColllecitons
		auto &[k, v] = item;
		// get initial probing position from the hash value of the key
		auto i = (tuple_hash(k, capacity) + group.thread_rank()) % capacity;
		insert_result status{insert_result::HT_CONTINUE};
		while (true) {
			auto &bucket = buckets[i];
			// load the content of the bucket at the current probe position
			auto old_item = bucket.load(cuda::std::memory_order_relaxed);
			auto [old_k, old_v] = old_item;
			// input key is already present in the map
			if (group.any(old_k == k)) return false;

			// each rank checks if its current bucket is empty, i.e., a candidate bucket for insertion
			auto const empty_mask = group.ballot(old_item == sentinel);

			// if the bucket is empty we can attempt to insert the pair
			if(empty_mask) {
				// elect a candidate rank (here: thread with lowest rank in mask)
				auto const candidate = __ffs(empty_mask) - 1;
				if(group.thread_rank() == candidate) {
					// attempt atomically swapping the input pair into the bucket
					bool const success = buckets[i].compare_exchange_strong(
													old_item, item, cuda::std::memory_order_relaxed);
					if (success) {
						// insertion went successful
						status = insert_result::HT_SUCCESS;
					} else if (old_k == k) {
						// else, re-check if a duplicate key has been inserted at the current probing position
						status = insert_result::HT_DUPLICATE;
					}
				}
				// broadcast the insertion result from the candidate rank to all other ranks
				auto const candidate_status = group.shfl(status, candidate);
				if(candidate_status == insert_result::HT_SUCCESS) return true;
				if(candidate_status == insert_result::HT_DUPLICATE) return false;
			} else {
				// else, move to the next (linear) probing window
				i = (i + group.size()) % capacity;
			}
		}
	}
	
	template <uint16_t GroupSize, typename TRet = bool>
	__device__ TRet contains_item(cg::thread_block_tile<GroupSize> group, TKey k) const {
		// get initial probing position from the hash value of the key
		auto i = (tuple_hash(k, capacity) + group.thread_rank()) % capacity;
		while (true) {
			auto &bucket = buckets[i];
			// load the content of the bucket at the current probe position
			auto old_item = bucket.load(cuda::std::memory_order_relaxed);
			auto [old_k, old_v] = old_item;
			// input key is present in the map
			if (group.any(old_k == k)) return true;

			// each rank checks if its current bucket is empty, i.e., a candidate bucket for insertion
			auto const empty_mask = group.ballot(old_item == sentinel);

			// if the bucket is empty we don't have the k in the hash map
			if (empty_mask) {
				return false;
			}

			// move to the next (linear) probing window
			i = (i + group.size()) % capacity;
		}
	}
	
	template <uint16_t GroupSize>
	__device__ TVal get_item(cg::thread_block_tile<GroupSize> group, const TKey k) const {
		// get initial probing position from the hash value of the key
		auto i = (tuple_hash(k, capacity) + group.thread_rank()) % capacity;
		while (true) {
			auto &bucket = buckets[i];
			// load the content of the bucket at the current probe position
			auto old_item = bucket.load(cuda::std::memory_order_relaxed);
			auto [old_k, old_v] = old_item;
			// input key is present in the map
			if (group.any(old_k == k))
				return old_v;

			// each rank checks if its current bucket is empty, i.e., a candidate bucket for insertion
			auto const empty_mask = group.ballot(old_item == sentinel);

			// if the bucket is empty we don't have the k in the hash map
			if (empty_mask) {
				// return sentinel.at<1>();
				// return sentinel.rest.first;
				return get<1>(sentinel);
			}

			// move to the next (linear) probing window
			i = (i + group.size()) % capacity;
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
__global__ void dump_data(const HashTable<TKey, TVal> ht, Tuple<TKey, TVal> *dest) {
	auto i = BlockSize * blockIdx.x + threadIdx.x;
	if (i >= ht.capacity) return;
	dest[i] = ht.buckets[i].load(cuda::std::memory_order_relaxed);
}

template <
	uint32_t BlockSize,
	uint32_t GroupSize, // Size of a cooperative group to update one item in parallel
	typename TKey,
	typename TVal>
__global__ void update_items(HashTable<TKey, TVal> ht, uint32_t n, Tuple<TKey, TVal> *items) {
	auto tile = cg::tiled_partition<GroupSize>(cg::this_thread_block());
	auto i = (BlockSize * blockIdx.x + threadIdx.x) / GroupSize;
	if (i >= n) return;
	ht.update_item<GroupSize>(tile, items[i]);
}

template <
	uint32_t BlockSize,
	uint32_t GroupSize,
	typename TKey,
	typename TVal,
	typename TRet = bool>
__global__ void contains_items(const HashTable<TKey, TVal> ht, uint32_t n, const TKey *keys, TRet *dest) {
	auto tile = cg::tiled_partition<GroupSize>(cg::this_thread_block());
	auto i = (BlockSize * blockIdx.x + threadIdx.x) / GroupSize;
	if (i >= n) return;
	dest[i] = ht.contains_item<GroupSize>(tile, keys[i]);
}

template <
	uint32_t BlockSize,
	uint32_t GroupSize,
	typename TKey,
	typename TVal>
__global__ void get_items(const HashTable<TKey, TVal> ht, uint32_t n, const TKey *keys, TVal *dest) {
	auto tile = cg::tiled_partition<GroupSize>(cg::this_thread_block());
	auto i = (BlockSize * blockIdx.x + threadIdx.x) / GroupSize;
	if (i >= n) return;
	dest[i] = ht.get_item<GroupSize>(tile, keys[i]);
}

template <typename T>
constexpr T *cuMalloc(const uint32_t n) {
	T *p;
	auto ret = cudaMalloc(&p, n * sizeof(T));
	gpuErrchk(ret);
	return p;
}

template <typename T>
constexpr auto cuToDev(const std::vector<T> &host) {
	T *p = cuMalloc<T>(host.size());
	cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
	return p;
}

template <typename T>
constexpr auto cuToHost(uint32_t n, const T *p) {
	auto host = std::vector<T>(n);
	cudaMemcpy(host.data(), p, host.size() * sizeof(T), cudaMemcpyDeviceToHost);
	return host;
}

template <typename T>
constexpr auto cuDel(T *p) { cudaFree(p); }

template <typename T>
struct DevVec {
	T *p;
	uint32_t n;
	DevVec(uint32_t n) : n(n) {
		p = cuMalloc<T>(n);
	}
	DevVec(const std::vector<T> &src) {
		p = cuToDev(src);
	}
	auto ToHost() {
		return cuToHost(n, p);
	}
	~DevVec() { cuDel(p); }

	operator T*() { return p; }
	operator std::vector<T>() { return ToHost(); }
};


const uint16_t BLOCK_SIZE = 256;

const uint16_t CG_SIZE = 4;

template <typename TKey, typename TVal>
struct CUDA_Dict {
	using TItem = Tuple<TKey, TVal>;
	using TBuk = cuda::std::atomic<TItem>;

	uint32_t size;
	uint32_t capacity;
	// TItem *buckets;
	TBuk *buckets;
	TKey sentinel_key;
	TVal sentinel_val;
	
	CUDA_Dict(const std::vector<TItem> &data,
			TKey sentinel_key, TVal sentinel_val)
	: sentinel_key(sentinel_key), sentinel_val(sentinel_val) {
		auto n = data.size();
		this->size = n;
		this->capacity = 2 * n;
		// cudaMalloc(&this->buckets, this->capacity * sizeof(TBuk));
		cudaMalloc(&this->buckets, this->capacity * sizeof(TBuk));

		auto const grid_size = (capacity + BLOCK_SIZE - 1/*Ceiling*/) / BLOCK_SIZE;
		initialize_ht<BLOCK_SIZE>
			<<<grid_size, BLOCK_SIZE>>>(hashtable_view());

		this->update(data);
	}

	auto hashtable_view() {
		return HashTable(2 * this->size, buckets, EmptyKey(sentinel_key), EmptyVal(sentinel_val));
	}

	const auto hashtable_view() const {
		return HashTable(2 * this->size, buckets, EmptyKey(sentinel_key), EmptyVal(sentinel_val));
	}

	void update(const std::vector<TItem> &data) {
		auto n = data.size();
		// TItem *temp_data;
		// cudaMalloc(&temp_data, n * sizeof(TItem));
		TItem *temp_data = cuMalloc<TItem>(n);
		cuda_ret = cudaMemcpy(temp_data, data.data(), n * sizeof(TItem), cudaMemcpyHostToDevice);
		gpuErrchk(cuda_ret);
		auto const grid_size = (n + BLOCK_SIZE - 1/*Ceiling*/) / BLOCK_SIZE;
		update_items<BLOCK_SIZE, CG_SIZE>
			<<<grid_size, BLOCK_SIZE>>>(hashtable_view(), n, temp_data);
		cudaFree(temp_data);
	}

	void dump_data(std::vector<TItem> &data) const {
		data.resize(2 * this->size);
		auto const grid_size = (capacity + BLOCK_SIZE - 1/*Ceiling*/) / BLOCK_SIZE;
		TItem *temp_data = cuMalloc<TItem>(capacity);
		// cudaMalloc(&temp_data, capacity * sizeof(TItem));
		::dump_data<BLOCK_SIZE>
			<<<grid_size, BLOCK_SIZE>>>(hashtable_view(), temp_data);
		cudaMemcpy(data.data(), temp_data, capacity * sizeof(TItem), cudaMemcpyDeviceToHost);
		cudaFree(temp_data);
	}

	// auto contains_items(const std::initializer_list<TKey> &arg) {
	// 	auto keys = std::vector<TKey>(arg);
	// 	return contains_items(keys);
	// }

	// auto get_items(const std::initializer_list<TKey> &arg) {
	// 	auto keys = std::vector<TKey>(arg);
	// 	return get_items(keys);
	// }

	auto contains_items(const std::vector<TKey> &keys) {
		using T = int;
		auto n = keys.size();
		auto dev_in = DevVec(keys);
		auto dev_out = DevVec<T>(n);
		auto const grid_size = (n + BLOCK_SIZE - 1/*Ceiling*/) / BLOCK_SIZE;
		::contains_items<BLOCK_SIZE, CG_SIZE>
			<<<grid_size, BLOCK_SIZE>>>(hashtable_view(), n, dev_in.p, dev_out.p);
		return dev_out.ToHost();
	}

	auto get_items(const std::vector<TKey> &keys) {
		using T = TVal;
		auto n = keys.size();
		auto dev_in = DevVec(keys);
		auto dev_out = DevVec<T>(n);
		auto const grid_size = (n + BLOCK_SIZE - 1/*Ceiling*/) / BLOCK_SIZE;
		::get_items<BLOCK_SIZE, CG_SIZE>
			<<<grid_size, BLOCK_SIZE>>>(hashtable_view(), n, dev_in.p, dev_out.p);
		return dev_out.ToHost();
	}


	friend std::ostream& operator<< <> (std::ostream&, const CUDA_Dict&);

};

template <typename TKey, typename TVal>
std::ostream &operator << (std::ostream &os, const CUDA_Dict<TKey, TVal> &d) {
	std::vector<Tuple<TKey, TVal>> dict_data;
	d.dump_data(dict_data);

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
#pragma pack(pop)

#endif  // __CUDA_DICT_H__
