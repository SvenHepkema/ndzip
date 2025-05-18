#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <utility>

#ifndef BENCHMARKER_CUH
#define BENCHMARKER_CUH

namespace custom {

constexpr size_t DEFAULT_N_VECTORS = 102400;

#define CUDA_SAFE_CALL(call)                                                   \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__,    \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace internal {
// All copied from main alp repo
using vbw_t = uint8_t;
using lane_t = uint16_t;
using si_t = uint32_t; // si = start index (of value within vector)
using vi_t = uint32_t; // vi = Vector Index

template <typename T> struct same_width_uint {
  using type = typename std::conditional<
      sizeof(T) == 8, uint64_t,
      typename std::conditional<
          sizeof(T) == 4, uint32_t,
          typename std::conditional<sizeof(T) == 2, uint16_t,
                                    uint8_t>::type>::type>::type;
};

template <typename T> struct Column {
  T *in;
  size_t n_values;
};

template <typename T, unsigned UNPACK_N_VECTORS = 1,
          unsigned UNPACK_N_VALUES = 1>
struct VectorLoader {
  using UINT_T = typename same_width_uint<T>::type;

  const UINT_T *in;

  __device__ __forceinline__ VectorLoader(const UINT_T *__restrict a_in,
                                          const lane_t lane)
      : in(a_in + lane){};

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) {
    constexpr int32_t N_LANES = 1024 / (sizeof(T) * 8);

#pragma unroll
    for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
#pragma unroll
      for (int j = 0; j < UNPACK_N_VALUES; ++j) {
        out[v * UNPACK_N_VALUES + j] = in[v * 1024 + j * N_LANES];
      }
    }

    in += UNPACK_N_VALUES * N_LANES;
  }
};

template <typename T> struct Loader {
  using UINT_T = typename same_width_uint<T>::type;
  VectorLoader<T> loader;

  __device__ __forceinline__ Loader(const Column<T> column,
                                    const vi_t vector_index, const lane_t lane)
      : loader(reinterpret_cast<UINT_T *>(column.in + vector_index * 1024),
               lane){};

  void __device__ unpack_next_into(T *__restrict out) {
    loader.unpack_next_into(out);
  }
};

template <typename T> constexpr int32_t get_n_lanes() {
  return 1024 / (sizeof(T) * 8);
}

template <typename T> constexpr int32_t get_values_per_lane() {
  return 1024 / get_n_lanes<T>();
}

constexpr size_t get_n_vecs_from_size(const size_t size) {
  return (size + 1024 - 1) / 1024;
}

template <typename T, unsigned UNPACK_N_VECTORS> struct VectorToWarpMapping {
  static constexpr uint32_t N_LANES = get_n_lanes<T>();
  static constexpr uint32_t N_VALUES_IN_LANE = get_values_per_lane<T>();

  __device__ __forceinline__ lane_t get_lane() const {
    return threadIdx.x % N_LANES;
  }

  __device__ __forceinline__ vi_t get_vector_index() const {
    // Concurrent vectors per block: how many vectors can be processed
    // by the block simultaneously, assuming that each thread is 1 lane

    const int32_t concurrent_vectors_per_block = blockDim.x / N_LANES;
    const int32_t vectors_per_block =
        concurrent_vectors_per_block * UNPACK_N_VECTORS;

    const int32_t concurrent_vector_index = threadIdx.x / N_LANES;
    const int32_t block_index = blockIdx.x;

    return vectors_per_block * block_index +
           concurrent_vector_index * UNPACK_N_VECTORS;
  }
};

template <typename T> struct ThreadblockMapping {
  static constexpr int32_t THREADS_PER_WARP = 32;
  static constexpr unsigned N_WARPS_PER_BLOCK =
      std::max(get_n_lanes<T>() / THREADS_PER_WARP, 2);
  static constexpr unsigned N_THREADS_PER_BLOCK =
      N_WARPS_PER_BLOCK * THREADS_PER_WARP;
  static constexpr unsigned N_CONCURRENT_VECTORS_PER_BLOCK =
      N_THREADS_PER_BLOCK / get_n_lanes<T>();

  const unsigned n_blocks;

  ThreadblockMapping(const size_t unpack_n_vecs, const size_t n_vecs)
      : n_blocks(n_vecs / (unpack_n_vecs * N_CONCURRENT_VECTORS_PER_BLOCK)) {}
};

template <typename T, unsigned N_VALUES> struct MagicChecker {
  const T magic_value;
  bool no_magic_found = true;

  __device__ __forceinline__ MagicChecker(const T magic_value)
      : magic_value(magic_value) {}

  __device__ __forceinline__ void check(const T *__restrict registers) {
#pragma unroll
    for (int i = 0; i < N_VALUES; ++i) {
      no_magic_found &= registers[i] != magic_value;
    }
  }

  __device__ __forceinline__ void write_result(bool *__restrict out) {
    // This is a branch, as we do not want to write 0s, only emit a write
    // if we found a magic value
    if (!no_magic_found) {
      *out = true;
    }
  }
};

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES,
          typename DecompressorT, typename ColumnT>
__global__ void query_column(const ColumnT column, bool *out,
                             const T magic_value) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToWarpMapping<T, UNPACK_N_VECTORS>();
  const lane_t lane = mapping.get_lane();
  const int32_t vector_index = mapping.get_vector_index();

  T registers[N_VALUES];
  auto checker = MagicChecker<T, N_VALUES>(magic_value);

  DecompressorT unpacker = DecompressorT(column, vector_index, lane);

  for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unpacker.unpack_next_into(registers);
    checker.check(registers);
  }

  checker.write_result(out);
}

template <typename T>
std::pair<T *, size_t> repeat_buffer_to_n_values(T *input_buffer,
                                                 const size_t input_n_values,
                                                 const size_t target_n_values) {
  T *output_buffer = new T[target_n_values];

	size_t output_buffer_offset = 0;
	size_t n_empty_values_column = target_n_values;
	while (n_empty_values_column != 0) {
		size_t n_values_to_copy = std::min(n_empty_values_column, input_n_values);
		std::memcpy(output_buffer + output_buffer_offset, input_buffer,
								n_values_to_copy * sizeof(T));
		output_buffer_offset += n_values_to_copy;
		n_empty_values_column -= n_values_to_copy;
	}

  return std::make_pair(output_buffer, target_n_values);
}

template <typename T = bool> T *allocate_query_result_buffer() {
  T *device_ptr = nullptr;
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void **>(&device_ptr), sizeof(T)));
  return device_ptr;
}

template <typename T = bool>
void deallocate_query_result_buffer(T *device_ptr) {
  if (device_ptr != nullptr) {
    CUDA_SAFE_CALL(cudaFree(device_ptr));
  }
  device_ptr = nullptr;
}
template <typename T>
void query_column(T *d_output_buffer, bool *d_query_result_buffer,
                  const size_t value_count) {
  internal::Column<T> d_column{d_output_buffer, value_count};

  const internal::ThreadblockMapping<T> mapping(
      1, internal::get_n_vecs_from_size(value_count));
  internal::query_column<T, 1, 1, internal::Loader<T>, internal::Column<T>>
      <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
          d_column, d_query_result_buffer, 0.231841231);
}

} // namespace internal

class Benchmark {
private:
  cudaEvent_t start_event, decompression_event, query_event;
  const cudaStream_t stream;
  bool *d_query_result_ptr;

public:
  Benchmark(cudaStream_t stream = 0) : stream(stream) {
    CUDA_SAFE_CALL(cudaEventCreate(&start_event));
    CUDA_SAFE_CALL(cudaEventCreate(&decompression_event));
    CUDA_SAFE_CALL(cudaEventCreate(&query_event));

    d_query_result_ptr = internal::allocate_query_result_buffer();
  }

  ~Benchmark() {
    CUDA_SAFE_CALL(cudaEventDestroy(start_event));
    CUDA_SAFE_CALL(cudaEventDestroy(decompression_event));
    CUDA_SAFE_CALL(cudaEventDestroy(query_event));

    internal::deallocate_query_result_buffer(d_query_result_ptr);
  }

  void start() {
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaEventRecord(start_event, stream));
  }

  template <typename T, typename input_T>
  void stop(input_T *decompressed_data_buffer, const size_t decompressed_size) {
    // WARNING: decompressed_size and compressed_size should either be n_values
    // or n_bytes
    CUDA_SAFE_CALL(cudaEventRecord(decompression_event, stream));
    T *reinterpreted_data_buffer =
        reinterpret_cast<T *>(decompressed_data_buffer);
    size_t reinterpreted_data_buffer_size =
        (decompressed_size * sizeof(input_T)) / sizeof(T);
    internal::query_column<T>(reinterpreted_data_buffer, d_query_result_ptr,
                              reinterpreted_data_buffer_size);
    CUDA_SAFE_CALL(cudaEventRecord(query_event, stream));

    CUDA_SAFE_CALL(cudaEventSynchronize(decompression_event));
    CUDA_SAFE_CALL(cudaEventSynchronize(query_event));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    float decompression_duration_ms = 0.0f;
    float decompression_query_duration_ms = 0.0f;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&decompression_duration_ms, start_event,
                                        decompression_event));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&decompression_query_duration_ms,
                                        start_event, query_event));

    fprintf(stdout, "GREPTAG[DECOMPRESSION];%f\n", decompression_duration_ms);
    fprintf(stdout, "GREPTAG[DECOMPRESSION_QUERY];%f\n",
            decompression_query_duration_ms);
  }

};

template<typename T>
void print_compression_ratio(const T compressed_size,
														 const T decompressed_size) {
	const double compression_ratio = static_cast<double>(decompressed_size) / static_cast<double>(compressed_size);

	fprintf(stdout, "GREPTAG[COMPRESSION_RATIO];%f\n",
					compression_ratio);
}
template <typename T>
T read_env_var(const std::string variable_name, const T default_value) {
  const char *value = std::getenv(variable_name.c_str());

  const bool variable_not_set = !value;
  if (variable_not_set) {
    return default_value;
  }

  std::istringstream string_stream(value);
  T result;
  string_stream >> result;
  return string_stream.fail() ? default_value : result;
}

template <typename T, typename input_T>
void resize_buffer_to_n_values(input_T *&input_buffer,
                               size_t &input_buffer_size) {
  const size_t n_vectors = read_env_var("N_VECTORS", DEFAULT_N_VECTORS);
  const size_t n_values = n_vectors * 1024;

  T *reinterpreted_input_buffer = reinterpret_cast<double *>(input_buffer);
  size_t reinterpreted_input_buffer_size =
      (input_buffer_size * sizeof(input_T)) / sizeof(T);

  std::pair<T *, size_t> resized_buffer =
      internal::repeat_buffer_to_n_values<T>(reinterpreted_input_buffer,
                                             reinterpreted_input_buffer_size,
                                             n_values);

  input_T *reinterpreted_resized_buffer =
      reinterpret_cast<input_T *>(resized_buffer.first);
  size_t reinterpreted_resized_buffer_size =
      (resized_buffer.second * sizeof(T)) / sizeof(input_T);

  // WARNING
  // As we do not know how the original input buffer was allocated,
  // we do not free it. This is acceptable as leaking memory
  // on the CPU for these GPU benchmarks is not catastrophic
  input_buffer = reinterpreted_resized_buffer;
  input_buffer_size = reinterpreted_resized_buffer_size;
}

} // namespace custom

#endif
