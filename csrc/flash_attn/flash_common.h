#pragma once

#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include "flash.h"
#include "exception.h"
#include "static_switch.h"
#include "check.h"

enum ElementType { BF16, FP16, FP32 };

enum SimilarityType {
    sympower,
    softmax
};

void set_params_fprop(Flash_fwd_params &params,
					  ElementType element_type,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      void* q_ptr,
                      void* k_ptr,
                      void* v_ptr,
                      void* out_ptr,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      SimilarityType similarity,
                      int deg,
                      bool seqlenq_ngroups_swapped=false);

void set_params_splitkv(Flash_fwd_params &params, const int batch_size,
						const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
						const int head_size_rounded, const float p_dropout,
						const int num_splits, int multiProcessorCount, ElementType dtype);

template <typename T>
inline std::string Pack(const T& args) {
  return std::string(reinterpret_cast<const char*>(&args), sizeof(T));
}

template <typename T>
inline T Unpack(const void* opaque, size_t opaque_len) {
	T out;
	CHECK(sizeof(out)==opaque_len, "opaque len");
	memcpy(&out, opaque, opaque_len);
	return out;
}


template <typename T>
__forceinline__ __device__ T cuda_abs(T value);

// Specialization for __half (float16)
template <>
__forceinline__ __device__ __half cuda_abs<__half>(__half value) {
    return __habs(value);
}

// Specialization for __nv_bfloat16
template <>
__forceinline__ __device__ __nv_bfloat16 cuda_abs<__nv_bfloat16>(__nv_bfloat16 value) {
    return __habs(value);
}

// Specialization for float
template <>
__forceinline__ __device__ float cuda_abs<float>(float value) {
    return fabsf(value);
}


// template <typename T>
// __forceinline__ __device__ T convert_to_half(float value);

// // Specialization for __half (float16)
// template <>
// __forceinline__ __device__ __half convert_to_half<__half>(float value) {
//     return __float2half(value);
// }

// // Specialization for __nv_bfloat16
// template <>
// __forceinline__ __device__ __nv_bfloat16 convert_to_half<__nv_bfloat16>(float value) {
//     return __float2bfloat16(value);
// }

// // Specialization for __half (float16)
// template <>
// __forceinline__ __device__ __half convert_to_half<cutlass::bfloat16_t>(float value) {
//     return __float2bfloat16(value);
// }

// // Specialization for __nv_bfloat16
// template <>
// __forceinline__ __device__ __nv_bfloat16 convert_to_half<cutlass::half_t>(float value) {
//     return __float2bfloat16(value);
// }
