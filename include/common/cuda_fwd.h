/*
 * LinearAlgebra — forward declaration of CUDA's opaque stream handle.
 *
 * Lets linalg::gpu::* public headers accept a caller's stream parameter
 * without requiring the CUDA toolkit's own headers on every consumer's
 * include path — in particular, a CPU-only build of this library needs no
 * CUDA toolchain at all (see CMakeLists.txt), and these headers are always
 * compiled (only their .cxx bodies are guarded by LINALG_ENABLE_CUBLAS).
 * Matches <cuda_runtime_api.h>'s own `typedef struct CUstream_st*
 * cudaStream_t;` exactly, so this is a compatible redeclaration in any
 * translation unit that also includes the real header (every *_gpu.cxx
 * file under LINALG_ENABLE_CUBLAS does).
 */
#pragma once

struct CUstream_st;
using cudaStream_t = CUstream_st*;
