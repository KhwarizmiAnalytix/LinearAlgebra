/*
 * LinearAlgebra — forward declaration of HIP's opaque stream handle.
 *
 * Lets the ROCM_BLAS struct-template specializations accept a caller's
 * stream parameter without requiring the ROCm toolkit's own headers on
 * every consumer's include path — in particular, a CPU-only build of this
 * library needs no HIP toolchain at all (see CMakeLists.txt), and these
 * headers are always compiled (only their .cpp/.hip bodies are guarded by
 * LINALG_HAS_ROCBLAS). Matches <hip/hip_runtime_api.h>'s own
 * `typedef struct ihipStream_t* hipStream_t;` exactly, so this is a
 * compatible redeclaration in any translation unit that also includes the
 * real header (every hipblas TU under LINALG_HAS_ROCBLAS does).
 */
#pragma once

struct ihipStream_t;
using hipStream_t = ihipStream_t*;
