#pragma once

#include <cstddef>

#include "include/common/cuda_fwd.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{
namespace gpu
{

// Number of bytes `matrix_transpose` needs in its `workspace` argument for
// an `rows` x `columns` matrix; query this first and pass a caller-owned
// device buffer of at least this many bytes (the library never allocates
// its own GPU memory).
LINALG_API size_t matrix_transpose_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns);
LINALG_API size_t matrix_transpose_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns);

// `m`, `workspace` are CUDA device pointers (cudaMalloc'd); only available
// when built with LINALG_ENABLE_CUBLAS. No host memory is read or written by
// this call. `workspace` must point to at least
// `matrix_transpose_workspace_size(type_tag, rows, columns)` bytes; the
// caller must keep it alive and unmodified until work queued on `stream` has
// completed. `stream` (default: the legacy default stream) is the CUDA
// stream the underlying cuBLAS call and device-to-device copy are issued on;
// this call does not synchronize it. For host pointers, see
// linalg::matrix_transpose in include/matrix_operation/matrix_transpose.h.
LINALG_API void matrix_transpose(linalg_long rows, linalg_long columns, float* m, void* workspace,
    size_t workspace_size, cudaStream_t stream = nullptr);

LINALG_API void matrix_transpose(linalg_long rows, linalg_long columns, double* m, void* workspace,
    size_t workspace_size, cudaStream_t stream = nullptr);

}  // namespace gpu

namespace detail
{

// Raw transpose-in-place helper for other linalg::gpu::* translation units
// that already carved their own scratch (of exactly `rows * columns`
// elements of T) out of a workspace they were given, so they don't need a
// second, nested allocation just to call the public linalg::gpu::
// matrix_transpose above. `scratch` must point to at least `rows * columns`
// elements of device memory; `m` and `scratch` must not overlap.
LINALG_API void transpose_device(
    linalg_long rows, linalg_long columns, float* m, float* scratch, cudaStream_t stream);
LINALG_API void transpose_device(
    linalg_long rows, linalg_long columns, double* m, double* scratch, cudaStream_t stream);

}  // namespace detail
}  // namespace linalg
