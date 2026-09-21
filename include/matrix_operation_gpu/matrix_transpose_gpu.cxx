#include "include/matrix_operation_gpu/matrix_transpose_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "include/common/cuda_handle.h"
#include "include/util/exception.h"

namespace linalg
{
namespace gpu
{
namespace
{

// cublasSgeam/Dgeam are out-of-place (C = alpha*op(A) + beta*op(B)), and this
// API transposes `m` in place, so a scratch device buffer of the same size
// is used as the out-of-place destination and copied back device-to-device
// (never touching host memory). Both the geam call and the copy-back run on
// `stream`, so a caller-supplied non-default stream is honored end to end.
template <typename T, typename GeamFn>
void transpose_impl(
    GeamFn geam, linalg_long rows, linalg_long columns, T* m, cudaStream_t stream)
{
    if (m == nullptr)
    {
        LINALG_THROW("matrix_transpose: m must not be null");
    }
    if (rows == 0 || columns == 0)
    {
        LINALG_THROW("matrix_transpose: rows and columns must be positive");
    }
    const auto r = static_cast<int>(rows);
    const auto c = static_cast<int>(columns);

    T* scratch = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&scratch),
            sizeof(T) * static_cast<size_t>(r) * static_cast<size_t>(c)) != cudaSuccess)
    {
        LINALG_THROW("cudaMalloc failed");
    }

    auto handle = detail::cublas_handle_for_current_device();
    detail::set_stream(handle, stream);

    const T alpha = static_cast<T>(1);
    const T beta  = static_cast<T>(0);
    // Row-major `m` (rows x columns, ld=columns) is bit-identical to a
    // column-major (columns x rows, ld=columns) reading of the same buffer.
    // geam with transa=T treats that as the *pre-transpose* operand (so its
    // declared shape is n x m_out = columns x rows, matching `m`/ld exactly)
    // and produces a genuine (rows x columns, ld=rows) column-major result in
    // `scratch` — which is bit-identical to the desired row-major
    // (columns x rows, ld=rows) transposed output.
    if (geam(handle, CUBLAS_OP_T, CUBLAS_OP_T, r, c, &alpha, m, c, &beta, m, c, scratch, r) !=
        CUBLAS_STATUS_SUCCESS)
    {
        cudaFree(scratch);
        LINALG_THROW("cublasSgeam/Dgeam failed");
    }

    const auto copy_status = cudaMemcpyAsync(m,
        scratch,
        sizeof(T) * static_cast<size_t>(r) * static_cast<size_t>(c),
        cudaMemcpyDeviceToDevice,
        stream);
    // Waits for the copy above to actually finish reading `scratch` before
    // freeing it — see detail::synchronize_and_free. This makes the call
    // synchronous on its own internal cleanup, exactly like the blocking
    // cudaMemcpy it replaces; callers only get cross-call stream
    // composability, not intra-call concurrency, from this scratch buffer's
    // use of `stream`.
    cudaError_t sync_status = cudaSuccess;
    detail::synchronize_and_free(scratch, stream, &sync_status);
    if (copy_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync failed");
    }
    if (sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaStreamSynchronize failed");
    }
}

}  // namespace

void matrix_transpose(linalg_long rows, linalg_long columns, float* m, cudaStream_t stream)
{
    transpose_impl(cublasSgeam, rows, columns, m, stream);
}

void matrix_transpose(linalg_long rows, linalg_long columns, double* m, cudaStream_t stream)
{
    transpose_impl(cublasDgeam, rows, columns, m, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
