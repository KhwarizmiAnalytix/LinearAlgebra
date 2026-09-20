#include "include/matrix_operation_gpu/matrix_transpose_gpu.h"

#include "include/common/configure.h"  // IWYU pragma: keep

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
// (never touching host memory).
template <typename T, typename GeamFn>
void transpose_impl(GeamFn geam, quarisma_long rows, quarisma_long columns, T* m)
{
    const auto r = static_cast<int>(rows);
    const auto c = static_cast<int>(columns);

    T* scratch = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&scratch), sizeof(T) * static_cast<size_t>(r) * static_cast<size_t>(c));

    const T alpha = static_cast<T>(1);
    const T beta  = static_cast<T>(0);
    // Row-major `m` (rows x columns, ld=columns) is bit-identical to a
    // column-major (columns x rows, ld=columns) reading of the same buffer.
    // geam with transa=T treats that as the *pre-transpose* operand (so its
    // declared shape is n x m_out = columns x rows, matching `m`/ld exactly)
    // and produces a genuine (rows x columns, ld=rows) column-major result in
    // `scratch` — which is bit-identical to the desired row-major
    // (columns x rows, ld=rows) transposed output.
    if (geam(
            detail::cublas_handle(),
            CUBLAS_OP_T,
            CUBLAS_OP_T,
            r,
            c,
            &alpha,
            m,
            c,
            &beta,
            m,
            c,
            scratch,
            r) != CUBLAS_STATUS_SUCCESS)
    {
        cudaFree(scratch);
        LINALG_THROW("cublasSgeam/Dgeam failed");
    }

    cudaMemcpy(
        m, scratch, sizeof(T) * static_cast<size_t>(r) * static_cast<size_t>(c), cudaMemcpyDeviceToDevice);
    cudaFree(scratch);
}

}  // namespace

void matrix_transpose(quarisma_long rows, quarisma_long columns, float* m)
{
    transpose_impl(cublasSgeam, rows, columns, m);
}

void matrix_transpose(quarisma_long rows, quarisma_long columns, double* m)
{
    transpose_impl(cublasDgeam, rows, columns, m);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
