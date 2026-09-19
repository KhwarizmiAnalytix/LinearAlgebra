#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_CUBLAS

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "include/common/cuda_handle.h"
#include "include/matrix_operation/matrix_transpose_dispatch.h"
#include "include/util/exception.h"

namespace linalg
{
namespace detail
{
namespace
{

// cublasSgeam/Dgeam are out-of-place (C = alpha*op(A) + beta*op(B)), and this
// API transposes `m` in place, so a scratch device buffer of the same size
// is used as the out-of-place destination and copied back device-to-device.
template <typename T, typename GeamFn>
void transpose_cuda_impl(GeamFn geam, quarisma_long rows, quarisma_long columns, T* m)
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
            cublas_handle(),
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

void transpose_cuda_f32(quarisma_long rows, quarisma_long columns, float* m)
{
    transpose_cuda_impl(cublasSgeam, rows, columns, m);
}

void transpose_cuda_f64(quarisma_long rows, quarisma_long columns, double* m)
{
    transpose_cuda_impl(cublasDgeam, rows, columns, m);
}

LINALG_REGISTER_DISPATCH(transpose_f32_stub, cuda, transpose_cuda_f32);
LINALG_REGISTER_DISPATCH(transpose_f64_stub, cuda, transpose_cuda_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
