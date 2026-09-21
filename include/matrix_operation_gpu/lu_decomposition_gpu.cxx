#include "include/matrix_operation_gpu/lu_decomposition_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/matrix_operation_gpu/matrix_transpose_gpu.h"
#include "include/util/exception.h"

namespace linalg
{
namespace gpu
{

void lu_decomposition(float* m, quarisma_int lda, int* pivot, int* info, cudaStream_t stream)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    int lwork = 0;
    if (cusolverDnSgetrf_bufferSize(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnSgetrf_bufferSize failed");
    }

    float* workspace = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&workspace),
            sizeof(float) * static_cast<size_t>(lwork)) != cudaSuccess)
    {
        LINALG_THROW("cudaMalloc failed");
    }

    // cuSOLVER is column-major: transposing the row-major buffer first
    // (device-to-device, matrix_transpose_gpu.cxx) makes its column-major
    // reading of the result equal the caller's actual A rather than A^T
    // (see that file's row/column-major identity), so getrf below factors A
    // itself and its pivots are A's own row-interchange sequence — matching
    // lapacke_?getrf's convention exactly. Transposing the packed L\U result
    // back afterward converts cuSOLVER's column-major packed output into
    // the row-major packed layout callers expect.
    matrix_transpose(lda, lda, m, stream);

    const auto  status      = cusolverDnSgetrf(handle, n, n, m, n, workspace, pivot, info);
    cudaError_t sync_status = cudaSuccess;
    detail::synchronize_and_free(workspace, stream, &sync_status);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnSgetrf failed");
    }
    if (sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaStreamSynchronize failed");
    }

    matrix_transpose(lda, lda, m, stream);
}

void lu_decomposition(double* m, quarisma_int lda, int* pivot, int* info, cudaStream_t stream)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    int lwork = 0;
    if (cusolverDnDgetrf_bufferSize(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnDgetrf_bufferSize failed");
    }

    double* workspace = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&workspace),
            sizeof(double) * static_cast<size_t>(lwork)) != cudaSuccess)
    {
        LINALG_THROW("cudaMalloc failed");
    }

    // See the float overload above for why the transpose-getrf-transpose
    // sequence is needed.
    matrix_transpose(lda, lda, m, stream);

    const auto  status      = cusolverDnDgetrf(handle, n, n, m, n, workspace, pivot, info);
    cudaError_t sync_status = cudaSuccess;
    detail::synchronize_and_free(workspace, stream, &sync_status);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnDgetrf failed");
    }
    if (sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaStreamSynchronize failed");
    }

    matrix_transpose(lda, lda, m, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
