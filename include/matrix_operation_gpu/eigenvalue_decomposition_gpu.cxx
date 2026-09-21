#include "include/matrix_operation_gpu/eigenvalue_decomposition_gpu.h"

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
namespace
{

// See svd_decomposition_gpu.h / qr_decomposition_gpu.h's RESTRICTION note:
// every caller-visible buffer here is required to be tightly packed
// row-major.
template <typename T, typename BufferSizeFn, typename SyevdFn>
void symmetric_eigen_impl(BufferSizeFn buffer_size,
    SyevdFn                            syevd,
    const T*                           A,
    linalg_int                       n,
    linalg_int                       lda,
    T*                                 eigenvalues,
    T*                                 eigenvectors,
    linalg_int                       ldv,
    int*                               info,
    cudaStream_t                       stream)
{
    if (A == nullptr || eigenvalues == nullptr || eigenvectors == nullptr || info == nullptr)
    {
        LINALG_THROW(
            "symmetric_eigenvalue_decomposition: A, eigenvalues, eigenvectors, and info must not be "
            "null");
    }
    if (n <= 0)
    {
        LINALG_THROW("symmetric_eigenvalue_decomposition: n must be positive");
    }
    if (lda != n || ldv != n)
    {
        LINALG_THROW("linalg::gpu::symmetric_eigenvalue_decomposition requires tightly packed "
                     "lda/ldv (both == n)");
    }

    const auto nn      = static_cast<int>(n);
    auto       handle  = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    // Symmetric matrix: A == A^T, so (unlike qr/svd) no transpose is needed
    // on input — a row-major-LOWER-supplied matrix and a
    // column-major-UPPER-read of the same bytes describe the identical
    // symmetric matrix (same identity cholesky_decomposition_gpu.cxx's
    // to_fill_mode relies on). The CPU contract only ever references A's
    // lower triangle, so the fill mode passed to cuSOLVER is always UPPER.
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    const auto n2 = static_cast<size_t>(nn) * static_cast<size_t>(nn);
    T*         work_a = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&work_a), sizeof(T) * n2) != cudaSuccess)
    {
        LINALG_THROW("cudaMalloc failed");
    }
    if (cudaMemcpyAsync(work_a, A, sizeof(T) * n2, cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
    {
        cudaFree(work_a);
        LINALG_THROW("cudaMemcpyAsync failed");
    }

    int lwork = 0;
    if (buffer_size(handle, CUSOLVER_EIG_MODE_VECTOR, fill, nn, work_a, nn, eigenvalues, &lwork) !=
        CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(work_a);
        LINALG_THROW("cusolverDn*syevd_bufferSize failed");
    }

    T* workspace = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(T) * static_cast<size_t>(lwork)) !=
        cudaSuccess)
    {
        cudaFree(work_a);
        LINALG_THROW("cudaMalloc failed");
    }

    const auto status = syevd(
        handle, CUSOLVER_EIG_MODE_VECTOR, fill, nn, work_a, nn, eigenvalues, workspace, lwork, info);
    cudaFree(workspace);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(work_a);
        LINALG_THROW("cusolverDn*syevd failed");
    }

    // work_a now holds the eigenvectors column-major (n x n); the CPU
    // convention wants them row-major, one eigenvector per column.
    matrix_transpose(static_cast<linalg_long>(n), static_cast<linalg_long>(n), work_a, stream);
    const auto copy_status =
        cudaMemcpyAsync(eigenvectors, work_a, sizeof(T) * n2, cudaMemcpyDeviceToDevice, stream);
    cudaFree(work_a);
    if (copy_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync failed");
    }
}

}  // namespace

void symmetric_eigenvalue_decomposition(const float* A,
    linalg_int                                     n,
    linalg_int                                     lda,
    float*                                           eigenvalues,
    float*                                           eigenvectors,
    linalg_int                                     ldv,
    int*                                              info,
    cudaStream_t                                      stream)
{
    symmetric_eigen_impl(
        cusolverDnSsyevd_bufferSize, cusolverDnSsyevd, A, n, lda, eigenvalues, eigenvectors, ldv, info, stream);
}

void symmetric_eigenvalue_decomposition(const double* A,
    linalg_int                                      n,
    linalg_int                                      lda,
    double*                                           eigenvalues,
    double*                                           eigenvectors,
    linalg_int                                      ldv,
    int*                                               info,
    cudaStream_t                                       stream)
{
    symmetric_eigen_impl(
        cusolverDnDsyevd_bufferSize, cusolverDnDsyevd, A, n, lda, eigenvalues, eigenvectors, ldv, info, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
