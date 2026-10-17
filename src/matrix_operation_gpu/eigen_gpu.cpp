#include "include/matrix_operation_gpu/eigen_gpu.h"

#if LINALG_HAS_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/matrix_operation_gpu/transpose_gpu.h"
#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"

#include "include/common/linalg_profiler.h"

namespace linalg
{
namespace gpu
{
namespace
{

template <typename T>
struct symmetric_eigen_layout
{
    T* work_a;
    T* workspace;
    T* transpose_scratch;
};

// Three scratch buffers: the row-major-to-column-major copy of A,
// cuSOLVER's own syevd workspace, and the transpose scratch
// detail::transpose_device needs for the post-syevd eigenvector transpose.
// Shared by the size query and the real call (see
// include/common/cuda_workspace.h).
template <typename T>
symmetric_eigen_layout<T> symmetric_eigen_plan(detail::workspace_arena& arena, int n, int lwork)
{
    symmetric_eigen_layout<T> p;
    p.work_a            = arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    p.workspace         = arena.next<T>(static_cast<size_t>(lwork));
    p.transpose_scratch = arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    return p;
}

template <typename T, typename BufferSizeFn>
size_t symmetric_eigen_workspace_size_impl(BufferSizeFn buffer_size, linalg_int n)
{
    LINALG_CHECK(n > 0, "symmetric_eigenvalue_decomposition_workspace_size: n must be positive");
    const auto nn     = detail::checked_narrow<int>(n);
    auto       handle = detail::cusolver_handle_for_current_device();
    const auto fill   = CUBLAS_FILL_MODE_UPPER;
    int        lwork  = 0;
    LINALG_CHECK(buffer_size(handle, CUSOLVER_EIG_MODE_VECTOR, fill, nn, static_cast<T*>(nullptr), nn,
                      static_cast<T*>(nullptr), &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*syevd_bufferSize failed");
    detail::workspace_arena arena;
    symmetric_eigen_plan<T>(arena, nn, lwork);
    return arena.total_size();
}

// See svd_decomposition_gpu.h / qr_decomposition_gpu.h's RESTRICTION note:
// every caller-visible buffer here is required to be tightly packed
// row-major. `workspace`/`workspace_size` are caller-owned; the caller must
// keep them alive and unmodified until work queued on `stream` has
// completed.
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
    void*                              workspace,
    size_t                             workspace_size,
    cudaStream_t                       stream)
{
    LINALG_CHECK(A != nullptr && eigenvalues != nullptr && eigenvectors != nullptr && info != nullptr,
            "symmetric_eigenvalue_decomposition: A, eigenvalues, eigenvectors, and info must not be "
            "null");
    LINALG_CHECK(n > 0, "symmetric_eigenvalue_decomposition: n must be positive");
    LINALG_CHECK(!(lda != n) && !(ldv != n), "linalg::gpu::symmetric_eigenvalue_decomposition requires tightly packed "
                     "lda/ldv (both == n)");

    const auto nn      = detail::checked_narrow<int>(n);
    auto       handle  = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    // Symmetric matrix: A == A^T, so (unlike qr/svd) no transpose is needed
    // on input — a row-major-LOWER-supplied matrix and a
    // column-major-UPPER-read of the same bytes describe the identical
    // symmetric matrix (same identity cholesky_decomposition_gpu.cpp's
    // to_fill_mode relies on). The CPU contract only ever references A's
    // lower triangle, so the fill mode passed to cuSOLVER is always UPPER.
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    if (buffer_size(handle, CUSOLVER_EIG_MODE_VECTOR, fill, nn, A, nn, eigenvalues, &lwork) !=
        CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*syevd_bufferSize failed");
    }

    LINALG_CHECK(workspace_size >= symmetric_eigen_workspace_size_impl<T>(buffer_size, n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = symmetric_eigen_plan<T>(arena, nn, lwork);

    const auto n2 = static_cast<size_t>(nn) * static_cast<size_t>(nn);
    LINALG_CHECK(cudaMemcpyAsync(bufs.work_a, A, sizeof(T) * n2, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");

    const auto status = syevd(handle, CUSOLVER_EIG_MODE_VECTOR, fill, nn, bufs.work_a, nn,
        eigenvalues, bufs.workspace, lwork, info);
    LINALG_CHECK(!(status != CUSOLVER_STATUS_SUCCESS), "cusolverDn*syevd failed");

    // work_a now holds the eigenvectors column-major (n x n); the CPU
    // convention wants them row-major, one eigenvector per column.
    detail::transpose_device(static_cast<linalg_long>(n), static_cast<linalg_long>(n), bufs.work_a,
        bufs.transpose_scratch, stream);
    const auto copy_status =
        cudaMemcpyAsync(eigenvectors, bufs.work_a, sizeof(T) * n2, cudaMemcpyDeviceToDevice, stream);
    LINALG_CHECK(!(copy_status != cudaSuccess), "cudaMemcpyAsync failed");
}

}  // namespace

size_t symmetric_eigenvalue_decomposition_workspace_size(const float* /*type_tag*/, linalg_int n)
{
    return symmetric_eigen_workspace_size_impl<float>(cusolverDnSsyevd_bufferSize, n);
}

size_t symmetric_eigenvalue_decomposition_workspace_size(const double* /*type_tag*/, linalg_int n)
{
    return symmetric_eigen_workspace_size_impl<double>(cusolverDnDsyevd_bufferSize, n);
}

void symmetric_eigenvalue_decomposition(const float* A,
    linalg_int                                     n,
    linalg_int                                     lda,
    float*                                           eigenvalues,
    float*                                           eigenvectors,
    linalg_int                                     ldv,
    int*                                              info,
    void*                                             workspace,
    size_t                                            workspace_size,
    cudaStream_t                                      stream)
{
    LINALG_SCOPE("eigen/gpu<float>");
    symmetric_eigen_impl(cusolverDnSsyevd_bufferSize, cusolverDnSsyevd, A, n, lda, eigenvalues,
        eigenvectors, ldv, info, workspace, workspace_size, stream);
}

void symmetric_eigenvalue_decomposition(const double* A,
    linalg_int                                      n,
    linalg_int                                      lda,
    double*                                           eigenvalues,
    double*                                           eigenvectors,
    linalg_int                                      ldv,
    int*                                               info,
    void*                                              workspace,
    size_t                                             workspace_size,
    cudaStream_t                                       stream)
{
    LINALG_SCOPE("eigen/gpu<double>");
    symmetric_eigen_impl(cusolverDnDsyevd_bufferSize, cusolverDnDsyevd, A, n, lda, eigenvalues,
        eigenvectors, ldv, info, workspace, workspace_size, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
