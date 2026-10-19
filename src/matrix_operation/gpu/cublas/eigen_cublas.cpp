#if LINALG_HAS_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/matrix_operation/eigen.h"
#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

// Only symmetric_eigen_backend has a GPU implementation; there is no
// cuSOLVER geev-equivalent for general (non-symmetric) matrices.
namespace linalg
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
template <typename T>
symmetric_eigen_layout<T> symmetric_eigen_plan(detail::workspace_arena& arena, int n, int lwork)
{
    symmetric_eigen_layout<T> p;
    p.work_a            = arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    p.workspace         = arena.next<T>(static_cast<size_t>(lwork));
    p.transpose_scratch = arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    return p;
}

template <typename T, typename BufSizeFn>
size_t sym_eigen_workspace_size_impl(BufSizeFn buffer_size, linalg_int n)
{
    LINALG_CHECK(n > 0, "symmetric_eigenvalue_decomposition_workspace_size: n must be positive");
    const auto nn     = detail::checked_narrow<int>(n);
    auto       handle = detail::cusolver_handle_for_current_device();
    const auto fill   = CUBLAS_FILL_MODE_UPPER;
    int        lwork  = 0;
    LINALG_CHECK(buffer_size(handle, CUSOLVER_EIG_MODE_VECTOR, fill, nn, static_cast<T*>(nullptr),
                      nn, static_cast<T*>(nullptr), &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*syevd_bufferSize failed");
    detail::workspace_arena arena;
    symmetric_eigen_plan<T>(arena, nn, lwork);
    return arena.total_size();
}

// See svd_decomposition_cublas.cpp's RESTRICTION note: every caller-visible
// buffer here is required to be tightly packed row-major.
template <typename T, typename BufSizeFn, typename SyevdFn>
void sym_eigen_impl(BufSizeFn buffer_size,
    SyevdFn                   syevd,
    const T*                  A,
    linalg_int                n,
    linalg_int                lda,
    T*                        eigenvalues,
    T*                        eigenvectors,
    linalg_int                ldv,
    void*                     workspace,
    size_t                    workspace_size,
    int*                      info,
    void*                     stream)
{
    LINALG_CHECK(info != nullptr, "device info must not be null");
    const auto stream_t = static_cast<cudaStream_t>(stream);
    LINALG_CHECK(A != nullptr && eigenvalues != nullptr && eigenvectors != nullptr,
        "symmetric_eigenvalue_decomposition: A, eigenvalues, and eigenvectors must not be null");
    LINALG_CHECK(n > 0, "symmetric_eigenvalue_decomposition: n must be positive");
    LINALG_CHECK(!(lda != n) && !(ldv != n),
        "symmetric_eigenvalue_decomposition requires tightly packed lda/ldv (both == n)");

    const auto nn      = detail::checked_narrow<int>(n);
    auto       handle  = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream_t);
    // Symmetric matrix: A == A^T, so (unlike qr/svd) no transpose is needed
    // on input — a row-major-LOWER-supplied matrix and a
    // column-major-UPPER-read of the same bytes describe the identical
    // symmetric matrix. The fill mode passed to cuSOLVER is always UPPER.
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    if (buffer_size(handle, CUSOLVER_EIG_MODE_VECTOR, fill, nn, A, nn, eigenvalues, &lwork) !=
        CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*syevd_bufferSize failed");
    }

    LINALG_CHECK(workspace_size >= sym_eigen_workspace_size_impl<T>(buffer_size, n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = symmetric_eigen_plan<T>(arena, nn, lwork);

    const auto n2 = static_cast<size_t>(nn) * static_cast<size_t>(nn);
    LINALG_CHECK(cudaMemcpyAsync(bufs.work_a, A, sizeof(T) * n2, cudaMemcpyDeviceToDevice, stream_t) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");

    LINALG_CHECK(!(syevd(handle,
                       CUSOLVER_EIG_MODE_VECTOR,
                       fill,
                       nn,
                       bufs.work_a,
                       nn,
                       eigenvalues,
                       bufs.workspace,
                       lwork,
                       info) != CUSOLVER_STATUS_SUCCESS),
        "cusolverDn*syevd failed");

    // work_a now holds the eigenvectors column-major (n x n); the CPU
    // convention wants them row-major, one eigenvector per column.
    detail::transpose_device(static_cast<linalg_long>(n), static_cast<linalg_long>(n),
        bufs.work_a, bufs.transpose_scratch, stream_t);
    LINALG_CHECK(!(cudaMemcpyAsync(eigenvectors, bufs.work_a, sizeof(T) * n2,
                       cudaMemcpyDeviceToDevice, stream_t) != cudaSuccess),
        "cudaMemcpyAsync failed");
}

template <typename T>
size_t sym_eigen_ws_size(linalg_int n);

template <>
size_t sym_eigen_ws_size<float>(linalg_int n)
{
    const auto base = sym_eigen_workspace_size_impl<float>(cusolverDnSsyevd_bufferSize, n);
    return base;
}

template <>
size_t sym_eigen_ws_size<double>(linalg_int n)
{
    const auto base = sym_eigen_workspace_size_impl<double>(cusolverDnDsyevd_bufferSize, n);
    return base;
}

}  // namespace

template <>
void symmetric_eigen_backend<float, linalg_backend::cuda_blas>::apply(const float* A,
    linalg_int                                                                     n,
    linalg_int                                                                     lda,
    float*                                                                         eigenvalues,
    float*                                                                         eigenvectors,
    linalg_int                                                                     ldv,
    void*                                                                          workspace,
    size_t                                                                         workspace_size,
    int*                                                                           info,
    void*                                                                          stream)
{
    LINALG_SCOPE("symmetric_eigen/cuda_blas<float>");
    sym_eigen_impl(cusolverDnSsyevd_bufferSize,
        cusolverDnSsyevd,
        A,
        n,
        lda,
        eigenvalues,
        eigenvectors,
        ldv,
        workspace,
        workspace_size,
        info,
        stream);
}

template <>
void symmetric_eigen_backend<double, linalg_backend::cuda_blas>::apply(const double* A,
    linalg_int                                                                       n,
    linalg_int                                                                       lda,
    double*                                                                          eigenvalues,
    double*                                                                          eigenvectors,
    linalg_int                                                                       ldv,
    void*                                                                            workspace,
    size_t                                                                           workspace_size,
    int*                                                                             info,
    void*                                                                            stream)
{
    LINALG_SCOPE("symmetric_eigen/cuda_blas<double>");
    sym_eigen_impl(cusolverDnDsyevd_bufferSize,
        cusolverDnDsyevd,
        A,
        n,
        lda,
        eigenvalues,
        eigenvectors,
        ldv,
        workspace,
        workspace_size,
        info,
        stream);
}

template <>
size_t symmetric_eigen_backend<float, linalg_backend::cuda_blas>::workspace_size(linalg_int n)
{
    return sym_eigen_ws_size<float>(n);
}

template <>
size_t symmetric_eigen_backend<double, linalg_backend::cuda_blas>::workspace_size(linalg_int n)
{
    return sym_eigen_ws_size<double>(n);
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
