#if LINALG_HAS_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <vector>

#include "include/matrix_operation/lu_decomposition.h"
#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

namespace linalg
{
namespace
{

// cuSOLVER's getrf pivot and info are 32-bit; linalg_int (the primary
// template's pivot element type, matching the CPU backends) is 64-bit, so
// neither can be aliased onto the caller's device pivot buffer directly.
// dev_ipiv/dev_info are device scratch for cuSOLVER's own call; the pivot
// widening and info check happen host-side after one synchronizing
// round-trip, matching the primary (Backend-templated) template's bool-
// returning, host-synchronous signature exactly (see rank_cublas.cu).
template <typename T>
struct lu_decomposition_layout
{
    T*   workspace;
    T*   transpose_scratch;
    int* dev_ipiv;
    int* dev_info;
};

template <typename T>
lu_decomposition_layout<T> lu_decomposition_plan(
    detail::workspace_arena& arena, int lwork, int n)
{
    lu_decomposition_layout<T> p;
    p.workspace         = arena.next<T>(static_cast<size_t>(lwork));
    p.transpose_scratch = arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    p.dev_ipiv          = arena.next<int>(static_cast<size_t>(n));
    p.dev_info          = arena.next<int>(1);
    return p;
}

template <typename T, typename BufSizeFn>
size_t lu_ws_size_impl(BufSizeFn buffer_size, linalg_int lda)
{
    LINALG_CHECK(lda > 0, "lu_decomposition_workspace_size: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    int        lwork  = 0;
    LINALG_CHECK(buffer_size(handle, n, n, nullptr, n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*getrf_bufferSize failed");
    detail::workspace_arena arena;
    lu_decomposition_plan<T>(arena, lwork, n);
    return arena.total_size();
}

template <typename T>
size_t lu_ws_size(linalg_int lda);

template <>
size_t lu_ws_size<float>(linalg_int lda)
{
    return lu_ws_size_impl<float>(cusolverDnSgetrf_bufferSize, lda);
}

template <>
size_t lu_ws_size<double>(linalg_int lda)
{
    return lu_ws_size_impl<double>(cusolverDnDgetrf_bufferSize, lda);
}

template <typename T, typename BufSizeFn, typename GetrfFn>
bool lu_apply_impl(BufSizeFn buffer_size,
    GetrfFn                  getrf,
    T*                       m,
    linalg_int               lda,
    linalg_int*              pivot,
    void*                    workspace,
    size_t                   workspace_size,
    void*                    stream)
{
    LINALG_CHECK(m != nullptr && pivot != nullptr, "lu_decomposition: m and pivot must not be null");
    LINALG_CHECK(lda > 0, "lu_decomposition: lda must be positive", lda);
    const auto stream_t = static_cast<cudaStream_t>(stream);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream_t);

    int lwork = 0;
    LINALG_CHECK(buffer_size(handle, n, n, m, n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*getrf_bufferSize failed");
    LINALG_CHECK(workspace_size >= lu_ws_size_impl<T>(buffer_size, lda),
        "lu_decomposition: workspace_size too small");

    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = lu_decomposition_plan<T>(arena, lwork, n);

    // cuSOLVER is column-major: transposing the row-major buffer first
    // (detail::transpose_device, transpose_cublas.cpp) makes its column-major
    // reading of the result equal the caller's actual A rather than A^T,
    // so getrf below factors A itself and its pivots are A's own
    // row-interchange sequence. Transposing the packed L\U result back
    // afterward converts cuSOLVER's column-major packed output into the
    // row-major packed layout callers expect.
    detail::transpose_device(lda, lda, m, bufs.transpose_scratch, stream_t);
    LINALG_CHECK(
        getrf(handle, n, n, m, n, bufs.workspace, bufs.dev_ipiv, bufs.dev_info) ==
            CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*getrf failed");
    detail::transpose_device(lda, lda, m, bufs.transpose_scratch, stream_t);

    std::vector<int> host_ipiv(static_cast<size_t>(n));
    int               host_info = 0;
    LINALG_CHECK(cudaMemcpyAsync(host_ipiv.data(), bufs.dev_ipiv, sizeof(int) * static_cast<size_t>(n),
                     cudaMemcpyDeviceToHost, stream_t) == cudaSuccess,
        "lu_decomposition: pivot copy failed");
    LINALG_CHECK(cudaMemcpyAsync(&host_info, bufs.dev_info, sizeof(int), cudaMemcpyDeviceToHost,
                     stream_t) == cudaSuccess,
        "lu_decomposition: info copy failed");
    LINALG_CHECK(cudaStreamSynchronize(stream_t) == cudaSuccess, "lu_decomposition: stream sync failed");

    std::vector<linalg_int> host_pivot(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        host_pivot[static_cast<size_t>(i)] = static_cast<linalg_int>(host_ipiv[static_cast<size_t>(i)]);
    }
    LINALG_CHECK(cudaMemcpyAsync(pivot, host_pivot.data(), sizeof(linalg_int) * static_cast<size_t>(n),
                     cudaMemcpyHostToDevice, stream_t) == cudaSuccess,
        "lu_decomposition: pivot write-back failed");
    LINALG_CHECK(cudaStreamSynchronize(stream_t) == cudaSuccess, "lu_decomposition: stream sync failed");

    return host_info == 0;
}

}  // namespace

template <>
bool lu_decomposition<float, linalg_backend::CUDA_BLAS>::apply(float* m,
    linalg_int                                                        lda,
    linalg_int*                                                       pivot,
    void*                                                             workspace,
    size_t                                                            workspace_size,
    void*                                                             stream)
{
    LINALG_SCOPE("lu_decomposition/cuda_blas<float>");
    return lu_apply_impl(cusolverDnSgetrf_bufferSize,
        cusolverDnSgetrf,
        m,
        lda,
        pivot,
        workspace,
        workspace_size,
        stream);
}

template <>
bool lu_decomposition<double, linalg_backend::CUDA_BLAS>::apply(double* m,
    linalg_int                                                         lda,
    linalg_int*                                                        pivot,
    void*                                                              workspace,
    size_t                                                             workspace_size,
    void*                                                              stream)
{
    LINALG_SCOPE("lu_decomposition/cuda_blas<double>");
    return lu_apply_impl(cusolverDnDgetrf_bufferSize,
        cusolverDnDgetrf,
        m,
        lda,
        pivot,
        workspace,
        workspace_size,
        stream);
}

template <>
size_t lu_decomposition<float, linalg_backend::CUDA_BLAS>::workspace_size(linalg_int lda)
{
    return lu_ws_size<float>(lda);
}

template <>
size_t lu_decomposition<double, linalg_backend::CUDA_BLAS>::workspace_size(linalg_int lda)
{
    return lu_ws_size<double>(lda);
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
