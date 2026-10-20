#if LINALG_HAS_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

namespace linalg
{
namespace
{

cublasFillMode_t to_fill_mode(cholesky_decomposition_enum type)
{
    // cuSOLVER's potrf is column-major; a row-major LOWER-triangular factor
    // of a symmetric matrix occupies the same memory as the column-major
    // UPPER-triangular factor of the same matrix (and vice versa), so the
    // fill mode is flipped relative to the CPU (LAPACKE row-major) backend.
    return type == cholesky_decomposition_enum::LOWER_TRIANGULAR ? CUBLAS_FILL_MODE_UPPER
                                                                 : CUBLAS_FILL_MODE_LOWER;
}

// Workspace contains only potrf scratch; status is caller-owned device memory.
template <typename T>
size_t cholesky_ws_size(linalg_int lda, cholesky_decomposition_enum type);

template <>
size_t cholesky_ws_size<float>(linalg_int lda, cholesky_decomposition_enum type)
{
    LINALG_CHECK(lda > 0, "cholesky_decomposition_workspace_size: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    int        lwork  = 0;
    LINALG_CHECK(cusolverDnSpotrf_bufferSize(handle, to_fill_mode(type), n, nullptr, n, &lwork) ==
                     CUSOLVER_STATUS_SUCCESS,
        "cusolverDnSpotrf_bufferSize failed");
    detail::workspace_arena arena;
    arena.next<float>(static_cast<size_t>(lwork));
    return arena.total_size();
}

template <>
size_t cholesky_ws_size<double>(linalg_int lda, cholesky_decomposition_enum type)
{
    LINALG_CHECK(lda > 0, "cholesky_decomposition_workspace_size: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    int        lwork  = 0;
    LINALG_CHECK(cusolverDnDpotrf_bufferSize(handle, to_fill_mode(type), n, nullptr, n, &lwork) ==
                     CUSOLVER_STATUS_SUCCESS,
        "cusolverDnDpotrf_bufferSize failed");
    detail::workspace_arena arena;
    arena.next<double>(static_cast<size_t>(lwork));
    return arena.total_size();
}

template <typename T>
void cholesky_apply(T*          C,
    linalg_int                  lda,
    cholesky_decomposition_enum type,
    void*                       workspace,
    size_t                      workspace_size,
    int*                        info,
    void*                       stream);

template <>
void cholesky_apply<float>(float* C,
    linalg_int                    lda,
    cholesky_decomposition_enum   type,
    void*                         workspace,
    size_t                        workspace_size,
    int*                          info,
    void*                         stream)
{
    LINALG_CHECK(info != nullptr, "device info must not be null");
    const auto stream_t = static_cast<cudaStream_t>(stream);
    LINALG_CHECK(C != nullptr, "cholesky_decomposition: C must not be null");
    LINALG_CHECK(lda > 0, "cholesky_decomposition: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream_t);
    const auto fill = to_fill_mode(type);

    int lwork = 0;
    LINALG_CHECK(!(cusolverDnSpotrf_bufferSize(handle, fill, n, C, n, &lwork) !=
                     CUSOLVER_STATUS_SUCCESS),
        "cusolverDnSpotrf_bufferSize failed");
    LINALG_CHECK(workspace_size >= cholesky_ws_size<float>(lda, type),
        "cholesky_decomposition: workspace_size too small");

    detail::workspace_arena arena(workspace, workspace_size);
    float*                  ws_buf   = arena.next<float>(static_cast<size_t>(lwork));

    LINALG_CHECK(
        !(cusolverDnSpotrf(handle, fill, n, C, n, ws_buf, lwork, info) != CUSOLVER_STATUS_SUCCESS),
        "cusolverDnSpotrf failed");
}

template <>
void cholesky_apply<double>(double* C,
    linalg_int                      lda,
    cholesky_decomposition_enum     type,
    void*                           workspace,
    size_t                          workspace_size,
    int*                            info,
    void*                           stream)
{
    LINALG_CHECK(info != nullptr, "device info must not be null");
    const auto stream_t = static_cast<cudaStream_t>(stream);
    LINALG_CHECK(C != nullptr, "cholesky_decomposition: C must not be null");
    LINALG_CHECK(lda > 0, "cholesky_decomposition: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream_t);
    const auto fill = to_fill_mode(type);

    int lwork = 0;
    LINALG_CHECK(!(cusolverDnDpotrf_bufferSize(handle, fill, n, C, n, &lwork) !=
                     CUSOLVER_STATUS_SUCCESS),
        "cusolverDnDpotrf_bufferSize failed");
    LINALG_CHECK(workspace_size >= cholesky_ws_size<double>(lda, type),
        "cholesky_decomposition: workspace_size too small");

    detail::workspace_arena arena(workspace, workspace_size);
    double*                 ws_buf   = arena.next<double>(static_cast<size_t>(lwork));

    LINALG_CHECK(
        !(cusolverDnDpotrf(handle, fill, n, C, n, ws_buf, lwork, info) != CUSOLVER_STATUS_SUCCESS),
        "cusolverDnDpotrf failed");
}

}  // namespace

template <>
void cholesky_decomposition<float, linalg_backend::CUDA_BLAS>::apply(float* C,
    linalg_int                                                              lda,
    cholesky_decomposition_enum                                             type,
    void*                                                                   workspace,
    size_t                                                                  workspace_size,
    int*                                                                    info,
    void*                                                                   stream)
{
    cholesky_apply<float>(C, lda, type, workspace, workspace_size, info, stream);
}

template <>
size_t cholesky_decomposition<float, linalg_backend::CUDA_BLAS>::workspace_size(
    linalg_int lda, cholesky_decomposition_enum type)
{
    return cholesky_ws_size<float>(lda, type);
}

template <>
void cholesky_decomposition<double, linalg_backend::CUDA_BLAS>::apply(double* C,
    linalg_int                                                                lda,
    cholesky_decomposition_enum                                               type,
    void*                                                                     workspace,
    size_t                                                                    workspace_size,
    int*                                                                      info,
    void*                                                                     stream)
{
    cholesky_apply<double>(C, lda, type, workspace, workspace_size, info, stream);
}

template <>
size_t cholesky_decomposition<double, linalg_backend::CUDA_BLAS>::workspace_size(
    linalg_int lda, cholesky_decomposition_enum type)
{
    return cholesky_ws_size<double>(lda, type);
}

}  // namespace linalg
#endif
