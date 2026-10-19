#if LINALG_HAS_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

// Returning bool requires synchronizing the stream to read back info.
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

// Workspace layout: potrf buffer followed by one int for dev_info.
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
    arena.next<int>(1, alignof(int));
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
    arena.next<int>(1, alignof(int));
    return arena.total_size();
}

template <typename T>
bool cholesky_apply(T* C, linalg_int lda, cholesky_decomposition_enum type,
    void* workspace, size_t workspace_size, void* stream);

template <>
bool cholesky_apply<float>(float* C, linalg_int lda, cholesky_decomposition_enum type,
    void* workspace, size_t workspace_size, void* stream)
{
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
    int*                    dev_info = arena.next<int>(1, alignof(int));

    LINALG_CHECK(!(cusolverDnSpotrf(handle, fill, n, C, n, ws_buf, lwork, dev_info) !=
                     CUSOLVER_STATUS_SUCCESS),
        "cusolverDnSpotrf failed");
    LINALG_CHECK(cudaStreamSynchronize(stream_t) == cudaSuccess,
        "cholesky_decomposition<cuda_blas>: cudaStreamSynchronize failed");
    int host_info = 0;
    LINALG_CHECK(
        cudaMemcpy(&host_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess,
        "cholesky_decomposition<cuda_blas>: cudaMemcpy failed");
    return host_info == 0;
}

template <>
bool cholesky_apply<double>(double* C, linalg_int lda, cholesky_decomposition_enum type,
    void* workspace, size_t workspace_size, void* stream)
{
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
    int*                    dev_info = arena.next<int>(1, alignof(int));

    LINALG_CHECK(!(cusolverDnDpotrf(handle, fill, n, C, n, ws_buf, lwork, dev_info) !=
                     CUSOLVER_STATUS_SUCCESS),
        "cusolverDnDpotrf failed");
    LINALG_CHECK(cudaStreamSynchronize(stream_t) == cudaSuccess,
        "cholesky_decomposition<cuda_blas>: cudaStreamSynchronize failed");
    int host_info = 0;
    LINALG_CHECK(
        cudaMemcpy(&host_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess,
        "cholesky_decomposition<cuda_blas>: cudaMemcpy failed");
    return host_info == 0;
}

}  // namespace

template <>
bool cholesky_decomposition<float, linalg_backend::cuda_blas>::apply(
    float* C, linalg_int lda, cholesky_decomposition_enum type, void* stream)
{
    LINALG_SCOPE("cholesky_decomposition/cuda_blas<float>");
    // The struct interface has no workspace parameter; allocate internally.
    // The buffer is small (a few megabytes at most for typical matrix sizes)
    // and the call already synchronizes the stream, so the cudaMalloc
    // overhead is negligible compared to the factorization itself.
    const auto ws_size = cholesky_ws_size<float>(lda, type);
    void*      workspace = nullptr;
    LINALG_CHECK(cudaMalloc(&workspace, ws_size) == cudaSuccess,
        "cholesky_decomposition<cuda_blas>: cudaMalloc for workspace failed");
    const bool ok = cholesky_apply<float>(C, lda, type, workspace, ws_size, stream);
    cudaFree(workspace);
    return ok;
}

template <>
bool cholesky_decomposition<double, linalg_backend::cuda_blas>::apply(
    double* C, linalg_int lda, cholesky_decomposition_enum type, void* stream)
{
    LINALG_SCOPE("cholesky_decomposition/cuda_blas<double>");
    const auto ws_size = cholesky_ws_size<double>(lda, type);
    void*      workspace = nullptr;
    LINALG_CHECK(cudaMalloc(&workspace, ws_size) == cudaSuccess,
        "cholesky_decomposition<cuda_blas>: cudaMalloc for workspace failed");
    const bool ok = cholesky_apply<double>(C, lda, type, workspace, ws_size, stream);
    cudaFree(workspace);
    return ok;
}

template <>
bool cholesky_decomposition<float, linalg_backend::cuda_blas>::apply_aad(
    float* L_aad, const float* L, linalg_int lda, cholesky_decomposition_enum type, float* A_aad)
{
    (void)L_aad; (void)L; (void)lda; (void)type; (void)A_aad;
    LINALG_THROW("cholesky_decomposition<cuda_blas>: apply_aad is not supported on GPU");
}

template <>
bool cholesky_decomposition<double, linalg_backend::cuda_blas>::apply_aad(
    double* L_aad, const double* L, linalg_int lda, cholesky_decomposition_enum type,
    double* A_aad)
{
    (void)L_aad; (void)L; (void)lda; (void)type; (void)A_aad;
    LINALG_THROW("cholesky_decomposition<cuda_blas>: apply_aad is not supported on GPU");
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
