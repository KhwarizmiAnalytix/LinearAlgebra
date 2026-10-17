#include "include/matrix_operation_gpu/cholesky_decomposition_gpu.h"

#if LINALG_HAS_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"

#include "include/common/linalg_profiler.h"

namespace linalg
{
namespace gpu
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

// Single scratch buffer: cuSOLVER's own potrf workspace. Shared by the size
// query and the real call (see include/common/cuda_workspace.h).
template <typename T>
T* cholesky_decomposition_plan(detail::workspace_arena& arena, int lwork)
{
    return arena.next<T>(static_cast<size_t>(lwork));
}

}  // namespace

size_t cholesky_decomposition_workspace_size(
    const float* /*type_tag*/, linalg_int lda, cholesky_decomposition_enum type)
{
    LINALG_CHECK(lda > 0, "cholesky_decomposition_workspace_size: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    auto       fill   = to_fill_mode(type);
    int        lwork  = 0;
    LINALG_CHECK(cusolverDnSpotrf_bufferSize(handle, fill, n, nullptr, n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDnSpotrf_bufferSize failed");
    detail::workspace_arena arena;
    cholesky_decomposition_plan<float>(arena, lwork);
    return arena.total_size();
}

size_t cholesky_decomposition_workspace_size(
    const double* /*type_tag*/, linalg_int lda, cholesky_decomposition_enum type)
{
    LINALG_CHECK(lda > 0, "cholesky_decomposition_workspace_size: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    auto       fill   = to_fill_mode(type);
    int        lwork  = 0;
    LINALG_CHECK(cusolverDnDpotrf_bufferSize(handle, fill, n, nullptr, n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDnDpotrf_bufferSize failed");
    detail::workspace_arena arena;
    cholesky_decomposition_plan<double>(arena, lwork);
    return arena.total_size();
}

void cholesky_decomposition(float* C, linalg_int lda, cholesky_decomposition_enum type, int* info,
    void* workspace, size_t workspace_size, cudaStream_t stream)
{
    LINALG_SCOPE("cholesky_decomposition/gpu<float>");
    LINALG_CHECK(C != nullptr && info != nullptr, "cholesky_decomposition: C and info must not be null");
    LINALG_CHECK(lda > 0, "cholesky_decomposition: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    auto fill = to_fill_mode(type);

    int lwork = 0;
    LINALG_CHECK(!(cusolverDnSpotrf_bufferSize(handle, fill, n, C, n, &lwork) != CUSOLVER_STATUS_SUCCESS), "cusolverDnSpotrf_bufferSize failed");

    LINALG_CHECK(
        workspace_size >= cholesky_decomposition_workspace_size(static_cast<const float*>(nullptr), lda, type),
        "cholesky_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    float*                  workspace_buf = cholesky_decomposition_plan<float>(arena, lwork);

    const auto status = cusolverDnSpotrf(handle, fill, n, C, n, workspace_buf, lwork, info);
    LINALG_CHECK(!(status != CUSOLVER_STATUS_SUCCESS), "cusolverDnSpotrf failed");
}

void cholesky_decomposition(double* C, linalg_int lda, cholesky_decomposition_enum type, int* info,
    void* workspace, size_t workspace_size, cudaStream_t stream)
{
    LINALG_SCOPE("cholesky_decomposition/gpu<double>");
    LINALG_CHECK(C != nullptr && info != nullptr, "cholesky_decomposition: C and info must not be null");
    LINALG_CHECK(lda > 0, "cholesky_decomposition: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    auto fill = to_fill_mode(type);

    int lwork = 0;
    LINALG_CHECK(!(cusolverDnDpotrf_bufferSize(handle, fill, n, C, n, &lwork) != CUSOLVER_STATUS_SUCCESS), "cusolverDnDpotrf_bufferSize failed");

    LINALG_CHECK(
        workspace_size >= cholesky_decomposition_workspace_size(static_cast<const double*>(nullptr), lda, type),
        "cholesky_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    double*                 workspace_buf = cholesky_decomposition_plan<double>(arena, lwork);

    const auto status = cusolverDnDpotrf(handle, fill, n, C, n, workspace_buf, lwork, info);
    LINALG_CHECK(!(status != CUSOLVER_STATUS_SUCCESS), "cusolverDnDpotrf failed");
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
