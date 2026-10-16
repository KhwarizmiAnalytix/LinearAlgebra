#include "include/matrix_operation_gpu/rank_gpu.h"

#if LINALG_HAS_CUBLAS

#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <limits>
#include <vector>

#include "include/common/cuda_workspace.h"
#include "include/matrix_operation_gpu/svd_decomposition_gpu.h"
#include "include/logging.h"

namespace linalg
{
namespace gpu
{
namespace
{

template <typename T>
struct singular_values_layout
{
    T*             S;
    T*             U;
    T*             VT;
    int*           info;
    unsigned char* svd_workspace;
};

template <typename T>
singular_values_layout<T> singular_values_plan(
    detail::workspace_arena& arena, linalg_long rows, linalg_long columns, size_t svd_workspace_bytes)
{
    const auto k = std::min(rows, columns);
    singular_values_layout<T> p;
    p.S             = arena.next<T>(static_cast<size_t>(k));
    p.U             = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(k));
    p.VT            = arena.next<T>(static_cast<size_t>(k) * static_cast<size_t>(columns));
    p.info          = arena.next<int>(1);
    p.svd_workspace = arena.next<unsigned char>(svd_workspace_bytes);
    return p;
}

template <typename T>
size_t singular_values_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    const auto svd_bytes =
        svd_decomposition_workspace_size(static_cast<const T*>(nullptr), rows, columns);
    detail::workspace_arena arena;
    singular_values_plan<T>(arena, rows, columns, svd_bytes);
    return arena.total_size();
}

template <typename T> std::vector<T> download_singular_values(const T* A, linalg_long rows,
    linalg_long columns, linalg_long lda, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    LOGGING_CHECK(A != nullptr, "A must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0, "rows and columns must be positive");
    LOGGING_CHECK(!(static_cast<linalg_long>(lda) != columns), "requires tightly packed lda (lda == columns)");

    const auto k         = std::min(rows, columns);
    const auto svd_bytes = svd_decomposition_workspace_size(static_cast<const T*>(nullptr), rows, columns);

    LOGGING_CHECK(workspace_size >= singular_values_workspace_size_impl<T>(rows, columns),
        "rank/condition_number: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = singular_values_plan<T>(arena, rows, columns, svd_bytes);

    svd_decomposition(rows, columns, A, columns, bufs.S, bufs.U, k, bufs.VT, columns, bufs.info,
        bufs.svd_workspace, svd_bytes, stream);

    int svd_status = -1;
    std::vector<T> host_s(static_cast<std::size_t>(k));
    const auto     info_copy_status = cudaMemcpyAsync(&svd_status, bufs.info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    const auto     s_copy_status =
        cudaMemcpyAsync(host_s.data(), bufs.S, sizeof(T) * static_cast<size_t>(k), cudaMemcpyDeviceToHost, stream);
    const auto sync_status = cudaStreamSynchronize(stream);

    LOGGING_CHECK(!(info_copy_status != cudaSuccess) && !(s_copy_status != cudaSuccess) && !(sync_status != cudaSuccess), "cudaMemcpyAsync/cudaStreamSynchronize failed");
    LOGGING_CHECK(svd_status == 0, "linalg::gpu::svd_decomposition failed", svd_status);
    return host_s;
}

template <typename T> linalg_long matrix_rank_impl(const T* A, linalg_long rows, linalg_long columns,
    linalg_long lda, T tol, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    const auto S = download_singular_values(A, rows, columns, lda, workspace, workspace_size, stream);

    T smax = T(0);
    for (const auto& s : S)
    {
        smax = std::max(smax, std::fabs(s));
    }
    const T eff_tol =
        (tol >= T(0)) ? tol : std::numeric_limits<T>::epsilon() * static_cast<T>(std::max(rows, columns)) * smax;

    linalg_long rank = 0;
    for (const auto& s : S)
    {
        if (std::fabs(s) > eff_tol)
        {
            ++rank;
        }
    }
    return rank;
}

template <typename T> T matrix_condition_number_impl(const T* A, linalg_long rows,
    linalg_long columns, linalg_long lda, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    const auto S = download_singular_values(A, rows, columns, lda, workspace, workspace_size, stream);

    T smax = T(0);
    T smin = std::numeric_limits<T>::max();
    for (const auto& s : S)
    {
        const T abs_s = std::fabs(s);
        smax          = std::max(smax, abs_s);
        smin          = std::min(smin, abs_s);
    }
    if (smin == T(0))
    {
        return std::numeric_limits<T>::infinity();
    }
    return smax / smin;
}

}  // namespace

size_t matrix_rank_workspace_size(const float* /*type_tag*/, linalg_long rows, linalg_long columns)
{
    return singular_values_workspace_size_impl<float>(rows, columns);
}

size_t matrix_rank_workspace_size(const double* /*type_tag*/, linalg_long rows, linalg_long columns)
{
    return singular_values_workspace_size_impl<double>(rows, columns);
}

linalg_long rank(const float* A, linalg_long rows, linalg_long columns, linalg_long lda,
    float tol, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    return matrix_rank_impl(A, rows, columns, lda, tol, workspace, workspace_size, stream);
}

linalg_long rank(const double* A, linalg_long rows, linalg_long columns, linalg_long lda,
    double tol, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    return matrix_rank_impl(A, rows, columns, lda, tol, workspace, workspace_size, stream);
}

float condition_number(const float* A, linalg_long rows, linalg_long columns,
    linalg_long lda, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    return matrix_condition_number_impl(A, rows, columns, lda, workspace, workspace_size, stream);
}

double condition_number(const double* A, linalg_long rows, linalg_long columns,
    linalg_long lda, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    return matrix_condition_number_impl(A, rows, columns, lda, workspace, workspace_size, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
