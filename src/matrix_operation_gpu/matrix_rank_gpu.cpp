#include "include/matrix_operation_gpu/matrix_rank_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <limits>
#include <vector>

#include "include/matrix_operation_gpu/svd_decomposition_gpu.h"
#include <include/logging.h>

namespace linalg
{
namespace gpu
{
namespace
{

template <typename T> std::vector<T> download_singular_values(
    const T* A, linalg_long rows, linalg_long columns, linalg_long lda, cudaStream_t stream)
{
    LOGGING_CHECK(A != nullptr, "A must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0, "rows and columns must be positive");
    LOGGING_CHECK(!(static_cast<linalg_long>(lda) != columns), "requires tightly packed lda (lda == columns)");

    const auto k = std::min(rows, columns);

    T*  S    = nullptr;
    T*  U    = nullptr;
    T*  VT   = nullptr;
    int* info = nullptr;
    LOGGING_CHECK(!(cudaMalloc(reinterpret_cast<void**>(&S), sizeof(T) * static_cast<size_t>(k)) != cudaSuccess), "cudaMalloc failed");
    if (cudaMalloc(reinterpret_cast<void**>(&U),
            sizeof(T) * static_cast<size_t>(rows) * static_cast<size_t>(k)) != cudaSuccess)
    {
        cudaFree(S);
        LOGGING_THROW("cudaMalloc failed");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&VT),
            sizeof(T) * static_cast<size_t>(k) * static_cast<size_t>(columns)) != cudaSuccess)
    {
        cudaFree(S);
        cudaFree(U);
        LOGGING_THROW("cudaMalloc failed");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&info), sizeof(int)) != cudaSuccess)
    {
        cudaFree(S);
        cudaFree(U);
        cudaFree(VT);
        LOGGING_THROW("cudaMalloc failed");
    }

    svd_decomposition(rows, columns, A, columns, S, U, k, VT, columns, info, stream);

    int svd_status = -1;
    std::vector<T> host_s(static_cast<std::size_t>(k));
    const auto     info_copy_status = cudaMemcpyAsync(&svd_status, info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    const auto     s_copy_status =
        cudaMemcpyAsync(host_s.data(), S, sizeof(T) * static_cast<size_t>(k), cudaMemcpyDeviceToHost, stream);
    // U/VT are unused here (only the singular values matter), but they are
    // still being written by svd_decomposition's own asynchronous work on
    // `stream` until this sync — freeing them any earlier would race it.
    const auto sync_status = cudaStreamSynchronize(stream);
    cudaFree(U);
    cudaFree(VT);
    cudaFree(S);
    cudaFree(info);

    LOGGING_CHECK(!(info_copy_status != cudaSuccess) && !(s_copy_status != cudaSuccess) && !(sync_status != cudaSuccess), "cudaMemcpyAsync/cudaStreamSynchronize failed");
    LOGGING_CHECK(svd_status == 0, "linalg::gpu::svd_decomposition failed", svd_status);
    return host_s;
}

template <typename T> linalg_long matrix_rank_impl(
    const T* A, linalg_long rows, linalg_long columns, linalg_long lda, T tol, cudaStream_t stream)
{
    const auto S = download_singular_values(A, rows, columns, lda, stream);

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

template <typename T> T matrix_condition_number_impl(
    const T* A, linalg_long rows, linalg_long columns, linalg_long lda, cudaStream_t stream)
{
    const auto S = download_singular_values(A, rows, columns, lda, stream);

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

linalg_long matrix_rank(
    const float* A, linalg_long rows, linalg_long columns, linalg_long lda, float tol, cudaStream_t stream)
{
    return matrix_rank_impl(A, rows, columns, lda, tol, stream);
}

linalg_long matrix_rank(
    const double* A, linalg_long rows, linalg_long columns, linalg_long lda, double tol, cudaStream_t stream)
{
    return matrix_rank_impl(A, rows, columns, lda, tol, stream);
}

float matrix_condition_number(
    const float* A, linalg_long rows, linalg_long columns, linalg_long lda, cudaStream_t stream)
{
    return matrix_condition_number_impl(A, rows, columns, lda, stream);
}

double matrix_condition_number(
    const double* A, linalg_long rows, linalg_long columns, linalg_long lda, cudaStream_t stream)
{
    return matrix_condition_number_impl(A, rows, columns, lda, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
