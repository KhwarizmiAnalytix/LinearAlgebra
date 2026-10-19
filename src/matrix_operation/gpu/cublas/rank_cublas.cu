#if LINALG_HAS_CUBLAS

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <limits>

#include "include/matrix_operation/condition_number.h"
#include "include/matrix_operation/rank.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

namespace linalg
{

// Cross-TU helper: detail::svd_on_device* defined in svd_decomposition_cublas.cpp.
namespace detail
{
size_t svd_on_device_workspace_size(const float*, linalg_long rows, linalg_long columns);
size_t svd_on_device_workspace_size(const double*, linalg_long rows, linalg_long columns);
void svd_on_device(linalg_long rows, linalg_long columns, const float* A, linalg_long lda,
    float* S, float* U, linalg_long ldu, float* VT, linalg_long ldv,
    int* info, void* workspace, size_t workspace_size, cudaStream_t stream);
void svd_on_device(linalg_long rows, linalg_long columns, const double* A, linalg_long lda,
    double* S, double* U, linalg_long ldu, double* VT, linalg_long ldv,
    int* info, void* workspace, size_t workspace_size, cudaStream_t stream);
}  // namespace detail

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
    detail::workspace_arena& arena, linalg_long rows, linalg_long columns,
    size_t svd_workspace_bytes)
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
        detail::svd_on_device_workspace_size(static_cast<const T*>(nullptr), rows, columns);
    detail::workspace_arena arena;
    singular_values_plan<T>(arena, rows, columns, svd_bytes);
    return arena.total_size();
}

template <typename T>
static __global__ void singular_values_reduce(const T* S,
    int                                                k,
    linalg_long                                        rows,
    linalg_long                                        columns,
    T                                                  tol,
    T                                                  epsilon,
    const int*                                         info,
    linalg_long*                                       rank_result,
    T*                                                 condition_result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    if (*info != 0)
    {
        if (rank_result)
            *rank_result = static_cast<linalg_long>(-1);
        if (condition_result)
            *condition_result = T(NAN);
        return;
    }
    // cuSOLVER gesvd returns singular values in descending order.
    if (rank_result)
    {
        const T cutoff    = tol >= T(0) ? tol : T(rows > columns ? rows : columns) * S[0] * epsilon;
        linalg_long count = 0;
        for (int i = 0; i < k; ++i)
            if (S[i] > cutoff)
                ++count;
        *rank_result = count;
    }
    if (condition_result)
        *condition_result = S[k - 1] == T(0) ? T(INFINITY) : S[0] / S[k - 1];
}

template <typename T>
void reduce_singular_values(const T* A,
    linalg_long                      rows,
    linalg_long                      columns,
    linalg_long                      lda,
    T                                tol,
    void*                            workspace,
    size_t                           workspace_size,
    linalg_long*                     rank_result,
    T*                               condition_result,
    cudaStream_t                     stream)
{
    LINALG_CHECK(A != nullptr, "A must not be null");
    LINALG_CHECK(
        rank_result != nullptr || condition_result != nullptr, "device result must not be null");
    LINALG_CHECK(rows > 0 && columns > 0, "rows and columns must be positive");
    LINALG_CHECK(lda == columns, "rank/condition_number requires tightly packed lda");
    const auto k = std::min(rows, columns);
    const auto svd_bytes =
        detail::svd_on_device_workspace_size(static_cast<const T*>(nullptr), rows, columns);
    LINALG_CHECK(workspace_size >= singular_values_workspace_size_impl<T>(rows, columns),
        "rank/condition_number: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = singular_values_plan<T>(arena, rows, columns, svd_bytes);
    detail::svd_on_device(rows, columns, A, columns, bufs.S, bufs.U, k, bufs.VT, columns,
        bufs.info, bufs.svd_workspace, svd_bytes, stream);
    singular_values_reduce<T><<<1, 1, 0, stream>>>(bufs.S,
        detail::checked_narrow<int>(k),
        rows,
        columns,
        tol,
        std::numeric_limits<T>::epsilon(),
        bufs.info,
        rank_result,
        condition_result);
    LINALG_CHECK(cudaGetLastError() == cudaSuccess, "singular_values_reduce launch failed");
}

}  // namespace

// ---- rank<T, cuda_blas> specializations ----

template <>
void rank<float, linalg_backend::cuda_blas>::apply(const float* A,
    linalg_long                                                 rows,
    linalg_long                                                 columns,
    linalg_long                                                 lda,
    float                                                       tol,
    void*                                                       workspace,
    size_t                                                      workspace_size,
    linalg_long*                                                result,
    void*                                                       stream)
{
    LINALG_SCOPE("rank/cuda_blas<float>");
    reduce_singular_values(A,
        rows,
        columns,
        lda,
        tol,
        workspace,
        workspace_size,
        result,
        static_cast<float*>(nullptr),
        static_cast<cudaStream_t>(stream));
}

template <>
void rank<double, linalg_backend::cuda_blas>::apply(const double* A,
    linalg_long                                                   rows,
    linalg_long                                                   columns,
    linalg_long                                                   lda,
    double                                                        tol,
    void*                                                         workspace,
    size_t                                                        workspace_size,
    linalg_long*                                                  result,
    void*                                                         stream)
{
    LINALG_SCOPE("rank/cuda_blas<double>");
    reduce_singular_values(A,
        rows,
        columns,
        lda,
        tol,
        workspace,
        workspace_size,
        result,
        static_cast<double*>(nullptr),
        static_cast<cudaStream_t>(stream));
}

template <>
size_t rank<float, linalg_backend::cuda_blas>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return singular_values_workspace_size_impl<float>(rows, columns);
}

template <>
size_t rank<double, linalg_backend::cuda_blas>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return singular_values_workspace_size_impl<double>(rows, columns);
}

// ---- condition_number<T, cuda_blas> specializations ----

template <>
void condition_number<float, linalg_backend::cuda_blas>::apply(const float* A,
    linalg_long                                                             rows,
    linalg_long                                                             columns,
    linalg_long                                                             lda,
    void*                                                                   workspace,
    size_t                                                                  workspace_size,
    float*                                                                  result,
    void*                                                                   stream)
{
    LINALG_SCOPE("condition_number/cuda_blas<float>");
    reduce_singular_values(A,
        rows,
        columns,
        lda,
        float(-1),
        workspace,
        workspace_size,
        nullptr,
        result,
        static_cast<cudaStream_t>(stream));
}

template <>
void condition_number<double, linalg_backend::cuda_blas>::apply(const double* A,
    linalg_long                                                               rows,
    linalg_long                                                               columns,
    linalg_long                                                               lda,
    void*                                                                     workspace,
    size_t                                                                    workspace_size,
    double*                                                                   result,
    void*                                                                     stream)
{
    LINALG_SCOPE("condition_number/cuda_blas<double>");
    reduce_singular_values(A,
        rows,
        columns,
        lda,
        double(-1),
        workspace,
        workspace_size,
        nullptr,
        result,
        static_cast<cudaStream_t>(stream));
}

template <>
size_t condition_number<float, linalg_backend::cuda_blas>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return singular_values_workspace_size_impl<float>(rows, columns);
}

template <>
size_t condition_number<double, linalg_backend::cuda_blas>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return singular_values_workspace_size_impl<double>(rows, columns);
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
