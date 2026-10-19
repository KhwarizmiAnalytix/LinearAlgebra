#if LINALG_HAS_CUBLAS

#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "include/matrix_operation/norm.h"
#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

// One thread per column/row, looping serially over the other dimension.
template <typename scalar_t>
static __global__ void abs_col_sum_kernel(
    const scalar_t* A, int rows, int columns, scalar_t* col_sums)
{
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= columns)
    {
        return;
    }
    scalar_t sum = scalar_t(0);
    for (int i = 0; i < rows; ++i)
    {
        const scalar_t v = A[i * columns + j];
        sum += (v < scalar_t(0)) ? -v : v;
    }
    col_sums[j] = sum;
}

template <typename scalar_t>
static __global__ void abs_row_sum_kernel(
    const scalar_t* A, int rows, int columns, scalar_t* row_sums)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= rows)
    {
        return;
    }
    scalar_t sum = scalar_t(0);
    for (int j = 0; j < columns; ++j)
    {
        const scalar_t v = A[i * columns + j];
        sum += (v < scalar_t(0)) ? -v : v;
    }
    row_sums[i] = sum;
}

namespace linalg
{

// Cross-TU helper: detail::svd_on_device and detail::svd_on_device_workspace_size
// are defined in svd_decomposition_cublas.cpp.
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

template <typename T, typename Nrm2Fn>
T norm_frobenius(Nrm2Fn nrm2, const T* A, linalg_long rows, linalg_long columns,
    cudaStream_t stream)
{
    auto handle = detail::cublas_handle_for_current_device();
    detail::set_stream(handle, stream);
    T          result = T(0);
    const auto n       = detail::checked_narrow<int>(rows) * detail::checked_narrow<int>(columns);
    if (nrm2(handle, n, A, 1, &result) != CUBLAS_STATUS_SUCCESS)
    {
        LINALG_THROW("cublasX?nrm2 failed");
    }
    return result;
}

template <typename T, typename AmaxFn>
T reduce_max(AmaxFn amax, const T* values, int n, cudaStream_t stream)
{
    auto handle = detail::cublas_handle_for_current_device();
    detail::set_stream(handle, stream);
    int idx_1based = 1;
    if (amax(handle, n, values, 1, &idx_1based) != CUBLAS_STATUS_SUCCESS)
    {
        LINALG_THROW("cublasI*amax failed");
    }
    T result = T(0);
    if (cudaMemcpyAsync(&result, values + (idx_1based - 1), sizeof(T), cudaMemcpyDeviceToHost,
            stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
    return result;
}

template <typename T>
size_t matrix_norm_workspace_size_impl(linalg_long rows, linalg_long columns, norm_type type)
{
    const auto               c = detail::checked_narrow<int>(columns);
    const auto               r = detail::checked_narrow<int>(rows);
    detail::workspace_arena arena;
    switch (type)
    {
    case norm_type::FROBENIUS:
        break;
    case norm_type::ONE:
        arena.next<T>(static_cast<size_t>(c));
        break;
    case norm_type::INFINITY_NORM:
        arena.next<T>(static_cast<size_t>(r));
        break;
    case norm_type::TWO:
    {
        const auto k         = std::min(rows, columns);
        const auto svd_bytes =
            detail::svd_on_device_workspace_size(static_cast<const T*>(nullptr), rows, columns);
        arena.next<T>(static_cast<size_t>(k));
        arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(k));
        arena.next<T>(static_cast<size_t>(k) * static_cast<size_t>(columns));
        arena.next<int>(1);
        arena.next<unsigned char>(svd_bytes);
        break;
    }
    default:
        LINALG_THROW("unsupported norm_type", static_cast<linalg_int>(type));
    }
    return arena.total_size();
}

template <typename T, typename AmaxFn>
T norm_one(AmaxFn amax, const T* A, linalg_long rows, linalg_long columns, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);
    LINALG_CHECK(workspace_size >= matrix_norm_workspace_size_impl<T>(rows, columns, norm_type::ONE),
        "norm: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    T*                        col_sums = arena.next<T>(static_cast<size_t>(c));
    const dim3 block(256);
    const dim3 grid((static_cast<unsigned>(c) + block.x - 1) / block.x);
    abs_col_sum_kernel<T><<<grid, block, 0, stream>>>(A, r, c, col_sums);
    LINALG_CHECK(cudaGetLastError() == cudaSuccess, "abs_col_sum_kernel launch failed");
    return reduce_max(amax, col_sums, c, stream);
}

template <typename T, typename AmaxFn>
T norm_infinity(AmaxFn amax, const T* A, linalg_long rows, linalg_long columns, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);
    LINALG_CHECK(
        workspace_size >= matrix_norm_workspace_size_impl<T>(rows, columns, norm_type::INFINITY_NORM),
        "norm: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    T*                        row_sums = arena.next<T>(static_cast<size_t>(r));
    const dim3 block(256);
    const dim3 grid((static_cast<unsigned>(r) + block.x - 1) / block.x);
    abs_row_sum_kernel<T><<<grid, block, 0, stream>>>(A, r, c, row_sums);
    LINALG_CHECK(cudaGetLastError() == cudaSuccess, "abs_row_sum_kernel launch failed");
    return reduce_max(amax, row_sums, r, stream);
}

template <typename T, typename AmaxFn>
T norm_two(AmaxFn amax, const T* A, linalg_long rows, linalg_long columns, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    const auto k         = std::min(rows, columns);
    const auto svd_bytes =
        detail::svd_on_device_workspace_size(static_cast<const T*>(nullptr), rows, columns);
    LINALG_CHECK(workspace_size >= matrix_norm_workspace_size_impl<T>(rows, columns, norm_type::TWO),
        "norm: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    T*              S             = arena.next<T>(static_cast<size_t>(k));
    T*              U             = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(k));
    T*              VT            = arena.next<T>(static_cast<size_t>(k) * static_cast<size_t>(columns));
    int*            info          = arena.next<int>(1);
    unsigned char*  svd_workspace = arena.next<unsigned char>(svd_bytes);

    detail::svd_on_device(rows, columns, A, columns, S, U, k, VT, columns, info, svd_workspace,
        svd_bytes, stream);

    int svd_status = -1;
    LINALG_CHECK(
        cudaMemcpyAsync(&svd_status, info, sizeof(int), cudaMemcpyDeviceToHost, stream) ==
                cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess,
        "cudaMemcpyAsync/cudaStreamSynchronize failed");
    LINALG_CHECK(svd_status == 0, "svd_on_device failed inside norm_two", svd_status);

    const T result = reduce_max(amax, S, detail::checked_narrow<int>(k), stream);
    return (result < T(0)) ? -result : result;
}

template <typename T, typename Nrm2Fn, typename AmaxFn>
T matrix_norm_impl(Nrm2Fn nrm2,
    AmaxFn                amax,
    const T*              A,
    linalg_long           rows,
    linalg_long           columns,
    linalg_long           lda,
    norm_type             type,
    void*                 workspace,
    size_t                workspace_size,
    cudaStream_t          stream)
{
    if (A == nullptr)
    {
        LINALG_THROW("norm: A must not be null");
    }
    if (rows == 0 || columns == 0)
    {
        LINALG_THROW("norm: rows and columns must be positive");
    }
    if (static_cast<linalg_long>(lda) != columns)
    {
        LINALG_THROW("norm<cuda_blas> requires tightly packed lda (lda == columns)");
    }

    switch (type)
    {
    case norm_type::FROBENIUS:
        return norm_frobenius(nrm2, A, rows, columns, stream);
    case norm_type::ONE:
        return norm_one(amax, A, rows, columns, workspace, workspace_size, stream);
    case norm_type::INFINITY_NORM:
        return norm_infinity(amax, A, rows, columns, workspace, workspace_size, stream);
    case norm_type::TWO:
        return norm_two(amax, A, rows, columns, workspace, workspace_size, stream);
    default:
        LINALG_THROW("unsupported norm_type", static_cast<linalg_int>(type));
    }
}

}  // namespace

template <>
float norm<float, linalg_backend::cuda_blas>::apply(const float* A,
    linalg_long  rows,
    linalg_long  columns,
    linalg_long  lda,
    norm_type    type,
    void*        workspace,
    size_t       workspace_size,
    void*        stream)
{
    LINALG_SCOPE("norm/cuda_blas<float>");
    return matrix_norm_impl(cublasSnrm2, cublasIsamax, A, rows, columns, lda, type, workspace,
        workspace_size, static_cast<cudaStream_t>(stream));
}

template <>
double norm<double, linalg_backend::cuda_blas>::apply(const double* A,
    linalg_long   rows,
    linalg_long   columns,
    linalg_long   lda,
    norm_type     type,
    void*         workspace,
    size_t        workspace_size,
    void*         stream)
{
    LINALG_SCOPE("norm/cuda_blas<double>");
    return matrix_norm_impl(cublasDnrm2, cublasIdamax, A, rows, columns, lda, type, workspace,
        workspace_size, static_cast<cudaStream_t>(stream));
}

template <>
size_t norm<float, linalg_backend::cuda_blas>::workspace_size(
    linalg_long rows, linalg_long columns, norm_type type)
{
    return matrix_norm_workspace_size_impl<float>(rows, columns, type);
}

template <>
size_t norm<double, linalg_backend::cuda_blas>::workspace_size(
    linalg_long rows, linalg_long columns, norm_type type)
{
    return matrix_norm_workspace_size_impl<double>(rows, columns, type);
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
