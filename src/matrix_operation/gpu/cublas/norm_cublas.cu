#if LINALG_HAS_CUBLAS

#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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

template <typename T>
static __global__ void max_value_kernel(
    const T* values, int n, T* result, const int* info = nullptr)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    if (info && *info != 0)
    {
        *result = T(NAN);
        return;
    }
    T maximum = T(0);
    for (int i = 0; i < n; ++i)
        maximum = values[i] > maximum ? values[i] : maximum;
    *result = maximum;
}

template <typename T>
void reduce_max(const T* values, int n, T* result, cudaStream_t stream, const int* info = nullptr)
{
    max_value_kernel<T><<<1, 1, 0, stream>>>(values, n, result, info);
    LINALG_CHECK(cudaGetLastError() == cudaSuccess, "max_value_kernel launch failed");
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
        arena.next<T>(1);  // device scratch for the reduced result
        break;
    case norm_type::INFINITY_NORM:
        arena.next<T>(static_cast<size_t>(r));
        arena.next<T>(1);  // device scratch for the reduced result
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
        arena.next<T>(1);  // device scratch for the reduced result
        break;
    }
    default:
        LINALG_THROW("unsupported norm_type", static_cast<linalg_int>(type));
    }
    return arena.total_size();
}

template <typename T, typename Nrm2Fn>
T matrix_norm_impl(Nrm2Fn nrm2,
    const T*             A,
    linalg_long          rows,
    linalg_long          columns,
    linalg_long          lda,
    norm_type            type,
    void*                workspace,
    size_t               workspace_size,
    cudaStream_t         stream)
{
    LINALG_CHECK(A != nullptr, "norm: device input must not be null");
    LINALG_CHECK(rows > 0 && columns > 0, "norm: dimensions must be positive");
    LINALG_CHECK(lda == columns, "norm requires tightly packed lda");
    const int r = detail::checked_narrow<int>(rows);
    const int c = detail::checked_narrow<int>(columns);
    LINALG_CHECK(workspace_size >= matrix_norm_workspace_size_impl<T>(rows, columns, type),
        "norm: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    T                        host_result = T(0);
    if (type == norm_type::FROBENIUS)
    {
        const int count =
            detail::checked_narrow<int>(static_cast<size_t>(r) * static_cast<size_t>(c));
        auto handle = detail::cublas_handle_for_current_device();
        detail::set_stream(handle, stream);
        // Default (host) cublas pointer mode: nrm2 synchronizes internally
        // and writes straight to host memory, so no device scratch or
        // explicit stream sync is needed for this branch.
        LINALG_CHECK(nrm2(handle, count, A, 1, &host_result) == CUBLAS_STATUS_SUCCESS,
            "cublas nrm2 failed");
    }
    else if (type == norm_type::ONE || type == norm_type::INFINITY_NORM)
    {
        const int  count = type == norm_type::ONE ? c : r;
        T*         sums  = arena.next<T>(static_cast<size_t>(count));
        T*         dev_result = arena.next<T>(1);
        const dim3 block(256), grid((static_cast<unsigned>(count) + 255) / 256);
        if (type == norm_type::ONE)
            abs_col_sum_kernel<T><<<grid, block, 0, stream>>>(A, r, c, sums);
        else
            abs_row_sum_kernel<T><<<grid, block, 0, stream>>>(A, r, c, sums);
        LINALG_CHECK(cudaGetLastError() == cudaSuccess, "norm sum kernel launch failed");
        reduce_max(sums, count, dev_result, stream);
        LINALG_CHECK(cudaMemcpyAsync(&host_result, dev_result, sizeof(T), cudaMemcpyDeviceToHost,
                         stream) == cudaSuccess,
            "norm: result copy failed");
        LINALG_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "norm: stream sync failed");
    }
    else if (type == norm_type::TWO)
    {
        const auto k = std::min(rows, columns);
        const auto svd_bytes =
            detail::svd_on_device_workspace_size(static_cast<const T*>(nullptr), rows, columns);
        T*    S             = arena.next<T>(static_cast<size_t>(k));
        T*    U             = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(k));
        T*    VT            = arena.next<T>(static_cast<size_t>(k) * static_cast<size_t>(columns));
        int*  info          = arena.next<int>(1);
        auto* svd_workspace = arena.next<unsigned char>(svd_bytes);
        T*    dev_result    = arena.next<T>(1);
        detail::svd_on_device(rows,
            columns,
            A,
            columns,
            S,
            U,
            k,
            VT,
            columns,
            info,
            svd_workspace,
            svd_bytes,
            stream);
        reduce_max(S, detail::checked_narrow<int>(k), dev_result, stream, info);
        LINALG_CHECK(cudaMemcpyAsync(&host_result, dev_result, sizeof(T), cudaMemcpyDeviceToHost,
                         stream) == cudaSuccess,
            "norm: result copy failed");
        LINALG_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "norm: stream sync failed");
    }
    else
        LINALG_THROW("unsupported norm_type");
    return host_result;
}

}  // namespace

template <>
float norm<float, linalg_backend::CUDA_BLAS>::apply(const float* A,
    linalg_long                                                  rows,
    linalg_long                                                  columns,
    linalg_long                                                  lda,
    norm_type                                                    type,
    void*                                                        workspace,
    size_t                                                       workspace_size,
    void*                                                        stream)
{
    return matrix_norm_impl(cublasSnrm2,
        A,
        rows,
        columns,
        lda,
        type,
        workspace,
        workspace_size,
        static_cast<cudaStream_t>(stream));
}

template <>
size_t norm<float, linalg_backend::CUDA_BLAS>::workspace_size(
    linalg_long rows, linalg_long columns, norm_type type)
{
    return matrix_norm_workspace_size_impl<float>(rows, columns, type);
}

template <>
double norm<double, linalg_backend::CUDA_BLAS>::apply(const double* A,
    linalg_long                                                    rows,
    linalg_long                                                    columns,
    linalg_long                                                    lda,
    norm_type                                                      type,
    void*                                                          workspace,
    size_t                                                         workspace_size,
    void*                                                          stream)
{
    return matrix_norm_impl(cublasDnrm2,
        A,
        rows,
        columns,
        lda,
        type,
        workspace,
        workspace_size,
        static_cast<cudaStream_t>(stream));
}

template <>
size_t norm<double, linalg_backend::CUDA_BLAS>::workspace_size(
    linalg_long rows, linalg_long columns, norm_type type)
{
    return matrix_norm_workspace_size_impl<double>(rows, columns, type);
}

}  // namespace linalg
#endif
