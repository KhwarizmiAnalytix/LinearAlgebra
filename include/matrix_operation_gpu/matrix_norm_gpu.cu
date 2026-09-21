#include "include/matrix_operation_gpu/matrix_norm_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "include/common/cuda_handle.h"
#include "include/matrix_operation_gpu/svd_decomposition_gpu.h"
#include <include/logging.h>

// One thread per column/row, looping serially over the other dimension —
// simple and obviously correct rather than a tree reduction, matching this
// codebase's "portable/simple over fastest" fallback philosophy elsewhere
// (see e.g. matrix_inversion_gpu.cu's lu_determinant_kernel, also a single-
// purpose file-scope kernel). `A` is row-major rows x columns, tightly
// packed.
template <typename scalar_t>
static __global__ void abs_col_sum_kernel(const scalar_t* A, int rows, int columns, scalar_t* col_sums)
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
static __global__ void abs_row_sum_kernel(const scalar_t* A, int rows, int columns, scalar_t* row_sums)
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
namespace gpu
{
namespace
{

template <typename T, typename Nrm2Fn>
T norm_frobenius(Nrm2Fn nrm2, const T* A, linalg_long rows, linalg_long columns, cudaStream_t stream)
{
    auto handle = detail::cublas_handle_for_current_device();
    detail::set_stream(handle, stream);
    T          result = T(0);
    const auto n       = static_cast<int>(rows) * static_cast<int>(columns);
    if (nrm2(handle, n, A, 1, &result) != CUBLAS_STATUS_SUCCESS)
    {
        LOGGING_THROW("cublasX?nrm2 failed");
    }
    return result;
}

// Shared by ONE/INFINITY_NORM: reduce `values` (length n, all >= 0, already
// device-resident) down to its largest entry via cublasI?amax, then fetch
// that single element back to the host.
template <typename T, typename AmaxFn>
T reduce_max(AmaxFn amax, const T* values, int n, cudaStream_t stream)
{
    auto handle = detail::cublas_handle_for_current_device();
    detail::set_stream(handle, stream);
    int idx_1based = 1;
    if (amax(handle, n, values, 1, &idx_1based) != CUBLAS_STATUS_SUCCESS)
    {
        LOGGING_THROW("cublasI*amax failed");
    }
    T result = T(0);
    if (cudaMemcpyAsync(&result, values + (idx_1based - 1), sizeof(T), cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
    {
        LOGGING_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
    return result;
}

template <typename T, typename AmaxFn>
T norm_one(AmaxFn amax, const T* A, linalg_long rows, linalg_long columns, cudaStream_t stream)
{
    const auto r = static_cast<int>(rows);
    const auto c = static_cast<int>(columns);

    T* col_sums = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&col_sums), sizeof(T) * static_cast<size_t>(c)) != cudaSuccess)
    {
        LOGGING_THROW("cudaMalloc failed");
    }
    const dim3 block(256);
    const dim3 grid((static_cast<unsigned>(c) + block.x - 1) / block.x);
    abs_col_sum_kernel<T><<<grid, block, 0, stream>>>(A, r, c, col_sums);
    if (cudaGetLastError() != cudaSuccess)
    {
        cudaFree(col_sums);
        LOGGING_THROW("abs_col_sum_kernel launch failed");
    }
    const T result = reduce_max(amax, col_sums, c, stream);
    cudaFree(col_sums);
    return result;
}

template <typename T, typename AmaxFn>
T norm_infinity(AmaxFn amax, const T* A, linalg_long rows, linalg_long columns, cudaStream_t stream)
{
    const auto r = static_cast<int>(rows);
    const auto c = static_cast<int>(columns);

    T* row_sums = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&row_sums), sizeof(T) * static_cast<size_t>(r)) != cudaSuccess)
    {
        LOGGING_THROW("cudaMalloc failed");
    }
    const dim3 block(256);
    const dim3 grid((static_cast<unsigned>(r) + block.x - 1) / block.x);
    abs_row_sum_kernel<T><<<grid, block, 0, stream>>>(A, r, c, row_sums);
    if (cudaGetLastError() != cudaSuccess)
    {
        cudaFree(row_sums);
        LOGGING_THROW("abs_row_sum_kernel launch failed");
    }
    const T result = reduce_max(amax, row_sums, r, stream);
    cudaFree(row_sums);
    return result;
}

template <typename T, typename AmaxFn>
T norm_two(AmaxFn amax, const T* A, linalg_long rows, linalg_long columns, cudaStream_t stream)
{
    const auto k = std::min(rows, columns);

    T*  S    = nullptr;
    T*  U    = nullptr;
    T*  VT   = nullptr;
    int* info = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&S), sizeof(T) * static_cast<size_t>(k)) != cudaSuccess)
    {
        LOGGING_THROW("cudaMalloc failed");
    }
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
    const auto info_copy_status =
        cudaMemcpyAsync(&svd_status, info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    // U/VT are unused here (only the singular values matter), but they are
    // still being written by svd_decomposition's own asynchronous work on
    // `stream` until this sync — freeing them any earlier would race it.
    const auto sync_status = cudaStreamSynchronize(stream);
    cudaFree(U);
    cudaFree(VT);
    cudaFree(info);
    if (info_copy_status != cudaSuccess || sync_status != cudaSuccess)
    {
        cudaFree(S);
        LOGGING_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
    if (svd_status != 0)
    {
        cudaFree(S);
        LOGGING_THROW("linalg::gpu::svd_decomposition failed", svd_status);
    }

    const T result = reduce_max(amax, S, static_cast<int>(k), stream);
    cudaFree(S);
    return (result < T(0)) ? -result : result;
}

template <typename T, typename Nrm2Fn, typename AmaxFn>
T matrix_norm_impl(Nrm2Fn nrm2,
    AmaxFn                amax,
    const T*              A,
    linalg_long         rows,
    linalg_long         columns,
    linalg_long         lda,
    matrix_norm_type      type,
    cudaStream_t          stream)
{
    if (A == nullptr)
    {
        LOGGING_THROW("matrix_norm: A must not be null");
    }
    if (rows == 0 || columns == 0)
    {
        LOGGING_THROW("matrix_norm: rows and columns must be positive");
    }
    if (static_cast<linalg_long>(lda) != columns)
    {
        LOGGING_THROW("linalg::gpu::matrix_norm requires tightly packed lda (lda == columns)");
    }

    switch (type)
    {
    case matrix_norm_type::FROBENIUS:
        return norm_frobenius(nrm2, A, rows, columns, stream);
    case matrix_norm_type::ONE:
        return norm_one(amax, A, rows, columns, stream);
    case matrix_norm_type::INFINITY_NORM:
        return norm_infinity(amax, A, rows, columns, stream);
    case matrix_norm_type::TWO:
        return norm_two(amax, A, rows, columns, stream);
    default:
        LOGGING_THROW("unsupported matrix_norm_type", static_cast<linalg_int>(type));
    }
}

}  // namespace

float matrix_norm(
    const float* A, linalg_long rows, linalg_long columns, linalg_long lda, matrix_norm_type type,
    cudaStream_t stream)
{
    return matrix_norm_impl(cublasSnrm2, cublasIsamax, A, rows, columns, lda, type, stream);
}

double matrix_norm(
    const double* A, linalg_long rows, linalg_long columns, linalg_long lda, matrix_norm_type type,
    cudaStream_t stream)
{
    return matrix_norm_impl(cublasDnrm2, cublasIdamax, A, rows, columns, lda, type, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
