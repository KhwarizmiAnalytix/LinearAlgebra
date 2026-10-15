#include "include/matrix_operation_gpu/norm_gpu.h"

#if LINALG_HAS_CUBLAS

#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/matrix_operation_gpu/svd_decomposition_gpu.h"
#include <include/logging.h>
#include "include/common/narrow.h"

// One thread per column/row, looping serially over the other dimension —
// simple and obviously correct rather than a tree reduction, matching this
// codebase's "portable/simple over fastest" fallback philosophy elsewhere
// (see e.g. inversion_gpu.cu's lu_determinant_kernel, also a single-
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
    const auto n       = detail::checked_narrow<int>(rows) * detail::checked_narrow<int>(columns);
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
        const auto svd_bytes = svd_decomposition_workspace_size(static_cast<const T*>(nullptr), rows, columns);
        arena.next<T>(static_cast<size_t>(k));
        arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(k));
        arena.next<T>(static_cast<size_t>(k) * static_cast<size_t>(columns));
        arena.next<int>(1);
        arena.next<unsigned char>(svd_bytes);
        break;
    }
    default:
        LOGGING_THROW("unsupported norm_type", static_cast<linalg_int>(type));
    }
    return arena.total_size();
}

template <typename T, typename AmaxFn>
T norm_one(AmaxFn amax, const T* A, linalg_long rows, linalg_long columns, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);

    LOGGING_CHECK(workspace_size >= matrix_norm_workspace_size_impl<T>(rows, columns, norm_type::ONE),
        "norm: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    T*                        col_sums = arena.next<T>(static_cast<size_t>(c));

    const dim3 block(256);
    const dim3 grid((static_cast<unsigned>(c) + block.x - 1) / block.x);
    abs_col_sum_kernel<T><<<grid, block, 0, stream>>>(A, r, c, col_sums);
    LOGGING_CHECK(cudaGetLastError() == cudaSuccess, "abs_col_sum_kernel launch failed");
    return reduce_max(amax, col_sums, c, stream);
}

template <typename T, typename AmaxFn>
T norm_infinity(AmaxFn amax, const T* A, linalg_long rows, linalg_long columns, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);

    LOGGING_CHECK(workspace_size >= matrix_norm_workspace_size_impl<T>(rows, columns, norm_type::INFINITY_NORM),
        "norm: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    T*                        row_sums = arena.next<T>(static_cast<size_t>(r));

    const dim3 block(256);
    const dim3 grid((static_cast<unsigned>(r) + block.x - 1) / block.x);
    abs_row_sum_kernel<T><<<grid, block, 0, stream>>>(A, r, c, row_sums);
    LOGGING_CHECK(cudaGetLastError() == cudaSuccess, "abs_row_sum_kernel launch failed");
    return reduce_max(amax, row_sums, r, stream);
}

template <typename T, typename AmaxFn>
T norm_two(AmaxFn amax, const T* A, linalg_long rows, linalg_long columns, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    const auto k         = std::min(rows, columns);
    const auto svd_bytes = svd_decomposition_workspace_size(static_cast<const T*>(nullptr), rows, columns);

    LOGGING_CHECK(workspace_size >= matrix_norm_workspace_size_impl<T>(rows, columns, norm_type::TWO),
        "norm: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    T*                        S    = arena.next<T>(static_cast<size_t>(k));
    T*                        U    = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(k));
    T*                        VT   = arena.next<T>(static_cast<size_t>(k) * static_cast<size_t>(columns));
    int*                       info = arena.next<int>(1);
    unsigned char*              svd_workspace = arena.next<unsigned char>(svd_bytes);

    svd_decomposition(rows, columns, A, columns, S, U, k, VT, columns, info, svd_workspace,
        svd_bytes, stream);

    int svd_status = -1;
    const auto info_copy_status =
        cudaMemcpyAsync(&svd_status, info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    const auto sync_status = cudaStreamSynchronize(stream);
    LOGGING_CHECK(!(info_copy_status != cudaSuccess) && !(sync_status != cudaSuccess),
        "cudaMemcpyAsync/cudaStreamSynchronize failed");
    LOGGING_CHECK(svd_status == 0, "linalg::gpu::svd_decomposition failed", svd_status);

    const T result = reduce_max(amax, S, detail::checked_narrow<int>(k), stream);
    return (result < T(0)) ? -result : result;
}

template <typename T, typename Nrm2Fn, typename AmaxFn>
T matrix_norm_impl(Nrm2Fn nrm2,
    AmaxFn                amax,
    const T*              A,
    linalg_long         rows,
    linalg_long         columns,
    linalg_long         lda,
    norm_type      type,
    void*                 workspace,
    size_t                workspace_size,
    cudaStream_t          stream)
{
    if (A == nullptr)
    {
        LOGGING_THROW("norm: A must not be null");
    }
    if (rows == 0 || columns == 0)
    {
        LOGGING_THROW("norm: rows and columns must be positive");
    }
    if (static_cast<linalg_long>(lda) != columns)
    {
        LOGGING_THROW("linalg::gpu::norm requires tightly packed lda (lda == columns)");
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
        LOGGING_THROW("unsupported norm_type", static_cast<linalg_int>(type));
    }
}

}  // namespace

size_t matrix_norm_workspace_size(const float* /*type_tag*/, linalg_long rows, linalg_long columns,
    norm_type type)
{
    return matrix_norm_workspace_size_impl<float>(rows, columns, type);
}

size_t matrix_norm_workspace_size(const double* /*type_tag*/, linalg_long rows, linalg_long columns,
    norm_type type)
{
    return matrix_norm_workspace_size_impl<double>(rows, columns, type);
}

float norm(const float* A, linalg_long rows, linalg_long columns, linalg_long lda,
    norm_type type, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    return matrix_norm_impl(
        cublasSnrm2, cublasIsamax, A, rows, columns, lda, type, workspace, workspace_size, stream);
}

double norm(const double* A, linalg_long rows, linalg_long columns, linalg_long lda,
    norm_type type, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    return matrix_norm_impl(
        cublasDnrm2, cublasIdamax, A, rows, columns, lda, type, workspace, workspace_size, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
