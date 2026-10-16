#include "include/matrix_operation_gpu/transpose_gpu.h"

#if LINALG_HAS_CUBLAS

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/logging.h"
#include "include/common/narrow.h"

namespace linalg
{
namespace gpu
{
namespace
{

template <typename T>
size_t transpose_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);
    detail::workspace_arena arena;
    arena.next<T>(static_cast<size_t>(r) * static_cast<size_t>(c));
    return arena.total_size();
}

}  // namespace

size_t matrix_transpose_workspace_size(const float* /*type_tag*/, linalg_long rows, linalg_long columns)
{
    return transpose_workspace_size_impl<float>(rows, columns);
}

size_t matrix_transpose_workspace_size(const double* /*type_tag*/, linalg_long rows, linalg_long columns)
{
    return transpose_workspace_size_impl<double>(rows, columns);
}

void transpose(linalg_long rows, linalg_long columns, float* m, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    LOGGING_CHECK(m != nullptr, "transpose: m must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0, "transpose: rows and columns must be positive");
    LOGGING_CHECK(workspace_size >= matrix_transpose_workspace_size(static_cast<const float*>(nullptr), rows, columns),
        "transpose: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    float*                  scratch =
        arena.next<float>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    detail::transpose_device(rows, columns, m, scratch, stream);
}

void transpose(linalg_long rows, linalg_long columns, double* m, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    LOGGING_CHECK(m != nullptr, "transpose: m must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0, "transpose: rows and columns must be positive");
    LOGGING_CHECK(workspace_size >= matrix_transpose_workspace_size(static_cast<const double*>(nullptr), rows, columns),
        "transpose: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    double*                 scratch =
        arena.next<double>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    detail::transpose_device(rows, columns, m, scratch, stream);
}

}  // namespace gpu

namespace detail
{

namespace
{

// cublasSgeam/Dgeam are out-of-place (C = alpha*op(A) + beta*op(B)), and this
// API transposes `m` in place, so a caller-supplied scratch device buffer of
// the same size is used as the out-of-place destination and copied back
// device-to-device (never touching host memory). Both the geam call and the
// copy-back run on `stream`, so a caller-supplied non-default stream is
// honored end to end.
template <typename T, typename GeamFn>
void transpose_device_impl(
    GeamFn geam, linalg_long rows, linalg_long columns, T* m, T* scratch, cudaStream_t stream)
{
    LOGGING_CHECK(m != nullptr && scratch != nullptr, "transpose_device: m and scratch must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0, "transpose_device: rows and columns must be positive");
    const auto r = checked_narrow<int>(rows);
    const auto c = checked_narrow<int>(columns);

    auto handle = cublas_handle_for_current_device();
    set_stream(handle, stream);

    const T alpha = static_cast<T>(1);
    const T beta  = static_cast<T>(0);
    // Row-major `m` (rows x columns, ld=columns) is bit-identical to a
    // column-major (columns x rows, ld=columns) reading of the same buffer.
    // geam with transa=T treats that as the *pre-transpose* operand (so its
    // declared shape is n x m_out = columns x rows, matching `m`/ld exactly)
    // and produces a genuine (rows x columns, ld=rows) column-major result in
    // `scratch` — which is bit-identical to the desired row-major
    // (columns x rows, ld=rows) transposed output.
    LOGGING_CHECK(geam(handle, CUBLAS_OP_T, CUBLAS_OP_T, r, c, &alpha, m, c, &beta, m, c, scratch, r) ==
            CUBLAS_STATUS_SUCCESS,
        "cublasSgeam/Dgeam failed");

    const auto copy_status = cudaMemcpyAsync(m,
        scratch,
        sizeof(T) * static_cast<size_t>(r) * static_cast<size_t>(c),
        cudaMemcpyDeviceToDevice,
        stream);
    LOGGING_CHECK(!(copy_status != cudaSuccess), "cudaMemcpyAsync failed");
}

}  // namespace

void transpose_device(
    linalg_long rows, linalg_long columns, float* m, float* scratch, cudaStream_t stream)
{
    transpose_device_impl(cublasSgeam, rows, columns, m, scratch, stream);
}

void transpose_device(
    linalg_long rows, linalg_long columns, double* m, double* scratch, cudaStream_t stream)
{
    transpose_device_impl(cublasDgeam, rows, columns, m, scratch, stream);
}

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
