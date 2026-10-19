#if LINALG_HAS_CUBLAS

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "include/matrix_operation/transpose.h"
#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

namespace linalg
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
    LINALG_CHECK(m != nullptr && scratch != nullptr, "transpose_device: m and scratch must not be null");
    LINALG_CHECK(rows != 0 && columns != 0, "transpose_device: rows and columns must be positive");
    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);

    auto handle = detail::cublas_handle_for_current_device();
    detail::set_stream(handle, stream);

    const T alpha = static_cast<T>(1);
    const T beta  = static_cast<T>(0);
    // Row-major `m` (rows x columns, ld=columns) is bit-identical to a
    // column-major (columns x rows, ld=columns) reading of the same buffer.
    // geam with transa=T treats that as the *pre-transpose* operand (so its
    // declared shape is n x m_out = columns x rows, matching `m`/ld exactly)
    // and produces a genuine (rows x columns, ld=rows) column-major result in
    // `scratch` — which is bit-identical to the desired row-major
    // (columns x rows, ld=rows) transposed output.
    LINALG_CHECK(geam(handle, CUBLAS_OP_T, CUBLAS_OP_T, r, c, &alpha, m, c, &beta, m, c, scratch, r) ==
            CUBLAS_STATUS_SUCCESS,
        "cublasSgeam/Dgeam failed");

    const auto copy_status = cudaMemcpyAsync(m,
        scratch,
        sizeof(T) * static_cast<size_t>(r) * static_cast<size_t>(c),
        cudaMemcpyDeviceToDevice,
        stream);
    LINALG_CHECK(!(copy_status != cudaSuccess), "cudaMemcpyAsync failed");
}

template <typename T>
size_t transpose_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);
    detail::workspace_arena arena;
    arena.next<T>(static_cast<size_t>(r) * static_cast<size_t>(c));
    return arena.total_size();
}

template <typename T>
void transpose_impl(linalg_long rows, linalg_long columns, T* m, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    LINALG_CHECK(m != nullptr, "transpose: m must not be null");
    LINALG_CHECK(rows != 0 && columns != 0, "transpose: rows and columns must be positive");
    LINALG_CHECK(workspace_size >= transpose_workspace_size_impl<T>(rows, columns),
        "transpose: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    T*                      scratch =
        arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    detail::transpose_device(rows, columns, m, scratch, stream);
}

}  // namespace

// detail::transpose_device is used by several other cublas translation units
// (eigen, lu, linear_solver, inversion, qr, svd). It lives in namespace
// linalg::detail with non-inline external linkage so those TUs can call it
// across the link boundary. Forward declarations are in cuda_handle.h.
namespace detail
{

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

template <>
void transpose<float, linalg_backend::cuda_blas>::apply(linalg_long rows,
    linalg_long  columns,
    float*       m,
    void*        workspace,
    size_t       workspace_size,
    void*        stream)
{
    LINALG_SCOPE("transpose/cuda_blas<float>");
    transpose_impl(rows, columns, m, workspace, workspace_size,
        static_cast<cudaStream_t>(stream));
}

template <>
void transpose<double, linalg_backend::cuda_blas>::apply(linalg_long rows,
    linalg_long  columns,
    double*      m,
    void*        workspace,
    size_t       workspace_size,
    void*        stream)
{
    LINALG_SCOPE("transpose/cuda_blas<double>");
    transpose_impl(rows, columns, m, workspace, workspace_size,
        static_cast<cudaStream_t>(stream));
}

template <>
size_t transpose<float, linalg_backend::cuda_blas>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return transpose_workspace_size_impl<float>(rows, columns);
}

template <>
size_t transpose<double, linalg_backend::cuda_blas>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return transpose_workspace_size_impl<double>(rows, columns);
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
