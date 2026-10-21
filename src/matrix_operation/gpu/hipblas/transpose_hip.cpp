#if LINALG_HAS_ROCBLAS

#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

#include "include/matrix_operation/transpose.h"
#include "include/common/hip_handle.h"
#include "include/common/hip_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

namespace linalg
{
namespace
{

// hipblasSgeam/Dgeam are out-of-place (C = alpha*op(A) + beta*op(B)), and this
// API transposes `m` in place, so a caller-supplied scratch device buffer of
// the same size is used as the out-of-place destination and copied back
// device-to-device (never touching host memory). Both the geam call and the
// copy-back run on `stream`, so a caller-supplied non-default stream is
// honored end to end.
template <typename T, typename GeamFn>
void transpose_device_impl(
    GeamFn geam, linalg_long rows, linalg_long columns, T* m, T* scratch, hipStream_t stream)
{
    LINALG_CHECK(m != nullptr && scratch != nullptr, "transpose_device: m and scratch must not be null");
    LINALG_CHECK(rows != 0 && columns != 0, "transpose_device: rows and columns must be positive");
    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);

    auto handle = detail::hipblas_handle_for_current_device();
    detail::set_hipblas_stream(handle, stream);

    const T alpha = static_cast<T>(1);
    const T beta  = static_cast<T>(0);
    // Row-major `m` (rows x columns, ld=columns) is bit-identical to a
    // column-major (columns x rows, ld=columns) reading of the same buffer.
    // geam with transa=T treats that as the *pre-transpose* operand (so its
    // declared shape is n x m_out = columns x rows, matching `m`/ld exactly)
    // and produces a genuine (rows x columns, ld=rows) column-major result in
    // `scratch` — which is bit-identical to the desired row-major
    // (columns x rows, ld=rows) transposed output.
    LINALG_CHECK(geam(handle, HIPBLAS_OP_T, HIPBLAS_OP_T, r, c, &alpha, m, c, &beta, m, c, scratch, r) ==
            HIPBLAS_STATUS_SUCCESS,
        "hipblasSgeam/Dgeam failed");

    const auto copy_status = hipMemcpyAsync(m,
        scratch,
        sizeof(T) * static_cast<size_t>(r) * static_cast<size_t>(c),
        hipMemcpyDeviceToDevice,
        stream);
    LINALG_CHECK(!(copy_status != hipSuccess), "hipMemcpyAsync failed");
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
    size_t workspace_size, hipStream_t stream)
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

// detail::transpose_device is used by several other hipblas translation units
// (eigen, lu, linear_solver, inversion, qr, svd). It lives in namespace
// linalg::detail with non-inline external linkage so those TUs can call it
// across the link boundary. Forward declarations are in hip_handle.h.
namespace detail
{

void transpose_device(
    linalg_long rows, linalg_long columns, float* m, float* scratch, hipStream_t stream)
{
    transpose_device_impl(hipblasSgeam, rows, columns, m, scratch, stream);
}

void transpose_device(
    linalg_long rows, linalg_long columns, double* m, double* scratch, hipStream_t stream)
{
    transpose_device_impl(hipblasDgeam, rows, columns, m, scratch, stream);
}

}  // namespace detail

template <>
void transpose<float, linalg_backend::ROCM_BLAS>::apply(linalg_long rows,
    linalg_long  columns,
    float*       m,
    void*        workspace,
    size_t       workspace_size,
    void*        stream)
{
    LINALG_SCOPE("transpose/rocm_blas<float>");
    transpose_impl(rows, columns, m, workspace, workspace_size,
        static_cast<hipStream_t>(stream));
}

template <>
void transpose<double, linalg_backend::ROCM_BLAS>::apply(linalg_long rows,
    linalg_long  columns,
    double*      m,
    void*        workspace,
    size_t       workspace_size,
    void*        stream)
{
    LINALG_SCOPE("transpose/rocm_blas<double>");
    transpose_impl(rows, columns, m, workspace, workspace_size,
        static_cast<hipStream_t>(stream));
}

template <>
size_t transpose<float, linalg_backend::ROCM_BLAS>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return transpose_workspace_size_impl<float>(rows, columns);
}

template <>
size_t transpose<double, linalg_backend::ROCM_BLAS>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return transpose_workspace_size_impl<double>(rows, columns);
}

}  // namespace linalg

#endif  // LINALG_HAS_ROCBLAS
