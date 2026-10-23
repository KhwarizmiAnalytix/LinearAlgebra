#if LINALG_HAS_CUBLAS
#include "include/common/cuda_handle.h"
#include "include/common/linalg_profiler.h"
#include "include/matrix_operation/triangular_solver.h"
#include "src/common/triangular_solver_checks.h"
#include <algorithm>

namespace linalg
{

template <>
void triangular_solver<float, linalg_backend::CUDA_BLAS>::apply(bool upper,
    bool                                                             transpose,
    bool                                                             unit,
    linalg_int                                                       n,
    linalg_int                                                       nrhs,
    const float*                                                     a,
    linalg_int                                                       lda,
    const float*                                                     b,
    linalg_int                                                       ldb,
    float*                                                           x,
    linalg_int                                                       ldx,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    void* stream)
{
    LINALG_FUNCTION();
    detail::check_triangular_arguments(a, b, x, n, nrhs, lda, ldb, ldx);
    const auto device_stream = static_cast<cudaStream_t>(stream);
    // Preserve the RHS using a device-to-device copy on the caller's stream.
    LINALG_CHECK(cudaMemcpy2DAsync(x,
                     static_cast<size_t>(ldx) * sizeof(float),
                     b,
                     static_cast<size_t>(ldb) * sizeof(float),
                     static_cast<size_t>(nrhs) * sizeof(float),
                     static_cast<size_t>(n),
                     cudaMemcpyDeviceToDevice,
                     device_stream) == cudaSuccess,
        "triangular_solver: device copy failed");
    auto handle = detail::cublas_handle_for_current_device();
    detail::set_stream(handle, device_stream);
    const float alpha = float(1);
    // Row-major A*X=B is column-major X^T*A^T=B^T: side and triangle swap.
    LINALG_CHECK(cublasStrsm(handle,
                     CUBLAS_SIDE_RIGHT,
                     upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                     transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
                     unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
                     detail::checked_narrow<int>(nrhs),
                     detail::checked_narrow<int>(n),
                     &alpha,
                     a,
                     detail::checked_narrow<int>(lda),
                     x,
                     detail::checked_narrow<int>(ldx)) == CUBLAS_STATUS_SUCCESS,
        "triangular_solver: TRSM failed");
}

template <>
size_t triangular_solver<float, linalg_backend::CUDA_BLAS>::workspace_size(
    linalg_int n, linalg_int nrhs)
{
    detail::check_triangular_dimensions(n, nrhs);
    return 0;
}

template <>
void triangular_solver<double, linalg_backend::CUDA_BLAS>::apply(bool upper,
    bool                                                              transpose,
    bool                                                              unit,
    linalg_int                                                        n,
    linalg_int                                                        nrhs,
    const double*                                                     a,
    linalg_int                                                        lda,
    const double*                                                     b,
    linalg_int                                                        ldb,
    double*                                                           x,
    linalg_int                                                        ldx,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    void* stream)
{
    LINALG_FUNCTION();
    detail::check_triangular_arguments(a, b, x, n, nrhs, lda, ldb, ldx);
    const auto device_stream = static_cast<cudaStream_t>(stream);
    // Preserve the RHS using a device-to-device copy on the caller's stream.
    LINALG_CHECK(cudaMemcpy2DAsync(x,
                     static_cast<size_t>(ldx) * sizeof(double),
                     b,
                     static_cast<size_t>(ldb) * sizeof(double),
                     static_cast<size_t>(nrhs) * sizeof(double),
                     static_cast<size_t>(n),
                     cudaMemcpyDeviceToDevice,
                     device_stream) == cudaSuccess,
        "triangular_solver: device copy failed");
    auto handle = detail::cublas_handle_for_current_device();
    detail::set_stream(handle, device_stream);
    const double alpha = double(1);
    // Row-major A*X=B is column-major X^T*A^T=B^T: side and triangle swap.
    LINALG_CHECK(cublasDtrsm(handle,
                     CUBLAS_SIDE_RIGHT,
                     upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                     transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
                     unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
                     detail::checked_narrow<int>(nrhs),
                     detail::checked_narrow<int>(n),
                     &alpha,
                     a,
                     detail::checked_narrow<int>(lda),
                     x,
                     detail::checked_narrow<int>(ldx)) == CUBLAS_STATUS_SUCCESS,
        "triangular_solver: TRSM failed");
}

template <>
size_t triangular_solver<double, linalg_backend::CUDA_BLAS>::workspace_size(
    linalg_int n, linalg_int nrhs)
{
    detail::check_triangular_dimensions(n, nrhs);
    return 0;
}
}  // namespace linalg
#endif  // LINALG_HAS_CUBLAS
