#if LINALG_HAS_ROCBLAS

#include <algorithm>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <limits>

#include "include/matrix_operation/pseudo_inverse.h"
#include "include/common/hip_handle.h"
#include "include/common/hip_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

// Combines U (rows x k row-major), S (length k), VT (k x columns row-major)
// into Ainv = V * diag(S+) * U^T (columns x rows row-major), one thread per
// output element.
template <typename scalar_t>
static __global__ void pinv_combine_kernel(const scalar_t* U,
    const scalar_t*                                        S,
    const scalar_t*                                        VT,
    scalar_t                                               tol,
    scalar_t                                               default_scale,
    const int*                                             info,
    int                                                    rows,
    int                                                    columns,
    int                                                    k,
    scalar_t*                                              Ainv)
{
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (c >= columns || r >= rows)
    {
        return;
    }
    if (*info != 0)
    {
        Ainv[c * rows + r] = scalar_t(NAN);
        return;
    }
    if (tol < scalar_t(0))
    {
        tol = default_scale * S[0];
    }
    scalar_t sum = scalar_t(0);
    for (int i = 0; i < k; ++i)
    {
        const scalar_t s   = S[i];
        const scalar_t inv = (s > tol) ? (scalar_t(1) / s) : scalar_t(0);
        sum += VT[i * columns + c] * inv * U[r * k + i];
    }
    Ainv[c * rows + r] = sum;
}

namespace linalg
{

// Cross-TU helper: detail::svd_on_device_rocm* defined in svd_decomposition_hip.cpp.
namespace detail
{
size_t svd_on_device_rocm_workspace_size(const float*, linalg_long rows, linalg_long columns);
size_t svd_on_device_rocm_workspace_size(const double*, linalg_long rows, linalg_long columns);
void svd_on_device_rocm(linalg_long rows, linalg_long columns, const float* A, linalg_long lda,
    float* S, float* U, linalg_long ldu, float* VT, linalg_long ldv,
    int* info, void* workspace, size_t workspace_size, hipStream_t stream);
void svd_on_device_rocm(linalg_long rows, linalg_long columns, const double* A, linalg_long lda,
    double* S, double* U, linalg_long ldu, double* VT, linalg_long ldv,
    int* info, void* workspace, size_t workspace_size, hipStream_t stream);
}  // namespace detail

namespace
{

dim3 grid_2d(int width, int height, dim3 block)
{
    return dim3((static_cast<unsigned>(width) + block.x - 1) / block.x,
        (static_cast<unsigned>(height) + block.y - 1) / block.y);
}

template <typename T>
struct pseudo_inverse_layout
{
    T*             S;
    T*             U;
    T*             VT;
    int*           svd_info;
    unsigned char* svd_workspace;
};

template <typename T>
pseudo_inverse_layout<T> pseudo_inverse_plan(
    detail::workspace_arena& arena, int r, int c, int k, size_t svd_workspace_bytes)
{
    pseudo_inverse_layout<T> p;
    p.S             = arena.next<T>(static_cast<size_t>(k));
    p.U             = arena.next<T>(static_cast<size_t>(r) * static_cast<size_t>(k));
    p.VT            = arena.next<T>(static_cast<size_t>(k) * static_cast<size_t>(c));
    p.svd_info      = arena.next<int>(1);
    p.svd_workspace = arena.next<unsigned char>(svd_workspace_bytes);
    return p;
}

template <typename T>
size_t pseudo_inverse_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    const auto r            = detail::checked_narrow<int>(rows);
    const auto c            = detail::checked_narrow<int>(columns);
    const auto k            = std::min(r, c);
    const auto svd_bytes    =
        detail::svd_on_device_rocm_workspace_size(static_cast<const T*>(nullptr), rows, columns);
    detail::workspace_arena arena;
    pseudo_inverse_plan<T>(arena, r, c, k, svd_bytes);
    return arena.total_size();
}

template <typename T>
void pseudo_inverse_impl(linalg_long rows,
    linalg_long                      columns,
    const T*                         A,
    linalg_long                      lda,
    T*                                Ainv,
    linalg_long                      ldai,
    T                                tol,
    void*                            workspace,
    size_t                           workspace_size,
    hipStream_t                      stream)
{
    if (A == nullptr || Ainv == nullptr)
    {
        LINALG_THROW("pseudo_inverse: A and Ainv must not be null");
    }
    if (rows == 0 || columns == 0)
    {
        LINALG_THROW("pseudo_inverse: rows and columns must be positive");
    }
    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);
    if (detail::checked_narrow<int>(lda) != c || detail::checked_narrow<int>(ldai) != r)
    {
        LINALG_THROW(
            "pseudo_inverse<rocm_blas> requires tightly packed lda/ldai "
            "(lda == columns, ldai == rows)");
    }

    const auto k         = std::min(r, c);
    const auto svd_bytes =
        detail::svd_on_device_rocm_workspace_size(static_cast<const T*>(nullptr), rows, columns);

    LINALG_CHECK(workspace_size >= pseudo_inverse_workspace_size_impl<T>(rows, columns),
        "pseudo_inverse: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = pseudo_inverse_plan<T>(arena, r, c, k, svd_bytes);

    detail::svd_on_device_rocm(rows, columns, A, columns, bufs.S, bufs.U, static_cast<linalg_long>(k),
        bufs.VT, columns, bufs.svd_info, bufs.svd_workspace, svd_bytes, stream);

    const dim3 block(16, 16);
    pinv_combine_kernel<T><<<grid_2d(c, r, block), block, 0, stream>>>(bufs.U,
        bufs.S,
        bufs.VT,
        tol,
        std::numeric_limits<T>::epsilon() * static_cast<T>(std::max(r, c)),
        bufs.svd_info,
        r,
        c,
        k,
        Ainv);
    LINALG_CHECK(hipGetLastError() == hipSuccess, "pinv_combine_kernel launch failed");
}

}  // namespace

template <>
void pseudo_inverse<float, linalg_backend::ROCM_BLAS>::apply(linalg_long rows,
    linalg_long   columns,
    const float*  A,
    linalg_long   lda,
    float*        Ainv,
    linalg_long   ldai,
    float         tol,
    void*         workspace,
    size_t        workspace_size,
    void*         stream)
{
    LINALG_SCOPE("pseudo_inverse/rocm_blas<float>");
    pseudo_inverse_impl(rows,
        columns,
        A,
        lda,
        Ainv,
        ldai,
        tol,
        workspace,
        workspace_size,
        static_cast<hipStream_t>(stream));
}

template <>
void pseudo_inverse<double, linalg_backend::ROCM_BLAS>::apply(linalg_long rows,
    linalg_long    columns,
    const double*  A,
    linalg_long    lda,
    double*        Ainv,
    linalg_long    ldai,
    double         tol,
    void*          workspace,
    size_t         workspace_size,
    void*          stream)
{
    LINALG_SCOPE("pseudo_inverse/rocm_blas<double>");
    pseudo_inverse_impl(rows,
        columns,
        A,
        lda,
        Ainv,
        ldai,
        tol,
        workspace,
        workspace_size,
        static_cast<hipStream_t>(stream));
}

template <>
size_t pseudo_inverse<float, linalg_backend::ROCM_BLAS>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return pseudo_inverse_workspace_size_impl<float>(rows, columns);
}

template <>
size_t pseudo_inverse<double, linalg_backend::ROCM_BLAS>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return pseudo_inverse_workspace_size_impl<double>(rows, columns);
}

}  // namespace linalg

#endif  // LINALG_HAS_ROCBLAS
