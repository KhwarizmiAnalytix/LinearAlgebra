#include "include/matrix_operation_gpu/lu_decomposition_gpu.h"

#if LINALG_HAS_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/matrix_operation_gpu/transpose_gpu.h"
#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"
#include "include/common/linalg_profiler.h"

namespace linalg
{
namespace gpu
{
namespace
{

// Two scratch buffers: cuSOLVER's own getrf workspace, and the n x n
// transpose scratch the pre/post getrf transposes need (detail::
// transpose_device, transpose_gpu.cpp — reused across both calls
// since they run sequentially on the same stream). Shared by the size query
// and the real call so they can never disagree on layout (see
// include/common/cuda_workspace.h).
template <typename T>
struct lu_decomposition_layout
{
    T* workspace;
    T* transpose_scratch;
};

template <typename T>
lu_decomposition_layout<T> lu_decomposition_plan(detail::workspace_arena& arena, int lwork, int n)
{
    lu_decomposition_layout<T> p;
    p.workspace         = arena.next<T>(static_cast<size_t>(lwork));
    p.transpose_scratch = arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    return p;
}

}  // namespace

size_t lu_decomposition_workspace_size(const float* /*type_tag*/, linalg_int lda)
{
    LINALG_CHECK(lda > 0, "lu_decomposition_workspace_size: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    int        lwork  = 0;
    LINALG_CHECK(cusolverDnSgetrf_bufferSize(handle, n, n, nullptr, n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDnSgetrf_bufferSize failed");
    detail::workspace_arena arena;
    lu_decomposition_plan<float>(arena, lwork, n);
    return arena.total_size();
}

size_t lu_decomposition_workspace_size(const double* /*type_tag*/, linalg_int lda)
{
    LINALG_CHECK(lda > 0, "lu_decomposition_workspace_size: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    int        lwork  = 0;
    LINALG_CHECK(cusolverDnDgetrf_bufferSize(handle, n, n, nullptr, n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDnDgetrf_bufferSize failed");
    detail::workspace_arena arena;
    lu_decomposition_plan<double>(arena, lwork, n);
    return arena.total_size();
}

void lu_decomposition(float* m, linalg_int lda, int* pivot, int* info, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    LINALG_SCOPE("lu_decomposition/gpu<float>");
    LINALG_CHECK(m != nullptr && pivot != nullptr && info != nullptr, "lu_decomposition: m, pivot, and info must not be null");
    LINALG_CHECK(lda > 0, "lu_decomposition: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    int lwork = 0;
    LINALG_CHECK(!(cusolverDnSgetrf_bufferSize(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS), "cusolverDnSgetrf_bufferSize failed");

    LINALG_CHECK(workspace_size >= lu_decomposition_workspace_size(static_cast<const float*>(nullptr), lda),
        "lu_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = lu_decomposition_plan<float>(arena, lwork, n);

    // cuSOLVER is column-major: transposing the row-major buffer first
    // (device-to-device, transpose_gpu.cpp) makes its column-major
    // reading of the result equal the caller's actual A rather than A^T
    // (see that file's row/column-major identity), so getrf below factors A
    // itself and its pivots are A's own row-interchange sequence — matching
    // lapacke_?getrf's convention exactly. Transposing the packed L\U result
    // back afterward converts cuSOLVER's column-major packed output into
    // the row-major packed layout callers expect.
    detail::transpose_device(lda, lda, m, bufs.transpose_scratch, stream);

    const auto status = cusolverDnSgetrf(handle, n, n, m, n, bufs.workspace, pivot, info);
    LINALG_CHECK(!(status != CUSOLVER_STATUS_SUCCESS), "cusolverDnSgetrf failed");

    detail::transpose_device(lda, lda, m, bufs.transpose_scratch, stream);
}

void lu_decomposition(double* m, linalg_int lda, int* pivot, int* info, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    LINALG_SCOPE("lu_decomposition/gpu<double>");
    LINALG_CHECK(m != nullptr && pivot != nullptr && info != nullptr, "lu_decomposition: m, pivot, and info must not be null");
    LINALG_CHECK(lda > 0, "lu_decomposition: lda must be positive", lda);
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    int lwork = 0;
    LINALG_CHECK(!(cusolverDnDgetrf_bufferSize(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS), "cusolverDnDgetrf_bufferSize failed");

    LINALG_CHECK(workspace_size >= lu_decomposition_workspace_size(static_cast<const double*>(nullptr), lda),
        "lu_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = lu_decomposition_plan<double>(arena, lwork, n);

    // See the float overload above for why the transpose-getrf-transpose
    // sequence is needed.
    detail::transpose_device(lda, lda, m, bufs.transpose_scratch, stream);

    const auto status = cusolverDnDgetrf(handle, n, n, m, n, bufs.workspace, pivot, info);
    LINALG_CHECK(!(status != CUSOLVER_STATUS_SUCCESS), "cusolverDnDgetrf failed");

    detail::transpose_device(lda, lda, m, bufs.transpose_scratch, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
