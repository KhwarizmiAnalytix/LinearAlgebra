#pragma once

#include <cstddef>  // for linalg_long

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"  // for LINALG_API
#include "include/common/types.h"

namespace linalg
{

// Host-memory-only CPU entry point. The CPU backend (scalar / blas_lapacke /
// mkl) is a compile-time choice fixed by linalg::kActiveBackend
// (include/common/backend.h) — MKL takes precedence over BLAS+LAPACKE over
// the scalar fallback, with no runtime dispatch or per-call override.
//
// `apply` and `workspace_size` are declared here but deliberately have NO body:
// this class template is never specialized as a whole, only these members are,
// via explicit member specializations defined once in svd_decomposition.cpp
// (for whichever backend LINALG_HAS_MKL / LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
// selects — mkl.h / lapacke.h stay out of this public header). Every caller —
// including tests — sees only this declaration and calls
// `svd_backend<T, kActiveBackend>::apply(...)`; the actual function body is
// resolved by the linker against the one definition in svd_decomposition.cpp,
// so LINALG_API matters here (it must cross the shared-library boundary to
// reach the test binary) and there is no inline fallback body to silently
// instantiate per call site — calling this for a (T, Backend) nobody defined
// is a link error, not silently wrong behavior. For device (CUDA) pointers,
// see linalg::gpu::svd_decomposition in include/matrix_operation_gpu/svd_decomposition_gpu.h.
template <typename T, linalg_backend Backend> struct svd_backend
{
    LINALG_API static void apply(linalg_long rows,
        linalg_long                          columns,
        T*                                   A,
        linalg_long                          lda,
        T*                                   S,
        T*                                   U,
        linalg_long                          ldu,
        T*                                   VT,
        linalg_long                          ldv,
        void*                                workspace,
        size_t                               workspace_size);

    LINALG_API static size_t workspace_size(linalg_long rows, linalg_long columns);
};

}  // namespace linalg
