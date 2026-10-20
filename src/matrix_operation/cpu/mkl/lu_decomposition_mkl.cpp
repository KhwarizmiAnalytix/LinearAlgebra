#if LINALG_HAS_MKL

#include "include/matrix_operation/lu_decomposition.h"

#include "include/common/linalg_checks.h"

#include "include/common/backend.h"

#include <mkl_lapacke.h>

#include "include/common/linalg_profiler.h"

// MKL specializations for lu_decomposition.h (the native backend lives in
// lu_decomposition.cpp). The build compiles this file only when LINALG_ENABLE_MKL is
// on; the #if below is kept as a second guard.
namespace linalg
{

// Same checks as lu_decomposition.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_lu_inputs(const void* m, const void* pivot, linalg_int lda)
{
    LINALG_CHECK(
        m != nullptr && pivot != nullptr, "lu_decomposition::apply: m and pivot must not be null");
    LINALG_CHECK(lda > 0, "lu_decomposition::apply: lda must be positive", lda);
}

// No scratch needed: an ILP64 MKL build has MKL_INT == linalg_int, so the
// caller's pivot buffer is passed straight through to getrf.
template <>
bool lu_decomposition<float, linalg_backend::MKL>::apply(
    float* m, linalg_int lda, linalg_int* pivot, void* /*workspace*/, size_t /*workspace_size*/, void* /*stream*/)
{
    LINALG_SCOPE("lu_decomposition/mkl<float>");
    check_lu_inputs(m, pivot, lda);
    return LAPACKE_sgetrf(LAPACK_ROW_MAJOR, lda, lda, m, lda, pivot) == 0;
}

template <>
bool lu_decomposition<double, linalg_backend::MKL>::apply(
    double* m, linalg_int lda, linalg_int* pivot, void* /*workspace*/, size_t /*workspace_size*/, void* /*stream*/)
{
    LINALG_SCOPE("lu_decomposition/mkl<double>");
    check_lu_inputs(m, pivot, lda);
    return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, lda, lda, m, lda, pivot) == 0;
}

template <> size_t lu_decomposition<float, linalg_backend::MKL>::workspace_size(linalg_int /*lda*/)
{
    return 0;
}

template <> size_t lu_decomposition<double, linalg_backend::MKL>::workspace_size(linalg_int /*lda*/)
{
    return 0;
}

}  // namespace linalg
#endif  // LINALG_HAS_MKL
