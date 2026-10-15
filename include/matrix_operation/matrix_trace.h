#pragma once

#include <cstddef>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{
namespace detail
{

// Host-memory-only CPU entry point. Sum of the diagonal of a square,
// row-major n x n matrix. There is no vendor-library "trace" routine to
// call into (same rationale as matrix_determinant in matrix_inversion.cpp),
// so this always runs the same direct reduction regardless of
// linalg::kActiveBackend — unlike every other matrix_operation op, there is
// only ever one body, so only the `native` specialization is ever defined
// and callers always name it explicitly (never kActiveBackend, which may
// be `mkl` or `blas_lapacke`). The template still exists, with the same
// explicit-member-specialization shape as cholesky_backend and friends,
// purely so this op's call site looks like every other op's.
//
// `apply` is declared here but deliberately has NO body: the explicit
// member specialization is defined once in matrix_trace.cpp. Every caller
// — including tests — sees only this declaration and calls
// `matrix_trace_backend<T, linalg_backend::native>::apply(...)`; the actual
// function body is resolved by the linker against that one definition, so
// LINALG_API matters here (it must cross the shared-library boundary to
// reach the test binary) and there is no inline fallback body to silently
// instantiate per call site.
template <typename T, linalg_backend Backend>
struct matrix_trace_backend
{
    LINALG_API static T apply(const T* A, linalg_int n, linalg_int lda);
};

}  // namespace detail
}  // namespace linalg
