#if LINALG_HAS_MKL

#include "include/matrix_operation/transpose.h"

#include "include/common/linalg_checks.h"

#include <mkl.h>

#include "include/common/linalg_profiler.h"

// MKL specializations for transpose.h (the native backend lives in
// transpose.cpp). The build compiles this file only when LINALG_ENABLE_MKL is
// on; the #if below is kept as a second guard.
namespace linalg
{

// Same checks as transpose.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_transpose_inputs(const void* m, linalg_long rows, linalg_long columns)
{
    LINALG_CHECK(m != nullptr, "transpose::apply: m must not be null");
    LINALG_CHECK(
        rows != 0 && columns != 0, "transpose::apply: rows and columns must be positive");
}

template <>
void transpose<float, linalg_backend::mkl>::apply(
    linalg_long rows, linalg_long columns, float* m, void* /*workspace*/, size_t /*workspace_size*/)
{
    LINALG_SCOPE("transpose/mkl<float>");
    check_transpose_inputs(m, rows, columns);
    mkl_simatcopy('R', 'T', rows, columns, 1.F, m, columns, rows);
}

template <>
void transpose<double, linalg_backend::mkl>::apply(linalg_long rows,
    linalg_long                                                        columns,
    double*                                                            m,
    void* /*workspace*/,
    size_t /*workspace_size*/)
{
    LINALG_SCOPE("transpose/mkl<double>");
    check_transpose_inputs(m, rows, columns);
    mkl_dimatcopy('R', 'T', rows, columns, 1., m, columns, rows);
}

template <>
size_t transpose<float, linalg_backend::mkl>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return 0;
}

template <>
size_t transpose<double, linalg_backend::mkl>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return 0;
}

}  // namespace linalg
#endif  // LINALG_HAS_MKL
