#if LINALG_HAS_MKL

#include "include/matrix_operation/matrix_transpose.h"

#include <include/logging.h>

#include <mkl.h>

// MKL specializations for matrix_transpose.h (the native backend lives in
// matrix_transpose.cpp). The build compiles this file only when LINALG_ENABLE_MKL is
// on; the #if below is kept as a second guard.
namespace linalg
{

// Same checks as matrix_transpose.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_transpose_inputs(const void* m, linalg_long rows, linalg_long columns)
{
    LOGGING_CHECK(m != nullptr, "transpose_backend::apply: m must not be null");
    LOGGING_CHECK(
        rows != 0 && columns != 0, "transpose_backend::apply: rows and columns must be positive");
}

template <>
void transpose_backend<float, linalg_backend::mkl>::apply(
    linalg_long rows, linalg_long columns, float* m, void* /*workspace*/, size_t /*workspace_size*/)
{
    check_transpose_inputs(m, rows, columns);
    mkl_simatcopy('R', 'T', rows, columns, 1.F, m, columns, rows);
}

template <>
void transpose_backend<double, linalg_backend::mkl>::apply(linalg_long rows,
    linalg_long                                                        columns,
    double*                                                            m,
    void* /*workspace*/,
    size_t /*workspace_size*/)
{
    check_transpose_inputs(m, rows, columns);
    mkl_dimatcopy('R', 'T', rows, columns, 1., m, columns, rows);
}

template <>
size_t transpose_backend<float, linalg_backend::mkl>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return 0;
}

template <>
size_t transpose_backend<double, linalg_backend::mkl>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return 0;
}

}  // namespace linalg
#endif  // LINALG_HAS_MKL
