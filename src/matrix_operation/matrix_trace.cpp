#include "include/matrix_operation/matrix_trace.h"

#include <include/logging.h>

namespace linalg
{
namespace detail
{

template <typename T> T matrix_trace_impl(const T* A, linalg_int n, linalg_int lda)
{
    LOGGING_CHECK(A != nullptr, "matrix_trace_backend::apply: A must not be null");
    LOGGING_CHECK(n > 0 && lda > 0, "matrix_trace_backend::apply: n/lda must be positive");
    T sum = T(0);
    for (linalg_int i = 0; i < n; ++i)
    {
        sum += A[i * lda + i];
    }
    return sum;
}

template <>
float matrix_trace_backend<float, linalg_backend::native>::apply(const float* A, linalg_int n, linalg_int lda)
{
    return matrix_trace_impl(A, n, lda);
}

template <>
double matrix_trace_backend<double, linalg_backend::native>::apply(const double* A, linalg_int n, linalg_int lda)
{
    return matrix_trace_impl(A, n, lda);
}

}  // namespace detail
}  // namespace linalg
