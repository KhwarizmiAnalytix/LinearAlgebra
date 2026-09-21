#include "include/matrix_operation/matrix_trace.h"

#include "include/util/exception.h"

namespace linalg
{
namespace detail
{

template <typename T> T matrix_trace_impl(const T* A, linalg_int n, linalg_int lda)
{
    T sum = T(0);
    for (linalg_int i = 0; i < n; ++i)
    {
        sum += A[i * lda + i];
    }
    return sum;
}

}  // namespace detail

//-----------------------------------------------------------------------------
float matrix_trace(const float* A, linalg_int n, linalg_int lda)
{
    if (A == nullptr)
    {
        LINALG_THROW("matrix_trace: A must not be null");
    }
    if (n <= 0 || lda <= 0)
    {
        LINALG_THROW("matrix_trace: n/lda must be positive");
    }
    return detail::matrix_trace_impl(A, n, lda);
}

//-----------------------------------------------------------------------------
double matrix_trace(const double* A, linalg_int n, linalg_int lda)
{
    if (A == nullptr)
    {
        LINALG_THROW("matrix_trace: A must not be null");
    }
    if (n <= 0 || lda <= 0)
    {
        LINALG_THROW("matrix_trace: n/lda must be positive");
    }
    return detail::matrix_trace_impl(A, n, lda);
}

}  // namespace linalg
