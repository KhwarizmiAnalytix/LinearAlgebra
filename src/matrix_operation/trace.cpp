#include "include/matrix_operation/trace.h"

#include "include/common/linalg_checks.h"

#include "include/common/linalg_profiler.h"

namespace linalg
{

template <typename T> T matrix_trace_impl(const T* A, linalg_int n, linalg_int lda)
{
    LINALG_CHECK(A != nullptr, "trace::apply: A must not be null");
    LINALG_CHECK(n > 0 && lda > 0, "trace::apply: n/lda must be positive");
    T sum = T(0);
    for (linalg_int i = 0; i < n; ++i)
    {
        sum += A[i * lda + i];
    }
    return sum;
}

template <>
float trace<float, linalg_backend::native>::apply(
    const float* A, linalg_int n, linalg_int lda)
{
    LINALG_SCOPE("trace/native<float>");
    return matrix_trace_impl(A, n, lda);
}

template <>
double trace<double, linalg_backend::native>::apply(
    const double* A, linalg_int n, linalg_int lda)
{
    LINALG_SCOPE("trace/native<double>");
    return matrix_trace_impl(A, n, lda);
}

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

template <>
float trace<float, linalg_backend::blas_lapacke>::apply(
    const float* A, linalg_int n, linalg_int lda)
{
    LINALG_SCOPE("trace/blas_lapacke<float>");
    return matrix_trace_impl(A, n, lda);
}

template <>
double trace<double, linalg_backend::blas_lapacke>::apply(
    const double* A, linalg_int n, linalg_int lda)
{
    LINALG_SCOPE("trace/blas_lapacke<double>");
    return matrix_trace_impl(A, n, lda);
}

#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

#if LINALG_HAS_MKL

template <>
float trace<float, linalg_backend::mkl>::apply(
    const float* A, linalg_int n, linalg_int lda)
{
    LINALG_SCOPE("trace/mkl<float>");
    return matrix_trace_impl(A, n, lda);
}

template <>
double trace<double, linalg_backend::mkl>::apply(
    const double* A, linalg_int n, linalg_int lda)
{
    LINALG_SCOPE("trace/mkl<double>");
    return matrix_trace_impl(A, n, lda);
}

#endif  // LINALG_HAS_MKL

}  // namespace linalg
