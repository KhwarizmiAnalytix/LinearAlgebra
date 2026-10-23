#include "include/matrix_operation/linear_solver.h"

#include "include/common/linalg_checks.h"
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"

#include <algorithm>

#include "include/common/linalg_macros.h"

#include "include/common/linalg_profiler.h"
#if LINALG_HAS_VECTORIZATION
#include "src/common/packet.h"
#endif

namespace linalg
{

// Null/positivity checks shared by every linear_solver<T, Backend>::apply
// below, so callers (tests included) get the same validation regardless of
// which backend is active. There is no non-detail linear_solver() wrapper
// to put this in any more — callers use
// linear_solver<T, linalg_backend::NATIVE>::apply directly.
static void check_solver_inputs(const void* m, const void* x, linalg_int lda)
{
    LINALG_CHECK(m != nullptr && x != nullptr, "linear_solver::apply: m and x must not be null");
    LINALG_CHECK(lda > 0, "linear_solver::apply: lda must be positive", lda);
}

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/linear_solver_mkl.cpp and
// blas/linear_solver_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

// m, pivot, and x are __restrict: linear_solver.h documents them as three
// distinct caller-owned buffers (the matrix/factors, the pivot array, and
// the in/out right-hand side), never aliasing each other.
template <typename T> T lu_dot(const T* a, const T* b, linalg_int count)
{
    linalg_int j = 0;
#if LINALG_HAS_VECTORIZATION
    using P = packet<T>;
    // Independent accumulators hide FMA latency across the contiguous row dot product.
    typename P::packet_t sum0, sum1, sum2, sum3;
    P::set(T(0), sum0);
    P::set(T(0), sum1);
    P::set(T(0), sum2);
    P::set(T(0), sum3);
    for (; j + 4 * P::size <= count; j += 4 * P::size)
    {
        typename P::packet_t av, bv;
        P::loadu(a + j, av);
        P::loadu(b + j, bv);
        P::fma(av, bv, sum0, sum0);
        P::loadu(a + j + P::size, av);
        P::loadu(b + j + P::size, bv);
        P::fma(av, bv, sum1, sum1);
        P::loadu(a + j + 2 * P::size, av);
        P::loadu(b + j + 2 * P::size, bv);
        P::fma(av, bv, sum2, sum2);
        P::loadu(a + j + 3 * P::size, av);
        P::loadu(b + j + 3 * P::size, bv);
        P::fma(av, bv, sum3, sum3);
    }
    for (; j + P::size <= count; j += P::size)
    {
        typename P::packet_t av, bv;
        P::loadu(a + j, av);
        P::loadu(b + j, bv);
        P::fma(av, bv, sum0, sum0);
    }
    typename P::packet_t sum01, sum23, packet_sum;
    P::add(sum0, sum1, sum01);
    P::add(sum2, sum3, sum23);
    P::add(sum01, sum23, packet_sum);
    T sum = P::predux(packet_sum);
#else
    T sum = T(0);
#endif
    for (; j < count; ++j)
    {
        sum += a[j] * b[j];
    }
    return sum;
}

template <typename T>
void lu_solve(T* __restrict m,
    LINALG_UNUSED const linalg_int* __restrict pivot,
    linalg_int lda,
    T* __restrict x)
{
#if LINALG_HAS_LU_PIVOTING
    for (int i = 0; i < lda; i++)
    {
        std::swap(x[i], x[pivot[i] - 1]);
    }
#endif
    for (linalg_int i = 0; i < lda; i++)
    {
        const T* const row_i = m + i * lda;
        x[i] -= lu_dot(row_i, x, i);
    }

    for (linalg_int i = lda - 1; i >= 0; i--)
    {
        const T* const row_i = m + i * lda;
        x[i]                 = (x[i] - lu_dot(row_i + i + 1, x + i + 1, lda - i - 1)) / row_i[i];
    }
}

// m and x are __restrict, same reasoning as lu_solve above.
template <typename T> void cholesky_solve(T* __restrict m, linalg_int lda, T* __restrict x)
{
    // Forward substitution reads row i (contiguous): row_i is hoisted once
    // per i, reused for every j.
    for (linalg_int i = 0; i < lda; i++)
    {
        const T* const row_i = m + i * lda;
        auto           sum   = x[i];
        for (linalg_int j = 0; j < i; ++j)
        {
            sum -= row_i[j] * x[j];
        }

        x[i] = sum / row_i[i];
    }

    // Back substitution reads column i (stride lda, since this walks the
    // transposed triangle): col_ptr steps by lda per j instead of
    // recomputing j * lda + i from scratch each time.
    for (linalg_int i = lda - 1; i >= 0; i--)
    {
        auto     sum     = x[i];  //
        const T* col_ptr = m + (i + 1) * lda + i;
        for (linalg_int j = i + 1; j < lda; ++j, col_ptr += lda)
        {
            sum -= (*col_ptr) * x[j];
        }

        x[i] = sum / m[i * lda + i];
    }
}

// m, pivot, and x are __restrict, same reasoning as lu_solve above.
template <typename T>
void linear_solver_scalar_impl(T* __restrict m,
    linalg_int* __restrict pivot,
    linalg_int lda,
    T* __restrict x,
    linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    {
        // Scalar lu_decomposition needs no scratch of its own.
        if (lu_decomposition<T, linalg_backend::NATIVE>::apply(m, lda, pivot, nullptr, 0))
        {
            lu_solve(m, pivot, lda, x);
        }
    }
    break;

    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        lu_solve(m, pivot, lda, x);
        break;

    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    {
        if (cholesky_decomposition<T, linalg_backend::NATIVE>::apply(
                m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR, nullptr, 0, nullptr))
        {
            cholesky_solve(m, lda, x);
        }
    }
    break;

    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        cholesky_solve(m, lda, x);
        break;
    }
}

}  // namespace

template <>
void linear_solver<float, linalg_backend::NATIVE>::apply(float* __restrict m,
    linalg_int* __restrict pivot,
    linalg_int lda,
    float* __restrict x,
    linear_solver_type type,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    void* /*stream*/)
{
    LINALG_SCOPE("linear_solver/native<float>");
    check_solver_inputs(m, x, lda);
    linear_solver_scalar_impl(m, pivot, lda, x, type);
}

template <>
void linear_solver<double, linalg_backend::NATIVE>::apply(double* __restrict m,
    linalg_int* __restrict pivot,
    linalg_int lda,
    double* __restrict x,
    linear_solver_type type,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    void* /*stream*/)
{
    LINALG_SCOPE("linear_solver/native<double>");
    check_solver_inputs(m, x, lda);
    linear_solver_scalar_impl(m, pivot, lda, x, type);
}

template <>
size_t linear_solver<float, linalg_backend::NATIVE>::workspace_size(
    linalg_int /*lda*/, linear_solver_type /*type*/)
{
    return 0;
}

template <>
size_t linear_solver<double, linalg_backend::NATIVE>::workspace_size(
    linalg_int /*lda*/, linear_solver_type /*type*/)
{
    return 0;
}
}  // namespace linalg
