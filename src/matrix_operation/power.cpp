#include "include/matrix_operation/power.h"

#include <algorithm>
#include <cstdlib>

#include "include/common/linalg_checks.h"
#include "include/common/workspace_arena.h"
#include "include/matrix_operation/inversion.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/multiplication.h"
#if LINALG_HAS_VECTORIZATION
#include "src/common/packet.h"
#endif

namespace linalg
{
namespace
{

// dst[0..count) = src[0..count). Uses linalg::packet<T> (src/common/packet.h), the same
// Eigen-derived SIMD abstraction multiplication.cpp's native GEMM kernel is built on, instead
// of a plain scalar loop.
template <typename T> void copy_n(const T* src, T* dst, linalg_int count)
{
#if LINALG_HAS_VECTORIZATION
    using P                    = packet<T>;
    constexpr linalg_int width = P::size;
    linalg_int           i     = 0;
    for (; i + width <= count; i += width)
    {
        P::copy(src + i, dst + i);
    }
    for (; i < count; ++i)
    {
        dst[i] = src[i];
    }
#else
    std::copy(src, src + count, dst);
#endif
}

// dst[0..count) = 0.
template <typename T> void zero_n(T* dst, linalg_int count)
{
#if LINALG_HAS_VECTORIZATION
    using P                    = packet<T>;
    constexpr linalg_int width = P::size;
    typename P::packet_t z;
    P::set(T(0), z);
    linalg_int i = 0;
    for (; i + width <= count; i += width)
    {
        P::storeu(z, dst + i);
    }
    for (; i < count; ++i)
    {
        dst[i] = T(0);
    }
#else
    std::fill(dst, dst + count, T(0));
#endif
}

// Copies the n x n leading block of a row-major matrix with row stride `stride_in` into a
// tightly packed n x n row-major buffer. Each row is contiguous in both layouts, so every row
// is copied with the vectorized copy_n() above.
template <typename T> void copy_in(const T* src, linalg_int stride_in, T* dst, linalg_int n)
{
    for (linalg_int i = 0; i < n; ++i)
    {
        copy_n(src + i * stride_in, dst + i * n, n);
    }
}

// Copies a tightly packed n x n row-major buffer into the n x n leading block of a row-major
// matrix with row stride `stride_out`.
template <typename T> void copy_out(const T* src, linalg_int n, T* dst, linalg_int stride_out)
{
    for (linalg_int i = 0; i < n; ++i)
    {
        copy_n(src + i * n, dst + i * stride_out, n);
    }
}

// Writes the n x n identity into a row-major matrix with row stride `stride_out`: vectorized
// zero-fill per row (the dominant O(n^2) cost) followed by a single scalar write for that
// row's diagonal entry.
template <typename T> void set_identity(T* dst, linalg_int n, linalg_int stride_out)
{
    for (linalg_int i = 0; i < n; ++i)
    {
        T* row = dst + i * stride_out;
        zero_n(row, n);
        row[i] = T(1);
    }
}

template <typename T> struct power_layout
{
    T*             a;
    T*             z;
    T*             z_tmp;
    T*             result;
    T*             result_tmp;
    linalg_int*    pivot;
    unsigned char* invert_ws;
    size_t         invert_bytes;
    unsigned char* gemm_ws;
    size_t         gemm_bytes;
};

template <typename T> power_layout<T> power_plan(detail::workspace_arena& arena, linalg_int n)
{
    const size_t    nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    power_layout<T> p;
    p.a            = arena.next<T>(nn);
    p.z            = arena.next<T>(nn);
    p.z_tmp        = arena.next<T>(nn);
    p.result       = arena.next<T>(nn);
    p.result_tmp   = arena.next<T>(nn);
    p.pivot        = arena.next<linalg_int>(static_cast<size_t>(n));
    p.invert_bytes = linalg::inversion<T, linalg::linalg_backend::native>::workspace_size(
        n, linalg::linear_solver_type::LU_LINEAR_SOLVER);
    p.invert_ws  = arena.next<unsigned char>(p.invert_bytes);
    p.gemm_bytes = linalg::multiplication<T, linalg::kMatmulActiveBackend>::workspace_size(n, n, n);
    p.gemm_ws    = arena.next<unsigned char>(p.gemm_bytes);
    return p;
}

template <typename T> size_t matrix_power_workspace_size_impl(linalg_int n)
{
    detail::workspace_arena arena;
    power_plan<T>(arena, n);
    return arena.total_size();
}

template <typename T>
void matrix_power_impl(const T* A_in,
    linalg_int                  n,
    linalg_int                  lda,
    linalg_int                  exponent,
    T*                          result,
    linalg_int                  ldc,
    void*                       workspace,
    size_t                      workspace_size)
{
    LINALG_CHECK(workspace_size >= matrix_power_workspace_size_impl<T>(n),
        "power: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = power_plan<T>(arena, n);

    if (exponent == 0)
    {
        set_identity(result, n, ldc);
        return;
    }

    copy_in(A_in, lda, bufs.a, n);

    if (exponent == 1)
    {
        copy_out(bufs.a, n, result, ldc);
        return;
    }
    if (exponent == -1)
    {
        linalg::inversion<T, linalg::linalg_backend::native>::apply(bufs.a,
            bufs.pivot,
            n,
            linalg::linear_solver_type::LU_LINEAR_SOLVER,
            bufs.invert_ws,
            bufs.invert_bytes);
        copy_out(bufs.a, n, result, ldc);
        return;
    }

    T* a = bufs.a;
    if (exponent < 0)
    {
        linalg::inversion<T, linalg::linalg_backend::native>::apply(a,
            bufs.pivot,
            n,
            linalg::linear_solver_type::LU_LINEAR_SOLVER,
            bufs.invert_ws,
            bufs.invert_bytes);
    }
    linalg_int e = std::abs(exponent);

    // Binary (square-and-multiply) exponentiation, scanning bits from least to most
    // significant: raises `a` to successive powers of two by repeated squaring, and
    // accumulates the factors whose bit is set. Same decomposition PyTorch's
    // linalg_matrix_power uses to reach O(log n) matrix multiplications.
    bool z_defined      = false;
    bool result_defined = false;
    T*   z              = nullptr;
    T*   r              = nullptr;

    while (e > 0)
    {
        const linalg_int bit = e % 2;
        e /= 2;

        if (!z_defined)
        {
            z         = a;
            z_defined = true;
        }
        else
        {
            T* z_next = (z == bufs.z) ? bufs.z_tmp : bufs.z;
            linalg::multiplication<T, linalg::kMatmulActiveBackend>::apply(
                false, false, n, n, n, z, n, z, n, z_next, n, bufs.gemm_ws, bufs.gemm_bytes);
            z = z_next;
        }

        if (bit == 1)
        {
            if (!result_defined)
            {
                copy_n(z, bufs.result, n * n);
                r              = bufs.result;
                result_defined = true;
            }
            else
            {
                T* r_next = (r == bufs.result) ? bufs.result_tmp : bufs.result;
                linalg::multiplication<T, linalg::kMatmulActiveBackend>::apply(
                    false, false, n, n, n, r, n, z, n, r_next, n, bufs.gemm_ws, bufs.gemm_bytes);
                r = r_next;
            }
        }
    }

    copy_out(r, n, result, ldc);
}

}  // namespace

template <>
void power<float, linalg_backend::native>::apply(const float* A,
    linalg_int                                                n,
    linalg_int                                                lda,
    linalg_int                                                exponent,
    float*                                                    result,
    linalg_int                                                ldc,
    void*                                                     workspace,
    size_t                                                    workspace_size)
{
    LINALG_CHECK(A != nullptr && result != nullptr, "power::apply: A and result must not be null");
    LINALG_CHECK(n > 0 && lda > 0 && ldc > 0, "power::apply: n/lda/ldc must be positive");
    matrix_power_impl(A, n, lda, exponent, result, ldc, workspace, workspace_size);
}

template <>
void power<double, linalg_backend::native>::apply(const double* A,
    linalg_int                                                  n,
    linalg_int                                                  lda,
    linalg_int                                                  exponent,
    double*                                                     result,
    linalg_int                                                  ldc,
    void*                                                       workspace,
    size_t                                                      workspace_size)
{
    LINALG_CHECK(A != nullptr && result != nullptr, "power::apply: A and result must not be null");
    LINALG_CHECK(n > 0 && lda > 0 && ldc > 0, "power::apply: n/lda/ldc must be positive");
    matrix_power_impl(A, n, lda, exponent, result, ldc, workspace, workspace_size);
}

template <> size_t power<float, linalg_backend::native>::workspace_size(linalg_int n)
{
    return matrix_power_workspace_size_impl<float>(n);
}

template <> size_t power<double, linalg_backend::native>::workspace_size(linalg_int n)
{
    return matrix_power_workspace_size_impl<double>(n);
}

}  // namespace linalg
