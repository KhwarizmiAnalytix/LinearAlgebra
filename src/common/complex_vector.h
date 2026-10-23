#pragma once

#include <cstddef>

// Complex vector primitives shared by Schur reduction and the square-root
// recurrence. Native CPU code uses the library's existing SIMD packet kernels.
// CUDA/HIP and embedded Metal keep their portable device implementations.
#if LINALG_HAS_VECTORIZATION && !defined(__CUDACC__) && !defined(__HIPCC__) && !defined(__OBJC__)
#include "src/common/packet.h"

namespace linalg_schur
{
template <typename T>
inline complex_value<T> complex_dot(const complex_value<T>* x,
    int                                                     sx,
    const complex_value<T>*                                 y,
    int                                                     sy,
    int                                                     count,
    bool                                                    conjugate_x = false)
{
    using P = linalg::packet<T>;
    typename P::packet_t rr, ii, zero, sign;
    P::set(T(0), zero);
    rr = zero;
    ii = zero;
    P::set(conjugate_x ? T(-1) : T(1), sign);
    int k = 0;
    for (; k + P::size <= count; k += P::size)
    {
        const std::ptrdiff_t ox = static_cast<std::ptrdiff_t>(k) * sx;
        const std::ptrdiff_t oy = static_cast<std::ptrdiff_t>(k) * sy;
        typename P::packet_t xr, xi, yr, yi, negative_xi, conjugated_xi;
        P::gather(&x[ox].real, 2 * sx, xr);
        P::gather(&x[ox].imag, 2 * sx, xi);
        P::gather(&y[oy].real, 2 * sy, yr);
        P::gather(&y[oy].imag, 2 * sy, yi);
        P::fma(xi, sign, zero, conjugated_xi);
        typename P::packet_t minus_one;
        P::set(T(-1), minus_one);
        P::fma(conjugated_xi, minus_one, zero, negative_xi);
        P::fma(xr, yr, rr, rr);
        P::fma(negative_xi, yi, rr, rr);
        P::fma(xr, yi, ii, ii);
        P::fma(conjugated_xi, yr, ii, ii);
    }
    auto sum = value(P::predux(rr), P::predux(ii));
    for (; k < count; ++k)
    {
        const std::ptrdiff_t ox = static_cast<std::ptrdiff_t>(k) * sx;
        const std::ptrdiff_t oy = static_cast<std::ptrdiff_t>(k) * sy;
        sum = sum + (conjugate_x ? conjugate(x[ox]) : x[ox]) * y[oy];
    }
    return sum;
}

template <typename T>
inline void complex_axpy(complex_value<T>* y,
    int                                    sy,
    const complex_value<T>*                x,
    int                                    sx,
    int                                    count,
    complex_value<T>                       alpha,
    bool                                   conjugate_x = false)
{
    using P = linalg::packet<T>;
    typename P::packet_t ar, ai, negative_ai, imaginary_ar;
    P::set(alpha.real, ar);
    P::set(alpha.imag, ai);
    P::set(conjugate_x ? alpha.imag : -alpha.imag, negative_ai);
    P::set(conjugate_x ? -alpha.real : alpha.real, imaginary_ar);
    int k = 0;
    for (; k + P::size <= count; k += P::size)
    {
        const std::ptrdiff_t ox = static_cast<std::ptrdiff_t>(k) * sx;
        const std::ptrdiff_t oy = static_cast<std::ptrdiff_t>(k) * sy;
        typename P::packet_t xr, xi, yr, yi;
        P::gather(&x[ox].real, 2 * sx, xr);
        P::gather(&x[ox].imag, 2 * sx, xi);
        P::gather(&y[oy].real, 2 * sy, yr);
        P::gather(&y[oy].imag, 2 * sy, yi);
        P::fma(ar, xr, yr, yr);
        P::fma(negative_ai, xi, yr, yr);
        P::fma(ai, xr, yi, yi);
        P::fma(imaginary_ar, xi, yi, yi);
        P::scatter(yr, 2 * sy, &y[oy].real);
        P::scatter(yi, 2 * sy, &y[oy].imag);
    }
    for (; k < count; ++k)
    {
        const std::ptrdiff_t ox = static_cast<std::ptrdiff_t>(k) * sx;
        const std::ptrdiff_t oy = static_cast<std::ptrdiff_t>(k) * sy;
        y[oy] = y[oy] + alpha * (conjugate_x ? conjugate(x[ox]) : x[ox]);
    }
}

template <typename T> inline void complex_scale(complex_value<T>* x, int stride, int count, T scale)
{
    using P = linalg::packet<T>;
    typename P::packet_t factor, zero;
    P::set(scale, factor);
    P::set(T(0), zero);
    int k = 0;
    for (; k + P::size <= count; k += P::size)
    {
        const std::ptrdiff_t ox = static_cast<std::ptrdiff_t>(k) * stride;
        typename P::packet_t real, imag;
        P::gather(&x[ox].real, 2 * stride, real);
        P::gather(&x[ox].imag, 2 * stride, imag);
        P::fma(factor, real, zero, real);
        P::fma(factor, imag, zero, imag);
        P::scatter(real, 2 * stride, &x[ox].real);
        P::scatter(imag, 2 * stride, &x[ox].imag);
    }
    for (; k < count; ++k)
    {
        const std::ptrdiff_t ox = static_cast<std::ptrdiff_t>(k) * stride;
        x[ox] = x[ox] * scale;
    }
}
}  // namespace linalg_schur
#else
// clang-format off
LINALG_SCHUR_CORE(
namespace linalg_schur
{
template <typename T>
LINALG_INLINE complex_value<T> complex_dot(LINALG_ADDRESS const complex_value<T>* x,
    int                                                                           sx,
    LINALG_ADDRESS const complex_value<T>* y,
    int                                    sy,
    int                                    count,
    bool                                   conjugate_x = false)
{
    auto sum = value(T(0));
    for (int k = 0; k < count; ++k)
    {
        const std::ptrdiff_t ox = static_cast<std::ptrdiff_t>(k) * sx;
        const std::ptrdiff_t oy = static_cast<std::ptrdiff_t>(k) * sy;
        sum = sum + (conjugate_x ? conjugate(x[ox]) : x[ox]) * y[oy];
    }
    return sum;
}
template <typename T>
LINALG_INLINE void complex_axpy(LINALG_ADDRESS complex_value<T>* y,
    int                                                          sy,
    LINALG_ADDRESS const complex_value<T>* x,
    int                                    sx,
    int                                    count,
    complex_value<T>                       alpha,
    bool                                   conjugate_x = false)
{
    for (int k = 0; k < count; ++k)
    {
        const std::ptrdiff_t ox = static_cast<std::ptrdiff_t>(k) * sx;
        const std::ptrdiff_t oy = static_cast<std::ptrdiff_t>(k) * sy;
        y[oy] = y[oy] + alpha * (conjugate_x ? conjugate(x[ox]) : x[ox]);
    }
}
template <typename T>
LINALG_INLINE void complex_scale(LINALG_ADDRESS complex_value<T>* x, int stride, int count, T scale)
{
    for (int k = 0; k < count; ++k)
    {
        const std::ptrdiff_t ox = static_cast<std::ptrdiff_t>(k) * stride;
        x[ox] = x[ox] * scale;
    }
}
}  // namespace linalg_schur
)
// clang-format on
#endif

// clang-format off
LINALG_SCHUR_CORE(
namespace linalg_schur
{
template <typename T>
LINALG_INLINE void complex_rotate(LINALG_ADDRESS complex_value<T>* x,
    int                                                            sx,
    LINALG_ADDRESS complex_value<T>* y,
    int                              sy,
    int                              count,
    T                                c,
    complex_value<T>                 s,
    LINALG_ADDRESS complex_value<T>* temporary)
{
    for (int k = 0; k < count; ++k)
    {
        temporary[k] = x[static_cast<std::ptrdiff_t>(k) * sx];
    }
    complex_scale(x, sx, count, c);
    complex_axpy(x, sx, y, sy, count, s);
    complex_scale(y, sy, count, c);
    complex_axpy(y, sy, temporary, 1, count, conjugate(s) * T(-1));
}
}  // namespace linalg_schur
)
// clang-format on
