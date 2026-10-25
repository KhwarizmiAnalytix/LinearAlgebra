#pragma once

// Adapted from Eigen's matrix function helpers.
// Copyright (C) 2011, 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2011 Chen-Pang He <jdh8@ms63.hinet.net>
// This Source Code Form is subject to the Mozilla Public License, v. 2.0.
// A copy is available in LICENSES/MPL-2.0.txt.

// Implements the Scalars API for runtime-compiled Metal source. C++ kernels
// include Scalars directly; Metal cannot include the dependency's C++ headers.
// clang-format off
LINALG_SCHUR_CORE(
// MSL has no hypot overload. Scale before squaring to preserve range.
template <typename T> LINALG_INLINE T hypot(T a, T b)
{
    a = fabs(a);
    b = fabs(b);
    const T scale = fmax(a, b);
    if (scale == T(0))
    {
        return T(0);
    }
    const T ratio = fmin(a, b) / scale;
    return scale * metal::sqrt(T(1) + ratio * ratio);
}
namespace scalars
{
template <typename T> struct complex
{
    T real_;
    T imag_;
    complex() = default;
    complex(T real, T imag = T(0)) : real_(real), imag_(imag) {}
    T real() const { return real_; }
    T imag() const { return imag_; }
};

template <typename T>
LINALG_INLINE complex<T> operator+(complex<T> a, complex<T> b)
{
    return complex<T>(a.real() + b.real(), a.imag() + b.imag());
}
template <typename T>
LINALG_INLINE complex<T> operator-(complex<T> a, complex<T> b)
{
    return complex<T>(a.real() - b.real(), a.imag() - b.imag());
}
template <typename T>
LINALG_INLINE complex<T> operator*(complex<T> a, complex<T> b)
{
    return complex<T>(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
template <typename T> LINALG_INLINE complex<T> operator*(complex<T> a, T b)
{
    return complex<T>(a.real() * b, a.imag() * b);
}
template <typename T> LINALG_INLINE complex<T> operator/(complex<T> a, T b)
{
    return complex<T>(a.real() / b, a.imag() / b);
}
template <typename T> LINALG_INLINE complex<T> conj(complex<T> a)
{
    return complex<T>(a.real(), -a.imag());
}
template <typename T> LINALG_INLINE T abs(complex<T> a)
{
    return hypot(a.real(), a.imag());
}
template <typename T>
LINALG_INLINE complex<T> operator/(complex<T> a, complex<T> b)
{
    const T scale       = fmax(fabs(b.real()), fabs(b.imag()));
    const T br          = b.real() / scale;
    const T bi          = b.imag() / scale;
    const T denominator = br * br + bi * bi;
    a                   = complex<T>(a.real() / scale, a.imag() / scale);
    return complex<T>(
        (a.real() * br + a.imag() * bi) / denominator, (a.imag() * br - a.real() * bi) / denominator);
}
template <typename T> LINALG_INLINE complex<T> sqrt(complex<T> a)
{
    const T radius = abs(a);
    if (radius == T(0))
    {
        return complex<T>(T(0));
    }
    const T t = metal::sqrt(radius / T(2) + fabs(a.real()) / T(2));
    return a.real() >= T(0) ? complex<T>(t, a.imag() / (T(2) * t))
                          : complex<T>(fabs(a.imag()) / (T(2) * t), copysign(t, a.imag()));
}
}  // namespace scalars
)
// clang-format on

// clang-format off
LINALG_SCHUR_CORE(
namespace scalars
{
template <typename T>
LINALG_INLINE complex<T> complex_dot(LINALG_ADDRESS const complex<T>* x,
    int                                                                           sx,
    LINALG_ADDRESS const complex<T>* y,
    int                                    sy,
    int                                    count,
    bool                                   conjugate_x = false)
{
    auto sum = complex<T>(T(0));
    for (int k = 0; k < count; ++k)
    {
        const int ox = k * sx;
        const int oy = k * sy;
        sum = sum + (conjugate_x ? conj(x[ox]) : x[ox]) * y[oy];
    }
    return sum;
}
template <typename T>
LINALG_INLINE void complex_axpy(LINALG_ADDRESS complex<T>* y,
    int                                                          sy,
    LINALG_ADDRESS const complex<T>* x,
    int                                    sx,
    int                                    count,
    complex<T>                       alpha,
    bool                                   conjugate_x = false)
{
    for (int k = 0; k < count; ++k)
    {
        const int ox = k * sx;
        const int oy = k * sy;
        y[oy] = y[oy] + alpha * (conjugate_x ? conj(x[ox]) : x[ox]);
    }
}
template <typename T>
LINALG_INLINE void complex_scale(LINALG_ADDRESS complex<T>* x, int stride, int count, T scale)
{
    for (int k = 0; k < count; ++k)
    {
        const int ox = k * stride;
        x[ox] = x[ox] * scale;
    }
}
}  // namespace scalars
)
// clang-format on
// clang-format off
LINALG_SCHUR_CORE(
namespace scalars
{
template <typename T>
LINALG_INLINE void complex_rotate(LINALG_ADDRESS complex<T>* x,
    int                                                            sx,
    LINALG_ADDRESS complex<T>* y,
    int                              sy,
    int                              count,
    T                                c,
    complex<T>                 s)
{
    // Read both inputs before either store. Fusing the rotation avoids a
    // scratch copy and four separate read/modify/write passes over the vectors.
    for (int k = 0; k < count; ++k)
    {
        const int ix = k * sx;
        const int iy = k * sy;
        const auto a = x[ix];
        const auto b = y[iy];
        x[ix] = a * c + s * b;
        y[iy] = b * c - conj(s) * a;
    }
}
}  // namespace scalars
)
// clang-format on
