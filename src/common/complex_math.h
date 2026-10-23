#pragma once

// Adapted from Eigen's MatrixSquareRoot.h and MatrixLogarithm.h, used by
// TensorFlow's MatrixSquareRoot and MatrixLogarithm kernels (v2.20.0).
// Copyright (C) 2011, 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2011 Chen-Pang He <jdh8@ms63.hinet.net>
// This Source Code Form is subject to the Mozilla Public License, v. 2.0.
// A copy is available in LICENSES/MPL-2.0.txt or https://mozilla.org/MPL/2.0/.
// Changes: caller-owned storage, complex Hessenberg/Schur reduction, fixed
// degree-seven quadrature, a common CPU/CUDA/HIP/Metal implementation, and
// reconstruction through existing backend GEMM operations.

// Only portability definitions: this body is also stringified into Metal source.
#include "include/common/linalg_macros.h"

// clang-format off
LINALG_SCHUR_CORE(
namespace linalg_schur
{
template <typename T> LINALG_INLINE T real_hypot(T a, T b)
{
    a             = fabs(a);
    b             = fabs(b);
    const T scale = fmax(a, b);
    if (scale == T(0))
    {
        return T(0);
    }
    const T ratio = fmin(a, b) / scale;
    return scale * sqrt(T(1) + ratio * ratio);
}

template <typename T> struct complex_value
{
    T real;
    T imag;
};

template <typename T> LINALG_INLINE complex_value<T> value(T real, T imag = T(0))
{
    return {real, imag};
}
template <typename T>
LINALG_INLINE complex_value<T> operator+(complex_value<T> a, complex_value<T> b)
{
    return value(a.real + b.real, a.imag + b.imag);
}
template <typename T>
LINALG_INLINE complex_value<T> operator-(complex_value<T> a, complex_value<T> b)
{
    return value(a.real - b.real, a.imag - b.imag);
}
template <typename T>
LINALG_INLINE complex_value<T> operator*(complex_value<T> a, complex_value<T> b)
{
    return value(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real);
}
template <typename T> LINALG_INLINE complex_value<T> operator*(complex_value<T> a, T b)
{
    return value(a.real * b, a.imag * b);
}
template <typename T> LINALG_INLINE complex_value<T> operator/(complex_value<T> a, T b)
{
    return value(a.real / b, a.imag / b);
}
template <typename T> LINALG_INLINE complex_value<T> conjugate(complex_value<T> a)
{
    return value(a.real, -a.imag);
}
template <typename T> LINALG_INLINE T magnitude(complex_value<T> a)
{
    return real_hypot(a.real, a.imag);
}
template <typename T>
LINALG_INLINE complex_value<T> operator/(complex_value<T> a, complex_value<T> b)
{
    const T scale       = fmax(fabs(b.real), fabs(b.imag));
    const T br          = b.real / scale;
    const T bi          = b.imag / scale;
    const T denominator = br * br + bi * bi;
    a                   = value(a.real / scale, a.imag / scale);
    return value(
        (a.real * br + a.imag * bi) / denominator, (a.imag * br - a.real * bi) / denominator);
}
template <typename T> LINALG_INLINE complex_value<T> square_root(complex_value<T> a)
{
    const T radius = magnitude(a);
    if (radius == T(0))
    {
        return value(T(0));
    }
    const T t = sqrt(radius / T(2) + fabs(a.real) / T(2));
    return a.real >= T(0) ? value(t, a.imag / (T(2) * t))
                          : value(fabs(a.imag) / (T(2) * t), copysign(t, a.imag));
}
}  // namespace linalg_schur
)
// clang-format on
