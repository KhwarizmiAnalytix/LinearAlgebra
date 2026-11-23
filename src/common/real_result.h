// Adapted from Eigen's MatrixSquareRoot.h and MatrixLogarithm.h, used by
// TensorFlow's MatrixSquareRoot and MatrixLogarithm kernels (v2.20.0).
// Copyright (C) 2011, 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2011 Chen-Pang He <jdh8@ms63.hinet.net>
// This Source Code Form is subject to the Mozilla Public License, v. 2.0.
// A copy is available in LICENSES/MPL-2.0.txt or https://mozilla.org/MPL/2.0/.
// Changes: caller-owned storage, complex Hessenberg/Schur reduction, fixed
// degree-seven quadrature, a common CPU/CUDA/HIP/Metal implementation, and
// reconstruction through existing backend GEMM operations.

#include "include/common/linalg_macros.h"

#if defined(__OBJC__)
#include "src/matrix_operation/gpu/metal/scalars_metal.h"
#else
#include <cmath>
#include <types/complex.h>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include <math/complex_vector.h>
#endif
namespace linalg_schur
{
#if defined(__CUDACC__) || defined(__HIPCC__)
LINALG_INLINE float atan2(float lhs, float rhs)
{
    return ::atan2f(lhs, rhs);
}
LINALG_INLINE double atan2(double lhs, double rhs)
{
    return ::atan2(lhs, rhs);
}
LINALG_INLINE float exp(float value)
{
    return ::expf(value);
}
LINALG_INLINE double exp(double value)
{
    return ::exp(value);
}
LINALG_INLINE float fabs(float value)
{
    return ::fabsf(value);
}
LINALG_INLINE double fabs(double value)
{
    return ::fabs(value);
}
LINALG_INLINE float fmax(float lhs, float rhs)
{
    return ::fmaxf(lhs, rhs);
}
LINALG_INLINE double fmax(double lhs, double rhs)
{
    return ::fmax(lhs, rhs);
}
LINALG_INLINE float fmin(float lhs, float rhs)
{
    return ::fminf(lhs, rhs);
}
LINALG_INLINE double fmin(double lhs, double rhs)
{
    return ::fmin(lhs, rhs);
}
LINALG_INLINE float hypot(float lhs, float rhs)
{
    return ::hypotf(lhs, rhs);
}
LINALG_INLINE double hypot(double lhs, double rhs)
{
    return ::hypot(lhs, rhs);
}
LINALG_INLINE float log(float value)
{
    return ::logf(value);
}
LINALG_INLINE double log(double value)
{
    return ::log(value);
}
LINALG_INLINE float sin(float value)
{
    return ::sinf(value);
}
LINALG_INLINE double sin(double value)
{
    return ::sin(value);
}
LINALG_INLINE float sqrt(float value)
{
    return ::sqrtf(value);
}
LINALG_INLINE double sqrt(double value)
{
    return ::sqrt(value);
}
#else
using std::atan2;
using std::exp;
using std::fabs;
using std::fmax;
using std::fmin;
using std::hypot;
using std::log;
using std::sin;
using std::sqrt;
#endif
#if defined(__CUDACC__) || defined(__HIPCC__)
// MSVC's std::isfinite template is host-only; use the GPU runtime overloads.
using ::isfinite;
#else
using std::isfinite;
#endif
}  // namespace linalg_schur
#endif

// clang-format off
LINALG_SCHUR_CORE(
namespace linalg_schur
{
// Unpack a Schur triangular factor delivered as separate real/imaginary
// planes (schur_decomposition<T, Backend>'s sr/si output) into the
// interleaved complex form the SIMD complex_dot/complex_axpy helpers expect.
template <typename T>
LINALG_INLINE void unpack_triangular(LINALG_ADDRESS const T* real,
    LINALG_ADDRESS const T*                                  imag,
    int                                                       n,
    LINALG_ADDRESS scalars::complex<T>*                          h)
{
    for (int i = 0; i < n * n; ++i)
    {
        h[i] = scalars::complex<T>(real[i], imag[i]);
    }
}

// Split a complex result into real matrices for the existing backend GEMM.
// Invalid preparation poisons the result; device callers need no status readback.
template <typename T>
LINALG_INLINE void pack_result(int n,
    LINALG_ADDRESS const scalars::complex<T>* f,
    LINALG_ADDRESS T*                      fr,
    LINALG_ADDRESS T*                      fi,
    bool                                   success)
{
    for (int i = 0; i < n * n; ++i)
    {
        fr[i] = success ? f[i].real() : T(NAN);
        fi[i] = success ? f[i].imag() : T(NAN);
    }
}

template <typename T>
LINALG_INLINE void combine(LINALG_ADDRESS const T* a,
    LINALG_ADDRESS const T*                        b,
    LINALG_ADDRESS T*                              result,
    int                                            count,
    T                                              sign)
{
    for (int i = 0; i < count; ++i)
    {
        result[i] = a[i] + sign * b[i];
    }
}

// Validate before publishing any output, preserving exact input/output aliasing.
template <typename T>
LINALG_INLINE bool publish(LINALG_ADDRESS const T* real,
    LINALG_ADDRESS const T*                        imag,
    int                                            n,
    LINALG_ADDRESS T*                              result,
    int                                            ldc,
    T                                              epsilon)
{
    T real_norm = T(0);
    T imag_norm = T(0);
    for (int i = 0; i < n * n; ++i)
    {
        if (!isfinite(real[i]) || !isfinite(imag[i]))
        {
            return false;
        }
        real_norm = fmax(real_norm, fabs(real[i]));
        imag_norm = fmax(imag_norm, fabs(imag[i]));
    }
    if (imag_norm > T(512) * T(n) * epsilon * fmax(T(1), real_norm))
    {
        return false;
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            result[i * ldc + j] = real[i * n + j];
        }
    }
    return true;
}
}  // namespace linalg_schur
)
// clang-format on
