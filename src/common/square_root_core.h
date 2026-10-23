// Adapted from Eigen's MatrixSquareRoot.h, used by TensorFlow's
// MatrixSquareRoot kernel (v2.20.0).
// Copyright (C) 2011, 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2011 Chen-Pang He <jdh8@ms63.hinet.net>
// This Source Code Form is subject to the Mozilla Public License, v. 2.0.
// A copy is available in LICENSES/MPL-2.0.txt or https://mozilla.org/MPL/2.0/.
// Changes: caller-owned storage, a common CPU/CUDA/HIP/Metal implementation,
// and reuse of the dedicated schur_decomposition kernel for the complex
// Schur reduction (see src/matrix_operation/schur_decomposition_impl.h)
// instead of an embedded copy of it.

// No standard includes: this body is also stringified into Metal source.
#include "src/common/complex_math.h"
#include "src/common/complex_vector.h"
#include "src/common/real_result.h"

// clang-format off
LINALG_SQUARE_ROOT_CORE(
namespace linalg_schur
{
// Eigen's triangular square-root recurrence, evaluated in place.
template <typename T> LINALG_INLINE bool triangular_sqrt(LINALG_ADDRESS complex_value<T>* h, int n)
{
    for (int i = 0; i < n; ++i)
    {
        h[i * n + i] = square_root(h[i * n + i]);
    }
    for (int distance = 1; distance < n; ++distance)
    {
        for (int i = 0; i + distance < n; ++i)
        {
            const int j         = i + distance;
            auto      numerator = h[i * n + j];
            numerator =
                numerator - complex_dot(h + i * n + i + 1, 1, h + (i + 1) * n + j, n, j - i - 1);
            const auto denominator = h[i * n + i] + h[j * n + j];
            if (magnitude(denominator) == T(0))
            {
                if (magnitude(numerator) != T(0))
                {
                    return false;
                }
                h[i * n + j] = value(T(0));
            }
            else
            {
                h[i * n + j] = numerator / denominator;
            }
        }
    }
    return true;
}

// h is the complex Schur triangular factor already produced by the dedicated
// schur_decomposition<T, Backend> kernel (unpacked from its sr/si planes by
// unpack_triangular in src/common/real_result.h). Rejects a real negative
// eigenvalue, for which the principal square root is not real, then
// evaluates the triangular recurrence over the upper triangle in place.
// Failure is communicated without host access, so GPU callers remain async.
template <typename T>
LINALG_INLINE bool square_root_triangular(LINALG_ADDRESS complex_value<T>* h, int n, T epsilon)
{
    for (int i = 0; i < n; ++i)
    {
        const auto eigenvalue = h[i * n + i];
        if (eigenvalue.real < T(0) &&
            fabs(eigenvalue.imag) <= T(64) * epsilon * fabs(eigenvalue.real))
        {
            return false;
        }
    }
    return triangular_sqrt(h, n);
}

// The sequential remainder of square_root_schur once h already holds the
// unpacked triangular factor: the triangular square-root recurrence itself
// (inherently sequential across anti-diagonals; see triangular_sqrt above).
// GPU backends run unpack_triangular_element and pack_result_element as
// their own parallel dispatches (one thread per element) and call this only
// for the genuinely sequential middle step, writing the outcome to
// success_flag (a single caller-owned T) so the later parallel pack_result
// dispatch can read it without a host round-trip.
template <typename T>
LINALG_INLINE void square_root_triangular_step(LINALG_ADDRESS T* h_buffer,
    int                                                          n,
    T                                                            epsilon,
    LINALG_ADDRESS T*                                            success_flag)
{
    auto h        = reinterpret_cast<LINALG_ADDRESS complex_value<T>*>(h_buffer);
    *success_flag = square_root_triangular(h, n, epsilon) ? T(1) : T(0);
}

// Combines unpack_triangular, square_root_triangular, and pack_result behind
// one call that only ever names plain real/imaginary planes, so CUDA/HIP
// kernel wrappers and the stringified Metal kernel never need to spell out
// the interleaved complex_value<T> scratch type themselves. h_buffer is
// n*n complex_value<T> scratch (2*n*n reals), caller-owned. This sequential
// form (unpack and pack_result folded into the same single-thread call) is
// what the CPU path uses; GPU backends call unpack_triangular_element,
// square_root_triangular_step and pack_result_element as three separate
// dispatches instead (see above).
template <typename T>
LINALG_INLINE bool square_root_schur(LINALG_ADDRESS const T* sr,
    LINALG_ADDRESS const T*                                  si,
    int                                                       n,
    LINALG_ADDRESS T*                                         h_buffer,
    LINALG_ADDRESS T*                                         fr,
    LINALG_ADDRESS T*                                         fi,
    T                                                         epsilon)
{
    auto       h       = reinterpret_cast<LINALG_ADDRESS complex_value<T>*>(h_buffer);
    unpack_triangular(sr, si, n, h);
    const bool success = square_root_triangular(h, n, epsilon);
    pack_result(n, h, fr, fi, success);
    return success;
}
}  // namespace linalg_schur
)
// clang-format on
