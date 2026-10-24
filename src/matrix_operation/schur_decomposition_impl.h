// Adapted from Eigen's MatrixSquareRoot.h and MatrixLogarithm.h, used by
// TensorFlow's MatrixSquareRoot and MatrixLogarithm kernels (v2.20.0).
// Copyright (C) 2011, 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2011 Chen-Pang He <jdh8@ms63.hinet.net>
// This Source Code Form is subject to the Mozilla Public License, v. 2.0.
// A copy is available in LICENSES/MPL-2.0.txt or https://mozilla.org/MPL/2.0/.
// Changes: caller-owned storage, complex Hessenberg/Schur reduction, fixed
// degree-seven quadrature, a common CPU/CUDA/HIP/Metal implementation, and
// reconstruction through existing backend GEMM operations.

// Operation-owned portable kernel, compiled for native, CUDA/HIP, and Metal.
#include "src/common/complex_math.h"
#include "src/common/complex_vector.h"

// clang-format off
LINALG_SCHUR_CORE(
namespace linalg_schur
{
// Complex Schur form avoids the real 2x2 block cases and also handles
// nonsymmetric matrices, conjugate eigenvalues, and defective Jordan blocks.
template <typename T, bool ComputeVectors = true>
LINALG_INLINE bool schur_iteration(LINALG_ADDRESS complex_value<T>* h,
    LINALG_ADDRESS complex_value<T>* q,
    LINALG_ADDRESS complex_value<T>* v,
    int                              n,
    T                                epsilon)
{
    if (ComputeVectors)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                q[i * n + j] = value(i == j ? T(1) : T(0));
            }
        }
    }
    // Householder reduction to Hessenberg form; accumulate the unitary factor.
    for (int k = 0; k + 2 < n; ++k)
    {
        T norm = T(0);
        for (int i = k + 1; i < n; ++i)
        {
            v[i] = h[i * n + k];
            norm = real_hypot(norm, magnitude(v[i]));
        }
        if (norm == T(0))
        {
            continue;
        }
        const T    first = magnitude(v[k + 1]);
        const auto phase = first == T(0) ? value(T(1)) : v[k + 1] / first;
        v[k + 1]         = v[k + 1] + phase * norm;
        T vnorm          = T(0);
        for (int i = k + 1; i < n; ++i)
        {
            vnorm = real_hypot(vnorm, magnitude(v[i]));
        }
        for (int i = k + 1; i < n; ++i)
        {
            v[i] = v[i] / vnorm;
        }
        constexpr int column_tile_size = 32;
        complex_value<T> projection[column_tile_size];
        for (int j = k; j < n; j += column_tile_size)
        {
            const int tile_width = n - j < column_tile_size ? n - j : column_tile_size;
            for (int jj = 0; jj < tile_width; ++jj)
            {
                projection[jj] = value(T(0));
            }
            for (int i = k + 1; i < n; ++i)
            {
                const auto conjugated_v = conjugate(v[i]);
                const auto* h_i = h + i * n + j;
                for (int jj = 0; jj < tile_width; ++jj)
                {
                    projection[jj] = projection[jj] + conjugated_v * h_i[jj];
                }
            }
            for (int jj = 0; jj < tile_width; ++jj)
            {
                projection[jj] = projection[jj] * T(-2);
            }
            for (int i = k + 1; i < n; ++i)
            {
                auto* h_i = h + i * n + j;
                for (int jj = 0; jj < tile_width; ++jj)
                {
                    h_i[jj] = h_i[jj] + v[i] * projection[jj];
                }
            }
        }
        for (int i = 0; i < n; ++i)
        {
            const auto hd = complex_dot(h + i * n + k + 1, 1, v + k + 1, 1, n - k - 1);
            complex_axpy(h + i * n + k + 1, 1, v + k + 1, 1, n - k - 1, hd * T(-2), true);
            if (ComputeVectors)
            {
                const auto qd = complex_dot(q + i * n + k + 1, 1, v + k + 1, 1, n - k - 1);
                complex_axpy(q + i * n + k + 1, 1, v + k + 1, 1, n - k - 1, qd * T(-2), true);
            }
        }
        for (int i = k + 2; i < n; ++i)
        {
            h[i * n + k] = value(T(0));
        }
    }

    int end        = n - 1;
    int iterations = 0;
    while (end > 0)
    {
        int begin = end;
        for (; begin > 0; --begin)
        {
            const T scale =
                magnitude(h[(begin - 1) * n + begin - 1]) + magnitude(h[begin * n + begin]);
            if (magnitude(h[begin * n + begin - 1]) <= T(8) * epsilon * scale)
            {
                h[begin * n + begin - 1] = value(T(0));
                break;
            }
        }
        if (begin == end)
        {
            --end;
            iterations = 0;
            continue;
        }
        if (++iterations > 2000 * n)
        {
            return false;
        }
        const auto a     = h[(end - 1) * n + end - 1];
        const auto d     = h[end * n + end];
        const auto delta = (a - d) * T(0.5);
        const auto root =
            square_root(delta * delta + h[(end - 1) * n + end] * h[end * n + end - 1]);
        const auto mu1   = (a + d) * T(0.5) + root;
        const auto mu2   = (a + d) * T(0.5) - root;
        auto       shift = magnitude(mu1 - d) < magnitude(mu2 - d) ? mu1 : mu2;
        if (iterations % 20 == 0)
        {
            const T bump = magnitude(h[end * n + end - 1]);
            shift        = d + value(bump, bump * T(0.25));
        }
        auto x = h[begin * n + begin] - shift;
        auto y = h[(begin + 1) * n + begin];
        for (int k = begin; k < end; ++k)
        {
            const T    ax           = magnitude(x);
            const T    ay           = magnitude(y);
            const T    norm         = real_hypot(ax, ay);
            const T    c            = norm == T(0) ? T(1) : ax / norm;
            const auto phase        = ax == T(0) ? value(T(1)) : x / ax;
            const auto s            = norm == T(0) ? value(T(0)) : (phase * conjugate(y)) / norm;
            const int  first_column = k > begin ? k - 1 : k;
            complex_rotate(h + k * n + first_column,
                1,
                h + (k + 1) * n + first_column,
                1,
                n - first_column,
                c,
                s,
                v);
            complex_rotate(h + k, n, h + k + 1, n, n, c, conjugate(s), v);
            if (ComputeVectors)
            {
                complex_rotate(q + k, n, q + k + 1, n, n, c, conjugate(s), v);
            }
            if (k > begin)
            {
                h[(k + 1) * n + k - 1] = value(T(0));
            }
            if (k + 1 < end)
            {
                x = h[(k + 1) * n + k];
                y = h[(k + 2) * n + k];
            }
        }
    }
    return true;
}


template <typename T>
LINALG_INLINE bool decompose_real(LINALG_ADDRESS const T* a, int n, int lda,
    LINALG_ADDRESS T* sr, LINALG_ADDRESS T* si, int lds,
    LINALG_ADDRESS T* qr, LINALG_ADDRESS T* qi, int ldq,
    LINALG_ADDRESS T* workspace, T epsilon)
{
    T scale = T(0);
    bool diagonal = true;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (!isfinite(a[i * lda + j])) return false;
            scale = fmax(scale, fabs(a[i * lda + j]));
            if (i != j && a[i * lda + j] != T(0)) diagonal = false;
        }
    }
    if (diagonal)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                sr[i * lds + j] = i == j ? a[i * lda + j] : T(0);
                si[i * lds + j] = T(0);
                qr[i * ldq + j] = i == j ? T(1) : T(0);
                qi[i * ldq + j] = T(0);
            }
        }
        return true;
    }
    auto h = reinterpret_cast<LINALG_ADDRESS complex_value<T>*>(workspace);
    auto q = h + n * n;
    auto vector = q + n * n;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            h[i * n + j] = value(a[i * lda + j] / scale);
        }
    }
    if (!schur_iteration(h, q, vector, n, epsilon)) return false;
    // Validate both factors while publishing output, in one pass. Tiny
    // subdiagonals left by finite-precision QR are explicitly removed from
    // the triangular factor. Output may hold unfinished data on failure;
    // callers never read sr/si/qr/qi after a false return.
    bool finite = true;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            const auto t = i <= j ? h[i * n + j] * scale : value(T(0));
            finite = finite && isfinite(t.real) && isfinite(t.imag) &&
                      isfinite(q[i * n + j].real) && isfinite(q[i * n + j].imag);
            sr[i * lds + j] = t.real;
            si[i * lds + j] = t.imag;
            qr[i * ldq + j] = q[i * n + j].real;
            qi[i * ldq + j] = q[i * n + j].imag;
        }
    }
    return finite;
}

template <typename T>
LINALG_INLINE void poison_schur(int n, LINALG_ADDRESS T* sr, LINALG_ADDRESS T* si, int lds,
    LINALG_ADDRESS T* qr, LINALG_ADDRESS T* qi, int ldq)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            sr[i * lds + j] = si[i * lds + j] = T(NAN);
            qr[i * ldq + j] = qi[i * ldq + j] = T(NAN);
        }
    }
}
}  // namespace linalg_schur
)
// clang-format on
