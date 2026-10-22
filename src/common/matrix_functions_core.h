// Adapted from Eigen's MatrixSquareRoot.h and MatrixLogarithm.h, used by
// TensorFlow's MatrixSquareRoot and MatrixLogarithm kernels (v2.20.0).
// Copyright (C) 2011, 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2011 Chen-Pang He <jdh8@ms63.hinet.net>
// This Source Code Form is subject to the Mozilla Public License, v. 2.0.
// A copy is available in LICENSES/MPL-2.0.txt or https://mozilla.org/MPL/2.0/.
// Changes: caller-owned storage, complex Hessenberg/Schur reduction, fixed
// degree-seven quadrature, and a common CPU/CUDA/HIP/Metal implementation.

// Intentionally no includes: the same body is compiled as C++/CUDA/HIP and
// stringified into Metal source. MF_ADDRESS is empty on C++ and device on MSL.
#ifndef LINALG_MATRIX_FUNCTIONS_CORE
#define LINALG_MATRIX_FUNCTIONS_CORE(...) __VA_ARGS__
#endif

LINALG_MATRIX_FUNCTIONS_CORE(namespace linalg_matrix_functions {
    template <typename T> MF_INLINE T real_hypot(T a, T b)
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

    template <typename T> MF_INLINE complex_value<T> value(T real, T imag = T(0))
    {
        return {real, imag};
    }
    template <typename T>
    MF_INLINE complex_value<T> operator+(complex_value<T> a, complex_value<T> b)
    {
        return value(a.real + b.real, a.imag + b.imag);
    }
    template <typename T>
    MF_INLINE complex_value<T> operator-(complex_value<T> a, complex_value<T> b)
    {
        return value(a.real - b.real, a.imag - b.imag);
    }
    template <typename T>
    MF_INLINE complex_value<T> operator*(complex_value<T> a, complex_value<T> b)
    {
        return value(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real);
    }
    template <typename T> MF_INLINE complex_value<T> operator*(complex_value<T> a, T b)
    {
        return value(a.real * b, a.imag * b);
    }
    template <typename T> MF_INLINE complex_value<T> operator/(complex_value<T> a, T b)
    {
        return value(a.real / b, a.imag / b);
    }
    template <typename T> MF_INLINE complex_value<T> conjugate(complex_value<T> a)
    {
        return value(a.real, -a.imag);
    }
    template <typename T> MF_INLINE T magnitude(complex_value<T> a)
    {
        return real_hypot(a.real, a.imag);
    }
    template <typename T>
    MF_INLINE complex_value<T> operator/(complex_value<T> a, complex_value<T> b)
    {
        const T scale       = fmax(fabs(b.real), fabs(b.imag));
        const T br          = b.real / scale;
        const T bi          = b.imag / scale;
        const T denominator = br * br + bi * bi;
        a                   = value(a.real / scale, a.imag / scale);
        return value(
            (a.real * br + a.imag * bi) / denominator, (a.imag * br - a.real * bi) / denominator);
    }
    template <typename T> MF_INLINE complex_value<T> square_root(complex_value<T> a)
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
    template <typename T> MF_INLINE complex_value<T> logarithm(complex_value<T> a)
    {
        return value(log(magnitude(a)), atan2(a.imag, a.real));
    }
    template <typename T> MF_INLINE complex_value<T> exponential_minus_one(complex_value<T> a)
    {
        const T half_sine = sin(a.imag / T(2));
        // MSL has no expm1 intrinsic. A short Taylor series retains tiny diagonal
        // logarithms without subtracting two nearly equal floating-point numbers.
        T real_expm1 = exp(a.real) - T(1);
        if (fabs(a.real) < T(0.5))
        {
            T term     = a.real;
            real_expm1 = term;
            for (int k = 2; k <= 24; ++k)
            {
                term *= a.real / T(k);
                real_expm1 += term;
            }
        }
        return value(
            real_expm1 * cos(a.imag) - T(2) * half_sine * half_sine, exp(a.real) * sin(a.imag));
    }

    // Complex Schur form avoids the real 2x2 block cases and also handles
    // nonsymmetric matrices, conjugate eigenvalues, and defective Jordan blocks.
    template <typename T>
    MF_INLINE bool schur(MF_ADDRESS complex_value<T> * h,
        MF_ADDRESS                  complex_value<T> * q,
        MF_ADDRESS                  complex_value<T> * v,
        int                         n,
        T                           epsilon)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                q[i * n + j] = value(i == j ? T(1) : T(0));
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
            for (int j = k; j < n; ++j)
            {
                auto dot = value(T(0));
                for (int i = k + 1; i < n; ++i)
                {
                    dot = dot + conjugate(v[i]) * h[i * n + j];
                }
                for (int i = k + 1; i < n; ++i)
                {
                    h[i * n + j] = h[i * n + j] - v[i] * dot * T(2);
                }
            }
            for (int i = 0; i < n; ++i)
            {
                auto hd = value(T(0));
                auto qd = value(T(0));
                for (int j = k + 1; j < n; ++j)
                {
                    hd = hd + h[i * n + j] * v[j];
                    qd = qd + q[i * n + j] * v[j];
                }
                for (int j = k + 1; j < n; ++j)
                {
                    h[i * n + j] = h[i * n + j] - hd * conjugate(v[j]) * T(2);
                    q[i * n + j] = q[i * n + j] - qd * conjugate(v[j]) * T(2);
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
                const T    ax    = magnitude(x);
                const T    ay    = magnitude(y);
                const T    norm  = real_hypot(ax, ay);
                const T    c     = norm == T(0) ? T(1) : ax / norm;
                const auto phase = ax == T(0) ? value(T(1)) : x / ax;
                const auto s     = norm == T(0) ? value(T(0)) : (phase * conjugate(y)) / norm;
                for (int j = k > begin ? k - 1 : k; j < n; ++j)
                {
                    const auto u       = h[k * n + j];
                    const auto w       = h[(k + 1) * n + j];
                    h[k * n + j]       = u * c + s * w;
                    h[(k + 1) * n + j] = w * c - conjugate(s) * u;
                }
                for (int i = 0; i < n; ++i)
                {
                    const auto u     = h[i * n + k];
                    const auto w     = h[i * n + k + 1];
                    h[i * n + k]     = u * c + conjugate(s) * w;
                    h[i * n + k + 1] = w * c - s * u;
                    const auto qu    = q[i * n + k];
                    const auto qw    = q[i * n + k + 1];
                    q[i * n + k]     = qu * c + conjugate(s) * qw;
                    q[i * n + k + 1] = qw * c - s * qu;
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

    // Eigen's triangular square-root recurrence, evaluated in place.
    template <typename T> MF_INLINE bool triangular_sqrt(MF_ADDRESS complex_value<T> * h, int n)
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
                for (int k = i + 1; k < j; ++k)
                {
                    numerator = numerator - h[i * n + k] * h[k * n + j];
                }
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

    // Fixed degree-seven Gauss-Legendre/Pade evaluation from Eigen's logarithm.
    // Inverse scaling by triangular square roots puts ||T-I||_1 <= 1/4, below
    // Eigen's double-precision degree-seven threshold (0.2642960831111435).
    template <typename T>
    MF_INLINE bool triangular_log(MF_ADDRESS complex_value<T> * h,
        MF_ADDRESS                           complex_value<T> * f,
        MF_ADDRESS                           complex_value<T> * solve,
        MF_ADDRESS                           complex_value<T> * diagonal,
        int                                  n,
        T                                    input_scale)
    {
        for (int i = 0; i < n; ++i)
        {
            diagonal[i] = logarithm(h[i * n + i]);
        }
        int roots      = 0;
        T   multiplier = T(1);
        for (;;)
        {
            T norm = T(0);
            for (int j = 0; j < n; ++j)
            {
                T column = T(0);
                for (int i = 0; i <= j; ++i)
                {
                    column += magnitude(i == j ? h[i * n + j] - value(T(1)) : h[i * n + j]);
                }
                norm = fmax(norm, column);
            }
            if (norm <= T(0.25))
            {
                break;
            }
            if (++roots > 64 || !triangular_sqrt(h, n))
            {
                return false;
            }
            multiplier *= T(2);
        }
        for (int i = 0; i < n; ++i)
        {
            h[i * n + i] = exponential_minus_one(diagonal[i] * (T(1) / multiplier));
        }
        const T nodes[7]   = {T(0.0254460438286207377),
            T(0.1292344072003027801),
            T(0.2970774243113014165),
            T(0.5),
            T(0.7029225756886985835),
            T(0.8707655927996972199),
            T(0.9745539561713792623)};
        const T weights[7] = {T(0.0647424830844348466),
            T(0.1398526957446383340),
            T(0.1909150252525594725),
            T(0.2089795918367346939),
            T(0.1909150252525594725),
            T(0.1398526957446383340),
            T(0.0647424830844348466)};
        for (int i = 0; i < n * n; ++i)
        {
            f[i] = value(T(0));
        }
        for (int node = 0; node < 7; ++node)
        {
            for (int j = 0; j < n; ++j)
            {
                for (int i = j; i >= 0; --i)
                {
                    auto rhs = h[i * n + j];
                    for (int k = i + 1; k <= j; ++k)
                    {
                        rhs = rhs - h[i * n + k] * solve[k * n + j] * nodes[node];
                    }
                    solve[i * n + j] = rhs / (value(T(1)) + h[i * n + i] * nodes[node]);
                    f[i * n + j]     = f[i * n + j] + solve[i * n + j] * weights[node];
                }
            }
        }
        for (int i = 0; i < n * n; ++i)
        {
            f[i] = f[i] * multiplier;
        }
        for (int i = 0; i < n; ++i)
        {
            f[i * n + i] = diagonal[i] + value(log(input_scale));
        }
        return true;
    }

    // Five n*n complex matrices and one n-element complex vector; no allocations.
    // Failure is communicated without host access, so GPU callers remain async.
    template <typename T>
    MF_INLINE bool compute(MF_ADDRESS const T* a,
        int                                    n,
        int                                    lda,
        MF_ADDRESS T*                          result,
        int                                    ldc,
        MF_ADDRESS T*                          workspace,
        bool                                   is_log,
        T                                      epsilon)
    {
        auto h             = reinterpret_cast<MF_ADDRESS complex_value<T>*>(workspace);
        auto q             = h + n * n;
        auto f             = q + n * n;
        auto temporary     = f + n * n;
        auto solve         = temporary + n * n;
        auto diagonal      = solve + n * n;
        T    scale         = T(0);
        bool diagonal_only = true;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (!isfinite(a[i * lda + j]))
                {
                    return false;
                }
                scale = fmax(scale, fabs(a[i * lda + j]));
                if (i != j && a[i * lda + j] != T(0))
                {
                    diagonal_only = false;
                }
            }
        }
        // Scalar evaluation also preserves extreme diagonal dynamic ranges that
        // would underflow if every entry were divided by the largest entry.
        if (diagonal_only)
        {
            for (int i = 0; i < n; ++i)
            {
                if (is_log ? a[i * lda + i] <= T(0) : a[i * lda + i] < T(0))
                {
                    return false;
                }
                diagonal[i] = value(is_log ? log(a[i * lda + i]) : sqrt(a[i * lda + i]));
            }
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    result[i * ldc + j] = i == j ? diagonal[i].real : T(0);
                }
            }
            return true;
        }
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                h[i * n + j] = value(a[i * lda + j] / scale);
            }
        }
        if (!schur(h, q, diagonal, n, epsilon))
        {
            return false;
        }
        for (int i = 0; i < n; ++i)
        {
            const auto eigenvalue = h[i * n + i];
            if ((is_log && magnitude(eigenvalue) == T(0)) ||
                (eigenvalue.real < T(0) &&
                    fabs(eigenvalue.imag) <= T(64) * epsilon * fabs(eigenvalue.real)))
            {
                return false;
            }
            for (int j = 0; j < i; ++j)
            {
                h[i * n + j] = value(T(0));
            }
        }
        if (is_log)
        {
            if (!triangular_log(h, f, solve, diagonal, n, scale))
            {
                return false;
            }
        }
        else
        {
            if (!triangular_sqrt(h, n))
            {
                return false;
            }
            for (int i = 0; i < n * n; ++i)
            {
                f[i] = h[i] * sqrt(scale);
            }
        }
        // Reconstruct Q*f(T)*Q^*. Check the imaginary residual before publishing
        // any output, preserving input/output aliasing and rejecting complex roots.
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                auto sum = value(T(0));
                for (int k = 0; k <= j; ++k)
                {
                    sum = sum + q[i * n + k] * f[k * n + j];
                }
                temporary[i * n + j] = sum;
            }
        }
        T real_norm = T(0);
        T imag_norm = T(0);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                auto sum = value(T(0));
                for (int k = 0; k < n; ++k)
                {
                    sum = sum + temporary[i * n + k] * conjugate(q[j * n + k]);
                }
                if (!isfinite(sum.real) || !isfinite(sum.imag))
                {
                    return false;
                }
                real_norm    = fmax(real_norm, fabs(sum.real));
                imag_norm    = fmax(imag_norm, fabs(sum.imag));
                f[i * n + j] = sum;
            }
        }
        if (imag_norm > T(512) * T(n) * epsilon * fmax(T(1), real_norm))
        {
            return false;
        }
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                result[i * ldc + j] = f[i * n + j].real;
            }
        }
        return true;
    }
})

#undef LINALG_MATRIX_FUNCTIONS_CORE
