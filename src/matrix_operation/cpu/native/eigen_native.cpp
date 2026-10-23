#include "include/matrix_operation/eigen.h"

#include "include/common/linalg_checks.h"

#include "include/common/workspace_arena.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "include/common/linalg_profiler.h"

using std::isfinite;
#include "src/matrix_operation/schur_decomposition_impl.h"
#include "src/matrix_operation/schur_decomposition_workspace.h"

namespace linalg
{
namespace detail
{

// Null/positivity checks shared by every symmetric_eigen_backend<T,
// Backend>::apply below, so callers (tests included) get the same
// validation regardless of which backend is active. There is no
// non-detail symmetric_eigenvalue_decomposition() wrapper to put this in
// any more — callers use
// symmetric_eigen_backend<T, Backend>::apply directly, naming Backend explicitly.
static void check_symmetric_eigen_inputs(
    const void* eigenvectors, const void* eigenvalues, linalg_int n, linalg_int lda, linalg_int ldv)
{
    LINALG_CHECK(eigenvalues != nullptr && eigenvectors != nullptr,
        "symmetric_eigen_backend::apply: eigenvalues and eigenvectors must not be null");
    LINALG_CHECK(n > 0 && lda >= n && ldv >= n,
        "symmetric_eigen_backend::apply: n/lda/ldv must be positive");
}

// Same, for every general_eigen_backend<T, Backend>::apply below. There is
// no non-detail eigen() wrapper to put this in any more
// — callers use general_eigen_backend<T, Backend>::apply directly, naming
// Backend explicitly.
static void check_general_eigen_inputs(const void* eigenvectors,
    const void*                                    eigenvalues_real,
    const void*                                    eigenvalues_imag,
    bool                                           compute_eigenvectors,
    linalg_int                                     n,
    linalg_int                                     lda,
    linalg_int                                     ldv)
{
    LINALG_CHECK(
        eigenvectors != nullptr && eigenvalues_real != nullptr && eigenvalues_imag != nullptr,
        "general_eigen_backend::apply: eigenvectors, eigenvalues_real, and eigenvalues_imag "
        "must not be null");
    LINALG_CHECK(n > 0 && lda >= n && !(compute_eigenvectors && ldv < n),
        "general_eigen_backend::apply: n/lda/ldv must be positive");
}

}  // namespace detail

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/eigenvalue_decomposition_mkl.cpp and
// blas/eigenvalue_decomposition_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

// Householder reduction followed by implicit QL avoids repeated dense Jacobi
// sweeps. Eigenvectors are kept as contiguous rows during the QL rotations.
template <typename T> struct symmetric_layout
{
    T*          a;
    T*          vectors;
    T*          diagonal;
    T*          off_diagonal;
    T*          work;
    linalg_int* index;
};

template <typename T>
symmetric_layout<T> symmetric_plan(detail::workspace_arena& arena, linalg_int n)
{
    const auto count = static_cast<size_t>(n);
    return {arena.next<T>(count * count),
        arena.next<T>(count * count),
        arena.next<T>(count),
        arena.next<T>(count),
        arena.next<T>(count),
        arena.next<linalg_int>(count)};
}

template <typename T> size_t symmetric_workspace_size(linalg_int n)
{
    detail::workspace_arena arena;
    symmetric_plan<T>(arena, n);
    return arena.total_size();
}

template <typename T>
bool symmetric_eigen(
    T* output, linalg_int n, linalg_int lda, T* values, linalg_int ldv, symmetric_layout<T> buffers)
{
    auto [a, vectors, d, e, v, index] = buffers;
    T scale                           = T(0);
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j <= i; ++j)
        {
            if (!std::isfinite(output[i * lda + j]))
            {
                return false;
            }
            scale = std::max(scale, std::fabs(output[i * lda + j]));
        }
    }
    if (scale == T(0))
    {
        scale = T(1);
    }
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            a[i * n + j]       = (i >= j ? output[i * lda + j] : output[j * lda + i]) / scale;
            vectors[i * n + j] = i == j ? T(1) : T(0);
        }
    }
    for (linalg_int k = 0; k + 2 < n; ++k)
    {
        T norm = T(0);
        for (linalg_int i = k + 1; i < n; ++i)
        {
            v[i] = a[i * n + k];
            norm = std::hypot(norm, v[i]);
        }
        if (norm == T(0))
        {
            continue;
        }
        const T alpha = -std::copysign(norm, v[k + 1]);
        v[k + 1] -= alpha;
        T vnorm = T(0);
        for (linalg_int i = k + 1; i < n; ++i)
        {
            vnorm = std::hypot(vnorm, v[i]);
        }
        for (linalg_int i = k + 1; i < n; ++i)
        {
            v[i] /= vnorm;
        }
        // Symmetric rank-two update: H A H = A - v w^T - w v^T.
        T correction = T(0);
        for (linalg_int i = k + 1; i < n; ++i)
        {
            T dot = T(0);
            for (linalg_int j = k + 1; j < n; ++j)
            {
                dot += a[i * n + j] * v[j];
            }
            d[i] = T(2) * dot;
            correction += d[i] * v[i];
        }
        for (linalg_int i = k + 1; i < n; ++i)
        {
            d[i] -= correction * v[i];
        }
        for (linalg_int i = k + 1; i < n; ++i)
        {
            for (linalg_int j = k + 1; j <= i; ++j)
            {
                const T entry = a[i * n + j] - v[i] * d[j] - d[i] * v[j];
                a[i * n + j]  = entry;
                a[j * n + i]  = entry;
            }
        }
        a[(k + 1) * n + k] = alpha;
        for (linalg_int i = k + 2; i < n; ++i)
        {
            a[i * n + k] = T(0);
        }
        // Accumulate Q by applying each reflector to contiguous row segments.
        for (linalg_int i = 0; i < n; ++i)
        {
            T dot = T(0);
            for (linalg_int j = k + 1; j < n; ++j)
            {
                dot += vectors[i * n + j] * v[j];
            }
            for (linalg_int j = k + 1; j < n; ++j)
            {
                vectors[i * n + j] -= T(2) * dot * v[j];
            }
        }
    }
    for (linalg_int i = 0; i < n; ++i)
    {
        d[i] = a[i * n + i];
        e[i] = i + 1 < n ? a[(i + 1) * n + i] : T(0);
        for (linalg_int j = 0; j < i; ++j)
        {
            std::swap(vectors[i * n + j], vectors[j * n + i]);
        }
    }
    const T eps = std::numeric_limits<T>::epsilon();
    for (linalg_int l = 0; l < n; ++l)
    {
        int iterations = 0;
        while (true)
        {
            linalg_int m = l;
            for (; m + 1 < n; ++m)
            {
                if (std::fabs(e[m]) <= eps * (std::fabs(d[m]) + std::fabs(d[m + 1])))
                {
                    break;
                }
            }
            if (m == l)
            {
                break;
            }
            if (++iterations > 64)
            {
                return false;
            }
            T g       = (d[l + 1] - d[l]) / (T(2) * e[l]);
            T r       = std::hypot(g, T(1));
            g         = d[m] - d[l] + e[l] / (g + std::copysign(r, g));
            T    sine = T(1), cosine = T(1), shift = T(0);
            bool split = false;
            for (linalg_int i = m; i-- > l;)
            {
                const T f = sine * e[i];
                const T b = cosine * e[i];
                r         = std::hypot(f, g);
                e[i + 1]  = r;
                if (r == T(0))
                {
                    d[i + 1] -= shift;
                    e[m]  = T(0);
                    split = true;
                    break;
                }
                sine     = f / r;
                cosine   = g / r;
                g        = d[i + 1] - shift;
                r        = (d[i] - g) * sine + T(2) * cosine * b;
                shift    = sine * r;
                d[i + 1] = g + shift;
                g        = cosine * r - b;
                for (linalg_int j = 0; j < n; ++j)
                {
                    const T x                = vectors[i * n + j];
                    const T y                = vectors[(i + 1) * n + j];
                    vectors[i * n + j]       = cosine * x - sine * y;
                    vectors[(i + 1) * n + j] = sine * x + cosine * y;
                }
            }
            if (!split)
            {
                d[l] -= shift;
                e[l] = g;
                e[m] = T(0);
            }
        }
    }
    std::iota(index, index + n, static_cast<linalg_int>(0));
    std::sort(index,
        index + n,
        [diagonal = buffers.diagonal](linalg_int i, linalg_int j)
        { return diagonal[i] < diagonal[j]; });
    for (linalg_int j = 0; j < n; ++j)
    {
        values[j] = d[index[j]] * scale;
        for (linalg_int i = 0; i < n; ++i)
        {
            output[i * ldv + j] = vectors[index[j] * n + i];
        }
    }
    return true;
}

template <typename T> struct general_layout
{
    linalg_schur::complex_value<T>* h;
    linalg_schur::complex_value<T>* q;
    linalg_schur::complex_value<T>* work;
};

template <typename T>
general_layout<T> general_plan(detail::workspace_arena& arena, linalg_int n, bool vectors)
{
    const auto count = static_cast<size_t>(n);
    using C          = linalg_schur::complex_value<T>;
    return {arena.next<C>(count * count),
        vectors ? arena.next<C>(count * count) : nullptr,
        arena.next<C>(count)};
}

template <typename T> size_t general_workspace_size(linalg_int n, bool vectors)
{
    detail::schur_decomposition_order(n);
    detail::workspace_arena arena;
    general_plan<T>(arena, n, vectors);
    return arena.total_size();
}

// Reuse the portable Hessenberg/implicit shifted QR kernel. Each QR step
// costs O(n^2), replacing an explicit polynomial, dense QR and three GEMMs.
template <typename T>
bool general_eigen_impl(T* output,
    linalg_int             n,
    linalg_int             lda,
    T*                     real,
    T*                     imag,
    bool                   vectors,
    linalg_int             ldv,
    void*                  workspace,
    size_t                 bytes)
{
    LINALG_CHECK(bytes >= general_workspace_size<T>(n, vectors), "eigen: workspace_size too small");
    detail::workspace_arena arena(workspace, bytes);
    auto [h, q, work] = general_plan<T>(arena, n, vectors);
    using namespace linalg_schur;
    T scale = T(0);
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            if (!std::isfinite(output[i * lda + j]))
            {
                return false;
            }
            scale = std::max(scale, std::fabs(output[i * lda + j]));
        }
    }
    if (scale == T(0))
    {
        scale = T(1);
    }
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            h[i * n + j] = value(output[i * lda + j] / scale);
        }
    }
    const T    eps = std::numeric_limits<T>::epsilon();
    const bool ok  = vectors ? schur_iteration<T, true>(h, q, work, static_cast<int>(n), eps)
                             : schur_iteration<T, false>(h, q, work, static_cast<int>(n), eps);
    if (!ok)
    {
        return false;
    }
    for (linalg_int i = 0; i < n; ++i)
    {
        real[i] = h[i * n + i].real * scale;
        imag[i] = h[i * n + i].imag * scale;
    }
    if (!vectors)
    {
        return true;
    }
    // Back substitution in triangular Schur form, followed by Q*y, computes
    // all vectors in O(n^3), instead of an LU factorization per eigenvalue.
    for (linalg_int column = 0; column < n; ++column)
    {
        LINALG_CHECK(std::fabs(imag[column]) <= eps * (std::fabs(real[column]) + T(1)) * T(10),
            "eigen: scalar fallback cannot compute eigenvectors for a complex eigenvalue (index "
            "{})",
            column);
        std::fill_n(work, n, value(T(0)));
        work[column] = value(T(1));
        for (linalg_int row = column; row-- > 0;)
        {
            auto sum = value(T(0));
            for (linalg_int j = row + 1; j <= column; ++j)
            {
                sum = sum + h[row * n + j] * work[j];
            }
            auto denominator = h[row * n + row] - h[column * n + column];
            if (magnitude(denominator) < eps)
            {
                denominator = value(eps);
            }
            work[row]    = (value(T(0)) - sum) / denominator;
            const T norm = magnitude(work[row]);
            if (norm > T(1))
            {
                for (linalg_int j = row; j <= column; ++j)
                {
                    work[j] = work[j] / norm;
                }
            }
        }
        // A real eigenvector may carry a complex phase. Rotate its largest
        // component onto the real axis before returning the normalized real part.
        auto phase   = value(T(1));
        T    largest = T(0);
        for (linalg_int row = 0; row < n; ++row)
        {
            const auto component = complex_dot(q + row * n, 1, work, 1, column + 1);
            const T    norm      = magnitude(component);
            if (norm > largest)
            {
                largest = norm;
                phase   = conjugate(component) / norm;
            }
        }
        T norm = T(0);
        for (linalg_int row = 0; row < n; ++row)
        {
            const T component = (complex_dot(q + row * n, 1, work, 1, column + 1) * phase).real;
            output[row * ldv + column] = component;
            norm                       = std::hypot(norm, component);
        }
        if (norm == T(0))
        {
            return false;
        }
        for (linalg_int row = 0; row < n; ++row)
        {
            output[row * ldv + column] /= norm;
        }
    }
    return true;
}

}  // namespace

template <>
bool symmetric_eigen_backend<float, linalg_backend::NATIVE>::apply(float* eigenvectors,
    linalg_int                                                            n,
    linalg_int                                                            lda,
    float*                                                                eigenvalues,
    linalg_int                                                            ldv,
    void*                                                                 workspace,
    size_t                                                                workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("eigen/native<float>");
    detail::check_symmetric_eigen_inputs(eigenvectors, eigenvalues, n, lda, ldv);
    LINALG_CHECK(workspace_size >= symmetric_workspace_size<float>(n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = symmetric_plan<float>(arena, n);
    return symmetric_eigen(eigenvectors, n, lda, eigenvalues, ldv, bufs);
}

template <>
bool symmetric_eigen_backend<double, linalg_backend::NATIVE>::apply(double* eigenvectors,
    linalg_int                                                              n,
    linalg_int                                                              lda,
    double*                                                                 eigenvalues,
    linalg_int                                                              ldv,
    void*                                                                   workspace,
    size_t                                                                  workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("eigen/native<double>");
    detail::check_symmetric_eigen_inputs(eigenvectors, eigenvalues, n, lda, ldv);
    LINALG_CHECK(workspace_size >= symmetric_workspace_size<double>(n),
        "symmetric_eigenvalue_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = symmetric_plan<double>(arena, n);
    return symmetric_eigen(eigenvectors, n, lda, eigenvalues, ldv, bufs);
}

template <>
bool general_eigen_backend<float, linalg_backend::NATIVE>::apply(float* eigenvectors,
    linalg_int                                                          n,
    linalg_int                                                          lda,
    float*                                                              eigenvalues_real,
    float*                                                              eigenvalues_imag,
    bool                                                                compute_eigenvectors,
    linalg_int                                                          ldv,
    void*                                                               workspace,
    size_t                                                              workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("general_eigen/native<float>");
    detail::check_general_eigen_inputs(
        eigenvectors, eigenvalues_real, eigenvalues_imag, compute_eigenvectors, n, lda, ldv);
    return general_eigen_impl(eigenvectors,
        n,
        lda,
        eigenvalues_real,
        eigenvalues_imag,
        compute_eigenvectors,
        ldv,
        workspace,
        workspace_size);
}

template <>
bool general_eigen_backend<double, linalg_backend::NATIVE>::apply(double* eigenvectors,
    linalg_int                                                            n,
    linalg_int                                                            lda,
    double*                                                               eigenvalues_real,
    double*                                                               eigenvalues_imag,
    bool                                                                  compute_eigenvectors,
    linalg_int                                                            ldv,
    void*                                                                 workspace,
    size_t                                                                workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("general_eigen/native<double>");
    detail::check_general_eigen_inputs(
        eigenvectors, eigenvalues_real, eigenvalues_imag, compute_eigenvectors, n, lda, ldv);
    return general_eigen_impl(eigenvectors,
        n,
        lda,
        eigenvalues_real,
        eigenvalues_imag,
        compute_eigenvectors,
        ldv,
        workspace,
        workspace_size);
}

template <>
size_t symmetric_eigen_backend<float, linalg_backend::NATIVE>::workspace_size(linalg_int n)
{
    return symmetric_workspace_size<float>(n);
}

template <>
size_t symmetric_eigen_backend<double, linalg_backend::NATIVE>::workspace_size(linalg_int n)
{
    return symmetric_workspace_size<double>(n);
}

template <>
size_t general_eigen_backend<float, linalg_backend::NATIVE>::workspace_size(
    linalg_int n, bool compute_eigenvectors)
{
    return general_workspace_size<float>(n, compute_eigenvectors);
}

template <>
size_t general_eigen_backend<double, linalg_backend::NATIVE>::workspace_size(
    linalg_int n, bool compute_eigenvectors)
{
    return general_workspace_size<double>(n, compute_eigenvectors);
}
}  // namespace linalg
