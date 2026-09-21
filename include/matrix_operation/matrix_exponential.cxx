#include "include/matrix_operation/matrix_exponential.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "include/matrix_operation/matrix_inversion.h"
#include "include/matrix_operation/matrix_multiplication.h"
#include "include/matrix_operation/matrix_norm.h"
#include "ThirdParty/Logging/include/logging.h"

namespace linalg
{
namespace detail
{
namespace
{

// Diagonal Pade[m/m] numerator/denominator coefficients b_0..b_m and the
// ||A||_1 thresholds theta_m below which that degree's approximation error
// is within double-precision round-off, both from N. J. Higham, "The
// Scaling and Squaring Method for the Matrix Exponential Revisited", SIAM
// J. Matrix Anal. Appl. 26(4), 2005, Table 2.3 / Table 3.1 — identical to
// the tables PyTorch's and SciPy's matrix_exp/expm are built on.
constexpr double kTheta3  = 1.495585217958292e-2;
constexpr double kTheta5  = 2.539398330063230e-1;
constexpr double kTheta7  = 9.504178996162932e-1;
constexpr double kTheta9  = 2.097847961257068e0;
constexpr double kTheta13 = 5.371920351148152e0;

constexpr double kPade3[]  = {120., 60., 12., 1.};
constexpr double kPade5[]  = {30240., 15120., 3360., 420., 30., 1.};
constexpr double kPade7[]  = {17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.};
constexpr double kPade9[] = {
    17643225600., 8821612800., 2075673600., 302702400., 30270240., 2162160., 110880., 3960., 90., 1.};
constexpr double kPade13[] = {
    64764752532480000.,
    32382376266240000.,
    7771770303897600.,
    1187353796428800.,
    129060195264000.,
    10559470521600.,
    670442572800.,
    33522128640.,
    1323241920.,
    40840800.,
    960960.,
    16380.,
    182.,
    1.};

template <typename T>
void mat_mul(const T* a, const T* b, linalg_int n, T* c)
{
    linalg::matrix_multiplication(false, false, n, n, n, a, n, b, n, c, n);
}

// dst += c * src, both n x n and tightly packed.
template <typename T>
void add_scaled(T* dst, const T* src, linalg_int n, double c)
{
    const linalg_int nn = n * n;
    const T          cc = static_cast<T>(c);
    for (linalg_int i = 0; i < nn; ++i)
    {
        dst[i] += cc * src[i];
    }
}

// dst += c * I, n x n and tightly packed.
template <typename T>
void add_scaled_identity(T* dst, linalg_int n, double c)
{
    const T cc = static_cast<T>(c);
    for (linalg_int i = 0; i < n; ++i)
    {
        dst[i * n + i] += cc;
    }
}

template <typename T>
void zero(T* dst, linalg_int n)
{
    std::fill(dst, dst + n * n, T(0));
}

// Builds U, V (both n x n, tightly packed) such that exp(A) ~= (V-U)^-1 (V+U)
// for the diagonal Pade approximant of the given degree, from A and its
// precomputed even powers A2 = A*A, A4 = A2*A2, A6 = A2*A4 (A4/A6 may be
// null when the chosen degree doesn't need them).
template <typename T>
void build_pade(
    const T* A, const T* A2, const T* A4, const T* A6, linalg_int n, int degree, T* U, T* V)
{
    switch (degree)
    {
    case 3:
    {
        // V = b2*A2 + b0*I ; U = A*(b3*A2 + b1*I)
        zero(V, n);
        add_scaled(V, A2, n, kPade3[2]);
        add_scaled_identity(V, n, kPade3[0]);

        std::vector<T> tmp(static_cast<std::size_t>(n * n), T(0));
        add_scaled(tmp.data(), A2, n, kPade3[3]);
        add_scaled_identity(tmp.data(), n, kPade3[1]);
        mat_mul(A, tmp.data(), n, U);
        break;
    }
    case 5:
    {
        zero(V, n);
        add_scaled(V, A4, n, kPade5[4]);
        add_scaled(V, A2, n, kPade5[2]);
        add_scaled_identity(V, n, kPade5[0]);

        std::vector<T> tmp(static_cast<std::size_t>(n * n), T(0));
        add_scaled(tmp.data(), A4, n, kPade5[5]);
        add_scaled(tmp.data(), A2, n, kPade5[3]);
        add_scaled_identity(tmp.data(), n, kPade5[1]);
        mat_mul(A, tmp.data(), n, U);
        break;
    }
    case 7:
    {
        zero(V, n);
        add_scaled(V, A6, n, kPade7[6]);
        add_scaled(V, A4, n, kPade7[4]);
        add_scaled(V, A2, n, kPade7[2]);
        add_scaled_identity(V, n, kPade7[0]);

        std::vector<T> tmp(static_cast<std::size_t>(n * n), T(0));
        add_scaled(tmp.data(), A6, n, kPade7[7]);
        add_scaled(tmp.data(), A4, n, kPade7[5]);
        add_scaled(tmp.data(), A2, n, kPade7[3]);
        add_scaled_identity(tmp.data(), n, kPade7[1]);
        mat_mul(A, tmp.data(), n, U);
        break;
    }
    case 9:
    {
        std::vector<T> A8(static_cast<std::size_t>(n * n));
        mat_mul(A4, A4, n, A8.data());

        zero(V, n);
        add_scaled(V, A8.data(), n, kPade9[8]);
        add_scaled(V, A6, n, kPade9[6]);
        add_scaled(V, A4, n, kPade9[4]);
        add_scaled(V, A2, n, kPade9[2]);
        add_scaled_identity(V, n, kPade9[0]);

        std::vector<T> tmp(static_cast<std::size_t>(n * n), T(0));
        add_scaled(tmp.data(), A8.data(), n, kPade9[9]);
        add_scaled(tmp.data(), A6, n, kPade9[7]);
        add_scaled(tmp.data(), A4, n, kPade9[5]);
        add_scaled(tmp.data(), A2, n, kPade9[3]);
        add_scaled_identity(tmp.data(), n, kPade9[1]);
        mat_mul(A, tmp.data(), n, U);
        break;
    }
    case 13:
    default:
    {
        // U = A * (A6*(b13*A6 + b11*A4 + b9*A2) + b7*A6 + b5*A4 + b3*A2 + b1*I)
        std::vector<T> w1(static_cast<std::size_t>(n * n), T(0));
        add_scaled(w1.data(), A6, n, kPade13[13]);
        add_scaled(w1.data(), A4, n, kPade13[11]);
        add_scaled(w1.data(), A2, n, kPade13[9]);

        std::vector<T> w(static_cast<std::size_t>(n * n));
        mat_mul(A6, w1.data(), n, w.data());
        add_scaled(w.data(), A6, n, kPade13[7]);
        add_scaled(w.data(), A4, n, kPade13[5]);
        add_scaled(w.data(), A2, n, kPade13[3]);
        add_scaled_identity(w.data(), n, kPade13[1]);
        mat_mul(A, w.data(), n, U);

        // V = A6*(b12*A6 + b10*A4 + b8*A2) + b6*A6 + b4*A4 + b2*A2 + b0*I
        std::vector<T> z1(static_cast<std::size_t>(n * n), T(0));
        add_scaled(z1.data(), A6, n, kPade13[12]);
        add_scaled(z1.data(), A4, n, kPade13[10]);
        add_scaled(z1.data(), A2, n, kPade13[8]);

        mat_mul(A6, z1.data(), n, V);
        add_scaled(V, A6, n, kPade13[6]);
        add_scaled(V, A4, n, kPade13[4]);
        add_scaled(V, A2, n, kPade13[2]);
        add_scaled_identity(V, n, kPade13[0]);
        break;
    }
    }
}

template <typename T>
void matrix_exponential_impl(const T* A_in, linalg_int n, linalg_int lda, T* result, linalg_int ldc)
{
    const linalg_int nn = n * n;

    std::vector<T> A(static_cast<std::size_t>(nn));
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            A[static_cast<std::size_t>(i * n + j)] = A_in[i * lda + j];
        }
    }

    const double norm1 = static_cast<double>(
        linalg::matrix_norm(A.data(), static_cast<linalg_long>(n), static_cast<linalg_long>(n),
            static_cast<linalg_long>(n), linalg::matrix_norm_type::ONE));

    int degree = 13;
    int s      = 0;
    if (norm1 < kTheta3)
    {
        degree = 3;
    }
    else if (norm1 < kTheta5)
    {
        degree = 5;
    }
    else if (norm1 < kTheta7)
    {
        degree = 7;
    }
    else if (norm1 < kTheta9)
    {
        degree = 9;
    }
    else
    {
        degree = 13;
        if (norm1 > kTheta13)
        {
            s = static_cast<int>(std::ceil(std::log2(norm1 / kTheta13)));
            s = std::max(s, 0);
        }
    }

    if (s > 0)
    {
        const T scale = static_cast<T>(std::ldexp(1.0, -s));
        for (linalg_int i = 0; i < nn; ++i)
        {
            A[static_cast<std::size_t>(i)] *= scale;
        }
    }

    std::vector<T> A2(static_cast<std::size_t>(nn));
    mat_mul(A.data(), A.data(), n, A2.data());

    std::vector<T> A4;
    if (degree >= 5)
    {
        A4.resize(static_cast<std::size_t>(nn));
        mat_mul(A2.data(), A2.data(), n, A4.data());
    }

    std::vector<T> A6;
    if (degree >= 7)
    {
        A6.resize(static_cast<std::size_t>(nn));
        mat_mul(A2.data(), A4.data(), n, A6.data());
    }

    std::vector<T> U(static_cast<std::size_t>(nn));
    std::vector<T> V(static_cast<std::size_t>(nn));
    build_pade(A.data(), A2.data(), A4.empty() ? nullptr : A4.data(), A6.empty() ? nullptr : A6.data(),
        n, degree, U.data(), V.data());

    std::vector<T> denom(static_cast<std::size_t>(nn));
    std::vector<T> numer(static_cast<std::size_t>(nn));
    for (linalg_int i = 0; i < nn; ++i)
    {
        denom[static_cast<std::size_t>(i)] = V[static_cast<std::size_t>(i)] - U[static_cast<std::size_t>(i)];
        numer[static_cast<std::size_t>(i)] = V[static_cast<std::size_t>(i)] + U[static_cast<std::size_t>(i)];
    }

    std::vector<linalg_int> pivot(static_cast<std::size_t>(n));
    linalg::matrix_invert(denom.data(), pivot.data(), n);

    std::vector<T> R(static_cast<std::size_t>(nn));
    mat_mul(denom.data(), numer.data(), n, R.data());

    std::vector<T> tmp(static_cast<std::size_t>(nn));
    for (int k = 0; k < s; ++k)
    {
        mat_mul(R.data(), R.data(), n, tmp.data());
        R.swap(tmp);
    }

    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            result[i * ldc + j] = R[static_cast<std::size_t>(i * n + j)];
        }
    }
}

}  // namespace
}  // namespace detail

//-----------------------------------------------------------------------------
void matrix_exponential(const float* A, linalg_int n, linalg_int lda, float* result, linalg_int ldc)
{
    LOGGING_CHECK(A != nullptr && result != nullptr, "matrix_exponential: A and result must not be null");
    LOGGING_CHECK(n > 0 && lda > 0 && ldc > 0, "matrix_exponential: n/lda/ldc must be positive");
    detail::matrix_exponential_impl(A, n, lda, result, ldc);
}

//-----------------------------------------------------------------------------
void matrix_exponential(const double* A, linalg_int n, linalg_int lda, double* result, linalg_int ldc)
{
    LOGGING_CHECK(A != nullptr && result != nullptr, "matrix_exponential: A and result must not be null");
    LOGGING_CHECK(n > 0 && lda > 0 && ldc > 0, "matrix_exponential: n/lda/ldc must be positive");
    detail::matrix_exponential_impl(A, n, lda, result, ldc);
}

}  // namespace linalg
