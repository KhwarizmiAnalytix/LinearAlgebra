#include "include/matrix_operation/exponential.h"

#include <algorithm>
#include <cmath>

#include "include/common/workspace_arena.h"
#include "include/matrix_operation/inversion.h"
#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/norm.h"
#include "include/logging.h"

namespace linalg
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
constexpr double kPade9[]  = {17643225600.,
     8821612800.,
     2075673600.,
     302702400.,
     30270240.,
     2162160.,
     110880.,
     3960.,
     90.,
     1.};
constexpr double kPade13[] = {64764752532480000.,
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
void mat_mul(const T* a, const T* b, linalg_int n, T* c, void* gemm_ws, size_t gemm_bytes)
{
    linalg::multiplication<T, linalg::kMatmulActiveBackend>::apply(
        false, false, n, n, n, a, n, b, n, c, n, gemm_ws, gemm_bytes);
}

// dst += c * src, both n x n and tightly packed.
template <typename T> void add_scaled(T* dst, const T* src, linalg_int n, double c)
{
    const linalg_int nn = n * n;
    const T          cc = static_cast<T>(c);
    for (linalg_int i = 0; i < nn; ++i)
    {
        dst[i] += cc * src[i];
    }
}

// dst += c * I, n x n and tightly packed.
template <typename T> void add_scaled_identity(T* dst, linalg_int n, double c)
{
    const T cc = static_cast<T>(c);
    for (linalg_int i = 0; i < n; ++i)
    {
        dst[i * n + i] += cc;
    }
}

template <typename T> void zero(T* dst, linalg_int n)
{
    std::fill(dst, dst + n * n, T(0));
}

// Builds U, V (both n x n, tightly packed) such that exp(A) ~= (V-U)^-1 (V+U)
// for the diagonal Pade approximant of the given degree, from A and its
// precomputed even powers A2 = A*A, A4 = A2*A2, A6 = A2*A4 (A4/A6 may be
// null when the chosen degree doesn't need them).
template <typename T>
void build_pade(const T* A,
    const T*             A2,
    const T*             A4,
    const T*             A6,
    linalg_int           n,
    int                  degree,
    T*                   U,
    T*                   V,
    T*                   s1,
    T*                   s2,
    T*                   s3,
    void*                gemm_ws,
    size_t               gemm_bytes)
{
    const auto nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    switch (degree)
    {
    case 3:
    {
        // V = b2*A2 + b0*I ; U = A*(b3*A2 + b1*I)
        zero(V, n);
        add_scaled(V, A2, n, kPade3[2]);
        add_scaled_identity(V, n, kPade3[0]);

        std::fill_n(s1, nn, T(0));
        add_scaled(s1, A2, n, kPade3[3]);
        add_scaled_identity(s1, n, kPade3[1]);
        mat_mul(A, s1, n, U, gemm_ws, gemm_bytes);
        break;
    }
    case 5:
    {
        zero(V, n);
        add_scaled(V, A4, n, kPade5[4]);
        add_scaled(V, A2, n, kPade5[2]);
        add_scaled_identity(V, n, kPade5[0]);

        std::fill_n(s1, nn, T(0));
        add_scaled(s1, A4, n, kPade5[5]);
        add_scaled(s1, A2, n, kPade5[3]);
        add_scaled_identity(s1, n, kPade5[1]);
        mat_mul(A, s1, n, U, gemm_ws, gemm_bytes);
        break;
    }
    case 7:
    {
        zero(V, n);
        add_scaled(V, A6, n, kPade7[6]);
        add_scaled(V, A4, n, kPade7[4]);
        add_scaled(V, A2, n, kPade7[2]);
        add_scaled_identity(V, n, kPade7[0]);

        std::fill_n(s1, nn, T(0));
        add_scaled(s1, A6, n, kPade7[7]);
        add_scaled(s1, A4, n, kPade7[5]);
        add_scaled(s1, A2, n, kPade7[3]);
        add_scaled_identity(s1, n, kPade7[1]);
        mat_mul(A, s1, n, U, gemm_ws, gemm_bytes);
        break;
    }
    case 9:
    {
        mat_mul(A4, A4, n, s1, gemm_ws, gemm_bytes);

        zero(V, n);
        add_scaled(V, s1, n, kPade9[8]);
        add_scaled(V, A6, n, kPade9[6]);
        add_scaled(V, A4, n, kPade9[4]);
        add_scaled(V, A2, n, kPade9[2]);
        add_scaled_identity(V, n, kPade9[0]);

        std::fill_n(s2, nn, T(0));
        add_scaled(s2, s1, n, kPade9[9]);
        add_scaled(s2, A6, n, kPade9[7]);
        add_scaled(s2, A4, n, kPade9[5]);
        add_scaled(s2, A2, n, kPade9[3]);
        add_scaled_identity(s2, n, kPade9[1]);
        mat_mul(A, s2, n, U, gemm_ws, gemm_bytes);
        break;
    }
    case 13:
    default:
    {
        // U = A * (A6*(b13*A6 + b11*A4 + b9*A2) + b7*A6 + b5*A4 + b3*A2 + b1*I)
        std::fill_n(s1, nn, T(0));
        add_scaled(s1, A6, n, kPade13[13]);
        add_scaled(s1, A4, n, kPade13[11]);
        add_scaled(s1, A2, n, kPade13[9]);

        mat_mul(A6, s1, n, s2, gemm_ws, gemm_bytes);
        add_scaled(s2, A6, n, kPade13[7]);
        add_scaled(s2, A4, n, kPade13[5]);
        add_scaled(s2, A2, n, kPade13[3]);
        add_scaled_identity(s2, n, kPade13[1]);
        mat_mul(A, s2, n, U, gemm_ws, gemm_bytes);

        // V = A6*(b12*A6 + b10*A4 + b8*A2) + b6*A6 + b4*A4 + b2*A2 + b0*I
        std::fill_n(s3, nn, T(0));
        add_scaled(s3, A6, n, kPade13[12]);
        add_scaled(s3, A4, n, kPade13[10]);
        add_scaled(s3, A2, n, kPade13[8]);

        mat_mul(A6, s3, n, V, gemm_ws, gemm_bytes);
        add_scaled(V, A6, n, kPade13[6]);
        add_scaled(V, A4, n, kPade13[4]);
        add_scaled(V, A2, n, kPade13[2]);
        add_scaled_identity(V, n, kPade13[0]);
        break;
    }
    }
}

template <typename T> struct exponential_layout
{
    T*             A;
    T*             A2;
    T*             A4;
    T*             A6;
    T*             U;
    T*             V;
    T*             s1;
    T*             s2;
    T*             s3;
    linalg_int*    pivot;
    unsigned char* invert_ws;
    size_t         invert_bytes;
    unsigned char* gemm_ws;
    size_t         gemm_bytes;
};

template <typename T>
exponential_layout<T> exponential_plan(detail::workspace_arena& arena, linalg_int n)
{
    const size_t          nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    exponential_layout<T> p;
    p.A            = arena.next<T>(nn);
    p.A2           = arena.next<T>(nn);
    p.A4           = arena.next<T>(nn);
    p.A6           = arena.next<T>(nn);
    p.U            = arena.next<T>(nn);
    p.V            = arena.next<T>(nn);
    p.s1           = arena.next<T>(nn);
    p.s2           = arena.next<T>(nn);
    p.s3           = arena.next<T>(nn);
    p.pivot        = arena.next<linalg_int>(static_cast<size_t>(n));
    p.invert_bytes = linalg::inversion<T, linalg::linalg_backend::native>::workspace_size(
        n, linalg::linear_solver_type::LU_LINEAR_SOLVER);
    p.invert_ws  = arena.next<unsigned char>(p.invert_bytes);
    p.gemm_bytes = linalg::multiplication<T, linalg::kMatmulActiveBackend>::workspace_size(n, n, n);
    p.gemm_ws    = arena.next<unsigned char>(p.gemm_bytes);
    return p;
}

template <typename T> size_t matrix_exponential_workspace_size_impl(linalg_int n)
{
    detail::workspace_arena arena;
    exponential_plan<T>(arena, n);
    return arena.total_size();
}

template <typename T>
void matrix_exponential_impl(const T* A_in,
    linalg_int                        n,
    linalg_int                        lda,
    T*                                result,
    linalg_int                        ldc,
    void*                             workspace,
    size_t                            workspace_size)
{
    const linalg_int nn = n * n;
    LOGGING_CHECK(workspace_size >= matrix_exponential_workspace_size_impl<T>(n),
        "exponential: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = exponential_plan<T>(arena, n);
    T*                      A    = bufs.A;

    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            A[static_cast<std::size_t>(i * n + j)] = A_in[i * lda + j];
        }
    }

    const double norm1 =
        static_cast<double>(linalg::norm<T, linalg::linalg_backend::native>::apply(A,
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            linalg::norm_type::ONE,
            nullptr,
            0));

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

    T* A2 = bufs.A2;
    mat_mul(A, A, n, A2, bufs.gemm_ws, bufs.gemm_bytes);

    T* A4 = bufs.A4;
    if (degree >= 5)
    {
        mat_mul(A2, A2, n, A4, bufs.gemm_ws, bufs.gemm_bytes);
    }

    T* A6 = bufs.A6;
    if (degree >= 7)
    {
        mat_mul(A2, A4, n, A6, bufs.gemm_ws, bufs.gemm_bytes);
    }

    build_pade(A,
        A2,
        degree >= 5 ? A4 : nullptr,
        degree >= 7 ? A6 : nullptr,
        n,
        degree,
        bufs.U,
        bufs.V,
        bufs.s1,
        bufs.s2,
        bufs.s3,
        bufs.gemm_ws,
        bufs.gemm_bytes);

    T* denom     = bufs.s1;
    T* numerator = bufs.s2;
    for (linalg_int i = 0; i < nn; ++i)
    {
        denom[static_cast<std::size_t>(i)] =
            bufs.V[static_cast<std::size_t>(i)] - bufs.U[static_cast<std::size_t>(i)];
        numerator[static_cast<std::size_t>(i)] =
            bufs.V[static_cast<std::size_t>(i)] + bufs.U[static_cast<std::size_t>(i)];
    }

    linalg::inversion<T, linalg::linalg_backend::native>::apply(denom,
        bufs.pivot,
        n,
        linalg::linear_solver_type::LU_LINEAR_SOLVER,
        bufs.invert_ws,
        bufs.invert_bytes);

    T* R   = bufs.s3;
    T* tmp = A;
    mat_mul(denom, numerator, n, R, bufs.gemm_ws, bufs.gemm_bytes);

    for (int k = 0; k < s; ++k)
    {
        mat_mul(R, R, n, tmp, bufs.gemm_ws, bufs.gemm_bytes);
        std::swap(R, tmp);
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

template <>
void exponential<float, linalg_backend::native>::apply(const float* A,
    linalg_int                                                                     n,
    linalg_int                                                                     lda,
    float*                                                                         result,
    linalg_int                                                                     ldc,
    void*                                                                          workspace,
    size_t                                                                         workspace_size)
{
    LOGGING_CHECK(A != nullptr && result != nullptr,
        "exponential::apply: A and result must not be null");
    LOGGING_CHECK(n > 0 && lda > 0 && ldc > 0,
        "exponential::apply: n/lda/ldc must be positive");
    matrix_exponential_impl(A, n, lda, result, ldc, workspace, workspace_size);
}

template <>
void exponential<double, linalg_backend::native>::apply(const double* A,
    linalg_int                                                                       n,
    linalg_int                                                                       lda,
    double*                                                                          result,
    linalg_int                                                                       ldc,
    void*                                                                            workspace,
    size_t                                                                           workspace_size)
{
    LOGGING_CHECK(A != nullptr && result != nullptr,
        "exponential::apply: A and result must not be null");
    LOGGING_CHECK(n > 0 && lda > 0 && ldc > 0,
        "exponential::apply: n/lda/ldc must be positive");
    matrix_exponential_impl(A, n, lda, result, ldc, workspace, workspace_size);
}

template <>
size_t exponential<float, linalg_backend::native>::workspace_size(linalg_int n)
{
    return matrix_exponential_workspace_size_impl<float>(n);
}

template <>
size_t exponential<double, linalg_backend::native>::workspace_size(linalg_int n)
{
    return matrix_exponential_workspace_size_impl<double>(n);
}

}  // namespace linalg
