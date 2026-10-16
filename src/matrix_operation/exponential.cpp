#include "include/matrix_operation/exponential.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

#include "include/common/linalg_checks.h"
#include "include/common/workspace_arena.h"
#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/norm.h"
#if LINALG_HAS_VECTORIZATION
#include "src/common/packet.h"
#endif

namespace linalg
{
namespace
{

// Degree-{1,2,4,8,12} direct Taylor evaluation thresholds and the degree-18
// scale-and-square threshold, from A. H. Al-Mohy / Bader, Blanes & Casas,
// "Computing the Matrix Exponential with an Optimized Taylor Polynomial
// Approximation", SIAM J. Sci. Comput. 41(1), 2019 — the algorithm PyTorch's
// native matrix_exp (aten/src/ATen/native/LinearAlgebra.cpp, mexp_impl /
// compute_T1..compute_T18) is built on. Unlike the classic Higham (2005)
// diagonal-Pade scaling-and-squaring used by SciPy's expm, this algorithm is
// a pure polynomial (Taylor, Paterson-Stockmeyer) evaluation and never
// inverts a matrix.
template <typename T> struct mexp_theta;

template <> struct mexp_theta<float>
{
    static constexpr std::array<float, 6> values = {1.192092800768788e-07F,
        5.978858893805233e-04F,
        5.116619363445086e-02F,
        5.800524627688768e-01F,
        1.461661507209034e+00F,
        3.010066362817634e+00F};
};

template <> struct mexp_theta<double>
{
    static constexpr std::array<double, 6> values = {2.220446049250313e-16,
        2.580956802971767e-08,
        3.397168839976962e-04,
        4.991228871115323e-02,
        2.996158913811580e-01,
        1.090863719290036e+00};
};

// dst[0..count) = src[0..count). Uses linalg::packet<T> (src/common/packet.h), the same
// Eigen-derived SIMD abstraction multiplication.cpp's native GEMM kernel is built on, instead
// of hoping the compiler auto-vectorizes a scalar loop.
template <typename T> void copy_n(const T* src, T* dst, linalg_int count)
{
#if LINALG_HAS_VECTORIZATION
    using P                    = packet<T>;
    constexpr linalg_int width = P::size;
    linalg_int           i     = 0;
    for (; i + width <= count; i += width)
    {
        P::copy(src + i, dst + i);
    }
    for (; i < count; ++i)
    {
        dst[i] = src[i];
    }
#else
    std::copy(src, src + count, dst);
#endif
}

// dst[0..count) = 0.
template <typename T> void zero_n(T* dst, linalg_int count)
{
#if LINALG_HAS_VECTORIZATION
    using P                    = packet<T>;
    constexpr linalg_int width = P::size;
    typename P::packet_t z;
    P::set(T(0), z);
    linalg_int i = 0;
    for (; i + width <= count; i += width)
    {
        P::storeu(z, dst + i);
    }
    for (; i < count; ++i)
    {
        dst[i] = T(0);
    }
#else
    std::fill(dst, dst + count, T(0));
#endif
}

// dst[0..count) *= c. packet<T> exposes fma/add but no standalone multiply, so this uses fma
// against a zero accumulator (d*c + 0 == d*c) rather than introducing a multiply-only
// primitive for this one call site.
template <typename T> void scale_n(T* dst, linalg_int count, T c)
{
#if LINALG_HAS_VECTORIZATION
    using P                    = packet<T>;
    constexpr linalg_int width = P::size;
    typename P::packet_t cv;
    typename P::packet_t zero_v;
    P::set(c, cv);
    P::set(T(0), zero_v);
    linalg_int i = 0;
    for (; i + width <= count; i += width)
    {
        typename P::packet_t d;
        P::loadu(dst + i, d);
        P::fma(cv, d, zero_v, d);
        P::storeu(d, dst + i);
    }
    for (; i < count; ++i)
    {
        dst[i] *= c;
    }
#else
    for (linalg_int i = 0; i < count; ++i)
    {
        dst[i] *= c;
    }
#endif
}

// dst[0..count) += c * src[0..count).
template <typename T> void add_scaled_n(T* dst, const T* src, linalg_int count, T c)
{
#if LINALG_HAS_VECTORIZATION
    using P                    = packet<T>;
    constexpr linalg_int width = P::size;
    typename P::packet_t cv;
    P::set(c, cv);
    linalg_int i = 0;
    for (; i + width <= count; i += width)
    {
        typename P::packet_t d;
        typename P::packet_t s;
        P::loadu(dst + i, d);
        P::loadu(src + i, s);
        P::fma(cv, s, d, d);
        P::storeu(d, dst + i);
    }
    for (; i < count; ++i)
    {
        dst[i] += c * src[i];
    }
#else
    for (linalg_int i = 0; i < count; ++i)
    {
        dst[i] += c * src[i];
    }
#endif
}

// dst = src, dst = 0, dst += c * src — n x n and tightly packed.
template <typename T> void copy(const T* src, T* dst, linalg_int n)
{
    copy_n(src, dst, n * n);
}
template <typename T> void zero(T* dst, linalg_int n)
{
    zero_n(dst, n * n);
}
template <typename T> void add_scaled(T* dst, const T* src, linalg_int n, T c)
{
    add_scaled_n(dst, src, n * n, c);
}

// dst += c * I, n x n and tightly packed. The diagonal is strided (n + 1 elements apart), so
// this vectorizes via packet<T>::gather/scatter — the same strided-access primitives
// multiplication.cpp's GEBP kernel uses for its column-strided store into C — rather than
// packet<T>::loadu/storeu, which require contiguous lanes.
template <typename T> void add_scaled_identity(T* dst, linalg_int n, T c)
{
#if LINALG_HAS_VECTORIZATION
    using P                     = packet<T>;
    constexpr linalg_int width  = P::size;
    const linalg_int     stride = n + 1;
    typename P::packet_t cv;
    P::set(c, cv);
    linalg_int i = 0;
    for (; i + width <= n; i += width)
    {
        typename P::packet_t d;
        P::gather(dst + i * stride, stride, d);
        P::add(d, cv, d);
        P::scatter(d, stride, dst + i * stride);
    }
    for (; i < n; ++i)
    {
        dst[i * n + i] += c;
    }
#else
    for (linalg_int i = 0; i < n; ++i)
    {
        dst[i * n + i] += c;
    }
#endif
}

// Copies the n x n leading block of a row-major matrix with row stride `stride_in` into a
// tightly packed n x n row-major buffer. Each row is contiguous in both layouts, so every row
// is copied with the same vectorized copy_n() the whole-matrix helpers above use.
template <typename T> void copy_in(const T* src, linalg_int stride_in, T* dst, linalg_int n)
{
    for (linalg_int i = 0; i < n; ++i)
    {
        copy_n(src + i * stride_in, dst + i * n, n);
    }
}

// Copies a tightly packed n x n row-major buffer into the n x n leading block of a row-major
// matrix with row stride `stride_out`.
template <typename T> void copy_out(const T* src, linalg_int n, T* dst, linalg_int stride_out)
{
    for (linalg_int i = 0; i < n; ++i)
    {
        copy_n(src + i * n, dst + i * stride_out, n);
    }
}

template <typename T>
void mat_mul(const T* a, const T* b, linalg_int n, T* c, void* gemm_ws, size_t gemm_bytes)
{
    linalg::multiplication<T, linalg::kMatmulActiveBackend>::apply(
        false, false, n, n, n, a, n, b, n, c, n, gemm_ws, gemm_bytes);
}

// result = I + A.
template <typename T> void compute_T1(const T* A, linalg_int n, T* result)
{
    copy(A, result, n);
    add_scaled_identity(result, n, T(1));
}

// result = I + A + A2 / 2.
template <typename T> void compute_T2(const T* A, const T* A2, linalg_int n, T* result)
{
    copy(A, result, n);
    add_scaled(result, A2, n, T(0.5));
    add_scaled_identity(result, n, T(1));
}

// result = I + A + A2 * (I / 2 + A / 6 + A2 / 24).
template <typename T>
void compute_T4(
    const T* A, const T* A2, linalg_int n, T* result, T* tmp, void* gemm_ws, size_t gemm_bytes)
{
    zero(tmp, n);
    add_scaled_identity(tmp, n, T(1. / 2.));
    add_scaled(tmp, A, n, T(1. / 6.));
    add_scaled(tmp, A2, n, T(1. / 24.));

    mat_mul(A2, tmp, n, result, gemm_ws, gemm_bytes);
    add_scaled(result, A, n, T(1));
    add_scaled_identity(result, n, T(1));
}

// result = I + A + y2 * A2 + A8, where
// A4 = A2 * (x1*A + x2*A2),
// A8 = (x3*A2 + A4) * (x4*I + x5*A + x6*A2 + x7*A4).
template <typename T>
void compute_T8(const T* A,
    const T*             A2,
    linalg_int           n,
    T*                   result,
    T*                   tmp1,
    T*                   tmp2,
    T*                   tmp3,
    void*                gemm_ws,
    size_t               gemm_bytes)
{
    constexpr T sqrt_177 = T(0.1330413469565007072504e+2);
    constexpr T x3       = T(2.) / T(3.);
    constexpr T x1       = x3 * ((T(1.) + sqrt_177) / T(88.));
    constexpr T x2       = x3 * ((T(1.) + sqrt_177) / T(352.));
    constexpr T x4       = (T(-271.) + T(29.) * sqrt_177) / (T(315.) * x3);
    constexpr T x5       = (T(-11.) + T(11.) * sqrt_177) / (T(1260.) * x3);
    constexpr T x6       = (T(-99.) + T(11.) * sqrt_177) / (T(5040.) * x3);
    constexpr T x7       = (T(89.) - sqrt_177) / (T(5040.) * x3 * x3);
    constexpr T y2       = (T(857.) - T(58.) * sqrt_177) / T(630.);

    T* A4 = tmp2;
    zero(tmp1, n);
    add_scaled(tmp1, A, n, x1);
    add_scaled(tmp1, A2, n, x2);
    mat_mul(A2, tmp1, n, A4, gemm_ws, gemm_bytes);

    zero(tmp1, n);
    add_scaled(tmp1, A2, n, x3);
    add_scaled(tmp1, A4, n, T(1));

    zero(tmp3, n);
    add_scaled_identity(tmp3, n, x4);
    add_scaled(tmp3, A, n, x5);
    add_scaled(tmp3, A2, n, x6);
    add_scaled(tmp3, A4, n, x7);

    mat_mul(tmp1, tmp3, n, result, gemm_ws, gemm_bytes);
    add_scaled(result, A, n, T(1));
    add_scaled(result, A2, n, y2);
    add_scaled_identity(result, n, T(1));
}

// result = Bs0 + (Bs1 + Bs2) * Bs2, where Bs_k = sum_i b[k][i] * {I,A,A2,A3}[i]
// and Bs2 is updated in place with += Bs3 * Bs3 before that final product —
// the Paterson-Stockmeyer evaluation of the degree-12 Taylor approximant
// from Bader/Blanes/Casas, Table 1.
template <typename T>
void compute_T12(const T* A,
    const T*              A2,
    const T*              A3,
    linalg_int            n,
    T*                    result,
    T*                    Bs0,
    T*                    Bs1,
    T*                    Bs2,
    T*                    Bs3,
    T*                    tmp,
    void*                 gemm_ws,
    size_t                gemm_bytes)
{
    constexpr std::array<std::array<T, 4>, 4> b = {
        std::array<T, 4>{T(9.0198e-16),
            T(0.46932117595418237389),
            T(-0.20099424927047284052),
            T(-0.04623946134063071740)},
        std::array<T, 4>{T(5.31597895759871264183),
            T(1.19926790417132231573),
            T(0.01179296240992997031),
            T(0.01108844528519167989)},
        std::array<T, 4>{T(0.18188869982170434744),
            T(0.05502798439925399070),
            T(0.09351590770535414968),
            T(0.00610700528898058230)},
        std::array<T, 4>{T(-2.0861320e-13),
            T(-0.13181061013830184015),
            T(-0.02027855540589259079),
            T(-0.00675951846863086359)},
    };

    const std::array<T*, 4>       Bs     = {Bs0, Bs1, Bs2, Bs3};
    const std::array<const T*, 4> powers = {nullptr, A, A2, A3};
    for (int k = 0; k < 4; ++k)
    {
        zero(Bs[k], n);
        add_scaled_identity(Bs[k], n, b[k][0]);
        add_scaled(Bs[k], powers[1], n, b[k][1]);
        add_scaled(Bs[k], powers[2], n, b[k][2]);
        add_scaled(Bs[k], powers[3], n, b[k][3]);
    }

    mat_mul(Bs3, Bs3, n, tmp, gemm_ws, gemm_bytes);
    add_scaled(Bs2, tmp, n, T(1));
    add_scaled(Bs1, Bs2, n, T(1));
    mat_mul(Bs1, Bs2, n, tmp, gemm_ws, gemm_bytes);
    add_scaled(Bs0, tmp, n, T(1));

    copy(Bs0, result, n);
}

// result = Bs1 + (Bs2 + Bs3) * Bs3, where Bs_k = sum_i b[k][i] * {I,A,A2,A3,A6}[i]
// and Bs3 is updated in place with += Bs0 * Bs4 before Bs2 += Bs3 and the
// final product — the Paterson-Stockmeyer evaluation of the degree-18
// Taylor approximant from Bader/Blanes/Casas, Table 1.
template <typename T>
void compute_T18(const T* A,
    const T*              A2,
    const T*              A3,
    const T*              A6,
    linalg_int            n,
    T*                    result,
    T*                    Bs0,
    T*                    Bs1,
    T*                    Bs2,
    T*                    Bs3,
    T*                    Bs4,
    T*                    tmp,
    void*                 gemm_ws,
    size_t                gemm_bytes)
{
    constexpr std::array<std::array<T, 5>, 5> b = {
        std::array<T, 5>{T(0.),
            T(-1.00365581030144618291e-01),
            T(-8.02924648241156932449e-03),
            T(-8.92138498045729985177e-04),
            T(0.)},
        std::array<T, 5>{T(0.),
            T(3.97849749499645077844e-01),
            T(1.36783778460411720168e+00),
            T(4.98289622525382669416e-01),
            T(-6.37898194594723280150e-04)},
        std::array<T, 5>{T(-1.09676396052962061844e+01),
            T(1.68015813878906206114e+00),
            T(5.71779846478865511061e-02),
            T(-6.98210122488052056106e-03),
            T(3.34975017086070470649e-05)},
        std::array<T, 5>{T(-9.04316832390810593223e-02),
            T(-6.76404519071381882256e-02),
            T(6.75961301770459654925e-02),
            T(2.95552570429315521194e-02),
            T(-1.39180257516060693404e-05)},
        std::array<T, 5>{T(0.),
            T(0.),
            T(-9.23364619367118555360e-02),
            T(-1.69364939002081722752e-02),
            T(-1.40086798182036094347e-05)},
    };

    const std::array<T*, 5>       Bs     = {Bs0, Bs1, Bs2, Bs3, Bs4};
    const std::array<const T*, 5> powers = {nullptr, A, A2, A3, A6};
    for (int k = 0; k < 5; ++k)
    {
        zero(Bs[k], n);
        add_scaled_identity(Bs[k], n, b[k][0]);
        add_scaled(Bs[k], powers[1], n, b[k][1]);
        add_scaled(Bs[k], powers[2], n, b[k][2]);
        add_scaled(Bs[k], powers[3], n, b[k][3]);
        add_scaled(Bs[k], powers[4], n, b[k][4]);
    }

    mat_mul(Bs0, Bs4, n, tmp, gemm_ws, gemm_bytes);
    add_scaled(Bs3, tmp, n, T(1));
    add_scaled(Bs2, Bs3, n, T(1));
    mat_mul(Bs2, Bs3, n, tmp, gemm_ws, gemm_bytes);
    add_scaled(Bs1, tmp, n, T(1));

    copy(Bs1, result, n);
}

template <typename T> struct exponential_layout
{
    T*             A;
    T*             A2;
    T*             A3;
    T*             A6;
    T*             Bs0;
    T*             Bs1;
    T*             Bs2;
    T*             Bs3;
    T*             Bs4;
    T*             tmp;
    T*             R;
    T*             R_tmp;
    unsigned char* gemm_ws;
    size_t         gemm_bytes;
};

template <typename T>
exponential_layout<T> exponential_plan(detail::workspace_arena& arena, linalg_int n)
{
    const size_t          nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    exponential_layout<T> p;
    p.A          = arena.next<T>(nn);
    p.A2         = arena.next<T>(nn);
    p.A3         = arena.next<T>(nn);
    p.A6         = arena.next<T>(nn);
    p.Bs0        = arena.next<T>(nn);
    p.Bs1        = arena.next<T>(nn);
    p.Bs2        = arena.next<T>(nn);
    p.Bs3        = arena.next<T>(nn);
    p.Bs4        = arena.next<T>(nn);
    p.tmp        = arena.next<T>(nn);
    p.R          = arena.next<T>(nn);
    p.R_tmp      = arena.next<T>(nn);
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
    LINALG_CHECK(workspace_size >= matrix_exponential_workspace_size_impl<T>(n),
        "exponential: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              bufs = exponential_plan<T>(arena, n);
    T*                      A    = bufs.A;

    copy_in(A_in, lda, A, n);

    const double norm1 =
        static_cast<double>(linalg::norm<T, linalg::linalg_backend::native>::apply(A,
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            linalg::norm_type::ONE,
            nullptr,
            0));

    const auto& thetas = mexp_theta<T>::values;

    if (norm1 <= static_cast<double>(thetas[0]))
    {
        compute_T1(A, n, bufs.R);
    }
    else if (norm1 <= static_cast<double>(thetas[1]))
    {
        mat_mul(A, A, n, bufs.A2, bufs.gemm_ws, bufs.gemm_bytes);
        compute_T2(A, bufs.A2, n, bufs.R);
    }
    else if (norm1 <= static_cast<double>(thetas[2]))
    {
        mat_mul(A, A, n, bufs.A2, bufs.gemm_ws, bufs.gemm_bytes);
        compute_T4(A, bufs.A2, n, bufs.R, bufs.tmp, bufs.gemm_ws, bufs.gemm_bytes);
    }
    else if (norm1 <= static_cast<double>(thetas[3]))
    {
        mat_mul(A, A, n, bufs.A2, bufs.gemm_ws, bufs.gemm_bytes);
        compute_T8(
            A, bufs.A2, n, bufs.R, bufs.tmp, bufs.Bs0, bufs.Bs1, bufs.gemm_ws, bufs.gemm_bytes);
    }
    else if (norm1 <= static_cast<double>(thetas[4]))
    {
        mat_mul(A, A, n, bufs.A2, bufs.gemm_ws, bufs.gemm_bytes);
        mat_mul(A, bufs.A2, n, bufs.A3, bufs.gemm_ws, bufs.gemm_bytes);
        compute_T12(A,
            bufs.A2,
            bufs.A3,
            n,
            bufs.R,
            bufs.Bs0,
            bufs.Bs1,
            bufs.Bs2,
            bufs.Bs3,
            bufs.tmp,
            bufs.gemm_ws,
            bufs.gemm_bytes);
    }
    else
    {
        int s = static_cast<int>(std::ceil(std::log2(norm1 / static_cast<double>(thetas[5]))));
        s     = std::max(s, 0);

        if (s > 0)
        {
            const T scale = static_cast<T>(std::ldexp(1.0, -s));
            scale_n(A, n * n, scale);
        }

        mat_mul(A, A, n, bufs.A2, bufs.gemm_ws, bufs.gemm_bytes);
        mat_mul(A, bufs.A2, n, bufs.A3, bufs.gemm_ws, bufs.gemm_bytes);
        mat_mul(bufs.A3, bufs.A3, n, bufs.A6, bufs.gemm_ws, bufs.gemm_bytes);
        compute_T18(A,
            bufs.A2,
            bufs.A3,
            bufs.A6,
            n,
            bufs.R,
            bufs.Bs0,
            bufs.Bs1,
            bufs.Bs2,
            bufs.Bs3,
            bufs.Bs4,
            bufs.tmp,
            bufs.gemm_ws,
            bufs.gemm_bytes);

        T* R     = bufs.R;
        T* R_tmp = bufs.R_tmp;
        for (int k = 0; k < s; ++k)
        {
            mat_mul(R, R, n, R_tmp, bufs.gemm_ws, bufs.gemm_bytes);
            std::swap(R, R_tmp);
        }
        if (R != bufs.R)
        {
            copy(R, bufs.R, n);
        }
    }

    copy_out(bufs.R, n, result, ldc);
}

}  // namespace

template <>
void exponential<float, linalg_backend::native>::apply(const float* A,
    linalg_int                                                      n,
    linalg_int                                                      lda,
    float*                                                          result,
    linalg_int                                                      ldc,
    void*                                                           workspace,
    size_t                                                          workspace_size)
{
    LINALG_CHECK(
        A != nullptr && result != nullptr, "exponential::apply: A and result must not be null");
    LINALG_CHECK(n > 0 && lda > 0 && ldc > 0, "exponential::apply: n/lda/ldc must be positive");
    matrix_exponential_impl(A, n, lda, result, ldc, workspace, workspace_size);
}

template <>
void exponential<double, linalg_backend::native>::apply(const double* A,
    linalg_int                                                        n,
    linalg_int                                                        lda,
    double*                                                           result,
    linalg_int                                                        ldc,
    void*                                                             workspace,
    size_t                                                            workspace_size)
{
    LINALG_CHECK(
        A != nullptr && result != nullptr, "exponential::apply: A and result must not be null");
    LINALG_CHECK(n > 0 && lda > 0 && ldc > 0, "exponential::apply: n/lda/ldc must be positive");
    matrix_exponential_impl(A, n, lda, result, ldc, workspace, workspace_size);
}

template <> size_t exponential<float, linalg_backend::native>::workspace_size(linalg_int n)
{
    return matrix_exponential_workspace_size_impl<float>(n);
}

template <> size_t exponential<double, linalg_backend::native>::workspace_size(linalg_int n)
{
    return matrix_exponential_workspace_size_impl<double>(n);
}

}  // namespace linalg
