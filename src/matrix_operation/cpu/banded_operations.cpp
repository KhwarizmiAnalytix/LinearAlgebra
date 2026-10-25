#include "include/matrix_operation/banded_operations.h"

#include "include/common/linalg_macros.h"
#include "include/common/linalg_profiler.h"
#include "include/common/types.h"

#if LINALG_HAS_VECTORIZATION
#include <backend/simd.h>
#endif

namespace linalg::banded_operations
{

//=============================================================================
// decomposition<T, BandCount>
//=============================================================================
template <typename T, int BandCount>
void decomposition(std::vector<T>& output, size_t outer_dim, size_t dim, size_t inner_dim)
{
    static_assert(BandCount == 3 || BandCount == 5, "BandCount must be 3 or 5");

    const auto n      = dim * inner_dim;
    const auto stride = outer_dim * n;

    if (output.size() != static_cast<size_t>(BandCount) * stride)
    {
        throw std::invalid_argument("banded decomposition output has the wrong size!");
    }

    if constexpr (BandCount == 3)
    {
        LINALG_SCOPE("banded/decomposition/3");

#if LINALG_HAS_VECTORIZATION
        using P                = simd<T>;
        constexpr size_t width = static_cast<size_t>(P::size);
        // Same row-vs-lag reasoning as solve_decomposed above: the boundary row has no
        // recurrence at all (always safe to pack), but the main loop's lag is exactly
        // inner_dim, so packing it is only safe when width <= inner_dim. When inner_dim is
        // too narrow for that but outer_dim is wide enough, pack across outer_dim instead
        // (gather/scatter, stride n) -- see solve_decomposed and the _aad functions below.
        const bool       can_vectorize_inner = width <= inner_dim;
        const bool       can_vectorize_outer = !can_vectorize_inner && width <= outer_dim;
        const linalg_int gstride             = static_cast<linalg_int>(n);

        typename P::simd_t vzero, vone, vnegone;
        vzero   = P::set(T{0});
        vone    = P::set(T{1});
        vnegone = P::set(T{-1});
#endif

#if LINALG_HAS_VECTORIZATION
        if (can_vectorize_outer)
        {
            auto* __restrict band0 = output.data();
            auto* __restrict band1 = band0 + stride;
            auto* __restrict band2 = band1 + stride;

            for (size_t row = 0; row < inner_dim; ++row)
            {
                size_t l = 0;
                for (; l + width <= outer_dim; l += width)
                {
                    const size_t base = l * n + row;

                    typename P::simd_t v1, v2, vinv, vout;
                    v1   = P::gather(band1 + base, gstride);
                    v2   = P::gather(band2 + base, gstride);
                    vinv = P::div(vone, v1);
                    P::scatter(vinv, gstride, band0 + base);
                    P::scatter(vzero, gstride, band1 + base);
                    vout = P::fma(v2, vinv, vzero);
                    P::scatter(vout, gstride, band2 + base);
                }
                for (; l < outer_dim; ++l)
                {
                    const size_t base     = l * n + row;
                    const T      inv_diag = T{1} / band1[base];

                    band0[base] = inv_diag;
                    band1[base] = T{0};
                    band2[base] *= inv_diag;
                }
            }

            for (size_t row = inner_dim; row < n; ++row)
            {
                size_t l = 0;
                for (; l + width <= outer_dim; l += width)
                {
                    const size_t base = l * n + row;

                    typename P::simd_t v0, v1, v2, vlag, vneg0, vsum, vinv, vout;
                    v0    = P::gather(band0 + base, gstride);
                    v1    = P::gather(band1 + base, gstride);
                    v2    = P::gather(band2 + base, gstride);
                    vlag  = P::gather(band2 + base - inner_dim, gstride);
                    vneg0 = P::fma(v0, vnegone, vzero);
                    vsum  = P::fma(vneg0, vlag, v1);
                    vinv  = P::div(vone, vsum);
                    P::scatter(vinv, gstride, band0 + base);
                    vout = P::fma(vneg0, vinv, vzero);
                    P::scatter(vout, gstride, band1 + base);
                    vout = P::fma(v2, vinv, vzero);
                    P::scatter(vout, gstride, band2 + base);
                }
                for (; l < outer_dim; ++l)
                {
                    const size_t base     = l * n + row;
                    const auto   tmp      = band2[base - inner_dim];
                    const auto   alpha    = -band0[base];
                    const T      inv_diag = T{1} / (band1[base] + alpha * tmp);

                    band0[base] = inv_diag;
                    band1[base] = alpha * inv_diag;
                    band2[base] *= inv_diag;
                }
            }
        }
        else
#endif
        {
            for (size_t l = 0; l < outer_dim; ++l)
            {
                const auto offset = l * n;

                auto* __restrict itr0 = output.data() + offset;
                auto* __restrict itr1 = itr0 + stride;
                auto* __restrict itr2 = itr1 + stride;

                size_t i = 0;
#if LINALG_HAS_VECTORIZATION
                for (; i + width <= inner_dim;
                    i += width, itr0 += width, itr1 += width, itr2 += width)
                {
                    typename P::simd_t v1, v2, vinv, vout;
                    v1   = P::loadu(itr1);
                    v2   = P::loadu(itr2);
                    vinv = P::div(vone, v1);
                    P::storeu(vinv, itr0);
                    P::storeu(vzero, itr1);
                    vout = P::fma(v2, vinv, vzero);
                    P::storeu(vout, itr2);
                }
#endif
                for (; i < inner_dim; ++i)
                {
                    const T inv_diag = T{1} / (*itr1);

                    *itr0++ = inv_diag;
                    *itr1++ = T{0};
                    *itr2++ *= inv_diag;
                }

#if LINALG_HAS_VECTORIZATION
                if (can_vectorize_inner)
                {
                    for (; i + width <= n; i += width, itr0 += width, itr1 += width, itr2 += width)
                    {
                        typename P::simd_t v0, v1, v2, vlag, vneg0, vsum, vinv, vout;
                        v0    = P::loadu(itr0);
                        v1    = P::loadu(itr1);
                        v2    = P::loadu(itr2);
                        vlag  = P::loadu(itr2 - inner_dim);
                        vneg0 = P::fma(v0, vnegone, vzero);
                        vsum  = P::fma(vneg0, vlag, v1);
                        vinv  = P::div(vone, vsum);
                        P::storeu(vinv, itr0);
                        vout = P::fma(vneg0, vinv, vzero);
                        P::storeu(vout, itr1);
                        vout = P::fma(v2, vinv, vzero);
                        P::storeu(vout, itr2);
                    }
                }
#endif
                for (; i < n; ++i)
                {
                    const auto tmp      = *(itr2 - inner_dim);
                    const auto alpha    = -*itr0;
                    const T    inv_diag = T{1} / (*itr1 + alpha * tmp);

                    *itr0++ = inv_diag;
                    *itr1++ = alpha * inv_diag;
                    *itr2++ *= inv_diag;
                }
            }
        }

        size_t i = 0;
#if LINALG_HAS_VECTORIZATION
        for (; i + width <= stride; i += width)
        {
            typename P::simd_t v, vout;
            v    = P::loadu(output.data() + 2 * stride + i);
            vout = P::fma(v, vnegone, vzero);
            P::storeu(vout, output.data() + 2 * stride + i);
        }
#endif
        for (; i < stride; ++i)
        {
            output[2 * stride + i] = -output[2 * stride + i];
        }
    }
    else  // BandCount == 5
    {
        LINALG_SCOPE("banded/decomposition/5");

        const auto two_inner_dim = 2 * inner_dim;

#if LINALG_HAS_VECTORIZATION
        using P                = simd<T>;
        constexpr size_t width = static_cast<size_t>(P::size);
        // Phase 1 has no recurrence (always safe). Phases 2 and 3 read lags of inner_dim and
        // two_inner_dim, both >= inner_dim, so width <= inner_dim is the safety condition for
        // both, same reasoning as solve_decomposed<T,5> above. When inner_dim is too narrow
        // for that but outer_dim is wide enough, pack across outer_dim instead.
        const bool       can_vectorize_inner = width <= inner_dim;
        const bool       can_vectorize_outer = !can_vectorize_inner && width <= outer_dim;
        const linalg_int gstride             = static_cast<linalg_int>(n);

        typename P::simd_t vzero, vone, vnegone;
        vzero   = P::set(T{0});
        vone    = P::set(T{1});
        vnegone = P::set(T{-1});
#endif

#if LINALG_HAS_VECTORIZATION
        if (can_vectorize_outer)
        {
            auto* __restrict band0 = output.data();
            auto* __restrict band1 = band0 + stride;
            auto* __restrict band2 = band1 + stride;
            auto* __restrict band3 = band2 + stride;
            auto* __restrict band4 = band3 + stride;

            for (size_t row = 0; row < inner_dim; ++row)
            {
                size_t l = 0;
                for (; l + width <= outer_dim; l += width)
                {
                    const size_t base = l * n + row;

                    typename P::simd_t ve, vc, vd, va, vb, vmu, vnegc, vnege, vout;
                    ve = P::gather(band0 + base, gstride);
                    vc = P::gather(band1 + base, gstride);
                    vd = P::gather(band2 + base, gstride);
                    va = P::gather(band3 + base, gstride);
                    vb = P::gather(band4 + base, gstride);

                    vmu = P::div(vone, vd);
                    P::scatter(vmu, gstride, band0 + base);

                    vnegc = P::fma(vc, vnegone, vzero);
                    vout  = P::fma(vnegc, vmu, vzero);
                    P::scatter(vout, gstride, band1 + base);

                    vnege = P::fma(ve, vnegone, vzero);
                    vout  = P::fma(vnege, vmu, vzero);
                    P::scatter(vout, gstride, band2 + base);

                    vout = P::fma(va, vmu, vzero);
                    P::scatter(vout, gstride, band3 + base);

                    vout = P::fma(vb, vmu, vzero);
                    P::scatter(vout, gstride, band4 + base);
                }
                for (; l < outer_dim; ++l)
                {
                    const size_t base = l * n + row;

                    const auto e = band0[base];
                    const auto c = band1[base];
                    const auto d = band2[base];
                    const auto a = band3[base];
                    const auto b = band4[base];

                    const auto gamma = c;
                    const T    mu    = T{1} / d;

                    band0[base] = mu;
                    band1[base] = -gamma * mu;
                    band2[base] = -e * mu;
                    band3[base] = a * mu;
                    band4[base] = b * mu;
                }
            }

            for (size_t row = inner_dim; row < two_inner_dim; ++row)
            {
                size_t l = 0;
                for (; l + width <= outer_dim; l += width)
                {
                    const size_t base = l * n + row;

                    typename P::simd_t ve, vc, vd, va, vb, valpha, vbeta, vnegalpha, vdenom, vmu,
                        vnegc, vnege, vnegbeta, vt3, vout;
                    ve     = P::gather(band0 + base, gstride);
                    vc     = P::gather(band1 + base, gstride);
                    vd     = P::gather(band2 + base, gstride);
                    va     = P::gather(band3 + base, gstride);
                    vb     = P::gather(band4 + base, gstride);
                    valpha = P::gather(band3 + base - inner_dim, gstride);
                    vbeta  = P::gather(band4 + base - inner_dim, gstride);

                    vnegalpha = P::fma(valpha, vnegone, vzero);
                    vdenom    = P::fma(vnegalpha, vc, vd);
                    vmu       = P::div(vone, vdenom);
                    P::scatter(vmu, gstride, band0 + base);

                    vnegc = P::fma(vc, vnegone, vzero);
                    vout  = P::fma(vnegc, vmu, vzero);
                    P::scatter(vout, gstride, band1 + base);

                    vnege = P::fma(ve, vnegone, vzero);
                    vout  = P::fma(vnege, vmu, vzero);
                    P::scatter(vout, gstride, band2 + base);

                    vnegbeta = P::fma(vbeta, vnegone, vzero);
                    vt3      = P::fma(vnegbeta, vc, va);
                    vout     = P::fma(vt3, vmu, vzero);
                    P::scatter(vout, gstride, band3 + base);

                    vout = P::fma(vb, vmu, vzero);
                    P::scatter(vout, gstride, band4 + base);
                }
                for (; l < outer_dim; ++l)
                {
                    const size_t base = l * n + row;

                    const auto e = band0[base];
                    const auto c = band1[base];
                    const auto d = band2[base];
                    const auto a = band3[base];
                    const auto b = band4[base];

                    const auto alpha = band3[base - inner_dim];
                    const auto beta  = band4[base - inner_dim];

                    const auto gamma = c;
                    const T    mu    = T{1} / (d - alpha * gamma);

                    band0[base] = mu;
                    band1[base] = -gamma * mu;
                    band2[base] = -e * mu;
                    band3[base] = (a - beta * gamma) * mu;
                    band4[base] = b * mu;
                }
            }

            for (size_t row = two_inner_dim; row < n; ++row)
            {
                size_t l = 0;
                for (; l + width <= outer_dim; l += width)
                {
                    const size_t base = l * n + row;

                    typename P::simd_t ve, vc, vd, va, vb, valpha, vbeta, valpha_prev, vbeta_prev,
                        vneg_alpha_prev, vgamma, vneg_alpha, vt1, vneg_beta_prev, vdenom, vmu,
                        vneg_gamma, vnege, vneg_beta, vt3, vout;
                    ve          = P::gather(band0 + base, gstride);
                    vc          = P::gather(band1 + base, gstride);
                    vd          = P::gather(band2 + base, gstride);
                    va          = P::gather(band3 + base, gstride);
                    vb          = P::gather(band4 + base, gstride);
                    valpha      = P::gather(band3 + base - inner_dim, gstride);
                    vbeta       = P::gather(band4 + base - inner_dim, gstride);
                    valpha_prev = P::gather(band3 + base - two_inner_dim, gstride);
                    vbeta_prev  = P::gather(band4 + base - two_inner_dim, gstride);

                    // gamma = c - alpha_prev * e
                    vneg_alpha_prev = P::fma(valpha_prev, vnegone, vzero);
                    vgamma          = P::fma(vneg_alpha_prev, ve, vc);

                    // mu = 1 / (d - alpha * gamma - beta_prev * e)
                    vneg_alpha     = P::fma(valpha, vnegone, vzero);
                    vt1            = P::fma(vneg_alpha, vgamma, vd);
                    vneg_beta_prev = P::fma(vbeta_prev, vnegone, vzero);
                    vdenom         = P::fma(vneg_beta_prev, ve, vt1);
                    vmu            = P::div(vone, vdenom);
                    P::scatter(vmu, gstride, band0 + base);

                    vneg_gamma = P::fma(vgamma, vnegone, vzero);
                    vout       = P::fma(vneg_gamma, vmu, vzero);
                    P::scatter(vout, gstride, band1 + base);

                    vnege = P::fma(ve, vnegone, vzero);
                    vout  = P::fma(vnege, vmu, vzero);
                    P::scatter(vout, gstride, band2 + base);

                    // (a - beta * gamma) * mu
                    vneg_beta = P::fma(vbeta, vnegone, vzero);
                    vt3       = P::fma(vneg_beta, vgamma, va);
                    vout      = P::fma(vt3, vmu, vzero);
                    P::scatter(vout, gstride, band3 + base);

                    vout = P::fma(vb, vmu, vzero);
                    P::scatter(vout, gstride, band4 + base);
                }
                for (; l < outer_dim; ++l)
                {
                    const size_t base = l * n + row;

                    const auto e = band0[base];
                    const auto c = band1[base];
                    const auto d = band2[base];
                    const auto a = band3[base];
                    const auto b = band4[base];

                    const auto alpha      = band3[base - inner_dim];
                    const auto beta       = band4[base - inner_dim];
                    const auto alpha_prev = band3[base - two_inner_dim];
                    const auto beta_prev  = band4[base - two_inner_dim];

                    const auto gamma = c - alpha_prev * e;
                    const T    mu    = T{1} / (d - alpha * gamma - beta_prev * e);

                    band0[base] = mu;
                    band1[base] = -gamma * mu;
                    band2[base] = -e * mu;
                    band3[base] = (a - beta * gamma) * mu;
                    band4[base] = b * mu;
                }
            }
        }
        else
#endif
        {
            for (size_t l = 0; l < outer_dim; ++l)
            {
                const auto offset = l * n;

                auto* __restrict itr0 = output.data() + offset;
                auto* __restrict itr1 = itr0 + stride;
                auto* __restrict itr2 = itr1 + stride;
                auto* __restrict itr3 = itr2 + stride;
                auto* __restrict itr4 = itr3 + stride;

                size_t i = 0;
#if LINALG_HAS_VECTORIZATION
                for (; i + width <= inner_dim; i += width,
                    itr0 += width,
                    itr1 += width,
                    itr2 += width,
                    itr3 += width,
                    itr4 += width)
                {
                    typename P::simd_t ve, vc, vd, va, vb, vmu, vnegc, vnege, vout;
                    ve = P::loadu(itr0);
                    vc = P::loadu(itr1);
                    vd = P::loadu(itr2);
                    va = P::loadu(itr3);
                    vb = P::loadu(itr4);

                    vmu = P::div(vone, vd);
                    P::storeu(vmu, itr0);

                    vnegc = P::fma(vc, vnegone, vzero);
                    vout  = P::fma(vnegc, vmu, vzero);
                    P::storeu(vout, itr1);

                    vnege = P::fma(ve, vnegone, vzero);
                    vout  = P::fma(vnege, vmu, vzero);
                    P::storeu(vout, itr2);

                    vout = P::fma(va, vmu, vzero);
                    P::storeu(vout, itr3);

                    vout = P::fma(vb, vmu, vzero);
                    P::storeu(vout, itr4);
                }
#endif
                for (; i < inner_dim; ++i)
                {
                    const auto e = *itr0;
                    const auto c = *itr1;
                    const auto d = *itr2;
                    const auto a = *itr3;
                    const auto b = *itr4;

                    const auto gamma = c;
                    const T    mu    = T{1} / d;

                    *itr0++ = mu;
                    *itr1++ = -gamma * mu;
                    *itr2++ = -e * mu;
                    *itr3++ = a * mu;
                    *itr4++ = b * mu;
                }

#if LINALG_HAS_VECTORIZATION
                if (can_vectorize_inner)
                {
                    for (; i + width <= two_inner_dim; i += width,
                        itr0 += width,
                        itr1 += width,
                        itr2 += width,
                        itr3 += width,
                        itr4 += width)
                    {
                        typename P::simd_t ve, vc, vd, va, vb, valpha, vbeta, vnegalpha, vdenom,
                            vmu, vnegc, vnege, vnegbeta, vt3, vout;
                        ve     = P::loadu(itr0);
                        vc     = P::loadu(itr1);
                        vd     = P::loadu(itr2);
                        va     = P::loadu(itr3);
                        vb     = P::loadu(itr4);
                        valpha = P::loadu(itr3 - inner_dim);
                        vbeta  = P::loadu(itr4 - inner_dim);

                        vnegalpha = P::fma(valpha, vnegone, vzero);
                        vdenom    = P::fma(vnegalpha, vc, vd);
                        vmu       = P::div(vone, vdenom);
                        P::storeu(vmu, itr0);

                        vnegc = P::fma(vc, vnegone, vzero);
                        vout  = P::fma(vnegc, vmu, vzero);
                        P::storeu(vout, itr1);

                        vnege = P::fma(ve, vnegone, vzero);
                        vout  = P::fma(vnege, vmu, vzero);
                        P::storeu(vout, itr2);

                        vnegbeta = P::fma(vbeta, vnegone, vzero);
                        vt3      = P::fma(vnegbeta, vc, va);
                        vout     = P::fma(vt3, vmu, vzero);
                        P::storeu(vout, itr3);

                        vout = P::fma(vb, vmu, vzero);
                        P::storeu(vout, itr4);
                    }
                }
#endif
                for (; i < two_inner_dim; ++i)
                {
                    const auto e = *itr0;
                    const auto c = *itr1;
                    const auto d = *itr2;
                    const auto a = *itr3;
                    const auto b = *itr4;

                    const auto alpha = *(itr3 - inner_dim);
                    const auto beta  = *(itr4 - inner_dim);

                    const auto gamma = c;
                    const T    mu    = T{1} / (d - alpha * gamma);

                    *itr0++ = mu;
                    *itr1++ = -gamma * mu;
                    *itr2++ = -e * mu;
                    *itr3++ = (a - beta * gamma) * mu;
                    *itr4++ = b * mu;
                }

#if LINALG_HAS_VECTORIZATION
                if (can_vectorize_inner)
                {
                    for (; i + width <= n; i += width,
                        itr0 += width,
                        itr1 += width,
                        itr2 += width,
                        itr3 += width,
                        itr4 += width)
                    {
                        typename P::simd_t ve, vc, vd, va, vb, valpha, vbeta, valpha_prev,
                            vbeta_prev, vneg_alpha_prev, vgamma, vneg_alpha, vt1, vneg_beta_prev,
                            vdenom, vmu, vneg_gamma, vnege, vneg_beta, vt3, vout;
                        ve          = P::loadu(itr0);
                        vc          = P::loadu(itr1);
                        vd          = P::loadu(itr2);
                        va          = P::loadu(itr3);
                        vb          = P::loadu(itr4);
                        valpha      = P::loadu(itr3 - inner_dim);
                        vbeta       = P::loadu(itr4 - inner_dim);
                        valpha_prev = P::loadu(itr3 - two_inner_dim);
                        vbeta_prev  = P::loadu(itr4 - two_inner_dim);

                        // gamma = c - alpha_prev * e
                        vneg_alpha_prev = P::fma(valpha_prev, vnegone, vzero);
                        vgamma          = P::fma(vneg_alpha_prev, ve, vc);

                        // mu = 1 / (d - alpha * gamma - beta_prev * e)
                        vneg_alpha     = P::fma(valpha, vnegone, vzero);
                        vt1            = P::fma(vneg_alpha, vgamma, vd);
                        vneg_beta_prev = P::fma(vbeta_prev, vnegone, vzero);
                        vdenom         = P::fma(vneg_beta_prev, ve, vt1);
                        vmu            = P::div(vone, vdenom);
                        P::storeu(vmu, itr0);

                        vneg_gamma = P::fma(vgamma, vnegone, vzero);
                        vout       = P::fma(vneg_gamma, vmu, vzero);
                        P::storeu(vout, itr1);

                        vnege = P::fma(ve, vnegone, vzero);
                        vout  = P::fma(vnege, vmu, vzero);
                        P::storeu(vout, itr2);

                        // (a - beta * gamma) * mu
                        vneg_beta = P::fma(vbeta, vnegone, vzero);
                        vt3       = P::fma(vneg_beta, vgamma, va);
                        vout      = P::fma(vt3, vmu, vzero);
                        P::storeu(vout, itr3);

                        vout = P::fma(vb, vmu, vzero);
                        P::storeu(vout, itr4);
                    }
                }
#endif
                for (; i < n; ++i)
                {
                    const auto e = *itr0;
                    const auto c = *itr1;
                    const auto d = *itr2;
                    const auto a = *itr3;
                    const auto b = *itr4;

                    const auto alpha      = *(itr3 - inner_dim);
                    const auto beta       = *(itr4 - inner_dim);
                    const auto alpha_prev = *(itr3 - 2 * inner_dim);
                    const auto beta_prev  = *(itr4 - 2 * inner_dim);

                    const auto gamma = c - alpha_prev * e;
                    const T    mu    = T{1} / (d - alpha * gamma - beta_prev * e);

                    *itr0++ = mu;
                    *itr1++ = -gamma * mu;
                    *itr2++ = -e * mu;
                    *itr3++ = (a - beta * gamma) * mu;
                    *itr4++ = b * mu;
                }
            }
        }
    }
}

//=============================================================================
// solve_decomposed<T, BandCount>
//=============================================================================
template <typename T, int BandCount>
void solve_decomposed(std::vector<T>& x,
    const std::vector<T>&             decomposed,
    size_t                            outer_dim,
    size_t                            dim,
    size_t                            inner_dim)
{
    static_assert(BandCount == 3 || BandCount == 5, "BandCount must be 3 or 5");

    if constexpr (BandCount == 3)
    {
        LINALG_SCOPE("banded/solve_decomposed/3");

        const auto n      = inner_dim * dim;
        const int  end    = static_cast<int>(n) - static_cast<int>(inner_dim) - 1;
        const auto stride = outer_dim * n;

        for (size_t i = 0; i < stride; ++i)
        {
            x[i] *= decomposed[0 * stride + i];
        }

        auto* __restrict itr_x      = x.data();
        auto const* __restrict itr1 = decomposed.data() + stride;
        auto const* __restrict itr2 = itr1 + stride;

#if LINALG_HAS_VECTORIZATION
        using P                = simd<T>;
        constexpr size_t width = static_cast<size_t>(P::size);
        // Lanes 0..inner_dim-1 within a row are independent interleaved systems (see the
        // layout note in banded_operations.h); the only recurrence is row-to-row, at a fixed
        // lag of inner_dim (forward sweep) or -inner_dim (backward sweep). A packet of `width`
        // consecutive elements only ever reads a row that is strictly outside its own write
        // range when width <= inner_dim, so that is the condition under which packing across
        // inner_dim is safe. When inner_dim is too narrow for that but outer_dim is wide
        // enough, pack across outer_dim instead (gather/scatter, stride n): different l groups
        // never alias, so that axis has no such constraint and the row recurrence just stays
        // serial, exactly like the _aad functions below.
        const bool       can_vectorize_inner = width <= inner_dim;
        const bool       can_vectorize_outer = !can_vectorize_inner && width <= outer_dim;
        const linalg_int gstride             = static_cast<linalg_int>(n);
#endif

#if LINALG_HAS_VECTORIZATION
        if (can_vectorize_outer)
        {
            size_t r = inner_dim;
            for (; r < n; ++r)
            {
                size_t l = 0;
                for (; l + width <= outer_dim; l += width)
                {
                    const size_t base = l * n + r;

                    typename P::simd_t va, vprev, vcur, vout;
                    va    = P::gather(itr1 + base, gstride);
                    vprev = P::gather(itr_x + base - inner_dim, gstride);
                    vcur  = P::gather(itr_x + base, gstride);
                    vout  = P::fma(va, vprev, vcur);
                    P::scatter(vout, gstride, itr_x + base);
                }
                for (; l < outer_dim; ++l)
                {
                    const size_t base = l * n + r;
                    itr_x[base] += itr1[base] * itr_x[base - inner_dim];
                }
            }

            int rj = end;
            for (; rj >= 0; --rj)
            {
                size_t l = 0;
                for (; l + width <= outer_dim; l += width)
                {
                    const size_t base = l * n + static_cast<size_t>(rj);

                    typename P::simd_t va, vnext, vcur, vout;
                    va    = P::gather(itr2 + base, gstride);
                    vnext = P::gather(itr_x + base + inner_dim, gstride);
                    vcur  = P::gather(itr_x + base, gstride);
                    vout  = P::fma(va, vnext, vcur);
                    P::scatter(vout, gstride, itr_x + base);
                }
                for (; l < outer_dim; ++l)
                {
                    const size_t base = l * n + static_cast<size_t>(rj);
                    itr_x[base] += itr2[base] * itr_x[base + inner_dim];
                }
            }
        }
        else
#endif
        {
            for (size_t l = 0; l < outer_dim; ++l)
            {
                const auto offset               = l * n;
                auto* __restrict tmp            = itr_x + offset;
                auto const* __restrict tmp_itr1 = itr1 + offset;
                auto const* __restrict tmp_itr2 = itr2 + offset;

                size_t i = inner_dim;
#if LINALG_HAS_VECTORIZATION
                if (can_vectorize_inner)
                {
                    for (; i + width <= n; i += width)
                    {
                        typename P::simd_t va, vprev, vcur, vout;
                        va    = P::loadu(tmp_itr1 + i);
                        vprev = P::loadu(tmp + i - inner_dim);
                        vcur  = P::loadu(tmp + i);
                        vout  = P::fma(va, vprev, vcur);
                        P::storeu(vout, tmp + i);
                    }
                }
#endif
                for (; i < n; ++i)
                {
                    tmp[i] += tmp_itr1[i] * tmp[i - inner_dim];
                }

                int j = end;
#if LINALG_HAS_VECTORIZATION
                if (can_vectorize_inner)
                {
                    for (; j - static_cast<int>(width) + 1 >= 0; j -= static_cast<int>(width))
                    {
                        const size_t base = static_cast<size_t>(j) - width + 1;

                        typename P::simd_t va, vnext, vcur, vout;
                        va    = P::loadu(tmp_itr2 + base);
                        vnext = P::loadu(tmp + base + inner_dim);
                        vcur  = P::loadu(tmp + base);
                        vout  = P::fma(va, vnext, vcur);
                        P::storeu(vout, tmp + base);
                    }
                }
#endif
                for (; j >= 0; --j)
                {
                    tmp[j] += tmp_itr2[j] * tmp[j + inner_dim];
                }
            }
        }
    }
    else  // BandCount == 5
    {
        LINALG_SCOPE("banded/solve_decomposed/5");

        const int n             = static_cast<int>(inner_dim * dim);
        const int two_inner_dim = static_cast<int>(2 * inner_dim);

        const int end_1 = static_cast<int>(n) - static_cast<int>(inner_dim) - 1;
        const int end_2 = static_cast<int>(n) - static_cast<int>(two_inner_dim) - 1;

        const auto stride = outer_dim * static_cast<size_t>(n);

        if (x.size() != stride)
        {
            throw std::invalid_argument("banded solve_decomposed x has the wrong size!");
        }

        for (size_t i = 0; i < stride; ++i)
        {
            x[i] *= decomposed[0 * stride + i];
        }

#if LINALG_HAS_VECTORIZATION
        using P             = simd<T>;
        constexpr int width = P::size;
        // Same safety condition as the 3-band case above: a packet is only safe across a
        // recurrence lag of inner_dim (or two_inner_dim, which is always >= inner_dim) when
        // width <= inner_dim. When inner_dim is too narrow for that but outer_dim is wide
        // enough, pack across outer_dim instead (gather/scatter, stride n) -- see the 3-band
        // case and the _aad functions below for the full rationale.
        const bool can_vectorize_inner = width <= static_cast<int>(inner_dim);
        const bool can_vectorize_outer =
            !can_vectorize_inner && static_cast<size_t>(width) <= outer_dim;
        const linalg_int gstride = static_cast<linalg_int>(n);

        typename P::simd_t vzero, vnegone;
        vzero   = P::set(T{0});
        vnegone = P::set(T{-1});
#endif

#if LINALG_HAS_VECTORIZATION
        if (can_vectorize_outer)
        {
            auto const* __restrict band1 = decomposed.data() + 1 * stride;
            auto const* __restrict band2 = band1 + stride;
            auto const* __restrict band3 = band2 + stride;
            auto const* __restrict band4 = band3 + stride;

            int idx = static_cast<int>(inner_dim);
            for (; idx < two_inner_dim; ++idx)
            {
                size_t l = 0;
                for (; l + static_cast<size_t>(width) <= outer_dim; l += static_cast<size_t>(width))
                {
                    const size_t base = l * static_cast<size_t>(n) + static_cast<size_t>(idx);

                    typename P::simd_t va, vprev, vcur, vout;
                    va    = P::gather(band1 + base, gstride);
                    vprev = P::gather(x.data() + base - inner_dim, gstride);
                    vcur  = P::gather(x.data() + base, gstride);
                    vout  = P::fma(va, vprev, vcur);
                    P::scatter(vout, gstride, x.data() + base);
                }
                for (; l < outer_dim; ++l)
                {
                    const size_t base = l * static_cast<size_t>(n) + static_cast<size_t>(idx);
                    x[base] += band1[base] * x[base - inner_dim];
                }
            }

            for (; idx < n; ++idx)
            {
                size_t l = 0;
                for (; l + static_cast<size_t>(width) <= outer_dim; l += static_cast<size_t>(width))
                {
                    const size_t base = l * static_cast<size_t>(n) + static_cast<size_t>(idx);

                    typename P::simd_t v1, v2, vprev1, vprev2, vcur, vs, vout;
                    v1     = P::gather(band1 + base, gstride);
                    v2     = P::gather(band2 + base, gstride);
                    vprev1 = P::gather(x.data() + base - inner_dim, gstride);
                    vprev2 =
                        P::gather(x.data() + base - static_cast<size_t>(two_inner_dim), gstride);
                    vcur = P::gather(x.data() + base, gstride);
                    vs   = P::fma(v2, vprev2, vcur);
                    vout = P::fma(v1, vprev1, vs);
                    P::scatter(vout, gstride, x.data() + base);
                }
                for (; l < outer_dim; ++l)
                {
                    const size_t base = l * static_cast<size_t>(n) + static_cast<size_t>(idx);
                    x[base] += band1[base] * x[base - inner_dim] +
                               band2[base] * x[base - static_cast<size_t>(two_inner_dim)];
                }
            }

            idx = n - two_inner_dim;
            for (; idx <= end_1; ++idx)
            {
                size_t l = 0;
                for (; l + static_cast<size_t>(width) <= outer_dim; l += static_cast<size_t>(width))
                {
                    const size_t base = l * static_cast<size_t>(n) + static_cast<size_t>(idx);

                    typename P::simd_t v3, vnext, vcur, vs, vout;
                    v3    = P::gather(band3 + base, gstride);
                    vnext = P::gather(x.data() + base + inner_dim, gstride);
                    vcur  = P::gather(x.data() + base, gstride);
                    vs    = P::fma(v3, vnext, vzero);
                    vout  = P::fma(vs, vnegone, vcur);
                    P::scatter(vout, gstride, x.data() + base);
                }
                for (; l < outer_dim; ++l)
                {
                    const size_t base = l * static_cast<size_t>(n) + static_cast<size_t>(idx);
                    x[base] -= band3[base] * x[base + inner_dim];
                }
            }

            idx = end_2;
            for (; idx >= 0; --idx)
            {
                size_t l = 0;
                for (; l + static_cast<size_t>(width) <= outer_dim; l += static_cast<size_t>(width))
                {
                    const size_t base = l * static_cast<size_t>(n) + static_cast<size_t>(idx);

                    typename P::simd_t v3, v4, vnext1, vnext2, vcur, vs, vout;
                    v3     = P::gather(band3 + base, gstride);
                    v4     = P::gather(band4 + base, gstride);
                    vnext1 = P::gather(x.data() + base + inner_dim, gstride);
                    vnext2 =
                        P::gather(x.data() + base + static_cast<size_t>(two_inner_dim), gstride);
                    vcur = P::gather(x.data() + base, gstride);
                    vs   = P::fma(v4, vnext2, vzero);
                    vs   = P::fma(v3, vnext1, vs);
                    vout = P::fma(vs, vnegone, vcur);
                    P::scatter(vout, gstride, x.data() + base);
                }
                for (; l < outer_dim; ++l)
                {
                    const size_t base = l * static_cast<size_t>(n) + static_cast<size_t>(idx);
                    x[base] -= band3[base] * x[base + inner_dim] +
                               band4[base] * x[base + static_cast<size_t>(two_inner_dim)];
                }
            }
        }
        else
#endif
        {
            for (size_t l = 0; l < outer_dim; ++l)
            {
                const auto offset = l * static_cast<size_t>(n);

                auto* __restrict itr_x       = x.data() + offset;
                auto const* __restrict band1 = decomposed.data() + 1 * stride + offset;
                auto const* __restrict band2 = band1 + stride;
                auto const* __restrict band3 = band2 + stride;
                auto const* __restrict band4 = band3 + stride;

                // Forward sweep: idx in [inner_dim, two_inner_dim) depends only on
                // idx - inner_dim; idx in [two_inner_dim, n) also depends on idx -
                // two_inner_dim. Both lags are >= inner_dim.
                int idx = static_cast<int>(inner_dim);
#if LINALG_HAS_VECTORIZATION
                if (can_vectorize_inner)
                {
                    for (; idx + width <= two_inner_dim; idx += width)
                    {
                        typename P::simd_t va, vprev, vcur, vout;
                        va    = P::loadu(band1 + idx);
                        vprev = P::loadu(itr_x + idx - static_cast<int>(inner_dim));
                        vcur  = P::loadu(itr_x + idx);
                        vout  = P::fma(va, vprev, vcur);
                        P::storeu(vout, itr_x + idx);
                    }
                }
#endif
                for (; idx < two_inner_dim; ++idx)
                {
                    itr_x[idx] += band1[idx] * itr_x[idx - static_cast<int>(inner_dim)];
                }

#if LINALG_HAS_VECTORIZATION
                if (can_vectorize_inner)
                {
                    for (; idx + width <= n; idx += width)
                    {
                        typename P::simd_t v1, v2, vprev1, vprev2, vcur, vs, vout;
                        v1     = P::loadu(band1 + idx);
                        v2     = P::loadu(band2 + idx);
                        vprev1 = P::loadu(itr_x + idx - static_cast<int>(inner_dim));
                        vprev2 = P::loadu(itr_x + idx - two_inner_dim);
                        vcur   = P::loadu(itr_x + idx);
                        vs     = P::fma(v2, vprev2, vcur);
                        vout   = P::fma(v1, vprev1, vs);
                        P::storeu(vout, itr_x + idx);
                    }
                }
#endif
                for (; idx < n; ++idx)
                {
                    itr_x[idx] += band1[idx] * itr_x[idx - static_cast<int>(inner_dim)] +
                                  band2[idx] * itr_x[idx - two_inner_dim];
                }

                // Backward sweep. idx in [n - two_inner_dim, end_1] reads only the untouched
                // boundary rows (idx + inner_dim is never written by this sweep), so those
                // entries are mutually independent of each other and can vectorize
                // unconditionally. idx in [0, end_2] is a genuine backward recurrence with
                // lags inner_dim and two_inner_dim, so it needs the same width <= inner_dim
                // guard, processed from high idx to low idx.
                idx = n - two_inner_dim;
#if LINALG_HAS_VECTORIZATION
                for (; idx + width <= end_1 + 1; idx += width)
                {
                    typename P::simd_t v3, vnext, vcur, vs, vout;
                    v3    = P::loadu(band3 + idx);
                    vnext = P::loadu(itr_x + idx + static_cast<int>(inner_dim));
                    vcur  = P::loadu(itr_x + idx);
                    vs    = P::fma(v3, vnext, vzero);
                    vout  = P::fma(vs, vnegone, vcur);
                    P::storeu(vout, itr_x + idx);
                }
#endif
                for (; idx <= end_1; ++idx)
                {
                    itr_x[idx] -= band3[idx] * itr_x[idx + static_cast<int>(inner_dim)];
                }

                idx = end_2;
#if LINALG_HAS_VECTORIZATION
                if (can_vectorize_inner)
                {
                    for (; idx - width + 1 >= 0; idx -= width)
                    {
                        const int base = idx - width + 1;

                        typename P::simd_t v3, v4, vnext1, vnext2, vcur, vs, vout;
                        v3     = P::loadu(band3 + base);
                        v4     = P::loadu(band4 + base);
                        vnext1 = P::loadu(itr_x + base + static_cast<int>(inner_dim));
                        vnext2 = P::loadu(itr_x + base + two_inner_dim);
                        vcur   = P::loadu(itr_x + base);
                        vs     = P::fma(v4, vnext2, vzero);
                        vs     = P::fma(v3, vnext1, vs);
                        vout   = P::fma(vs, vnegone, vcur);
                        P::storeu(vout, itr_x + base);
                    }
                }
#endif
                for (; idx >= 0; --idx)
                {
                    itr_x[idx] -= band3[idx] * itr_x[idx + static_cast<int>(inner_dim)] +
                                  band4[idx] * itr_x[idx + two_inner_dim];
                }
            }
        }
    }
}

//=============================================================================
// multiply<T, BandCount>
//=============================================================================
template <typename T, int BandCount>
void multiply(std::vector<T>& result,
    const std::vector<T>&     in,
    const std::vector<T>&     mat,
    size_t                    outer_dim,
    size_t                    dim,
    size_t                    inner_dim,
    T                         time_multiplier)
{
    static_assert(BandCount == 3 || BandCount == 5, "BandCount must be 3 or 5");

    if constexpr (BandCount == 3)
    {
        LINALG_SCOPE("banded/multiply/3");

        const auto n      = inner_dim * dim;
        const auto stride = outer_dim * n;

#if LINALG_HAS_VECTORIZATION
        using P                = simd<T>;
        constexpr size_t width = static_cast<size_t>(P::size);

        typename P::simd_t vzero, vtm;
        vzero = P::set(T{0});
        vtm   = P::set(time_multiplier);
#endif

        for (size_t l = 0; l < outer_dim; ++l)
        {
            const auto offset = l * n;

            auto* __restrict itr           = result.data() + offset;
            auto const* __restrict itr_tmp = in.data() + offset;
            auto const* __restrict itr0    = mat.data() + offset;
            auto const* __restrict itr1    = itr0 + stride;
            auto const* __restrict itr2    = itr1 + stride;

            // Pure stencil accumulation (result/in/mat never alias each other and nothing fed
            // back into `in` or the mat bands), so unlike decomposition/solve_decomposed above
            // there is no recurrence hazard at all here: every row position is fully
            // independent and can be packed regardless of how wide the packet is relative to
            // inner_dim.
            size_t i = 0;
#if LINALG_HAS_VECTORIZATION
            for (; i + width <= inner_dim; i += width,
                itr += width,
                itr_tmp += width,
                itr0 += width,
                itr1 += width,
                itr2 += width)
            {
                typename P::simd_t vb, vc, vx0, vx1, vr, vs, vout;
                vb   = P::loadu(itr1);
                vc   = P::loadu(itr2);
                vx0  = P::loadu(itr_tmp);
                vx1  = P::loadu(itr_tmp + inner_dim);
                vr   = P::loadu(itr);
                vs   = P::fma(vc, vx1, vzero);
                vs   = P::fma(vb, vx0, vs);
                vout = P::fma(vtm, vs, vr);
                P::storeu(vout, itr);
            }
#endif
            for (; i < inner_dim; ++i, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2)
            {
                const auto b = *itr1;
                const auto c = *itr2;

                *itr += time_multiplier * (b * *itr_tmp + c * *(itr_tmp + inner_dim));
            }

#if LINALG_HAS_VECTORIZATION
            for (; i + width <= n - inner_dim; i += width,
                itr += width,
                itr_tmp += width,
                itr0 += width,
                itr1 += width,
                itr2 += width)
            {
                typename P::simd_t va, vb, vc, vxm1, vx0, vx1, vr, vs, vout;
                va   = P::loadu(itr0);
                vb   = P::loadu(itr1);
                vc   = P::loadu(itr2);
                vxm1 = P::loadu(itr_tmp - inner_dim);
                vx0  = P::loadu(itr_tmp);
                vx1  = P::loadu(itr_tmp + inner_dim);
                vr   = P::loadu(itr);
                vs   = P::fma(vc, vx1, vzero);
                vs   = P::fma(vb, vx0, vs);
                vs   = P::fma(va, vxm1, vs);
                vout = P::fma(vtm, vs, vr);
                P::storeu(vout, itr);
            }
#endif
            for (; i < n - inner_dim; ++i, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2)
            {
                const auto a = *itr0;
                const auto b = *itr1;
                const auto c = *itr2;

                *itr += time_multiplier *
                        (a * *(itr_tmp - inner_dim) + b * *itr_tmp + c * *(itr_tmp + inner_dim));
            }

#if LINALG_HAS_VECTORIZATION
            for (; i + width <= n; i += width,
                itr += width,
                itr_tmp += width,
                itr0 += width,
                itr1 += width,
                itr2 += width)
            {
                typename P::simd_t va, vb, vxm1, vx0, vr, vs, vout;
                va   = P::loadu(itr0);
                vb   = P::loadu(itr1);
                vxm1 = P::loadu(itr_tmp - inner_dim);
                vx0  = P::loadu(itr_tmp);
                vr   = P::loadu(itr);
                vs   = P::fma(vb, vx0, vzero);
                vs   = P::fma(va, vxm1, vs);
                vout = P::fma(vtm, vs, vr);
                P::storeu(vout, itr);
            }
#endif
            for (; i < n; ++i, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2)
            {
                const auto a = *itr0;
                const auto b = *itr1;

                *itr += time_multiplier * (a * *(itr_tmp - inner_dim) + b * *itr_tmp);
            }
        }
    }
    else  // BandCount == 5
    {
        LINALG_SCOPE("banded/multiply/5");

        const auto n             = inner_dim * dim;
        const auto two_inner_dim = 2 * inner_dim;
        const auto stride        = outer_dim * n;

#if LINALG_HAS_VECTORIZATION
        using P                = simd<T>;
        constexpr size_t width = static_cast<size_t>(P::size);

        typename P::simd_t vzero, vtm;
        vzero = P::set(T{0});
        vtm   = P::set(time_multiplier);
#endif

        // The unrolled block loop below assumes 5 distinct band-position regions (first/second/
        // middle/second-to-last/last block, each inner_dim wide), which only partitions a row of
        // n = dim * inner_dim elements correctly when dim >= 4. For dim < 4 the "second" and
        // "second-to-last" blocks overlap, so fall back to a bounds-checked per-element path.
        if (dim < 4)
        {
            for (size_t l = 0; l < outer_dim; ++l)
            {
                const auto offset = l * n;

                const T* __restrict band0 = mat.data() + offset;
                const T* __restrict band1 = band0 + stride;
                const T* __restrict band2 = band1 + stride;
                const T* __restrict band3 = band2 + stride;
                const T* __restrict band4 = band3 + stride;

                // Which bands apply depends only on b, not p, so check each condition once per
                // b (hoisting the `for p` loop inside each `if` instead of the other way
                // around) rather than inner_dim times per b. Each resulting p-loop is a plain,
                // branch-free elementwise accumulation over contiguous idx, which also leaves
                // the auto-vectorizer a clean loop to work with instead of a branchy one.
                for (size_t b = 0; b < dim; ++b)
                {
                    const size_t row = b * inner_dim;

                    if (b >= 2)
                    {
                        size_t p = 0;
#if LINALG_HAS_VECTORIZATION
                        for (; p + width <= inner_dim; p += width)
                        {
                            const size_t idx = row + p;

                            typename P::simd_t vband, vin, vr, vprod, vout;
                            vband = P::loadu(band0 + idx);
                            vin   = P::loadu(in.data() + offset + idx - two_inner_dim);
                            vr    = P::loadu(result.data() + offset + idx);
                            vprod = P::fma(vband, vin, vzero);
                            vout  = P::fma(vtm, vprod, vr);
                            P::storeu(vout, result.data() + offset + idx);
                        }
#endif
                        for (; p < inner_dim; ++p)
                        {
                            const size_t idx = row + p;
                            result[offset + idx] +=
                                time_multiplier * band0[idx] * in[offset + idx - two_inner_dim];
                        }
                    }
                    if (b >= 1)
                    {
                        size_t p = 0;
#if LINALG_HAS_VECTORIZATION
                        for (; p + width <= inner_dim; p += width)
                        {
                            const size_t idx = row + p;

                            typename P::simd_t vband, vin, vr, vprod, vout;
                            vband = P::loadu(band1 + idx);
                            vin   = P::loadu(in.data() + offset + idx - inner_dim);
                            vr    = P::loadu(result.data() + offset + idx);
                            vprod = P::fma(vband, vin, vzero);
                            vout  = P::fma(vtm, vprod, vr);
                            P::storeu(vout, result.data() + offset + idx);
                        }
#endif
                        for (; p < inner_dim; ++p)
                        {
                            const size_t idx = row + p;
                            result[offset + idx] +=
                                time_multiplier * band1[idx] * in[offset + idx - inner_dim];
                        }
                    }
                    {
                        size_t p = 0;
#if LINALG_HAS_VECTORIZATION
                        for (; p + width <= inner_dim; p += width)
                        {
                            const size_t idx = row + p;

                            typename P::simd_t vband, vin, vr, vprod, vout;
                            vband = P::loadu(band2 + idx);
                            vin   = P::loadu(in.data() + offset + idx);
                            vr    = P::loadu(result.data() + offset + idx);
                            vprod = P::fma(vband, vin, vzero);
                            vout  = P::fma(vtm, vprod, vr);
                            P::storeu(vout, result.data() + offset + idx);
                        }
#endif
                        for (; p < inner_dim; ++p)
                        {
                            const size_t idx = row + p;
                            result[offset + idx] += time_multiplier * band2[idx] * in[offset + idx];
                        }
                    }
                    if (b + 1 < dim)
                    {
                        size_t p = 0;
#if LINALG_HAS_VECTORIZATION
                        for (; p + width <= inner_dim; p += width)
                        {
                            const size_t idx = row + p;

                            typename P::simd_t vband, vin, vr, vprod, vout;
                            vband = P::loadu(band3 + idx);
                            vin   = P::loadu(in.data() + offset + idx + inner_dim);
                            vr    = P::loadu(result.data() + offset + idx);
                            vprod = P::fma(vband, vin, vzero);
                            vout  = P::fma(vtm, vprod, vr);
                            P::storeu(vout, result.data() + offset + idx);
                        }
#endif
                        for (; p < inner_dim; ++p)
                        {
                            const size_t idx = row + p;
                            result[offset + idx] +=
                                time_multiplier * band3[idx] * in[offset + idx + inner_dim];
                        }
                    }
                    if (b + 2 < dim)
                    {
                        size_t p = 0;
#if LINALG_HAS_VECTORIZATION
                        for (; p + width <= inner_dim; p += width)
                        {
                            const size_t idx = row + p;

                            typename P::simd_t vband, vin, vr, vprod, vout;
                            vband = P::loadu(band4 + idx);
                            vin   = P::loadu(in.data() + offset + idx + two_inner_dim);
                            vr    = P::loadu(result.data() + offset + idx);
                            vprod = P::fma(vband, vin, vzero);
                            vout  = P::fma(vtm, vprod, vr);
                            P::storeu(vout, result.data() + offset + idx);
                        }
#endif
                        for (; p < inner_dim; ++p)
                        {
                            const size_t idx = row + p;
                            result[offset + idx] +=
                                time_multiplier * band4[idx] * in[offset + idx + two_inner_dim];
                        }
                    }
                }
            }
            return;
        }

        for (size_t l = 0; l < outer_dim; ++l)
        {
            const auto offset = l * n;

            auto* __restrict itr           = result.data() + offset;
            auto const* __restrict itr_tmp = in.data() + offset;
            auto const* __restrict itr0    = mat.data() + offset;
            auto const* __restrict itr1    = itr0 + stride;
            auto const* __restrict itr2    = itr1 + stride;
            auto const* __restrict itr3    = itr2 + stride;
            auto const* __restrict itr4    = itr3 + stride;

            // Same pure-stencil reasoning as the 3-band case: result/in/mat never alias, so
            // every output position is independent and any packet width is safe.
            size_t i = 0;
#if LINALG_HAS_VECTORIZATION
            for (; i + width <= inner_dim; i += width,
                itr += width,
                itr_tmp += width,
                itr2 += width,
                itr3 += width,
                itr4 += width)
            {
                typename P::simd_t v2, v3, v4, vx0, vx1, vx2, vr, vs, vout;
                v2   = P::loadu(itr2);
                v3   = P::loadu(itr3);
                v4   = P::loadu(itr4);
                vx0  = P::loadu(itr_tmp);
                vx1  = P::loadu(itr_tmp + inner_dim);
                vx2  = P::loadu(itr_tmp + two_inner_dim);
                vr   = P::loadu(itr);
                vs   = P::fma(v4, vx2, vzero);
                vs   = P::fma(v3, vx1, vs);
                vs   = P::fma(v2, vx0, vs);
                vout = P::fma(vtm, vs, vr);
                P::storeu(vout, itr);
            }
#endif
            for (; i < inner_dim; ++i, ++itr, ++itr_tmp, ++itr2, ++itr3, ++itr4)
            {
                *itr += time_multiplier * (*itr2 * *itr_tmp + *itr3 * *(itr_tmp + inner_dim) +
                                              *itr4 * *(itr_tmp + two_inner_dim));
            }

            itr1 += inner_dim;
#if LINALG_HAS_VECTORIZATION
            for (; i + width <= two_inner_dim; i += width,
                itr += width,
                itr_tmp += width,
                itr1 += width,
                itr2 += width,
                itr3 += width,
                itr4 += width)
            {
                typename P::simd_t v1, v2, v3, v4, vxm1, vx0, vx1, vx2, vr, vs, vout;
                v1   = P::loadu(itr1);
                v2   = P::loadu(itr2);
                v3   = P::loadu(itr3);
                v4   = P::loadu(itr4);
                vxm1 = P::loadu(itr_tmp - inner_dim);
                vx0  = P::loadu(itr_tmp);
                vx1  = P::loadu(itr_tmp + inner_dim);
                vx2  = P::loadu(itr_tmp + two_inner_dim);
                vr   = P::loadu(itr);
                vs   = P::fma(v4, vx2, vzero);
                vs   = P::fma(v3, vx1, vs);
                vs   = P::fma(v2, vx0, vs);
                vs   = P::fma(v1, vxm1, vs);
                vout = P::fma(vtm, vs, vr);
                P::storeu(vout, itr);
            }
#endif
            for (; i < two_inner_dim; ++i, ++itr, ++itr_tmp, ++itr1, ++itr2, ++itr3, ++itr4)
            {
                *itr += time_multiplier *
                        (*itr1 * *(itr_tmp - inner_dim) + *itr2 * *itr_tmp +
                            *itr3 * *(itr_tmp + inner_dim) + *itr4 * *(itr_tmp + two_inner_dim));
            }

            itr0 += two_inner_dim;
#if LINALG_HAS_VECTORIZATION
            for (; i + width <= n - two_inner_dim; i += width,
                itr += width,
                itr_tmp += width,
                itr0 += width,
                itr1 += width,
                itr2 += width,
                itr3 += width,
                itr4 += width)
            {
                typename P::simd_t v0, v1, v2, v3, v4, vxm2, vxm1, vx0, vx1, vx2, vr, vs, vout;
                v0   = P::loadu(itr0);
                v1   = P::loadu(itr1);
                v2   = P::loadu(itr2);
                v3   = P::loadu(itr3);
                v4   = P::loadu(itr4);
                vxm2 = P::loadu(itr_tmp - two_inner_dim);
                vxm1 = P::loadu(itr_tmp - inner_dim);
                vx0  = P::loadu(itr_tmp);
                vx1  = P::loadu(itr_tmp + inner_dim);
                vx2  = P::loadu(itr_tmp + two_inner_dim);
                vr   = P::loadu(itr);
                vs   = P::fma(v4, vx2, vzero);
                vs   = P::fma(v3, vx1, vs);
                vs   = P::fma(v2, vx0, vs);
                vs   = P::fma(v1, vxm1, vs);
                vs   = P::fma(v0, vxm2, vs);
                vout = P::fma(vtm, vs, vr);
                P::storeu(vout, itr);
            }
#endif
            for (; i < n - two_inner_dim;
                ++i, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2, ++itr3, ++itr4)
            {
                *itr += time_multiplier *
                        (*itr0 * *(itr_tmp - two_inner_dim) + *itr1 * *(itr_tmp - inner_dim) +
                            *itr2 * *itr_tmp + *itr3 * *(itr_tmp + inner_dim) +
                            *itr4 * *(itr_tmp + two_inner_dim));
            }

            size_t k = 0;
#if LINALG_HAS_VECTORIZATION
            for (; k + width <= inner_dim; k += width,
                itr += width,
                itr_tmp += width,
                itr0 += width,
                itr1 += width,
                itr2 += width,
                itr3 += width)
            {
                typename P::simd_t v0, v1, v2, v3, vxm2, vxm1, vx0, vx1, vr, vs, vout;
                v0   = P::loadu(itr0);
                v1   = P::loadu(itr1);
                v2   = P::loadu(itr2);
                v3   = P::loadu(itr3);
                vxm2 = P::loadu(itr_tmp - two_inner_dim);
                vxm1 = P::loadu(itr_tmp - inner_dim);
                vx0  = P::loadu(itr_tmp);
                vx1  = P::loadu(itr_tmp + inner_dim);
                vr   = P::loadu(itr);
                vs   = P::fma(v3, vx1, vzero);
                vs   = P::fma(v2, vx0, vs);
                vs   = P::fma(v1, vxm1, vs);
                vs   = P::fma(v0, vxm2, vs);
                vout = P::fma(vtm, vs, vr);
                P::storeu(vout, itr);
            }
#endif
            for (; k < inner_dim; ++k, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2, ++itr3)
            {
                *itr += time_multiplier *
                        (*itr0 * *(itr_tmp - two_inner_dim) + *itr1 * *(itr_tmp - inner_dim) +
                            *itr2 * *itr_tmp + *itr3 * *(itr_tmp + inner_dim));
            }

            k = 0;
#if LINALG_HAS_VECTORIZATION
            for (; k + width <= inner_dim; k += width,
                itr += width,
                itr_tmp += width,
                itr0 += width,
                itr1 += width,
                itr2 += width)
            {
                typename P::simd_t v0, v1, v2, vxm2, vxm1, vx0, vr, vs, vout;
                v0   = P::loadu(itr0);
                v1   = P::loadu(itr1);
                v2   = P::loadu(itr2);
                vxm2 = P::loadu(itr_tmp - two_inner_dim);
                vxm1 = P::loadu(itr_tmp - inner_dim);
                vx0  = P::loadu(itr_tmp);
                vr   = P::loadu(itr);
                vs   = P::fma(v2, vx0, vzero);
                vs   = P::fma(v1, vxm1, vs);
                vs   = P::fma(v0, vxm2, vs);
                vout = P::fma(vtm, vs, vr);
                P::storeu(vout, itr);
            }
#endif
            for (; k < inner_dim; ++k, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2)
            {
                *itr += time_multiplier * (*itr0 * *(itr_tmp - two_inner_dim) +
                                              *itr1 * *(itr_tmp - inner_dim) + *itr2 * *itr_tmp);
            }
        }
    }
}

//=============================================================================
// Three-band AAD and vectorised operations
//=============================================================================

template <typename T>
void decomposition_aad(std::vector<T>& output_aad,
    const std::vector<T>&              output,
    size_t                             outer_dim,
    size_t                             dim,
    size_t                             inner_dim)
{
    const auto n      = dim * inner_dim;
    const auto stride = outer_dim * n;

#if LINALG_HAS_VECTORIZATION
    using P                  = simd<T>;
    constexpr size_t   width = static_cast<size_t>(P::size);
    typename P::simd_t vzero, vnegone;
    vzero   = P::set(T{0});
    vnegone = P::set(T{-1});
#endif

    size_t neg_i = 0;
#if LINALG_HAS_VECTORIZATION
    for (; neg_i + width <= stride; neg_i += width)
    {
        typename P::simd_t v, vout;
        v    = P::loadu(output_aad.data() + 2 * stride + neg_i);
        vout = P::fma(v, vnegone, vzero);
        P::storeu(vout, output_aad.data() + 2 * stride + neg_i);
    }
#endif
    for (; neg_i < stride; ++neg_i)
    {
        output_aad[2 * stride + neg_i] = -output_aad[2 * stride + neg_i];
    }

    if (output.size() != 3 * stride)
    {
        throw std::invalid_argument("banded decomposition_aad output has the wrong size!");
    }

    // Vectorized across outer_dim instead of within a row: different l groups never alias
    // each other (disjoint memory, offset by l*n), so gather/scatter with stride n lets us
    // batch `width` independent groups' row r together while keeping the recurrence over r
    // (which runs through *(itr2_aad - inner_dim), a genuine carry into a row this same
    // reverse sweep will visit later) fully serial, exactly as the scalar loop already does.
    // This sidesteps the within-row accumulation hazard that blocked vectorizing this
    // function along inner_dim instead.
    auto const* __restrict b0 = output.data();
    auto const* __restrict b1 = b0 + stride;
    auto const* __restrict b2 = b1 + stride;

    auto* __restrict b0a = output_aad.data();
    auto* __restrict b1a = b0a + stride;
    auto* __restrict b2a = b1a + stride;

#if LINALG_HAS_VECTORIZATION
    const bool       can_vectorize_outer = width <= outer_dim;
    const linalg_int gstride             = static_cast<linalg_int>(n);
#endif

    for (size_t r = n; r > inner_dim;)
    {
        --r;

        size_t l = 0;
#if LINALG_HAS_VECTORIZATION
        if (can_vectorize_outer)
        {
            for (; l + width <= outer_dim; l += width)
            {
                const size_t base = l * n;

                typename P::simd_t v_inv_diag, v_b1, v_b2, v_b2_lag, v_b0a, v_b1a, v_b2a;
                v_inv_diag = P::gather(b0 + base + r, gstride);
                v_b1       = P::gather(b1 + base + r, gstride);
                v_b2       = P::gather(b2 + base + r, gstride);
                v_b2_lag   = P::gather(b2 + base + (r - inner_dim), gstride);
                v_b0a      = P::gather(b0a + base + r, gstride);
                v_b1a      = P::gather(b1a + base + r, gstride);
                v_b2a      = P::gather(b2a + base + r, gstride);

                typename P::simd_t v_alpha, v_tmp, v_ratio, v_tmp_itr2;
                v_alpha    = P::div(v_b1, v_inv_diag);
                v_tmp      = P::fma(v_b2_lag, vnegone, vzero);
                v_ratio    = P::div(v_b2, v_inv_diag);
                v_tmp_itr2 = P::fma(v_ratio, vnegone, vzero);

                typename P::simd_t v_inv_diag_aad, v_alpha_aad;
                v_inv_diag_aad = P::fma(v_b2a, v_tmp_itr2, vzero);
                v_b2a          = P::fma(v_b2a, v_inv_diag, vzero);

                v_inv_diag_aad = P::fma(v_b1a, v_alpha, v_inv_diag_aad);
                v_alpha_aad    = P::fma(v_b1a, v_inv_diag, vzero);

                v_inv_diag_aad = P::add(v_inv_diag_aad, v_b0a);

                typename P::simd_t v_sq, v_negsq;
                v_sq           = P::fma(v_inv_diag, v_inv_diag, vzero);
                v_negsq        = P::fma(v_sq, vnegone, vzero);
                v_inv_diag_aad = P::fma(v_inv_diag_aad, v_negsq, vzero);

                v_b1a = v_inv_diag_aad;

                v_alpha_aad = P::fma(v_inv_diag_aad, v_tmp, v_alpha_aad);
                typename P::simd_t v_tmp_aad;
                v_tmp_aad = P::fma(v_inv_diag_aad, v_alpha, vzero);

                v_b0a = P::fma(v_alpha_aad, vnegone, vzero);

                P::scatter(v_b0a, gstride, b0a + base + r);
                P::scatter(v_b1a, gstride, b1a + base + r);
                P::scatter(v_b2a, gstride, b2a + base + r);

                typename P::simd_t v_b2a_lag;
                v_b2a_lag = P::gather(b2a + base + (r - inner_dim), gstride);
                v_b2a_lag = P::add(v_b2a_lag, v_tmp_aad);
                P::scatter(v_b2a_lag, gstride, b2a + base + (r - inner_dim));
            }
        }
#endif
        for (; l < outer_dim; ++l)
        {
            const auto base = l * n;

            const auto inv_diag = b0[base + r];
            const auto alpha    = b1[base + r] / inv_diag;
            const auto tmp      = -b2[base + (r - inner_dim)];
            const auto tmp_itr2 = -b2[base + r] / inv_diag;

            auto inv_diag_aad = b2a[base + r] * tmp_itr2;
            b2a[base + r] *= inv_diag;

            inv_diag_aad += b1a[base + r] * alpha;
            auto alpha_aad = b1a[base + r] * inv_diag;

            inv_diag_aad += b0a[base + r];

            inv_diag_aad *= -(inv_diag * inv_diag);
            b1a[base + r] = inv_diag_aad;
            alpha_aad += inv_diag_aad * tmp;
            auto tmp_aad = inv_diag_aad * alpha;

            b0a[base + r] = -alpha_aad;

            b2a[base + (r - inner_dim)] += tmp_aad;
        }
    }

    for (size_t r = inner_dim; r > 0;)
    {
        --r;

        size_t l = 0;
#if LINALG_HAS_VECTORIZATION
        if (can_vectorize_outer)
        {
            for (; l + width <= outer_dim; l += width)
            {
                const size_t base = l * n;

                typename P::simd_t v_inv_diag, v_b2, v_b2a, v_b0a;
                v_inv_diag = P::gather(b0 + base + r, gstride);
                v_b2       = P::gather(b2 + base + r, gstride);
                v_b2a      = P::gather(b2a + base + r, gstride);
                v_b0a      = P::gather(b0a + base + r, gstride);

                typename P::simd_t v_ratio, v_tmp_itr2;
                v_ratio    = P::div(v_b2, v_inv_diag);
                v_tmp_itr2 = P::fma(v_ratio, vnegone, vzero);

                typename P::simd_t v_inv_diag_aad;
                v_inv_diag_aad = P::fma(v_b2a, v_tmp_itr2, vzero);
                v_b2a          = P::fma(v_b2a, v_inv_diag, vzero);

                v_inv_diag_aad = P::add(v_inv_diag_aad, v_b0a);

                typename P::simd_t v_sq, v_b1a;
                v_sq  = P::fma(v_inv_diag, v_inv_diag, vzero);
                v_b1a = P::fma(v_inv_diag_aad, v_sq, vzero);
                v_b1a = P::fma(v_b1a, vnegone, vzero);

                P::scatter(v_b2a, gstride, b2a + base + r);
                P::scatter(v_b1a, gstride, b1a + base + r);
                P::scatter(vzero, gstride, b0a + base + r);
            }
        }
#endif
        for (; l < outer_dim; ++l)
        {
            const auto base     = l * n;
            const T    inv_diag = b0[base + r];

            auto tmp_itr2     = -b2[base + r] / inv_diag;
            auto inv_diag_aad = b2a[base + r] * tmp_itr2;
            b2a[base + r]     = b2a[base + r] * inv_diag;

            inv_diag_aad += b0a[base + r];

            b1a[base + r] = -inv_diag_aad * inv_diag * inv_diag;
            b0a[base + r] = T{0};
        }
    }
}

//-----------------------------------------------------------------------------
template <typename T>
void solve_decomposed_vectorised(std::vector<T>& x,
    const std::vector<T>&                        decomposed,
    size_t                                       rows,
    size_t                                       columns,
    size_t                                       outer_dim,
    size_t                                       dim,
    size_t                                       inner_dim)
{
    LINALG_SCOPE("banded/solve_decomposed_vectorised/3");

    const auto n      = inner_dim * dim;
    const int  end    = static_cast<int>(n) - static_cast<int>(inner_dim) - 1;
    const auto stride = outer_dim * n;

    for (size_t i = 0; i < rows; ++i)
    {
        const auto row_scale = decomposed[0 * stride + i];
        for (size_t j = 0; j < columns; ++j)
        {
            x[i * columns + j] *= row_scale;
        }
    }

    auto* __restrict itr_x      = x.data();
    auto const* __restrict itr1 = decomposed.data() + stride;
    auto const* __restrict itr2 = itr1 + stride;

#if LINALG_HAS_VECTORIZATION
    using P                = simd<T>;
    constexpr size_t width = static_cast<size_t>(P::size);
#endif

    for (size_t l = 0; l < outer_dim; ++l)
    {
        const auto offset               = l * n;
        auto const* __restrict tmp_itr1 = itr1 + offset;
        auto const* __restrict tmp_itr2 = itr2 + offset;

        auto* __restrict tmp = itr_x + offset * columns;

        for (size_t i = inner_dim; i < n; ++i)
        {
            const auto  a       = tmp_itr1[i];
            auto*       tmp_i   = tmp + i * columns;
            const auto* tmp_i_1 = tmp + (i - inner_dim) * columns;

            // The right-hand-side columns are independent of each other (only the row index i
            // carries the recurrence), so this axis vectorizes unconditionally regardless of
            // inner_dim.
            size_t j = 0;
#if LINALG_HAS_VECTORIZATION
            typename P::simd_t va;
            va = P::set(a);
            for (; j + width <= columns; j += width)
            {
                typename P::simd_t vprev, vcur, vout;
                vprev = P::loadu(tmp_i_1 + j);
                vcur  = P::loadu(tmp_i + j);
                vout  = P::fma(va, vprev, vcur);
                P::storeu(vout, tmp_i + j);
            }
#endif
            for (; j < columns; ++j)
            {
                tmp_i[j] += a * tmp_i_1[j];
            }
        }

        for (int i = end; i >= 0; --i)
        {
            const auto  a       = tmp_itr2[static_cast<size_t>(i)];
            auto*       tmp_i   = tmp + static_cast<size_t>(i) * columns;
            const auto* tmp_i_1 = tmp + (static_cast<size_t>(i) + inner_dim) * columns;

            size_t j = 0;
#if LINALG_HAS_VECTORIZATION
            typename P::simd_t va;
            va = P::set(a);
            for (; j + width <= columns; j += width)
            {
                typename P::simd_t vnext, vcur, vout;
                vnext = P::loadu(tmp_i_1 + j);
                vcur  = P::loadu(tmp_i + j);
                vout  = P::fma(va, vnext, vcur);
                P::storeu(vout, tmp_i + j);
            }
#endif
            for (; j < columns; ++j)
            {
                tmp_i[j] += a * tmp_i_1[j];
            }
        }
    }
}

//-----------------------------------------------------------------------------
template <typename T>
void solve_decomposed_aad(std::vector<T>& decomposed_aad,
    std::vector<T>&                       x_aad,
    std::vector<T>&                       x,
    const std::vector<T>&                 decomposed,
    size_t                                outer_dim,
    size_t                                dim,
    size_t                                inner_dim)
{
    const auto n      = inner_dim * dim;
    const int  end    = static_cast<int>(n) - static_cast<int>(inner_dim) - 1;
    const auto stride = outer_dim * n;

    auto* __restrict band1       = decomposed.data() + stride;
    auto const* __restrict band2 = band1 + stride;

    auto* __restrict band1_aad = decomposed_aad.data() + stride;
    auto* __restrict band2_aad = band1_aad + stride;

    // Vectorized across outer_dim (see decomposition_aad above for the full rationale): each
    // l group is independent, so for a fixed i we can gather/scatter `width` groups at once
    // while keeping the loop over i -- which carries real cross-row accumulation via
    // tmp_aad[i +/- inner_dim] -- exactly as serial as the scalar version. i becomes the
    // outer loop and l the (vectorized) inner one; this is a pure loop-nest transpose of two
    // independent axes, so it changes nothing about per-l correctness.
#if LINALG_HAS_VECTORIZATION
    using P                                = simd<T>;
    constexpr size_t   width               = static_cast<size_t>(P::size);
    const bool         can_vectorize_outer = width <= outer_dim;
    const linalg_int   gstride             = static_cast<linalg_int>(n);
    typename P::simd_t vzero, vnegone;
    vzero   = P::set(T{0});
    vnegone = P::set(T{-1});
#endif

    for (int i = 0; i <= end; ++i)
    {
        size_t l = 0;
#if LINALG_HAS_VECTORIZATION
        if (can_vectorize_outer)
        {
            for (; l + width <= outer_dim; l += width)
            {
                const size_t base = l * n + static_cast<size_t>(i);

                typename P::simd_t v_tmp_aad_i, v_tmp_itr2_i, v_tmp_ip, v_tmp_aad_ip,
                    v_tmp_itr2_aad_i, v_tmp_i;
                v_tmp_aad_i      = P::gather(x_aad.data() + base, gstride);
                v_tmp_itr2_i     = P::gather(band2 + base, gstride);
                v_tmp_ip         = P::gather(x.data() + base + inner_dim, gstride);
                v_tmp_aad_ip     = P::gather(x_aad.data() + base + inner_dim, gstride);
                v_tmp_itr2_aad_i = P::gather(band2_aad + base, gstride);
                v_tmp_i          = P::gather(x.data() + base, gstride);

                typename P::simd_t v_contrib1, v_contrib2, v_prod;
                v_contrib1       = P::fma(v_tmp_aad_i, v_tmp_ip, vzero);
                v_tmp_itr2_aad_i = P::add(v_tmp_itr2_aad_i, v_contrib1);
                P::scatter(v_tmp_itr2_aad_i, gstride, band2_aad + base);

                v_contrib2   = P::fma(v_tmp_aad_i, v_tmp_itr2_i, vzero);
                v_tmp_aad_ip = P::add(v_tmp_aad_ip, v_contrib2);
                P::scatter(v_tmp_aad_ip, gstride, x_aad.data() + base + inner_dim);

                v_prod  = P::fma(v_tmp_itr2_i, v_tmp_ip, vzero);
                v_tmp_i = P::fma(v_prod, vnegone, v_tmp_i);
                P::scatter(v_tmp_i, gstride, x.data() + base);
            }
        }
#endif
        for (; l < outer_dim; ++l)
        {
            const size_t base = l * n + static_cast<size_t>(i);

            band2_aad[base] += x_aad[base] * x[base + inner_dim];
            x_aad[base + inner_dim] += x_aad[base] * band2[base];

            x[base] -= band2[base] * x[base + inner_dim];
        }
    }

    for (size_t i = n - 1; i >= inner_dim; --i)
    {
        size_t l = 0;
#if LINALG_HAS_VECTORIZATION
        if (can_vectorize_outer)
        {
            for (; l + width <= outer_dim; l += width)
            {
                const size_t base = l * n + i;

                typename P::simd_t v_tmp_aad_i, v_tmp_itr1_i, v_tmp_im, v_tmp_aad_im,
                    v_tmp_itr1_aad_i, v_tmp_i;
                v_tmp_aad_i      = P::gather(x_aad.data() + base, gstride);
                v_tmp_itr1_i     = P::gather(band1 + base, gstride);
                v_tmp_im         = P::gather(x.data() + base - inner_dim, gstride);
                v_tmp_aad_im     = P::gather(x_aad.data() + base - inner_dim, gstride);
                v_tmp_itr1_aad_i = P::gather(band1_aad + base, gstride);
                v_tmp_i          = P::gather(x.data() + base, gstride);

                typename P::simd_t v_contrib1, v_contrib2, v_prod;
                v_contrib1       = P::fma(v_tmp_aad_i, v_tmp_im, vzero);
                v_tmp_itr1_aad_i = P::add(v_tmp_itr1_aad_i, v_contrib1);
                P::scatter(v_tmp_itr1_aad_i, gstride, band1_aad + base);

                v_contrib2   = P::fma(v_tmp_aad_i, v_tmp_itr1_i, vzero);
                v_tmp_aad_im = P::add(v_tmp_aad_im, v_contrib2);
                P::scatter(v_tmp_aad_im, gstride, x_aad.data() + base - inner_dim);

                v_prod  = P::fma(v_tmp_itr1_i, v_tmp_im, vzero);
                v_tmp_i = P::fma(v_prod, vnegone, v_tmp_i);
                P::scatter(v_tmp_i, gstride, x.data() + base);
            }
        }
#endif
        for (; l < outer_dim; ++l)
        {
            const size_t base = l * n + i;

            band1_aad[base] += x_aad[base] * x[base - inner_dim];
            x_aad[base - inner_dim] += x_aad[base] * band1[base];

            x[base] -= band1[base] * x[base - inner_dim];
        }
    }

    size_t i = 0;
#if LINALG_HAS_VECTORIZATION
    for (; i + width <= stride; i += width)
    {
        typename P::simd_t vx, vd, vout;
        vx   = P::loadu(x.data() + i);
        vd   = P::loadu(decomposed.data() + i);
        vout = P::div(vx, vd);
        P::storeu(vout, x.data() + i);
    }
#endif
    for (; i < stride; ++i)
    {
        x[i] /= decomposed[0 * stride + i];
    }

    i = 0;
#if LINALG_HAS_VECTORIZATION
    for (; i + width <= stride; i += width)
    {
        typename P::simd_t vxa, vx, vda, vout;
        vxa  = P::loadu(x_aad.data() + i);
        vx   = P::loadu(x.data() + i);
        vda  = P::loadu(decomposed_aad.data() + i);
        vout = P::fma(vxa, vx, vda);
        P::storeu(vout, decomposed_aad.data() + i);
    }
#endif
    for (; i < stride; ++i)
    {
        decomposed_aad[0 * stride + i] += x_aad[i] * x[i];
    }

    i = 0;
#if LINALG_HAS_VECTORIZATION
    for (; i + width <= stride; i += width)
    {
        typename P::simd_t vxa, vd, vout;
        vxa  = P::loadu(x_aad.data() + i);
        vd   = P::loadu(decomposed.data() + i);
        vout = P::fma(vxa, vd, vzero);
        P::storeu(vout, x_aad.data() + i);
    }
#endif
    for (; i < stride; ++i)
    {
        x_aad[i] *= decomposed[0 * stride + i];
    }
}

//-----------------------------------------------------------------------------
template <typename T>
void solve_decomposed_aad(std::vector<T>& x_aad,
    const std::vector<T>&                 decomposed,
    size_t                                outer_dim,
    size_t                                dim,
    size_t                                inner_dim)
{
    const auto n      = inner_dim * dim;
    const int  end    = static_cast<int>(n) - static_cast<int>(inner_dim) - 1;
    const auto stride = outer_dim * n;

    auto const* __restrict band1 = decomposed.data() + stride;
    auto const* __restrict band2 = band1 + stride;

#if LINALG_HAS_VECTORIZATION
    using P                                = simd<T>;
    constexpr size_t   width               = static_cast<size_t>(P::size);
    const bool         can_vectorize_outer = width <= outer_dim;
    const linalg_int   gstride             = static_cast<linalg_int>(n);
    typename P::simd_t vzero;
    vzero = P::set(T{0});
#endif

    for (int i = 0; i <= end; ++i)
    {
        size_t l = 0;
#if LINALG_HAS_VECTORIZATION
        if (can_vectorize_outer)
        {
            for (; l + width <= outer_dim; l += width)
            {
                const size_t base = l * n + static_cast<size_t>(i);

                typename P::simd_t v_tmp_aad_i, v_tmp_itr2_i, v_tmp_aad_ip, v_contrib;
                v_tmp_aad_i  = P::gather(x_aad.data() + base, gstride);
                v_tmp_itr2_i = P::gather(band2 + base, gstride);
                v_tmp_aad_ip = P::gather(x_aad.data() + base + inner_dim, gstride);

                v_contrib    = P::fma(v_tmp_aad_i, v_tmp_itr2_i, vzero);
                v_tmp_aad_ip = P::add(v_tmp_aad_ip, v_contrib);
                P::scatter(v_tmp_aad_ip, gstride, x_aad.data() + base + inner_dim);
            }
        }
#endif
        for (; l < outer_dim; ++l)
        {
            const size_t base = l * n + static_cast<size_t>(i);
            x_aad[base + inner_dim] += x_aad[base] * band2[base];
        }
    }

    for (size_t i = n - 1; i >= inner_dim; --i)
    {
        size_t l = 0;
#if LINALG_HAS_VECTORIZATION
        if (can_vectorize_outer)
        {
            for (; l + width <= outer_dim; l += width)
            {
                const size_t base = l * n + i;

                typename P::simd_t v_tmp_aad_i, v_tmp_itr1_i, v_tmp_aad_im, v_contrib;
                v_tmp_aad_i  = P::gather(x_aad.data() + base, gstride);
                v_tmp_itr1_i = P::gather(band1 + base, gstride);
                v_tmp_aad_im = P::gather(x_aad.data() + base - inner_dim, gstride);

                v_contrib    = P::fma(v_tmp_aad_i, v_tmp_itr1_i, vzero);
                v_tmp_aad_im = P::add(v_tmp_aad_im, v_contrib);
                P::scatter(v_tmp_aad_im, gstride, x_aad.data() + base - inner_dim);
            }
        }
#endif
        for (; l < outer_dim; ++l)
        {
            const size_t base = l * n + i;
            x_aad[base - inner_dim] += x_aad[base] * band1[base];
        }
    }

    size_t i = 0;
#if LINALG_HAS_VECTORIZATION
    for (; i + width <= stride; i += width)
    {
        typename P::simd_t vxa, vd, vout;
        vxa  = P::loadu(x_aad.data() + i);
        vd   = P::loadu(decomposed.data() + i);
        vout = P::fma(vxa, vd, vzero);
        P::storeu(vout, x_aad.data() + i);
    }
#endif
    for (; i < stride; ++i)
    {
        x_aad[i] *= decomposed[0 * stride + i];
    }
}

//-----------------------------------------------------------------------------
template <typename T>
void multiply_aad(std::vector<T>& mat_aad,
    std::vector<T>&               in_aad,
    const std::vector<T>&         result_aad,
    const std::vector<T>&         in,
    const std::vector<T>&         mat,
    size_t                        outer_dim,
    size_t                        dim,
    size_t                        inner_dim,
    T                             time_multiplier)
{
    const auto n      = inner_dim * dim;
    const auto stride = outer_dim * n;

    // Vectorized across outer_dim, same as the other _aad functions above: nothing this
    // function writes is ever read back by a later row (mat_aad[r] is written exactly once
    // per (l, r); in_aad accumulates from multiple rows but is never itself read), so the
    // original per-l, per-row loop has no ordering requirement at all -- but different rows
    // *within the same l* can still target the same in_aad slot (r +/- inner_dim), which is
    // exactly the scatter-add hazard that kept this function scalar before. Vectorizing
    // across l instead of across rows sidesteps it entirely: each lane is a different,
    // disjoint l group, so lanes can never collide, and every write becomes an ordinary
    // read-modify-write (gather, add this row's contribution, scatter) against that lane's
    // own memory.
    auto const* __restrict res_aad = result_aad.data();
    auto const* __restrict in_ro   = in.data();

    auto const* __restrict band0 = mat.data();
    auto const* __restrict band1 = band0 + stride;
    auto const* __restrict band2 = band1 + stride;

    auto* __restrict band0_aad = mat_aad.data();
    auto* __restrict band1_aad = band0_aad + stride;
    auto* __restrict band2_aad = band1_aad + stride;
    auto* __restrict in_aad_p  = in_aad.data();

#if LINALG_HAS_VECTORIZATION
    using P                                = simd<T>;
    constexpr size_t   width               = static_cast<size_t>(P::size);
    const bool         can_vectorize_outer = width <= outer_dim;
    const linalg_int   gstride             = static_cast<linalg_int>(n);
    typename P::simd_t vzero, vtm;
    vzero = P::set(T{0});
    vtm   = P::set(time_multiplier);
#endif

    for (size_t r = 0; r < n; ++r)
    {
        const bool has_below = r >= inner_dim;
        const bool has_above = r + inner_dim < n;

        size_t l = 0;
#if LINALG_HAS_VECTORIZATION
        if (can_vectorize_outer)
        {
            for (; l + width <= outer_dim; l += width)
            {
                const size_t base = l * n + r;

                typename P::simd_t v_res, v_tmp_aad;
                v_res     = P::gather(res_aad + base, gstride);
                v_tmp_aad = P::fma(v_res, vtm, vzero);

                if (has_below && has_above)
                {
                    typename P::simd_t v_a, v_b, v_c, v_in_m, v_in_0, v_in_p;
                    v_a    = P::gather(band0 + base, gstride);
                    v_b    = P::gather(band1 + base, gstride);
                    v_c    = P::gather(band2 + base, gstride);
                    v_in_m = P::gather(in_ro + base - inner_dim, gstride);
                    v_in_0 = P::gather(in_ro + base, gstride);
                    v_in_p = P::gather(in_ro + base + inner_dim, gstride);

                    typename P::simd_t v_acc;
                    v_acc = P::gather(band0_aad + base, gstride);
                    v_acc = P::fma(v_tmp_aad, v_in_m, v_acc);
                    P::scatter(v_acc, gstride, band0_aad + base);

                    v_acc = P::gather(band1_aad + base, gstride);
                    v_acc = P::fma(v_tmp_aad, v_in_0, v_acc);
                    P::scatter(v_acc, gstride, band1_aad + base);

                    v_acc = P::gather(band2_aad + base, gstride);
                    v_acc = P::fma(v_tmp_aad, v_in_p, v_acc);
                    P::scatter(v_acc, gstride, band2_aad + base);

                    v_acc = P::gather(in_aad_p + base - inner_dim, gstride);
                    v_acc = P::fma(v_tmp_aad, v_a, v_acc);
                    P::scatter(v_acc, gstride, in_aad_p + base - inner_dim);

                    v_acc = P::gather(in_aad_p + base, gstride);
                    v_acc = P::fma(v_tmp_aad, v_b, v_acc);
                    P::scatter(v_acc, gstride, in_aad_p + base);

                    v_acc = P::gather(in_aad_p + base + inner_dim, gstride);
                    v_acc = P::fma(v_tmp_aad, v_c, v_acc);
                    P::scatter(v_acc, gstride, in_aad_p + base + inner_dim);
                }
                else if (has_below)  // top block: no band2/c term
                {
                    typename P::simd_t v_a, v_b, v_in_m, v_in_0;
                    v_a    = P::gather(band0 + base, gstride);
                    v_b    = P::gather(band1 + base, gstride);
                    v_in_m = P::gather(in_ro + base - inner_dim, gstride);
                    v_in_0 = P::gather(in_ro + base, gstride);

                    typename P::simd_t v_acc;
                    v_acc = P::gather(band0_aad + base, gstride);
                    v_acc = P::fma(v_tmp_aad, v_in_m, v_acc);
                    P::scatter(v_acc, gstride, band0_aad + base);

                    v_acc = P::gather(band1_aad + base, gstride);
                    v_acc = P::fma(v_tmp_aad, v_in_0, v_acc);
                    P::scatter(v_acc, gstride, band1_aad + base);

                    v_acc = P::gather(in_aad_p + base - inner_dim, gstride);
                    v_acc = P::fma(v_tmp_aad, v_a, v_acc);
                    P::scatter(v_acc, gstride, in_aad_p + base - inner_dim);

                    v_acc = P::gather(in_aad_p + base, gstride);
                    v_acc = P::fma(v_tmp_aad, v_b, v_acc);
                    P::scatter(v_acc, gstride, in_aad_p + base);
                }
                else  // bottom block: no band0/a term
                {
                    typename P::simd_t v_b, v_c, v_in_0, v_in_p;
                    v_b    = P::gather(band1 + base, gstride);
                    v_c    = P::gather(band2 + base, gstride);
                    v_in_0 = P::gather(in_ro + base, gstride);
                    v_in_p = P::gather(in_ro + base + inner_dim, gstride);

                    typename P::simd_t v_acc;
                    v_acc = P::gather(band1_aad + base, gstride);
                    v_acc = P::fma(v_tmp_aad, v_in_0, v_acc);
                    P::scatter(v_acc, gstride, band1_aad + base);

                    v_acc = P::gather(band2_aad + base, gstride);
                    v_acc = P::fma(v_tmp_aad, v_in_p, v_acc);
                    P::scatter(v_acc, gstride, band2_aad + base);

                    v_acc = P::gather(in_aad_p + base, gstride);
                    v_acc = P::fma(v_tmp_aad, v_b, v_acc);
                    P::scatter(v_acc, gstride, in_aad_p + base);

                    v_acc = P::gather(in_aad_p + base + inner_dim, gstride);
                    v_acc = P::fma(v_tmp_aad, v_c, v_acc);
                    P::scatter(v_acc, gstride, in_aad_p + base + inner_dim);
                }
            }
        }
#endif
        for (; l < outer_dim; ++l)
        {
            const size_t base = l * n + r;

            const auto tmp_aad = res_aad[base] * time_multiplier;

            if (has_below && has_above)
            {
                const auto a = band0[base];
                const auto b = band1[base];
                const auto c = band2[base];

                band0_aad[base] += tmp_aad * in_ro[base - inner_dim];
                band1_aad[base] += tmp_aad * in_ro[base];
                band2_aad[base] += tmp_aad * in_ro[base + inner_dim];
                in_aad_p[base - inner_dim] += tmp_aad * a;
                in_aad_p[base] += tmp_aad * b;
                in_aad_p[base + inner_dim] += tmp_aad * c;
            }
            else if (has_below)
            {
                const auto a = band0[base];
                const auto b = band1[base];

                band0_aad[base] += tmp_aad * in_ro[base - inner_dim];
                band1_aad[base] += tmp_aad * in_ro[base];
                in_aad_p[base - inner_dim] += tmp_aad * a;
                in_aad_p[base] += tmp_aad * b;
            }
            else
            {
                const auto b = band1[base];
                const auto c = band2[base];

                band1_aad[base] += tmp_aad * in_ro[base];
                band2_aad[base] += tmp_aad * in_ro[base + inner_dim];
                in_aad_p[base] += tmp_aad * b;
                in_aad_p[base + inner_dim] += tmp_aad * c;
            }
        }
    }
}

//=============================================================================
// Explicit instantiations
//=============================================================================

template LINALG_API void decomposition<double, 3>(std::vector<double>&, size_t, size_t, size_t);
template LINALG_API void decomposition<float, 3>(std::vector<float>&, size_t, size_t, size_t);
template LINALG_API void decomposition<double, 5>(std::vector<double>&, size_t, size_t, size_t);
template LINALG_API void decomposition<float, 5>(std::vector<float>&, size_t, size_t, size_t);

template LINALG_API void solve_decomposed<double, 3>(
    std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t);
template LINALG_API void solve_decomposed<float, 3>(
    std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t);
template LINALG_API void solve_decomposed<double, 5>(
    std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t);
template LINALG_API void solve_decomposed<float, 5>(
    std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t);

template LINALG_API void multiply<double, 3>(std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    size_t,
    size_t,
    size_t,
    double);
template LINALG_API void multiply<float, 3>(std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    size_t,
    size_t,
    size_t,
    float);
template LINALG_API void multiply<double, 5>(std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    size_t,
    size_t,
    size_t,
    double);
template LINALG_API void multiply<float, 5>(std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    size_t,
    size_t,
    size_t,
    float);

template LINALG_API void decomposition_aad<double>(
    std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t);
template LINALG_API void decomposition_aad<float>(
    std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t);

template LINALG_API void solve_decomposed_vectorised<double>(
    std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t, size_t, size_t);
template LINALG_API void solve_decomposed_vectorised<float>(
    std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t, size_t, size_t);

template LINALG_API void solve_decomposed_aad<double>(std::vector<double>&,
    std::vector<double>&,
    std::vector<double>&,
    const std::vector<double>&,
    size_t,
    size_t,
    size_t);
template LINALG_API void solve_decomposed_aad<float>(std::vector<float>&,
    std::vector<float>&,
    std::vector<float>&,
    const std::vector<float>&,
    size_t,
    size_t,
    size_t);

template LINALG_API void solve_decomposed_aad<double>(
    std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t);
template LINALG_API void solve_decomposed_aad<float>(
    std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t);

template LINALG_API void multiply_aad<double>(std::vector<double>&,
    std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    size_t,
    size_t,
    size_t,
    double);
template LINALG_API void multiply_aad<float>(std::vector<float>&,
    std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    size_t,
    size_t,
    size_t,
    float);

}  // namespace linalg::banded_operations
