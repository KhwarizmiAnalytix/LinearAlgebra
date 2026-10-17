#include "include/matrix_operation/banded_operations.h"

#include "include/common/linalg_macros.h"
#include "include/common/linalg_profiler.h"

namespace linalg::banded_operations
{

//=============================================================================
// decomposition<T, BandCount>
//=============================================================================
template <typename T, int BandCount>
void decomposition(std::vector<T>& output,
    size_t                         outer_dim,
    size_t                         dim,
    size_t                         inner_dim)
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

        for (size_t l = 0; l < outer_dim; ++l)
        {
            const auto offset = l * n;

            auto* __restrict itr0 = output.data() + 0 * stride + offset;
            auto* __restrict itr1 = output.data() + 1 * stride + offset;
            auto* __restrict itr2 = output.data() + 2 * stride + offset;

            for (size_t i = 0; i < inner_dim; ++i)
            {
                const T inv_diag = T{1} / (*itr1);

                *itr0++ = inv_diag;
                *itr1++ = T{0};
                *itr2++ *= inv_diag;
            }
            for (size_t i = inner_dim; i < n; ++i)
            {
                const auto tmp      = *(itr2 - inner_dim);
                const auto alpha    = -*itr0;
                const T    inv_diag = T{1} / (*itr1 + alpha * tmp);

                *itr0++ = inv_diag;
                *itr1++ = alpha * inv_diag;
                *itr2++ *= inv_diag;
            }
        }

        for (size_t i = 0; i < stride; ++i)
        {
            output[2 * stride + i] = -output[2 * stride + i];
        }
    }
    else  // BandCount == 5
    {
        LINALG_SCOPE("banded/decomposition/5");

        for (size_t l = 0; l < outer_dim; ++l)
        {
            const auto offset = l * n;

            auto* itr0 = output.data() + 0 * stride + offset;
            auto* itr1 = output.data() + 1 * stride + offset;
            auto* itr2 = output.data() + 2 * stride + offset;
            auto* itr3 = output.data() + 3 * stride + offset;
            auto* itr4 = output.data() + 4 * stride + offset;

            size_t i = 0;
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
            for (; i < 2 * inner_dim; ++i)
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
        auto const* __restrict itr1 = decomposed.data() + 1 * stride;
        auto const* __restrict itr2 = decomposed.data() + 2 * stride;

        for (size_t l = 0; l < outer_dim; ++l)
        {
            const auto offset               = l * n;
            auto* __restrict tmp            = itr_x + offset;
            auto const* __restrict tmp_itr1 = itr1 + offset;
            auto const* __restrict tmp_itr2 = itr2 + offset;

            for (size_t i = inner_dim; i < n; ++i)
            {
                tmp[i] += tmp_itr1[i] * tmp[i - inner_dim];
            }

            for (int j = end; j >= 0; --j)
            {
                tmp[j] += tmp_itr2[j] * tmp[j + inner_dim];
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

        for (size_t l = 0; l < outer_dim; ++l)
        {
            const auto offset = l * static_cast<size_t>(n);

            auto*       itr_x = x.data() + offset;
            auto const* itr1  = decomposed.data() + 1 * stride + offset;
            auto const* itr2  = decomposed.data() + 2 * stride + offset;
            auto const* itr3  = decomposed.data() + 3 * stride + (offset + static_cast<size_t>(end_1));
            auto const* itr4  = decomposed.data() + 4 * stride + (offset + static_cast<size_t>(end_2));

            auto* itr_x_l = itr_x;

            int i = static_cast<int>(inner_dim);
            itr_x_l += inner_dim;

            auto const* itr1_l = itr1 + inner_dim;
            auto const* itr2_l = itr2 + inner_dim;

            for (; i < two_inner_dim; ++i, ++itr_x_l, ++itr1_l, ++itr2_l)
            {
                *itr_x_l += *itr1_l * *(itr_x_l - inner_dim);
            }

            for (; i < n; ++i, ++itr_x_l, ++itr1_l, ++itr2_l)
            {
                *itr_x_l += *itr1_l * *(itr_x_l - inner_dim) + *itr2_l * *(itr_x_l - two_inner_dim);
            }

            itr_x_l -= static_cast<int>(inner_dim) + 1;
            i -= static_cast<int>(inner_dim) + 1;

            auto const* itr3_l = itr3;
            auto const* itr4_l = itr4;

            for (int k = static_cast<int>(inner_dim) - 1; k >= 0; --k, --itr_x_l, --itr3_l)
            {
                *itr_x_l -= *itr3_l * *(itr_x_l + inner_dim);
            }

            i -= static_cast<int>(inner_dim);

            for (; i >= 0; --i, --itr_x_l, --itr3_l, --itr4_l)
            {
                *itr_x_l -= *itr3_l * *(itr_x_l + inner_dim) + *itr4_l * *(itr_x_l + two_inner_dim);
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

        for (size_t l = 0; l < outer_dim; ++l)
        {
            const auto offset = l * n;

            auto* __restrict itr           = result.data() + offset;
            auto const* __restrict itr_tmp = in.data() + offset;
            auto const* __restrict itr0    = mat.data() + 0 * stride + offset;
            auto const* __restrict itr1    = mat.data() + 1 * stride + offset;
            auto const* __restrict itr2    = mat.data() + 2 * stride + offset;

            size_t i = 0;
            for (; i < inner_dim; ++i, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2)
            {
                const auto b = *itr1;
                const auto c = *itr2;

                *itr += time_multiplier * (b * *itr_tmp + c * *(itr_tmp + inner_dim));
            }

            for (; i < n - inner_dim; ++i, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2)
            {
                const auto a = *itr0;
                const auto b = *itr1;
                const auto c = *itr2;

                *itr += time_multiplier *
                        (a * *(itr_tmp - inner_dim) + b * *itr_tmp + c * *(itr_tmp + inner_dim));
            }

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

        // The unrolled block loop below assumes 5 distinct band-position regions (first/second/
        // middle/second-to-last/last block, each inner_dim wide), which only partitions a row of
        // n = dim * inner_dim elements correctly when dim >= 4. For dim < 4 the "second" and
        // "second-to-last" blocks overlap, so fall back to a bounds-checked per-element path.
        if (dim < 4)
        {
            for (size_t l = 0; l < outer_dim; ++l)
            {
                const auto offset = l * n;
                for (size_t b = 0; b < dim; ++b)
                {
                    for (size_t p = 0; p < inner_dim; ++p)
                    {
                        const size_t idx  = b * inner_dim + p;
                        T            acc  = T{0};
                        if (b >= 2)
                        {
                            acc += mat[0 * stride + offset + idx] * in[offset + idx - two_inner_dim];
                        }
                        if (b >= 1)
                        {
                            acc += mat[1 * stride + offset + idx] * in[offset + idx - inner_dim];
                        }
                        acc += mat[2 * stride + offset + idx] * in[offset + idx];
                        if (b + 1 < dim)
                        {
                            acc += mat[3 * stride + offset + idx] * in[offset + idx + inner_dim];
                        }
                        if (b + 2 < dim)
                        {
                            acc += mat[4 * stride + offset + idx] * in[offset + idx + two_inner_dim];
                        }
                        result[offset + idx] += time_multiplier * acc;
                    }
                }
            }
            return;
        }

        for (size_t l = 0; l < outer_dim; ++l)
        {
            const auto offset = l * n;

            auto*       itr     = result.data() + offset;
            auto const* itr_tmp = in.data() + offset;
            auto const* itr0    = mat.data() + 0 * stride + offset;
            auto const* itr1    = mat.data() + 1 * stride + offset;
            auto const* itr2    = mat.data() + 2 * stride + offset;
            auto const* itr3    = mat.data() + 3 * stride + offset;
            auto const* itr4    = mat.data() + 4 * stride + offset;

            size_t i = 0;
            for (; i < inner_dim; ++i, ++itr, ++itr_tmp, ++itr2, ++itr3, ++itr4)
            {
                *itr += time_multiplier * (*itr2 * *itr_tmp + *itr3 * *(itr_tmp + inner_dim) +
                                              *itr4 * *(itr_tmp + two_inner_dim));
            }

            itr1 += inner_dim;
            for (; i < two_inner_dim; ++i, ++itr, ++itr_tmp, ++itr1, ++itr2, ++itr3, ++itr4)
            {
                *itr += time_multiplier *
                        (*itr1 * *(itr_tmp - inner_dim) + *itr2 * *itr_tmp +
                            *itr3 * *(itr_tmp + inner_dim) + *itr4 * *(itr_tmp + two_inner_dim));
            }

            itr0 += two_inner_dim;
            for (; i < n - two_inner_dim;
                ++i, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2, ++itr3, ++itr4)
            {
                *itr += time_multiplier *
                        (*itr0 * *(itr_tmp - two_inner_dim) + *itr1 * *(itr_tmp - inner_dim) +
                            *itr2 * *itr_tmp + *itr3 * *(itr_tmp + inner_dim) +
                            *itr4 * *(itr_tmp + two_inner_dim));
            }

            for (size_t k = 0; k < inner_dim;
                ++k, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2, ++itr3)
            {
                *itr += time_multiplier *
                        (*itr0 * *(itr_tmp - two_inner_dim) + *itr1 * *(itr_tmp - inner_dim) +
                            *itr2 * *itr_tmp + *itr3 * *(itr_tmp + inner_dim));
            }

            for (size_t k = 0; k < inner_dim; ++k, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2)
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

    for (size_t i = 0; i < stride; ++i)
    {
        output_aad[2 * stride + i] = -output_aad[2 * stride + i];
    }

    if (output.size() != 3 * stride)
    {
        throw std::invalid_argument("banded decomposition_aad output has the wrong size!");
    }

    for (size_t l = outer_dim; l > 0; --l)
    {
        const auto offset = (l - 1) * n + n - 1;

        auto const* __restrict itr0 = output.data() + 0 * stride + offset;
        auto const* __restrict itr1 = output.data() + 1 * stride + offset;
        auto const* __restrict itr2 = output.data() + 2 * stride + offset;

        auto* __restrict itr0_aad = output_aad.data() + 0 * stride + offset;
        auto* __restrict itr1_aad = output_aad.data() + 1 * stride + offset;
        auto* __restrict itr2_aad = output_aad.data() + 2 * stride + offset;

        for (size_t i = n; i > inner_dim;
            --i, --itr0, --itr1, --itr2, --itr0_aad, --itr1_aad, --itr2_aad)
        {
            const auto inv_diag = *itr0;
            const auto alpha    = *itr1 / inv_diag;
            const auto tmp      = -*(itr2 - inner_dim);
            const auto tmp_itr2 = -*itr2 / inv_diag;

            auto inv_diag_aad = *itr2_aad * tmp_itr2;
            *itr2_aad *= inv_diag;

            inv_diag_aad += *itr1_aad * alpha;
            auto alpha_aad = *itr1_aad * inv_diag;

            inv_diag_aad += *itr0_aad;

            inv_diag_aad *= -(inv_diag * inv_diag);
            *itr1_aad = inv_diag_aad;
            alpha_aad += inv_diag_aad * tmp;
            auto tmp_aad = inv_diag_aad * alpha;

            *itr0_aad = -alpha_aad;

            *(itr2_aad - inner_dim) += tmp_aad;
        }
        for (size_t i = inner_dim; i > 0;
            --i, --itr0, --itr1, --itr2, --itr0_aad, --itr1_aad, --itr2_aad)
        {
            const T inv_diag = *itr0;

            *itr1_aad = T{0};

            auto tmp_itr2     = -*itr2 / inv_diag;
            auto inv_diag_aad = *itr2_aad * tmp_itr2;
            *itr2_aad         = *itr2_aad * inv_diag;

            inv_diag_aad += *itr0_aad;

            *itr1_aad = -inv_diag_aad * inv_diag * inv_diag;
            *itr0_aad = T{0};
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
    auto const* __restrict itr1 = decomposed.data() + 1 * stride;
    auto const* __restrict itr2 = decomposed.data() + 2 * stride;

    for (size_t l = 0; l < outer_dim; ++l)
    {
        const auto offset               = l * n;
        auto const* __restrict tmp_itr1 = itr1 + offset;
        auto const* __restrict tmp_itr2 = itr2 + offset;

        auto* __restrict tmp = itr_x + offset * rows;

        for (size_t i = inner_dim; i < n; ++i)
        {
            const auto  a       = tmp_itr1[i];
            auto*       tmp_i   = tmp + i * columns;
            const auto* tmp_i_1 = tmp + (i - inner_dim) * columns;
            for (size_t j = 0; j < columns; ++j)
            {
                tmp_i[j] += a * tmp_i_1[j];
            }
        }

        for (int i = end; i >= 0; --i)
        {
            const auto  a       = tmp_itr2[static_cast<size_t>(i)];
            auto*       tmp_i   = tmp + static_cast<size_t>(i) * columns;
            const auto* tmp_i_1 = tmp + (static_cast<size_t>(i) + inner_dim) * columns;

            for (size_t j = 0; j < columns; ++j)
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

    auto* __restrict itr_x      = x.data();
    auto const* __restrict itr1 = decomposed.data() + 1 * stride;
    auto const* __restrict itr2 = decomposed.data() + 2 * stride;

    auto* __restrict itr_x_aad = x_aad.data();
    auto* __restrict itr1_aad  = decomposed_aad.data() + 1 * stride;
    auto* __restrict itr2_aad  = decomposed_aad.data() + 2 * stride;

    for (size_t l = outer_dim; l > 0; --l)
    {
        const auto offset               = (l - 1) * n;
        auto* __restrict tmp            = itr_x + offset;
        auto const* __restrict tmp_itr1 = itr1 + offset;
        auto const* __restrict tmp_itr2 = itr2 + offset;

        auto* __restrict tmp_aad      = itr_x_aad + offset;
        auto* __restrict tmp_itr1_aad = itr1_aad + offset;
        auto* __restrict tmp_itr2_aad = itr2_aad + offset;

        for (int i = 0; i <= end; ++i)
        {
            tmp_itr2_aad[i] += tmp_aad[i] * tmp[i + inner_dim];
            tmp_aad[i + inner_dim] += tmp_aad[i] * tmp_itr2[i];

            tmp[i] -= tmp_itr2[i] * tmp[i + inner_dim];
        }

        for (size_t i = n - 1; i >= inner_dim; --i)
        {
            tmp_itr1_aad[i] += tmp_aad[i] * tmp[i - inner_dim];
            tmp_aad[i - inner_dim] += tmp_aad[i] * tmp_itr1[i];

            tmp[i] -= tmp_itr1[i] * tmp[i - inner_dim];
        }
    }

    for (size_t i = 0; i < stride; ++i)
    {
        x[i] /= decomposed[0 * stride + i];
    }
    for (size_t i = 0; i < stride; ++i)
    {
        decomposed_aad[0 * stride + i] += x_aad[i] * x[i];
    }
    for (size_t i = 0; i < stride; ++i)
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

    auto const* __restrict itr1 = decomposed.data() + 1 * stride;
    auto const* __restrict itr2 = decomposed.data() + 2 * stride;

    auto* __restrict itr_x_aad = x_aad.data();

    for (int l = static_cast<int>(outer_dim); l > 0; --l)
    {
        const auto offset               = static_cast<size_t>(l - 1) * n;
        auto const* __restrict tmp_itr1 = itr1 + offset;
        auto const* __restrict tmp_itr2 = itr2 + offset;

        auto* __restrict tmp_aad = itr_x_aad + offset;

        for (int i = 0; i <= end; ++i)
        {
            tmp_aad[i + inner_dim] += tmp_aad[i] * tmp_itr2[i];
        }

        for (size_t i = n - 1; i >= inner_dim; --i)
        {
            tmp_aad[i - inner_dim] += tmp_aad[i] * tmp_itr1[i];
        }
    }

    for (size_t i = 0; i < stride; ++i)
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

    for (size_t l = outer_dim; l > 0; --l)
    {
        const auto offset = (l - 1) * n + n - 1;

        auto const* __restrict itr_aad = result_aad.data() + offset;
        auto const* __restrict itr_tmp = in.data() + offset;

        auto const* __restrict itr0 = mat.data() + 0 * stride + offset;
        auto const* __restrict itr1 = mat.data() + 1 * stride + offset;
        auto const* __restrict itr2 = mat.data() + 2 * stride + offset;

        auto* __restrict itr0_aad    = mat_aad.data() + 0 * stride + offset;
        auto* __restrict itr1_aad    = mat_aad.data() + 1 * stride + offset;
        auto* __restrict itr2_aad    = mat_aad.data() + 2 * stride + offset;
        auto* __restrict itr_tmp_aad = in_aad.data() + offset;

        size_t i = n;

        for (; i > n - inner_dim; --i,
            --itr_aad, --itr_tmp, --itr0, --itr1, --itr2,
            --itr_tmp_aad, --itr0_aad, --itr1_aad, --itr2_aad)
        {
            const auto a = *itr0;
            const auto b = *itr1;

            auto tmp_aad = *itr_aad * time_multiplier;
            *itr0_aad += tmp_aad * *(itr_tmp - inner_dim);
            *itr1_aad += tmp_aad * *itr_tmp;
            *itr_tmp_aad += tmp_aad * b;
            *(itr_tmp_aad - inner_dim) += tmp_aad * a;
        }

        for (; i > inner_dim; --i,
            --itr_aad, --itr_tmp, --itr0, --itr1, --itr2,
            --itr_tmp_aad, --itr0_aad, --itr1_aad, --itr2_aad)
        {
            const auto a = *itr0;
            const auto b = *itr1;
            const auto c = *itr2;

            auto tmp_aad = *itr_aad * time_multiplier;
            *itr0_aad += tmp_aad * *(itr_tmp - inner_dim);
            *itr1_aad += tmp_aad * *itr_tmp;
            *itr2_aad += tmp_aad * *(itr_tmp + inner_dim);
            *(itr_tmp_aad - inner_dim) += tmp_aad * a;
            *itr_tmp_aad += tmp_aad * b;
            *(itr_tmp_aad + inner_dim) += tmp_aad * c;
        }

        for (; i > 0; --i,
            --itr_aad, --itr_tmp, --itr0, --itr1, --itr2,
            --itr_tmp_aad, --itr0_aad, --itr1_aad, --itr2_aad)
        {
            const auto b = *itr1;
            const auto c = *itr2;

            auto tmp_aad = *itr_aad * time_multiplier;
            *itr1_aad += tmp_aad * *itr_tmp;
            *itr2_aad += tmp_aad * *(itr_tmp + inner_dim);
            *itr_tmp_aad += tmp_aad * b;
            *(itr_tmp_aad + inner_dim) += tmp_aad * c;
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

template LINALG_API void solve_decomposed<double, 3>(std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t);
template LINALG_API void solve_decomposed<float, 3>(std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t);
template LINALG_API void solve_decomposed<double, 5>(std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t);
template LINALG_API void solve_decomposed<float, 5>(std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t);

template LINALG_API void multiply<double, 3>(std::vector<double>&, const std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t, double);
template LINALG_API void multiply<float, 3>(std::vector<float>&, const std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t, float);
template LINALG_API void multiply<double, 5>(std::vector<double>&, const std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t, double);
template LINALG_API void multiply<float, 5>(std::vector<float>&, const std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t, float);

template LINALG_API void decomposition_aad<double>(std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t);
template LINALG_API void decomposition_aad<float>(std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t);

template LINALG_API void solve_decomposed_vectorised<double>(std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t, size_t, size_t);
template LINALG_API void solve_decomposed_vectorised<float>(std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t, size_t, size_t);

template LINALG_API void solve_decomposed_aad<double>(std::vector<double>&, std::vector<double>&, std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t);
template LINALG_API void solve_decomposed_aad<float>(std::vector<float>&, std::vector<float>&, std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t);

template LINALG_API void solve_decomposed_aad<double>(std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t);
template LINALG_API void solve_decomposed_aad<float>(std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t);

template LINALG_API void multiply_aad<double>(std::vector<double>&, std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, size_t, size_t, size_t, double);
template LINALG_API void multiply_aad<float>(std::vector<float>&, std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, size_t, size_t, size_t, float);

}  // namespace linalg::banded_operations
