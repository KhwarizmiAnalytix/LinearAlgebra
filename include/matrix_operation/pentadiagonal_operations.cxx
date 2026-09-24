#include "matrix_operation/pentadiagonal_operations.h"

#include "include/common/macros.h"
#include "include/parallel.h"

namespace linalg::pentadiagonal_operations
{
//-----------------------------------------------------------------------------
void decomposition(std::vector<double>& output,
    const size_t                        outer_dim,
    const size_t                        dim,
    const size_t                        inner_dim,
    const bool                          parallelize)
{
    const auto n      = dim * inner_dim;
    const auto stride = outer_dim * n;

    if (output.size() != 5 * stride)
    {
        throw std::invalid_argument("pentadiagonal decomposition output has the wrong size!");
    }

    auto parallel_eval = [&output, inner_dim, n, outer_dim, stride](int from, int to)
    {
        const auto offset = static_cast<size_t>(from) * n;

        auto* itr0 = output.data() + 0 * stride + offset;
        auto* itr1 = output.data() + 1 * stride + offset;
        auto* itr2 = output.data() + 2 * stride + offset;
        auto* itr3 = output.data() + 3 * stride + offset;
        auto* itr4 = output.data() + 4 * stride + offset;

        for (size_t l = from; l < to; l++)
        {
            size_t i = 0;
            for (; i < inner_dim; ++i)
            {
                const auto e = *itr0;
                const auto c = *itr1;
                const auto d = *itr2;
                const auto a = *itr3;
                const auto b = *itr4;

                const auto gamma = c;
                const auto mu    = 1. / d;

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
                const auto mu    = 1. / (d - alpha * gamma);

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

                const auto alpha = *(itr3 - inner_dim);
                const auto beta  = *(itr4 - inner_dim);

                const auto alpha_prev = *(itr3 - 2 * inner_dim);
                const auto beta_prev  = *(itr4 - 2 * inner_dim);

                const auto gamma = c - alpha_prev * e;
                const auto mu    = 1. / (d - alpha * gamma - beta_prev * e);

                *itr0++ = mu;
                *itr1++ = -gamma * mu;
                *itr2++ = -e * mu;
                *itr3++ = (a - beta * gamma) * mu;
                *itr4++ = b * mu;
            }
        }
    };

    if (parallelize && outer_dim > 1)
    {
        parallel::parallel_for(0, static_cast<int>(outer_dim), parallel_eval);
    }
    else
    {
        parallel_eval(0, static_cast<int>(outer_dim));
    }
}

//-----------------------------------------------------------------------------
void solve_decomposed(std::vector<double>& x,
    const std::vector<double>&             decomposed,
    const size_t                           outer_dim,
    const size_t                           dim,
    const size_t                           inner_dim,
    const bool                             parallelize)
{
    const int n             = (int)(inner_dim * dim);
    const int two_inner_dim = (int)(2 * inner_dim);

    const int end_1 = static_cast<int>(n) - static_cast<int>(inner_dim) - 1;
    const int end_2 = static_cast<int>(n) - static_cast<int>(two_inner_dim) - 1;

    const auto stride = outer_dim * static_cast<size_t>(n);

    if (x.size() != stride)
    {
        throw std::invalid_argument("pentadiagonal solve_decomposed x has the wrong size!");
    }

    for (size_t i = 0; i < stride; ++i)
    {
        x[i] *= decomposed[0 * stride + i];
    }

    auto parallel_eval =
        [&x, &decomposed, inner_dim, n, outer_dim, two_inner_dim, end_1, end_2, stride](
            int from, int to)
    {
        const auto offset = static_cast<size_t>(from) * n;

        auto*       itr_x = x.data() + offset;
        auto const* itr1  = decomposed.data() + 1 * stride + offset;
        auto const* itr2  = decomposed.data() + 2 * stride + offset;
        auto const* itr3  = decomposed.data() + 3 * stride + end_1 + offset;
        auto const* itr4  = decomposed.data() + 4 * stride + end_2 + offset;

        for (size_t l = from; l < to; ++l, itr_x += n, itr3 += n, itr4 += n)
        {
            auto* itr_x_l = itr_x;

            int i = static_cast<int>(inner_dim);
            itr_x_l += inner_dim;
            itr1 += inner_dim;
            itr2 += inner_dim;

            for (; i < two_inner_dim; ++i, ++itr_x_l, ++itr1, ++itr2)
            {
                *itr_x_l += *itr1 * *(itr_x_l - inner_dim);
            }

            for (; i < n; ++i, ++itr_x_l, ++itr1, ++itr2)
            {
                *itr_x_l += *itr1 * *(itr_x_l - inner_dim) + *itr2 * *(itr_x_l - two_inner_dim);
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
    };

    if (parallelize && outer_dim > 1)
    {
        parallel::parallel_for(0, static_cast<int>(outer_dim), parallel_eval);
    }
    else
    {
        parallel_eval(0, static_cast<int>(outer_dim));
    }
}

//-----------------------------------------------------------------------------
void multiply(std::vector<double>& result,
    const std::vector<double>&     in,
    const std::vector<double>&     mat,
    const size_t                   outer_dim,
    const size_t                   dim,
    const size_t                   inner_dim,
    double                         time_multiplier,
    const bool                     parallelize)
{
    const auto n             = inner_dim * dim;
    const auto two_inner_dim = 2 * inner_dim;
    const auto stride        = outer_dim * n;

    auto parallel_eval =
        [&result, &in, &mat, inner_dim, n, outer_dim, time_multiplier, two_inner_dim, stride](
            int from, int to)
    {
        const auto offset = static_cast<size_t>(from) * n;

        auto*       itr     = result.data() + offset;
        auto const* itr_tmp = in.data() + offset;
        auto const* itr0    = mat.data() + 0 * stride + offset;
        auto const* itr1    = mat.data() + 1 * stride + offset;
        auto const* itr2    = mat.data() + 2 * stride + offset;
        auto const* itr3    = mat.data() + 3 * stride + offset;
        auto const* itr4    = mat.data() + 4 * stride + offset;

        for (size_t l = from; l < to; ++l)
        {
            size_t i = 0;
            for (; i < inner_dim; ++i, ++itr, ++itr_tmp, ++itr2, ++itr3, ++itr4)
            {
                const auto a2 = *itr2;
                const auto a3 = *itr3;
                const auto a4 = *itr4;
                *itr += time_multiplier * (a2 * (*itr_tmp) + a3 * (*(itr_tmp + inner_dim)) +
                                              a4 * (*(itr_tmp + 2 * inner_dim)));
            }

            itr1 += inner_dim;
            for (; i < two_inner_dim; ++i, ++itr, ++itr_tmp, ++itr1, ++itr2, ++itr3, ++itr4)
            {
                const auto a1 = *itr1;
                const auto a2 = *itr2;
                const auto a3 = *itr3;
                const auto a4 = *itr4;
                *itr += time_multiplier *
                        (a1 * (*(itr_tmp - inner_dim)) + a2 * (*itr_tmp) +
                            a3 * (*(itr_tmp + inner_dim)) + a4 * (*(itr_tmp + two_inner_dim)));
            }

            itr0 += two_inner_dim;
            for (; i < n - two_inner_dim;
                ++i, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2, ++itr3, ++itr4)
            {
                const auto a0 = *itr0;
                const auto a1 = *itr1;
                const auto a2 = *itr2;
                const auto a3 = *itr3;
                const auto a4 = *itr4;

                *itr += time_multiplier *
                        (a0 * (*(itr_tmp - two_inner_dim)) + a1 * (*(itr_tmp - inner_dim)) +
                            a2 * (*itr_tmp) + a3 * (*(itr_tmp + inner_dim)) +
                            a4 * (*(itr_tmp + two_inner_dim)));
            }

            itr4 += two_inner_dim;
            for (size_t k = 0; k < inner_dim; ++k, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2, ++itr3)
            {
                const auto a0 = *itr0;
                const auto a1 = *itr1;
                const auto a2 = *itr2;
                const auto a3 = *itr3;

                *itr += time_multiplier *
                        (a0 * (*(itr_tmp - two_inner_dim)) + a1 * (*(itr_tmp - inner_dim)) +
                            a2 * (*itr_tmp) + a3 * (*(itr_tmp + inner_dim)));
            }

            itr3 += inner_dim;
            for (size_t k = 0; k < inner_dim; ++k, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2)
            {
                const auto a0 = *itr0;
                const auto a1 = *itr1;
                const auto a2 = *itr2;

                *itr += time_multiplier * (a0 * (*(itr_tmp - two_inner_dim)) +
                                              a1 * (*(itr_tmp - inner_dim)) + a2 * (*itr_tmp));
            }
        }
    };

    if (parallelize && outer_dim > 1)
    {
        parallel::parallel_for(0, static_cast<int>(outer_dim), parallel_eval);
    }
    else
    {
        parallel_eval(0, static_cast<int>(outer_dim));
    }
}
}  // namespace linalg::pentadiagonal_operations
