#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <vector>

#include "include/common/linear_algebra_export.h"

namespace linalg
{
namespace tridiagonal_operations
{

/**
 * @brief Perform decomposition on the matrix to separate it into parts.
 *
 * @p output holds 3 band-rows of length outer_dim*dim*inner_dim, stored back
 * to back in a single flat, row-major buffer (row k spans
 * [k * outer_dim*dim*inner_dim, (k+1) * outer_dim*dim*inner_dim)).
 *
 * @param output The output matrix after decomposition.
 * @param outer_dim The dimension of the outer part.
 * @param dim The dimension of the matrix.
 * @param inner_dim The dimension of the inner part.
 * @param parallelize Flag to indicate if decomposition should be parallelized.
 */
LINALG_API void decomposition(std::vector<double>& output,
    const size_t                                   outer_dim,
    const size_t                                   dim,
    const size_t                                   inner_dim,
    const bool                                     parallelize = false);

LINALG_API void decomposition_aad(std::vector<double>& output_aad,
    const std::vector<double>&                         output,
    const size_t                                       outer_dim,
    const size_t                                       dim,
    const size_t                                       inner_dim,
    const bool                                         parallelize = false);

LINALG_API void solve_decomposed(std::vector<double>& x,
    const std::vector<double>&                        decomposed,
    const size_t                                      outer_dim,
    const size_t                                      dim,
    const size_t                                      inner_dim,
    const bool                                        parallelize = false);

/**
 * @brief Vectorised variant of solve_decomposed solving several right-hand
 * sides at once.
 *
 * @p x is a flat row-major buffer of shape (rows, columns): rows equal
 * outer_dim*dim*inner_dim (one row per tridiagonal position, matching
 * @p decomposed's row length) and columns is the number of right-hand sides
 * solved in parallel.
 */
LINALG_API void solve_decomposed_vectorised(std::vector<double>& x,
    const std::vector<double>&                                       decomposed,
    size_t                                                           rows,
    size_t                                                           columns,
    const size_t                                                     outer_dim,
    const size_t                                                     dim,
    const size_t                                                     inner_dim,
    const bool                                                       parallelize = false);

LINALG_API void solve_decomposed_aad(std::vector<double>& decomposed_aad,
    std::vector<double>&                                      x_aad,
    std::vector<double>&                                      x,
    const std::vector<double>&                                decomposed,
    const size_t                                              outer_dim,
    const size_t                                              dim,
    const size_t                                              inner_dim,
    const bool                                                parallelize = false);

LINALG_API void solve_decomposed_aad(std::vector<double>& x_aad,
    const std::vector<double>&                                decomposed,
    const size_t                                              outer_dim,
    const size_t                                              dim,
    const size_t                                              inner_dim);

LINALG_API void multiply(std::vector<double>& result,
    const std::vector<double>&                    in,
    const std::vector<double>&                    mat,
    const size_t                                  outer_dim,
    const size_t                                  dim,
    const size_t                                  inner_dim,
    double                                        time_multiplier,
    const bool                                    parallelize = false);

LINALG_API void multiply_aad(std::vector<double>& mat_aad,
    std::vector<double>&                              in_aad,
    const std::vector<double>&                        result_aad,
    const std::vector<double>&                        in,
    const std::vector<double>&                        mat,
    const size_t                                      outer_dim,
    const size_t                                      dim,
    const size_t                                      inner_dim,
    double                                            time_multiplier,
    const bool                                        parallelize = false);

template <typename T>
void solve_tridiagonal(T* output, const size_t n, T const* L, T const* D, T* U)
{
    if (n <= 2)
    {
        throw std::invalid_argument("tridiagonal dimension is lower than 3!");
    }

    *U /= *D;
    *output /= *D;

    size_t i = 0;

    for (; i < n - 2; ++i, ++U, ++D, ++L, ++output)
    {
        const T m     = (*(D + 1) - *L * *U);
        *(output + 1) = (*(output + 1) - *L * *output) / m;
        *(U + 1) /= m;
    }
    {
        const T m     = (*(D + 1) - *L * *U);
        *(output + 1) = (*(output + 1) - *L * *output) / m;
    }

    for (auto k = static_cast<int>(n - 2); k >= 0; --k, --U, --output)
    {
        *output -= *U * *(output + 1);
    }
};

};  // namespace tridiagonal_operations
}  // namespace linalg
