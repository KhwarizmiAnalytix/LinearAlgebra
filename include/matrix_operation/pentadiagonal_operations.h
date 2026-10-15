#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <vector>

#include "include/common/linear_algebra_export.h"

namespace linalg
{
/**
 * @brief Operations on batches of pentadiagonal systems.
 *
 * Let N = outer_dim * dim * inner_dim. Matrix buffers store 5 consecutive bands (second lower,
 * lower, main, upper, second upper), each with N entries. The index (outer * dim + row) *
 * inner_dim + inner selects a position. Each of the outer_dim groups contains inner_dim
 * independent systems of order dim. All buffers are in host memory.
 */
namespace pentadiagonal_operations
{

/**
 * @brief Factors the band matrices in place for subsequent solves.
 *
 * @param[in,out] output Original band coefficients on input; factors on output. Must contain
 *     exactly 5 * N entries.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 * @param[in] parallelize Reserved parallel-execution flag; currently ignored.
 */
LINALG_API void decomposition(std::vector<double>& output,
    const size_t                                   outer_dim,
    const size_t                                   dim,
    const size_t                                   inner_dim,
    const bool                                     parallelize = false);

/**
 * @brief Solves the factored systems in place.
 *
 * @param[in,out] x Array of N right-hand-side values; overwritten with the solutions.
 * @param[in] decomposed Read-only factors from decomposition(), containing 5 * N entries.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 * @param[in] parallelize Reserved parallel-execution flag; currently ignored.
 */
LINALG_API void solve_decomposed(std::vector<double>& x,
    const std::vector<double>&                        decomposed,
    const size_t                                      outer_dim,
    const size_t                                      dim,
    const size_t                                      inner_dim,
    const bool                                        parallelize = false);

/**
 * @brief Accumulates result += time_multiplier * mat * in.
 *
 * @param[in,out] result Vector of N entries; the scaled product is added to its existing
 *     values.
 * @param[in] in Input vector containing N entries.
 * @param[in] mat Original matrix bands, containing 5 * N coefficients.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 * @param[in] time_multiplier Scale applied to the matrix-vector product.
 * @param[in] parallelize Reserved parallel-execution flag; currently ignored.
 */
LINALG_API void multiply(std::vector<double>& result,
    const std::vector<double>&                    in,
    const std::vector<double>&                    mat,
    const size_t                                  outer_dim,
    const size_t                                  dim,
    const size_t                                  inner_dim,
    double                                        time_multiplier,
    const bool                                    parallelize = false);
};  // namespace pentadiagonal_operations
}  // namespace linalg
