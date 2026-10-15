#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <vector>

#include "include/common/linear_algebra_export.h"

namespace linalg
{
/**
 * @brief Operations on batches of tridiagonal systems.
 *
 * Let N = outer_dim * dim * inner_dim. Matrix buffers store 3 consecutive bands (lower, main,
 * upper), each with N entries. The index (outer * dim + row) * inner_dim + inner selects a
 * position. Each of the outer_dim groups contains inner_dim independent systems of order dim.
 * All buffers are in host memory.
 */
namespace tridiagonal_operations
{

/**
 * @brief Factors the band matrices in place for subsequent solves.
 *
 * @param[in,out] output Original band coefficients on input; factors on output. Must contain
 *     exactly 3 * N entries.
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
 * @brief Propagates derivatives through the band factorization.
 *
 * @param[in,out] output_aad Factor derivatives on input; original-band derivatives on output.
 *     Contains 3 * N entries.
 * @param[in] output Factors from the forward decomposition, containing 3 * N entries.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 * @param[in] parallelize Reserved parallel-execution flag; currently ignored.
 */
LINALG_API void decomposition_aad(std::vector<double>& output_aad,
    const std::vector<double>&                         output,
    const size_t                                       outer_dim,
    const size_t                                       dim,
    const size_t                                       inner_dim,
    const bool                                         parallelize = false);

/**
 * @brief Solves the factored systems in place.
 *
 * @param[in,out] x Array of N right-hand-side values; overwritten with the solutions.
 * @param[in] decomposed Read-only factors from decomposition(), containing 3 * N entries.
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
 * @brief Solves several right-hand sides using the same factors.
 *
 * @param[in,out] x Row-major rows x columns right-hand-side matrix; overwritten with the
 *     solutions.
 * @param[in] decomposed Read-only factors from decomposition(), containing 3 * N entries.
 * @param[in] rows Number of matrix rows; must equal N.
 * @param[in] columns Number of right-hand sides.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 * @param[in] parallelize Reserved parallel-execution flag; currently ignored.
 */
LINALG_API void solve_decomposed_vectorised(std::vector<double>& x,
    const std::vector<double>&                                       decomposed,
    size_t                                                           rows,
    size_t                                                           columns,
    const size_t                                                     outer_dim,
    const size_t                                                     dim,
    const size_t                                                     inner_dim,
    const bool                                                       parallelize = false);

/**
 * @brief Propagates solution derivatives to the factors and right-hand sides.
 *
 * @param[in,out] decomposed_aad Array of 3 * N factor derivatives; contributions accumulate
 *     into existing values.
 * @param[in,out] x_aad Array of N solution derivatives; overwritten with right-hand-side
 *     derivatives.
 * @param[in,out] x Forward solution of length N; restored to the original right-hand side
 *     during propagation.
 * @param[in] decomposed Read-only factors from decomposition(), containing 3 * N entries.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 * @param[in] parallelize Reserved parallel-execution flag; currently ignored.
 */
LINALG_API void solve_decomposed_aad(std::vector<double>& decomposed_aad,
    std::vector<double>&                                      x_aad,
    std::vector<double>&                                      x,
    const std::vector<double>&                                decomposed,
    const size_t                                              outer_dim,
    const size_t                                              dim,
    const size_t                                              inner_dim,
    const bool                                                parallelize = false);

/**
 * @brief Propagates solution derivatives to the right-hand sides.
 *
 * @param[in,out] x_aad Array of N solution derivatives; overwritten with right-hand-side
 *     derivatives.
 * @param[in] decomposed Read-only factors from decomposition(), containing 3 * N entries.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 */
LINALG_API void solve_decomposed_aad(std::vector<double>& x_aad,
    const std::vector<double>&                                decomposed,
    const size_t                                              outer_dim,
    const size_t                                              dim,
    const size_t                                              inner_dim);

/**
 * @brief Accumulates result += time_multiplier * mat * in.
 *
 * @param[in,out] result Vector of N entries; the scaled product is added to its existing
 *     values.
 * @param[in] in Input vector containing N entries.
 * @param[in] mat Original matrix bands, containing 3 * N coefficients.
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

/**
 * @brief Accumulates derivatives of the scaled matrix-vector product.
 *
 * @param[in,out] mat_aad Array of 3 * N matrix derivatives; contributions accumulate into
 *     existing values.
 * @param[in,out] in_aad Array of N input-vector derivatives; contributions accumulate into
 *     existing values.
 * @param[in] result_aad Array of N output derivatives from the forward multiplication.
 * @param[in] in Input vector containing N entries.
 * @param[in] mat Original matrix bands, containing 3 * N coefficients.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 * @param[in] time_multiplier Scale applied to the matrix-vector product.
 * @param[in] parallelize Reserved parallel-execution flag; currently ignored.
 */
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

/**
 * @brief Solves a single tridiagonal system in place.
 *
 * @tparam T Scalar value type.
 *
 * @param[in,out] output Array of n right-hand-side values; overwritten with the solution.
 * @param[in] n Number of rows in the system; must be at least 3.
 * @param[in] L Array of n - 1 lower-diagonal entries.
 * @param[in] D Array of n main-diagonal entries.
 * @param[in,out] U Array of n - 1 upper-diagonal entries; modified during elimination.
 * @throws std::invalid_argument If n is less than 3.
 */
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
