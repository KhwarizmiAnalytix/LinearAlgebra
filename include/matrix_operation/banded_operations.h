#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "include/common/linear_algebra_export.h"

namespace linalg::banded_operations
{

/**
 * @brief Operations on batches of banded systems.
 *
 * @tparam T      Scalar type (float or double).
 * @tparam BandCount Number of bands: 3 (tridiagonal) or 5 (pentadiagonal).
 *
 * Let N = outer_dim * dim * inner_dim. Matrix buffers store BandCount consecutive bands, each
 * with N entries. The index (outer * dim + row) * inner_dim + inner selects a position. Each of
 * the outer_dim groups contains inner_dim independent systems of order dim. All buffers are in
 * host memory.
 */

// ---- Shared operations (BandCount = 3 or 5) --------------------------------

/**
 * @brief Factors the band matrices in place for subsequent solves.
 *
 * @param[in,out] output Original band coefficients on input; factors on output. Must contain
 *     exactly BandCount * N entries.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim       Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 */
template <typename T, int BandCount>
void decomposition(std::vector<T>& output,
    size_t                         outer_dim,
    size_t                         dim,
    size_t                         inner_dim);

/**
 * @brief Solves the factored systems in place.
 *
 * @param[in,out] x         Array of N right-hand-side values; overwritten with the solutions.
 * @param[in]     decomposed Read-only factors from decomposition(), containing BandCount * N
 *     entries.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim       Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 */
template <typename T, int BandCount>
void solve_decomposed(std::vector<T>& x,
    const std::vector<T>&            decomposed,
    size_t                           outer_dim,
    size_t                           dim,
    size_t                           inner_dim);

/**
 * @brief Accumulates result += time_multiplier * mat * in.
 *
 * @param[in,out] result         Vector of N entries; the scaled product is added to existing
 *     values.
 * @param[in]     in             Input vector containing N entries.
 * @param[in]     mat            Original matrix bands, containing BandCount * N coefficients.
 * @param[in]     outer_dim      Number of groups of independent systems.
 * @param[in]     dim            Number of rows in each system.
 * @param[in]     inner_dim      Number of interleaved independent systems in each group.
 * @param[in]     time_multiplier Scale applied to the matrix-vector product.
 */
template <typename T, int BandCount>
void multiply(std::vector<T>& result,
    const std::vector<T>&     in,
    const std::vector<T>&     mat,
    size_t                    outer_dim,
    size_t                    dim,
    size_t                    inner_dim,
    T                         time_multiplier);

// ---- Three-band (tridiagonal) only operations ------------------------------

/**
 * @brief Propagates derivatives through the three-band factorization.
 *
 * @param[in,out] output_aad Factor derivatives on input; original-band derivatives on output.
 *     Contains 3 * N entries.
 * @param[in]     output     Factors from the forward decomposition, containing 3 * N entries.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim       Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 */
template <typename T>
void decomposition_aad(std::vector<T>& output_aad,
    const std::vector<T>&              output,
    size_t                             outer_dim,
    size_t                             dim,
    size_t                             inner_dim);

/**
 * @brief Solves several right-hand sides using the same three-band factors.
 *
 * @param[in,out] x         Row-major rows x columns right-hand-side matrix; overwritten with
 *     the solutions.
 * @param[in]     decomposed Read-only factors from decomposition<T,3>(), containing 3 * N
 *     entries.
 * @param[in] rows      Number of matrix rows; must equal N.
 * @param[in] columns   Number of right-hand sides.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim       Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 */
template <typename T>
void solve_decomposed_vectorised(std::vector<T>& x,
    const std::vector<T>&                        decomposed,
    size_t                                       rows,
    size_t                                       columns,
    size_t                                       outer_dim,
    size_t                                       dim,
    size_t                                       inner_dim);

/**
 * @brief Propagates solution derivatives to the three-band factors and right-hand sides.
 *
 * @param[in,out] decomposed_aad Array of 3 * N factor derivatives; contributions accumulate.
 * @param[in,out] x_aad          Array of N solution derivatives; overwritten with rhs
 *     derivatives.
 * @param[in,out] x              Forward solution of length N; restored to the original rhs.
 * @param[in]     decomposed     Read-only factors from decomposition<T,3>(), 3 * N entries.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim       Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 */
template <typename T>
void solve_decomposed_aad(std::vector<T>& decomposed_aad,
    std::vector<T>&                       x_aad,
    std::vector<T>&                       x,
    const std::vector<T>&                 decomposed,
    size_t                                outer_dim,
    size_t                                dim,
    size_t                                inner_dim);

/**
 * @brief Propagates solution derivatives to the right-hand sides only.
 *
 * @param[in,out] x_aad    Array of N solution derivatives; overwritten with rhs derivatives.
 * @param[in]     decomposed Read-only factors from decomposition<T,3>(), 3 * N entries.
 * @param[in] outer_dim Number of groups of independent systems.
 * @param[in] dim       Number of rows in each system.
 * @param[in] inner_dim Number of interleaved independent systems in each group.
 */
template <typename T>
void solve_decomposed_aad(std::vector<T>& x_aad,
    const std::vector<T>&                 decomposed,
    size_t                                outer_dim,
    size_t                                dim,
    size_t                                inner_dim);

/**
 * @brief Accumulates derivatives of the scaled three-band matrix-vector product.
 *
 * @param[in,out] mat_aad       Array of 3 * N matrix derivatives; contributions accumulate.
 * @param[in,out] in_aad        Array of N input-vector derivatives; contributions accumulate.
 * @param[in]     result_aad    Array of N output derivatives from the forward multiplication.
 * @param[in]     in            Input vector containing N entries.
 * @param[in]     mat           Original matrix bands, containing 3 * N coefficients.
 * @param[in] outer_dim      Number of groups of independent systems.
 * @param[in] dim            Number of rows in each system.
 * @param[in] inner_dim      Number of interleaved independent systems in each group.
 * @param[in] time_multiplier Scale applied to the matrix-vector product.
 */
template <typename T>
void multiply_aad(std::vector<T>& mat_aad,
    std::vector<T>&               in_aad,
    const std::vector<T>&         result_aad,
    const std::vector<T>&         in,
    const std::vector<T>&         mat,
    size_t                        outer_dim,
    size_t                        dim,
    size_t                        inner_dim,
    T                             time_multiplier);

/**
 * @brief Solves a single tridiagonal system in place.
 *
 * @tparam T Scalar value type.
 *
 * @param[in,out] output Array of n right-hand-side values; overwritten with the solution.
 * @param[in]     n      Number of rows in the system; must be at least 3.
 * @param[in]     L      Array of n - 1 lower-diagonal entries.
 * @param[in]     D      Array of n main-diagonal entries.
 * @param[in,out] U      Array of n - 1 upper-diagonal entries; modified during elimination.
 * @throws std::invalid_argument If n is less than 3.
 */
template <typename T>
void solve_banded(T* output, size_t n, T const* L, T const* D, T* U)
{
    if (n <= 2)
    {
        throw std::invalid_argument("banded system dimension is lower than 3!");
    }

    *U /= *D;
    *output /= *D;

    size_t i = 0;
    for (; i < n - 2; ++i, ++U, ++D, ++L, ++output)
    {
        const T m     = *(D + 1) - *L * *U;
        *(output + 1) = (*(output + 1) - *L * *output) / m;
        *(U + 1) /= m;
    }
    {
        const T m     = *(D + 1) - *L * *U;
        *(output + 1) = (*(output + 1) - *L * *output) / m;
    }

    for (auto k = static_cast<int>(n - 2); k >= 0; --k, --U, --output)
    {
        *output -= *U * *(output + 1);
    }
}

}  // namespace linalg::banded_operations
