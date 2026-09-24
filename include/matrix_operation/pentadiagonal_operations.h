#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <vector>

#include "include/common/linear_algebra_export.h"

namespace linalg
{
namespace pentadiagonal_operations
{

/**
 * @brief 5 band-rows of length outer_dim*dim*inner_dim, stored back to back
 * in a single flat, row-major buffer (row k spans
 * [k * outer_dim*dim*inner_dim, (k+1) * outer_dim*dim*inner_dim)).
 */
LINALG_API void decomposition(std::vector<double>& output,
    const size_t                                   outer_dim,
    const size_t                                   dim,
    const size_t                                   inner_dim,
    const bool                                     parallelize = false);

LINALG_API void solve_decomposed(std::vector<double>& x,
    const std::vector<double>&                        decomposed,
    const size_t                                      outer_dim,
    const size_t                                      dim,
    const size_t                                      inner_dim,
    const bool                                        parallelize = false);

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
