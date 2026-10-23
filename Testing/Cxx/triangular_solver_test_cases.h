#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace linalg_test
{
// Independent known solutions cover block boundaries, SIMD tails, ignored
// triangle/diagonal NaNs, and three different padded row strides.
template <typename T, typename Check> void for_each_triangular_solver_case(Check check)
{
    for (const auto& dimensions :
        std::array<std::array<int, 2>, 6>{{{1, 1}, {5, 3}, {17, 7}, {35, 1}, {35, 9}, {65, 2}}})
    {
        const int n = dimensions[0], nrhs = dimensions[1];
        const int lda = n + 2, ldb = nrhs + 1, ldx = nrhs + 3;
        for (bool upper : {false, true})
            for (bool transpose : {false, true})
                for (bool unit : {false, true})
                {
                    std::vector<T> a(n * lda, std::numeric_limits<T>::quiet_NaN());
                    std::vector<T> b(n * ldb, T(-123)), expected(n * ldx, T(-123));
                    for (int i = 0; i < n; ++i)
                    {
                        for (int j = 0; j < n; ++j)
                            if (upper ? j >= i : j <= i)
                                a[i * lda + j] = i == j
                                                     ? (unit ? std::numeric_limits<T>::quiet_NaN()
                                                             : T(2. + 0.02 * i))
                                                     : T(0.01 * std::sin(double(i * n + j + 1)));
                        for (int j = 0; j < nrhs; ++j)
                            expected[i * ldx + j] = T(std::sin(double(i * nrhs + j + 1)));
                    }
                    for (int i = 0; i < n; ++i)
                        for (int j = 0; j < nrhs; ++j)
                        {
                            T sum = T(0);
                            for (int k = 0; k < n; ++k)
                            {
                                const int row = transpose ? k : i, col = transpose ? i : k;
                                if (upper ? col < row : col > row)
                                    continue;
                                const T coefficient =
                                    unit && row == col ? T(1) : a[row * lda + col];
                                sum += coefficient * expected[k * ldx + j];
                            }
                            b[i * ldb + j] = sum;
                        }
                    check(upper, transpose, unit, n, nrhs, lda, ldb, ldx, a, b, expected);
                }
    }
}
}  // namespace linalg_test
