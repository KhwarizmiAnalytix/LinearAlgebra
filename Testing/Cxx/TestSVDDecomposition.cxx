#if defined(_MSC_VER) && !defined(LINALG_DISPLAY_WIN32_WARNINGS)
#pragma warning(push)
#pragma warning(disable : 4305)
#endif  // _MSC_VER

#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "linear_algebra/matrix_operation/svd_decomposition.h"

namespace
{
using linalg_test::dense_matrix;

template <typename>
struct SVDTolerance
{
};

template <>
struct SVDTolerance<float>
{
    static constexpr double tolerance = 2.e-4;
};

template <>
struct SVDTolerance<double>
{
    static constexpr double tolerance = 5.e-13;
};

template <typename value_t>
void test_svd(std::size_t rows, std::size_t columns)
{
    dense_matrix<value_t> M(rows, columns);

    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            M(i, j) = static_cast<value_t>(20. * rand() / RAND_MAX);
        }
    }

    std::size_t ldu = (columns < rows ? columns : rows);

    std::vector<value_t>  D(ldu);
    dense_matrix<value_t> U(rows, ldu);
    dense_matrix<value_t> tV(ldu, columns);

    const auto ldm = columns;
    const auto ldv = columns;
    linalg::svd_decomposition(
        static_cast<quarisma_long>(rows),
        static_cast<quarisma_long>(columns),
        M.begin(),
        static_cast<quarisma_long>(ldm),
        D.data(),
        U.begin(),
        static_cast<quarisma_long>(ldu),
        tV.begin(),
        static_cast<quarisma_long>(ldv));

    value_t max_error = 0;
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            value_t sum = M(i, j);
            for (std::size_t k = 0; k < ldu; k++)
            {
                sum -= U(i, k) * D[k] * tV(k, j);
            }
            max_error = std::fmax(std::fabs(static_cast<double>(sum)), max_error);
        }
    }

    EXPECT_LT(max_error, SVDTolerance<value_t>::tolerance);
}
}  // namespace

TEST(Math, SVDDecomposition)
{
    std::size_t rows    = 6;
    std::size_t columns = 5;
    test_svd<float>(rows, columns);
    test_svd<double>(rows, columns);

    rows    = 5;
    columns = 6;
    test_svd<float>(rows, columns);
    test_svd<double>(rows, columns);

    rows    = 25;
    columns = 11;
    test_svd<float>(rows, columns);
    test_svd<double>(rows, columns);
}

#if defined(_MSC_VER) && !defined(LINALG_DISPLAY_WIN32_WARNINGS)
#pragma warning(pop)
#endif
