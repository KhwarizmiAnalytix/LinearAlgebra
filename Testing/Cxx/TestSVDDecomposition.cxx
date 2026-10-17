#if defined(_MSC_VER) && !LINALG_HAS_WIN32_WARNINGS
#pragma warning(push)
#pragma warning(disable : 4305)
#endif  // _MSC_VER

#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "test_backend.h"
#include "include/matrix_operation/svd_decomposition.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg_test::dense_matrix;

template <typename> struct SVDTolerance
{
};

template <> struct SVDTolerance<float>
{
    static constexpr double tolerance = 5.e-4;
};

template <> struct SVDTolerance<double>
{
    static constexpr double tolerance = 5.e-13;
};

template <typename value_t, linalg::linalg_backend Backend>
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

    const auto ldm     = columns;
    const auto ldv     = columns;
    const auto ws_size = linalg::svd_decomposition<value_t, Backend>::workspace_size(
        static_cast<linalg_long>(rows), static_cast<linalg_long>(columns));
    std::vector<unsigned char> ws(ws_size);
    linalg::svd_decomposition<value_t, Backend>::apply(static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        M.begin(),
        static_cast<linalg_long>(ldm),
        D.data(),
        U.begin(),
        static_cast<linalg_long>(ldu),
        tV.begin(),
        static_cast<linalg_long>(ldv),
        ws.data(),
        ws_size);

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

// Singular values must be non-negative (PyTorch test_svd_singular_values_nonneg).
template <typename value_t, linalg::linalg_backend Backend>
void test_svd_singular_values_nonneg(std::size_t rows, std::size_t columns)
{
    dense_matrix<value_t> M(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            M(i, j) = static_cast<value_t>(20. * rand() / RAND_MAX - 10.);
        }
    }

    const std::size_t     k   = std::min(rows, columns);
    std::vector<value_t>  S(k);
    dense_matrix<value_t> U(rows, k);
    dense_matrix<value_t> VT(k, columns);

    const auto ws_size = linalg::svd_decomposition<value_t, Backend>::workspace_size(
        static_cast<linalg_long>(rows), static_cast<linalg_long>(columns));
    std::vector<unsigned char> ws(ws_size);
    linalg::svd_decomposition<value_t, Backend>::apply(static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        M.begin(),
        static_cast<linalg_long>(columns),
        S.data(),
        U.begin(),
        static_cast<linalg_long>(k),
        VT.begin(),
        static_cast<linalg_long>(columns),
        ws.data(),
        ws_size);

    for (std::size_t i = 0; i < k; i++)
    {
        EXPECT_GE(static_cast<double>(S[i]), 0.0);
    }
}

// U and VT must have orthonormal columns/rows respectively.
// Mirrors torch.linalg.svd orthogonality assertions (PyTorch test_svd_orthogonality).
template <typename value_t, linalg::linalg_backend Backend>
void test_svd_orthogonality(std::size_t rows, std::size_t columns)
{
    dense_matrix<value_t> M(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            M(i, j) = static_cast<value_t>(20. * rand() / RAND_MAX - 10.);
        }
    }

    const std::size_t     k   = std::min(rows, columns);
    std::vector<value_t>  S(k);
    dense_matrix<value_t> U(rows, k);
    dense_matrix<value_t> VT(k, columns);

    const auto ws_size = linalg::svd_decomposition<value_t, Backend>::workspace_size(
        static_cast<linalg_long>(rows), static_cast<linalg_long>(columns));
    std::vector<unsigned char> ws(ws_size);
    linalg::svd_decomposition<value_t, Backend>::apply(static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        M.begin(),
        static_cast<linalg_long>(columns),
        S.data(),
        U.begin(),
        static_cast<linalg_long>(k),
        VT.begin(),
        static_cast<linalg_long>(columns),
        ws.data(),
        ws_size);

    // U^T * U == I_k
    value_t max_u_err = 0;
    for (std::size_t i = 0; i < k; i++)
    {
        for (std::size_t j = 0; j < k; j++)
        {
            value_t dot = 0;
            for (std::size_t r = 0; r < rows; r++)
            {
                dot += U(r, i) * U(r, j);
            }
            const value_t expected = (i == j) ? value_t(1) : value_t(0);
            max_u_err = std::fmax(max_u_err, std::fabs(static_cast<double>(dot - expected)));
        }
    }
    EXPECT_LT(max_u_err, SVDTolerance<value_t>::tolerance);

    // VT * VT^T == I_k
    value_t max_vt_err = 0;
    for (std::size_t i = 0; i < k; i++)
    {
        for (std::size_t j = 0; j < k; j++)
        {
            value_t dot = 0;
            for (std::size_t c = 0; c < columns; c++)
            {
                dot += VT(i, c) * VT(j, c);
            }
            const value_t expected = (i == j) ? value_t(1) : value_t(0);
            max_vt_err = std::fmax(max_vt_err, std::fabs(static_cast<double>(dot - expected)));
        }
    }
    EXPECT_LT(max_vt_err, SVDTolerance<value_t>::tolerance);
}

// Rank-deficient matrix: trailing singular values must be near zero.
// Built as A = u * v^T (rank 1) to give a known spectrum.
// Mirrors PyTorch test_svd_rank_deficient.
template <typename value_t, linalg::linalg_backend Backend>
void test_svd_rank_deficient()
{
    const std::size_t rows = 5, columns = 4;
    const std::size_t k    = std::min(rows, columns);

    std::vector<value_t> u(rows), v(columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        u[i] = static_cast<value_t>(i + 1);
    }
    for (std::size_t j = 0; j < columns; j++)
    {
        v[j] = static_cast<value_t>(j + 1);
    }

    dense_matrix<value_t> M(rows, columns);
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < columns; j++)
        {
            M(i, j) = u[i] * v[j];
        }
    }

    std::vector<value_t>  S(k);
    dense_matrix<value_t> U(rows, k);
    dense_matrix<value_t> VT(k, columns);

    const auto ws_size = linalg::svd_decomposition<value_t, Backend>::workspace_size(
        static_cast<linalg_long>(rows), static_cast<linalg_long>(columns));
    std::vector<unsigned char> ws(ws_size);
    linalg::svd_decomposition<value_t, Backend>::apply(static_cast<linalg_long>(rows),
        static_cast<linalg_long>(columns),
        M.begin(),
        static_cast<linalg_long>(columns),
        S.data(),
        U.begin(),
        static_cast<linalg_long>(k),
        VT.begin(),
        static_cast<linalg_long>(columns),
        ws.data(),
        ws_size);

    // Only 1 singular value should be non-negligible; the rest must be near 0.
    value_t smax = *std::max_element(S.begin(), S.end());
    int     nonzero_count = 0;
    for (std::size_t i = 0; i < k; i++)
    {
        if (S[i] > static_cast<value_t>(1e-3) * smax)
        {
            nonzero_count++;
        }
    }
    EXPECT_EQ(nonzero_count, 1);
}
}  // namespace

TEST(Math, SVDDecomposition)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr auto Backend = decltype(backend)::value;

        std::size_t rows    = 6;
        std::size_t columns = 5;
        test_svd<float, Backend>(rows, columns);
        test_svd<double, Backend>(rows, columns);

        rows    = 5;
        columns = 6;
        test_svd<float, Backend>(rows, columns);
        test_svd<double, Backend>(rows, columns);

        rows    = 25;
        columns = 11;
        test_svd<float, Backend>(rows, columns);
        test_svd<double, Backend>(rows, columns);
    });
}

TEST(Math, SVDSingularValuesNonneg)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr auto Backend = decltype(backend)::value;
        test_svd_singular_values_nonneg<float, Backend>(6, 5);
        test_svd_singular_values_nonneg<float, Backend>(5, 6);
        test_svd_singular_values_nonneg<double, Backend>(6, 5);
        test_svd_singular_values_nonneg<double, Backend>(5, 6);
    });
}

TEST(Math, SVDOrthogonality)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr auto Backend = decltype(backend)::value;
        test_svd_orthogonality<float, Backend>(8, 5);
        test_svd_orthogonality<float, Backend>(5, 8);
        test_svd_orthogonality<double, Backend>(8, 5);
        test_svd_orthogonality<double, Backend>(5, 8);
    });
}

TEST(Math, SVDRankDeficient)
{
    linalg_test::for_each_available_backend([](auto backend) {
        constexpr auto Backend = decltype(backend)::value;
        test_svd_rank_deficient<float, Backend>();
        test_svd_rank_deficient<double, Backend>();
    });
}

#if defined(_MSC_VER) && !LINALG_HAS_WIN32_WARNINGS
#pragma warning(pop)
#endif
