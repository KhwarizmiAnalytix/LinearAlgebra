#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "include/matrix_operation/eigenvalue_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/matrix_operation/matrix_inversion.h"
#include "include/matrix_operation/qr_decomposition.h"

namespace
{
using linalg_test::dense_matrix;

template <typename>
struct EigenTolerance
{
};

template <> struct EigenTolerance<float>
{
    static constexpr double tolerance = 2.e-3;
};

template <> struct EigenTolerance<double>
{
    static constexpr double tolerance = 1.e-8;
};

// Symmetric case: A = Q * diag(known eigenvalues) * Q^T for a random
// orthogonal Q (built from qr_decomposition). Eigenvalues must come back
// sorted ascending and V^T * A * V must reconstruct the diagonal.
template <typename value_t> void test_symmetric_eigen(std::size_t n)
{
    dense_matrix<value_t> R0(n, n);
    for (std::size_t i = 0; i < n * n; i++)
    {
        R0.begin()[i] = static_cast<value_t>(2. * rand() / RAND_MAX - 1.);
    }
    dense_matrix<value_t> Q(n, n);
    dense_matrix<value_t> R(n, n);
    linalg::qr_decomposition(
        static_cast<linalg_long>(n),
        static_cast<linalg_long>(n),
        R0.begin(),
        static_cast<linalg_long>(n),
        Q.begin(),
        static_cast<linalg_long>(n),
        R.begin(),
        static_cast<linalg_long>(n));

    std::vector<value_t> known(n);
    for (std::size_t i = 0; i < n; i++)
    {
        known[i] = static_cast<value_t>(i + 1);
    }

    dense_matrix<value_t> A(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < n; k++)
            {
                sum += Q(i, k) * known[k] * Q(j, k);
            }
            A(i, j) = sum;
        }
    }

    std::vector<value_t>  evals(n);
    dense_matrix<value_t> evecs(n, n);
    const bool             ok = linalg::symmetric_eigenvalue_decomposition(
        A.begin(), static_cast<linalg_int>(n), static_cast<linalg_int>(n), evals.data(), evecs.begin(),
        static_cast<linalg_int>(n));
    EXPECT_TRUE(ok);

    for (std::size_t i = 0; i < n; i++)
    {
        EXPECT_NEAR(static_cast<double>(evals[i]), static_cast<double>(known[i]), EigenTolerance<value_t>::tolerance);
    }

    // AV = A * evecs, then check evecs^T * AV == diag(evals).
    dense_matrix<value_t> AV(n, n);
    for (std::size_t p = 0; p < n; p++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            value_t sum = 0;
            for (std::size_t q = 0; q < n; q++)
            {
                sum += A(p, q) * evecs(q, j);
            }
            AV(p, j) = sum;
        }
    }

    value_t max_error = 0;
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            value_t sum = 0;
            for (std::size_t p = 0; p < n; p++)
            {
                sum += evecs(p, i) * AV(p, j);
            }
            const value_t expected = (i == j) ? evals[i] : value_t(0);
            max_error               = std::max(max_error, static_cast<value_t>(std::fabs(sum - expected)));
        }
    }
    EXPECT_LT(max_error, static_cast<value_t>(EigenTolerance<value_t>::tolerance));
}

// General (non-symmetric) case with a known real spectrum: A = P * D * P^-1
// for a diagonally-dominant (hence invertible) random P.
template <typename value_t> void test_general_eigen_real_spectrum(std::size_t n)
{
    dense_matrix<value_t> P(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            P(i, j) = static_cast<value_t>(2. * rand() / RAND_MAX - 1.);
        }
        P(i, i) += static_cast<value_t>(5);
    }

    dense_matrix<value_t> Pinv(n, n);
    for (std::size_t i = 0; i < n * n; i++)
    {
        Pinv.begin()[i] = P.begin()[i];
    }
    std::vector<linalg_int> pivot(n);
    linalg::matrix_invert(
        Pinv.begin(), pivot.data(), static_cast<linalg_int>(n), linalg::linear_solver_type::LU_LINEAR_SOLVER);

    std::vector<value_t> known(n);
    for (std::size_t i = 0; i < n; i++)
    {
        known[i] = static_cast<value_t>(2 * static_cast<int>(i) - 3);
    }

    dense_matrix<value_t> A(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < n; k++)
            {
                sum += P(i, k) * known[k] * Pinv(k, j);
            }
            A(i, j) = sum;
        }
    }

    std::vector<value_t> er(n);
    std::vector<value_t> ei(n);
    const bool             ok = linalg::eigenvalue_decomposition(
        A.begin(), static_cast<linalg_int>(n), static_cast<linalg_int>(n), er.data(), ei.data(), nullptr,
        static_cast<linalg_int>(n));
    EXPECT_TRUE(ok);

    std::vector<value_t> got = er;
    std::sort(got.begin(), got.end());
    for (std::size_t i = 0; i < n; i++)
    {
        EXPECT_NEAR(static_cast<double>(got[i]), static_cast<double>(known[i]), EigenTolerance<value_t>::tolerance);
        EXPECT_NEAR(static_cast<double>(ei[i]), 0.0, EigenTolerance<value_t>::tolerance);
    }
}

// Complex-conjugate pair: a 2x2 rotation-scale block [[a,-b],[b,a]] has
// eigenvalues a +- b*i exactly.
template <typename value_t> void test_general_eigen_complex_pair()
{
    const value_t a = static_cast<value_t>(1.5);
    const value_t b = static_cast<value_t>(2.0);
    dense_matrix<value_t> A(2, 2);
    A(0, 0) = a;
    A(0, 1) = -b;
    A(1, 0) = b;
    A(1, 1) = a;

    value_t er[2];
    value_t ei[2];
    const bool ok = linalg::eigenvalue_decomposition(A.begin(), 2, 2, er, ei, nullptr, 2);
    EXPECT_TRUE(ok);

    EXPECT_NEAR(static_cast<double>(er[0]), static_cast<double>(a), EigenTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(er[1]), static_cast<double>(a), EigenTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(std::fabs(ei[0])), static_cast<double>(b), EigenTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(std::fabs(ei[1])), static_cast<double>(b), EigenTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(ei[0] + ei[1]), 0.0, EigenTolerance<value_t>::tolerance);
}

// Real-eigenvector residual check on a lower-triangular matrix (eigenvalues
// are exactly its diagonal entries).
template <typename value_t> void test_general_eigen_eigenvectors()
{
    dense_matrix<value_t> A(3, 3);
    A(0, 0) = 2;
    A(1, 0) = 1;
    A(1, 1) = 3;
    A(2, 1) = 1;
    A(2, 2) = 4;

    value_t                er[3];
    value_t                ei[3];
    dense_matrix<value_t>  evecs(3, 3);
    const bool ok = linalg::eigenvalue_decomposition(A.begin(), 3, 3, er, ei, evecs.begin(), 3);
    EXPECT_TRUE(ok);

    for (std::size_t i = 0; i < 3; i++)
    {
        value_t max_error = 0;
        for (std::size_t r = 0; r < 3; r++)
        {
            value_t sum = 0;
            for (std::size_t c = 0; c < 3; c++)
            {
                sum += A(r, c) * evecs(c, i);
            }
            const value_t expected = er[i] * evecs(r, i);
            max_error               = std::max(max_error, static_cast<value_t>(std::fabs(sum - expected)));
        }
        EXPECT_LT(max_error, static_cast<value_t>(EigenTolerance<value_t>::tolerance));
    }
}
}  // namespace

TEST(Math, SymmetricEigenvalueDecomposition)
{
    test_symmetric_eigen<float>(5);
    test_symmetric_eigen<double>(5);
    test_symmetric_eigen<double>(8);
}

TEST(Math, GeneralEigenvalueDecompositionRealSpectrum)
{
    test_general_eigen_real_spectrum<float>(4);
    test_general_eigen_real_spectrum<double>(4);
    test_general_eigen_real_spectrum<double>(6);
}

TEST(Math, GeneralEigenvalueDecompositionComplexPair)
{
    test_general_eigen_complex_pair<float>();
    test_general_eigen_complex_pair<double>();
}

TEST(Math, GeneralEigenvalueDecompositionEigenvectors)
{
    test_general_eigen_eigenvectors<float>();
    test_general_eigen_eigenvectors<double>();
}

TEST(Math, GeneralEigenvalueDecompositionComplexEigenvectorThrows)
{
    dense_matrix<double> A(2, 2);
    A(0, 0) = 0;
    A(0, 1) = -1;
    A(1, 0) = 1;
    A(1, 1) = 0;

    double er[2];
    double ei[2];
    double evecs[4];
    EXPECT_THROW(linalg::eigenvalue_decomposition(A.begin(), 2, 2, er, ei, evecs, 2), std::exception);
}
