#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "include/matrix_operation/pseudo_inverse.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg_test::dense_matrix;

template <typename> struct PseudoInverseTolerance
{
};
template <> struct PseudoInverseTolerance<float>
{
    static constexpr double value = 1e-4;
};
template <> struct PseudoInverseTolerance<double>
{
    static constexpr double value = 1e-12;
};

template <typename T>
dense_matrix<T> mat_sub(const dense_matrix<T>& A, const dense_matrix<T>& B)
{
    dense_matrix<T> C(A.rows(), A.cols());
    for (std::size_t i = 0; i < A.rows(); i++)
        for (std::size_t j = 0; j < A.cols(); j++)
            C(i, j) = A(i, j) - B(i, j);
    return C;
}

template <typename T>
double frobenius(const dense_matrix<T>& M)
{
    double s = 0;
    for (std::size_t i = 0; i < M.rows(); i++)
        for (std::size_t j = 0; j < M.cols(); j++)
            s += static_cast<double>(M(i, j)) * static_cast<double>(M(i, j));
    return std::sqrt(s);
}

template <typename T>
dense_matrix<T> compute_pinv(const dense_matrix<T>& A, T tol = T(-1))
{
    dense_matrix<T> Ainv(A.cols(), A.rows());
    const auto      ws_size =
        linalg::pseudo_inverse<T, linalg::linalg_backend::native>::workspace_size(
            static_cast<linalg_long>(A.rows()), static_cast<linalg_long>(A.cols()));
    std::vector<unsigned char> ws(ws_size);
    linalg::pseudo_inverse<T, linalg::linalg_backend::native>::apply(
        static_cast<linalg_long>(A.rows()),
        static_cast<linalg_long>(A.cols()),
        A.begin(),
        static_cast<linalg_long>(A.cols()),
        Ainv.begin(),
        static_cast<linalg_long>(A.rows()),
        tol,
        ws.data(),
        ws_size);
    return Ainv;
}

// All four Moore–Penrose identities, verified via relative Frobenius residuals.
// Mirrors numpy.linalg.pinv validation (test_pinv_consistency in numpy test suite).
template <typename T>
void check_mp_identities(const dense_matrix<T>& A, const dense_matrix<T>& Ainv)
{
    const double tol = PseudoInverseTolerance<T>::value;

    // 1. A * Ainv * A == A
    {
        const double residual = frobenius(mat_sub(matmul(matmul(A, Ainv), A), A));
        EXPECT_LT(residual / frobenius(A), tol);
    }

    // 2. Ainv * A * Ainv == Ainv
    {
        const double residual = frobenius(mat_sub(matmul(matmul(Ainv, A), Ainv), Ainv));
        EXPECT_LT(residual / frobenius(Ainv), tol);
    }

    // 3. A * Ainv is symmetric
    {
        const auto   P        = matmul(A, Ainv);
        const double residual = frobenius(mat_sub(P, transpose(P)));
        EXPECT_LT(residual / frobenius(P), tol);
    }

    // 4. Ainv * A is symmetric
    {
        const auto   Q        = matmul(Ainv, A);
        const double residual = frobenius(mat_sub(Q, transpose(Q)));
        EXPECT_LT(residual / frobenius(Q), tol);
    }
}

// Full column-rank tall matrix: pinv is a left inverse, Ainv*A ≈ I_cols.
template <typename T>
void test_pseudo_inverse_tall(std::size_t rows, std::size_t cols)
{
    dense_matrix<T> A(rows, cols);
    for (std::size_t i = 0; i < rows; i++)
        for (std::size_t j = 0; j < cols; j++)
            A(i, j) = static_cast<T>(20. * rand() / RAND_MAX - 10.);

    const auto Ainv = compute_pinv(A);
    check_mp_identities(A, Ainv);

    const double tol = PseudoInverseTolerance<T>::value;
    const auto   P   = matmul(Ainv, A);
    for (std::size_t i = 0; i < cols; i++)
        for (std::size_t j = 0; j < cols; j++)
        {
            const double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(static_cast<double>(P(i, j)), expected, tol);
        }
}

// Full row-rank wide matrix: pinv is a right inverse, A*Ainv ≈ I_rows.
template <typename T>
void test_pseudo_inverse_wide(std::size_t rows, std::size_t cols)
{
    dense_matrix<T> A(rows, cols);
    for (std::size_t i = 0; i < rows; i++)
        for (std::size_t j = 0; j < cols; j++)
            A(i, j) = static_cast<T>(20. * rand() / RAND_MAX - 10.);

    const auto Ainv = compute_pinv(A);
    check_mp_identities(A, Ainv);

    const double tol = PseudoInverseTolerance<T>::value;
    const auto   P   = matmul(A, Ainv);
    for (std::size_t i = 0; i < rows; i++)
        for (std::size_t j = 0; j < rows; j++)
        {
            const double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(static_cast<double>(P(i, j)), expected, tol);
        }
}

// Rank-deficient: 4x4 block diag [[I_2, 0],[0, 0]], rank=2.
// pinv equals A itself; tests that zero singular values are correctly discarded.
template <typename T>
void test_pseudo_inverse_rank_deficient()
{
    dense_matrix<T> A(4, 4);
    A.fill(T(0));
    A(0, 0) = T(1);
    A(1, 1) = T(1);

    const auto   Ainv = compute_pinv(A);
    check_mp_identities(A, Ainv);

    const double tol = PseudoInverseTolerance<T>::value;
    for (std::size_t i = 0; i < 4; i++)
        for (std::size_t j = 0; j < 4; j++)
        {
            const double expected = (i == j && i < 2) ? 1.0 : 0.0;
            EXPECT_NEAR(static_cast<double>(Ainv(i, j)), expected, tol);
        }
}

// Known analytic case: A = [[1,0],[0,1],[0,0]] (3x2, rank 2).
// pinv(A) = [[1,0,0],[0,1,0]] (2x3).
template <typename T>
void test_pseudo_inverse_known()
{
    dense_matrix<T> A(3, 2);
    A.fill(T(0));
    A(0, 0) = T(1);
    A(1, 1) = T(1);

    const auto Ainv = compute_pinv(A);
    check_mp_identities(A, Ainv);

    const double tol                  = PseudoInverseTolerance<T>::value;
    const double expected[2][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    for (std::size_t i = 0; i < 2; i++)
        for (std::size_t j = 0; j < 3; j++)
            EXPECT_NEAR(static_cast<double>(Ainv(i, j)), expected[i][j], tol);
}

}  // namespace

TEST(Math, PseudoInverse)
{
    test_pseudo_inverse_tall<float>(6, 4);
    test_pseudo_inverse_tall<double>(6, 4);
    test_pseudo_inverse_tall<float>(9, 3);
    test_pseudo_inverse_tall<double>(9, 3);
}

TEST(Math, PseudoInverseWide)
{
    test_pseudo_inverse_wide<float>(4, 6);
    test_pseudo_inverse_wide<double>(4, 6);
    test_pseudo_inverse_wide<float>(3, 9);
    test_pseudo_inverse_wide<double>(3, 9);
}

TEST(Math, PseudoInverseRankDeficient)
{
    test_pseudo_inverse_rank_deficient<float>();
    test_pseudo_inverse_rank_deficient<double>();
}

TEST(Math, PseudoInverseKnown)
{
    test_pseudo_inverse_known<float>();
    test_pseudo_inverse_known<double>();
}
