#include <vector>

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/matrix_operation/matrix_inversion.h"
#include "include/matrix_operation/matrix_multiplication.h"
#include "include/matrix_operation/matrix_transpose.h"
#include "include/matrix_operation/svd_decomposition.h"
#include "gtest/gtest.h"

// Every public matrix_operation entry point throws (rather than exhibiting
// undefined behavior) on a null required pointer or a non-positive
// dimension/leading-dimension — see the LINALG_THROW checks added at the
// top of each op's .cxx. One representative case per op is enough here;
// this is a regression guard for that contract, not an exhaustive sweep.
TEST(Math, InputValidation)
{
    std::vector<double>       buf(9, 1.0);
    std::vector<double>       vec(3, 1.0);
    std::vector<linalg_int> pivot(4);

    EXPECT_ANY_THROW(linalg::cholesky_decomposition(
        static_cast<double*>(nullptr), 3, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR));
    EXPECT_ANY_THROW(linalg::cholesky_decomposition(
        buf.data(), 0, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR));

    EXPECT_ANY_THROW(linalg::lu_decomposition(static_cast<double*>(nullptr), 3, pivot.data()));
    EXPECT_ANY_THROW(linalg::lu_decomposition(buf.data(), 0, pivot.data()));
    EXPECT_ANY_THROW(linalg::lu_decomposition(buf.data(), 3, static_cast<linalg_int*>(nullptr)));

    EXPECT_ANY_THROW(linalg::linear_solver(static_cast<double*>(nullptr),
        pivot.data(),
        3,
        vec.data(),
        linalg::linear_solver_type::LU_LINEAR_SOLVER));
    EXPECT_ANY_THROW(linalg::linear_solver(
        buf.data(), pivot.data(), 0, vec.data(), linalg::linear_solver_type::LU_LINEAR_SOLVER));

    EXPECT_ANY_THROW(linalg::matrix_invert(static_cast<double*>(nullptr), pivot.data(), 3));
    EXPECT_ANY_THROW(linalg::matrix_invert(buf.data(), pivot.data(), 0));
    EXPECT_ANY_THROW(linalg::matrix_determinant(static_cast<double*>(nullptr), pivot.data(), 3));

    EXPECT_ANY_THROW(linalg::matrix_multiplication(false,
        false,
        3,
        3,
        3,
        static_cast<const double*>(nullptr),
        3,
        buf.data(),
        3,
        buf.data(),
        3));
    EXPECT_ANY_THROW(linalg::matrix_multiplication(
        false, false, 0, 3, 3, buf.data(), 3, buf.data(), 3, buf.data(), 3));

    EXPECT_ANY_THROW(linalg::matrix_transpose(3ULL, 3ULL, static_cast<double*>(nullptr)));
    EXPECT_ANY_THROW(linalg::matrix_transpose(0ULL, 3ULL, buf.data()));

    EXPECT_ANY_THROW(linalg::svd_decomposition(3ULL,
        3ULL,
        static_cast<double*>(nullptr),
        3ULL,
        vec.data(),
        buf.data(),
        3ULL,
        buf.data(),
        3ULL));
    EXPECT_ANY_THROW(linalg::svd_decomposition(
        0ULL, 3ULL, buf.data(), 3ULL, vec.data(), buf.data(), 3ULL, buf.data(), 3ULL));
}
