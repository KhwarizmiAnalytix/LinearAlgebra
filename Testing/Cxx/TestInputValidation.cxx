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
// dimension/leading-dimension — see the LOGGING_THROW checks added at the
// top of each op's .cxx. One representative case per op is enough here;
// this is a regression guard for that contract, not an exhaustive sweep.
TEST(Math, InputValidation)
{
    std::vector<double>       buf(9, 1.0);
    std::vector<double>       vec(3, 1.0);
    std::vector<linalg_int> pivot(4);

    EXPECT_ANY_THROW((linalg::detail::cholesky_backend<double, linalg::kActiveBackend>::apply(
        static_cast<double*>(nullptr), 3, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR)));
    EXPECT_ANY_THROW((linalg::detail::cholesky_backend<double, linalg::kActiveBackend>::apply(
        buf.data(), 0, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR)));

    EXPECT_ANY_THROW((linalg::detail::lu_backend<double, linalg::kActiveBackend>::apply(
        static_cast<double*>(nullptr), 3, pivot.data(), nullptr, 0)));
    EXPECT_ANY_THROW((linalg::detail::lu_backend<double, linalg::kActiveBackend>::apply(
        buf.data(), 0, pivot.data(), nullptr, 0)));
    EXPECT_ANY_THROW((linalg::detail::lu_backend<double, linalg::kActiveBackend>::apply(
        buf.data(), 3, static_cast<linalg_int*>(nullptr), nullptr, 0)));

    EXPECT_ANY_THROW((linalg::detail::solver_backend<double, linalg::kActiveBackend>::apply(
        static_cast<double*>(nullptr),
        pivot.data(),
        3,
        vec.data(),
        linalg::linear_solver_type::LU_LINEAR_SOLVER,
        nullptr,
        0)));
    EXPECT_ANY_THROW((linalg::detail::solver_backend<double, linalg::kActiveBackend>::apply(buf.data(),
        pivot.data(), 0, vec.data(), linalg::linear_solver_type::LU_LINEAR_SOLVER, nullptr, 0)));

    EXPECT_ANY_THROW((linalg::detail::invert_backend<double, linalg::kActiveBackend>::apply(
        static_cast<double*>(nullptr), pivot.data(), 3,
        linalg::linear_solver_type::LU_LINEAR_SOLVER, nullptr, 0)));
    EXPECT_ANY_THROW((linalg::detail::invert_backend<double, linalg::kActiveBackend>::apply(
        buf.data(), pivot.data(), 0, linalg::linear_solver_type::LU_LINEAR_SOLVER, nullptr, 0)));
    EXPECT_ANY_THROW(linalg::matrix_determinant(static_cast<double*>(nullptr), pivot.data(), 3,
        linalg::linear_solver_type::LU_LINEAR_SOLVER, nullptr, 0));

    EXPECT_ANY_THROW((linalg::detail::matmul_backend<double, linalg::kMatmulActiveBackend>::apply(false,
        false,
        3,
        3,
        3,
        static_cast<const double*>(nullptr),
        3,
        buf.data(),
        3,
        buf.data(),
        3,
        nullptr,
        0)));
    EXPECT_ANY_THROW((linalg::detail::matmul_backend<double, linalg::kMatmulActiveBackend>::apply(
        false, false, 0, 3, 3, buf.data(), 3, buf.data(), 3, buf.data(), 3, nullptr, 0)));

    EXPECT_ANY_THROW((linalg::detail::transpose_backend<double, linalg::kTransposeActiveBackend>::apply(
        3ULL, 3ULL, static_cast<double*>(nullptr), nullptr, 0)));
    EXPECT_ANY_THROW((linalg::detail::transpose_backend<double, linalg::kTransposeActiveBackend>::apply(
        0ULL, 3ULL, buf.data(), nullptr, 0)));

    EXPECT_ANY_THROW((linalg::detail::svd_backend<double, linalg::kActiveBackend>::apply(3ULL,
        3ULL,
        static_cast<double*>(nullptr),
        3ULL,
        vec.data(),
        buf.data(),
        3ULL,
        buf.data(),
        3ULL,
        nullptr,
        0)));
    EXPECT_ANY_THROW((linalg::detail::svd_backend<double, linalg::kActiveBackend>::apply(
        0ULL, 3ULL, buf.data(), 3ULL, vec.data(), buf.data(), 3ULL, buf.data(), 3ULL, nullptr, 0)));
}
