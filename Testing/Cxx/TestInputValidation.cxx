#include <vector>

#include "test_backend.h"
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/matrix_operation/inversion.h"
#include "include/matrix_operation/determinant.h"
#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/transpose.h"
#include "include/matrix_operation/svd_decomposition.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

// Every public matrix_operation entry point throws (rather than exhibiting
// undefined behavior) on a null required pointer or a non-positive
// dimension/leading-dimension — see the LINALG_THROW checks added at the
// top of each op's .cxx. One representative case per op is enough here;
// this is a regression guard for that contract, not an exhaustive sweep.
TEST(Math, InputValidation)
{
    std::vector<double>     buf(9, 1.0);
    std::vector<double>     vec(3, 1.0);
    std::vector<linalg_int> pivot(4);

    linalg_test::for_each_available_backend([&](auto backend) {
        using B = decltype(backend);

        EXPECT_ANY_THROW((linalg::cholesky_decomposition<double, B::value>::apply(
            static_cast<double*>(nullptr), 3, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR,
            nullptr, 0, nullptr)));
        EXPECT_ANY_THROW((linalg::cholesky_decomposition<double, B::value>::apply(
            buf.data(), 0, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR, nullptr, 0, nullptr)));

        EXPECT_ANY_THROW((linalg::lu_decomposition<double, B::value>::apply(
            static_cast<double*>(nullptr), 3, pivot.data(), nullptr, 0)));
        EXPECT_ANY_THROW((linalg::lu_decomposition<double, B::value>::apply(
            buf.data(), 0, pivot.data(), nullptr, 0)));
        EXPECT_ANY_THROW((linalg::lu_decomposition<double, B::value>::apply(
            buf.data(), 3, static_cast<linalg_int*>(nullptr), nullptr, 0)));

        EXPECT_ANY_THROW((linalg::linear_solver<double, B::value>::apply(
            static_cast<double*>(nullptr),
            pivot.data(),
            3,
            vec.data(),
            linalg::linear_solver_type::LU_LINEAR_SOLVER,
            nullptr,
            0)));
        EXPECT_ANY_THROW((linalg::linear_solver<double, B::value>::apply(buf.data(),
            pivot.data(),
            0,
            vec.data(),
            linalg::linear_solver_type::LU_LINEAR_SOLVER,
            nullptr,
            0)));

        EXPECT_ANY_THROW((linalg::inversion<double, B::value>::apply(static_cast<double*>(nullptr),
            pivot.data(),
            3,
            linalg::linear_solver_type::LU_LINEAR_SOLVER,
            nullptr,
            0)));
        EXPECT_ANY_THROW((linalg::inversion<double, B::value>::apply(
            buf.data(), pivot.data(), 0, linalg::linear_solver_type::LU_LINEAR_SOLVER, nullptr, 0)));

        EXPECT_ANY_THROW((linalg::svd_decomposition<double, B::value>::apply(3ULL,
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
        EXPECT_ANY_THROW((linalg::svd_decomposition<double, B::value>::apply(0ULL,
            3ULL,
            buf.data(),
            3ULL,
            vec.data(),
            buf.data(),
            3ULL,
            buf.data(),
            3ULL,
            nullptr,
            0)));
    });

    EXPECT_ANY_THROW(
        (linalg::determinant<double, linalg::linalg_backend::NATIVE>::apply(
            static_cast<double*>(nullptr),
            pivot.data(),
            3,
            linalg::linear_solver_type::LU_LINEAR_SOLVER,
            nullptr,
            0)));

    EXPECT_ANY_THROW((linalg::multiplication<double, linalg::linalg_backend::NATIVE>::apply(false,
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
    EXPECT_ANY_THROW((linalg::multiplication<double, linalg::linalg_backend::NATIVE>::apply(
        false, false, 0, 3, 3, buf.data(), 3, buf.data(), 3, buf.data(), 3, nullptr, 0)));

    EXPECT_ANY_THROW((linalg::transpose<double, linalg::linalg_backend::NATIVE>::apply(
        3ULL, 3ULL, static_cast<double*>(nullptr), nullptr, 0)));
    EXPECT_ANY_THROW((linalg::transpose<double, linalg::linalg_backend::NATIVE>::apply(
        0ULL, 3ULL, buf.data(), nullptr, 0)));
}
