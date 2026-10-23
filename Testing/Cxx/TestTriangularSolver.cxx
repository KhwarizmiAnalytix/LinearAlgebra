#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "include/matrix_operation/triangular_solver.h"
#include "test_reporting.h"
#include "triangular_solver_test_cases.h"
#include "gtest/gtest.h"
#include <type_traits>

namespace
{
template <typename T, linalg::linalg_backend Backend> void solver_cases()
{
    using Solver = linalg::triangular_solver<T, Backend>;
    linalg_test::for_each_triangular_solver_case<T>(
        [](bool                   upper,
            bool                  transpose,
            bool                  unit,
            int                   n,
            int                   nrhs,
            int                   lda,
            int                   ldb,
            int                   ldx,
            const std::vector<T>& a,
            const std::vector<T>& b,
            const std::vector<T>& expected)
        {
            SCOPED_TRACE(n);
            SCOPED_TRACE(nrhs);
            SCOPED_TRACE(static_cast<int>(Backend));
            SCOPED_TRACE(upper);
            SCOPED_TRACE(transpose);
            SCOPED_TRACE(unit);
            const auto     original_a = a, original_b = b;
            const auto     bytes = Solver::workspace_size(n, nrhs);
            std::vector<T> scratch((bytes + sizeof(T) - 1) / sizeof(T) + 8, T(-123));
            std::vector<T> result(n * ldx, T(-123));
            Solver::apply(upper,
                transpose,
                unit,
                n,
                nrhs,
                a.data(),
                lda,
                b.data(),
                ldb,
                result.data(),
                ldx,
                scratch.data(),
                bytes);
            double error  = 0.;
            bool   finite = true, padding = true;
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < nrhs; ++j)
                {
                    finite = finite && std::isfinite(result[i * ldx + j]);
                    error  = std::max(
                        error, std::fabs(double(result[i * ldx + j] - expected[i * ldx + j])));
                }
                for (int j = nrhs; j < ldx; ++j)
                    padding = padding && result[i * ldx + j] == T(-123);
            }
            EXPECT_TRUE(finite);
            EXPECT_TRUE(padding);
            EXPECT_LE(error, sizeof(T) == sizeof(float) ? 4.e-6 : 1.e-13);
            EXPECT_EQ(std::memcmp(a.data(), original_a.data(), a.size() * sizeof(T)), 0);
            EXPECT_EQ(b, original_b);
            for (size_t i = (bytes + sizeof(T) - 1) / sizeof(T); i < scratch.size(); ++i)
                EXPECT_EQ(scratch[i], T(-123));
        });
}

template <typename T, linalg::linalg_backend Backend> void invalid_cases()
{
    using Solver         = linalg::triangular_solver<T, Backend>;
    T              a[4]  = {T(1), T(0), T(0), T(1)}, b[4]{}, x[4]{};
    const auto     bytes = Solver::workspace_size(2, 2);
    std::vector<T> scratch((bytes + sizeof(T) - 1) / sizeof(T) + 1);
    EXPECT_ANY_THROW((Solver::workspace_size(0, 2)));
    EXPECT_ANY_THROW((Solver::workspace_size(2, -1)));
    EXPECT_ANY_THROW(
        (Solver::apply(true, false, false, 2, 2, nullptr, 2, b, 2, x, 2, scratch.data(), bytes)));
    EXPECT_ANY_THROW(
        (Solver::apply(true, false, false, 2, 2, a, 1, b, 2, x, 2, scratch.data(), bytes)));
    EXPECT_ANY_THROW(
        (Solver::apply(true, false, false, 2, 2, a, 2, b, 1, x, 2, scratch.data(), bytes)));
    EXPECT_ANY_THROW(
        (Solver::apply(true, false, false, 2, 2, a, 2, b, 2, x, 1, scratch.data(), bytes)));
    EXPECT_ANY_THROW(
        (Solver::apply(true, false, false, 2, 2, a, 2, b, 2, b, 2, scratch.data(), bytes)));
    if (Backend == linalg::linalg_backend::NATIVE)
    {
        std::vector<T> matrix(35 * 35), rhs(35 * 2), out(35 * 2);
        const auto     needed = Solver::workspace_size(35, 2);
        EXPECT_ANY_THROW((Solver::apply(true,
            false,
            false,
            35,
            2,
            matrix.data(),
            35,
            rhs.data(),
            2,
            out.data(),
            2,
            nullptr,
            needed)));
        EXPECT_ANY_THROW((Solver::apply(true,
            false,
            false,
            35,
            2,
            matrix.data(),
            35,
            rhs.data(),
            2,
            out.data(),
            2,
            scratch.data(),
            needed - 1)));
    }
}
}  // namespace

TEST(Math, TriangularSolver)
{
    const auto check = [](auto backend)
    {
        constexpr auto Backend = decltype(backend)::value;
        solver_cases<float, Backend>();
        solver_cases<double, Backend>();
        invalid_cases<float, Backend>();
        invalid_cases<double, Backend>();
    };
    check(std::integral_constant<linalg::linalg_backend, linalg::linalg_backend::NATIVE>{});
#if LINALG_HAS_BLAS
    check(std::integral_constant<linalg::linalg_backend, linalg::linalg_backend::BLAS_LAPACKE>{});
#endif
#if LINALG_HAS_MKL
    check(std::integral_constant<linalg::linalg_backend, linalg::linalg_backend::MKL>{});
#endif
}
