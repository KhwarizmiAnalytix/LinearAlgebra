#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "include/matrix_operation/exponential.h"
#include "include/matrix_operation/logarithm.h"
#include "include/matrix_operation/square_root.h"
#include "matrix_functions_test_cases.h"
#include "test_reporting.h"
#include "gtest/gtest.h"

namespace
{
template <typename T, linalg::linalg_backend Backend>
void apply_function(
    bool is_log, const T* a, int n, int lda, T* result, int ldc, void* workspace, size_t bytes)
{
    if (is_log)
    {
        linalg::logarithm<T, Backend>::apply(a, n, lda, result, ldc, workspace, bytes);
    }
    else
    {
        linalg::square_root<T, Backend>::apply(a, n, lda, result, ldc, workspace, bytes);
    }
}

template <typename T, linalg::linalg_backend Backend> void analytic_cases()
{
    linalg_test::for_each_matrix_function_case(
        [](int n, const double* input, const double* expected, bool is_log)
        {
            SCOPED_TRACE(is_log ? "logarithm" : "square_root");
            SCOPED_TRACE(static_cast<int>(Backend));
            // logarithm needs more scratch than square_root (its seven
            // triangular quadrature solves), so each queries its own size.
            const size_t   bytes    = is_log ? linalg::logarithm<T, Backend>::workspace_size(n)
                                             : linalg::square_root<T, Backend>::workspace_size(n);
            const T        sentinel = T(-123);
            std::vector<T> workspace(bytes / sizeof(T) + 8, sentinel);
            for (bool in_place : {false, true})
            {
                const int      lda = n + 2;
                const int      ldc = in_place ? lda : n + 3;
                std::vector<T> a(n * lda, sentinel), result(n * ldc, sentinel);
                for (int i = 0; i < n; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        a[i * lda + j] = static_cast<T>(input[i * n + j]);
                    }
                }
                const auto original = a;
                T*         output   = in_place ? a.data() : result.data();
                apply_function<T, Backend>(
                    is_log, a.data(), n, lda, output, ldc, workspace.data(), bytes);
                const double tolerance = std::is_same<T, float>::value ? 8.e-4 : 2.e-10;
                for (int i = 0; i < n; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        EXPECT_NEAR(output[i * ldc + j],
                            expected[i * n + j],
                            tolerance * std::max(1., std::fabs(expected[i * n + j])));
                    }
                    for (int j = n; j < ldc; ++j)
                    {
                        EXPECT_EQ(output[i * ldc + j], sentinel);
                    }
                }
                if (!in_place)
                {
                    EXPECT_EQ(a, original);
                }
                for (size_t i = bytes / sizeof(T); i < workspace.size(); ++i)
                {
                    EXPECT_EQ(workspace[i], sentinel);
                }
            }
        });
}

template <typename T, linalg::linalg_backend Backend> void round_trip()
{
    // An independently evaluated exponential supplies nonsymmetric inputs of
    // increasing size; principal log(exp(A)) = A in this small-norm domain.
    for (int n : {1, 4, 7, 12, 17, 33})
    {
        std::vector<T> a(n * n), input(n * n), result(n * n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                a[i * n + j] = T(0.04 * std::sin(double(i * n + j + 1)));
            }
        }
        const auto exp_bytes =
            linalg::exponential<T, linalg::linalg_backend::NATIVE>::workspace_size(n);
        const auto     bytes = linalg::logarithm<T, Backend>::workspace_size(n);
        std::vector<T> workspace((std::max(bytes, exp_bytes) + sizeof(T) - 1) / sizeof(T));
        linalg::exponential<T, linalg::linalg_backend::NATIVE>::apply(
            a.data(), n, n, input.data(), n, workspace.data(), exp_bytes);
        linalg::logarithm<T, Backend>::apply(
            input.data(), n, n, result.data(), n, workspace.data(), bytes);
        const T tolerance = std::is_same<T, float>::value ? T(3.e-4) : T(2.e-10);
        for (int i = 0; i < n * n; ++i)
        {
            EXPECT_NEAR(a[i], result[i], tolerance);
        }
        linalg::square_root<T, Backend>::apply(
            input.data(), n, n, result.data(), n, workspace.data(), bytes);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                T product = T(0);
                for (int k = 0; k < n; ++k)
                {
                    product += result[i * n + k] * result[k * n + j];
                }
                EXPECT_NEAR(product, input[i * n + j], tolerance);
            }
        }
    }
}

template <typename T, linalg::linalg_backend Backend> void invalid_inputs()
{
    // logarithm needs more scratch than square_root (its seven triangular
    // quadrature solves), so each queries its own size; the workspace buffer
    // is sized for the larger of the two and each call passes its own bytes.
    const auto     log_bytes  = linalg::logarithm<T, Backend>::workspace_size(2);
    const auto     sqrt_bytes = linalg::square_root<T, Backend>::workspace_size(2);
    std::vector<T> workspace(log_bytes / sizeof(T));
    T              a[] = {T(1), T(0), T(0), T(1)};
    T              result[4]{};
    for (bool is_log : {false, true})
    {
        const auto bytes = is_log ? log_bytes : sqrt_bytes;
        EXPECT_ANY_THROW((
            apply_function<T, Backend>(is_log, nullptr, 2, 2, result, 2, workspace.data(), bytes)));
        EXPECT_ANY_THROW(
            (apply_function<T, Backend>(is_log, a, 2, 2, nullptr, 2, workspace.data(), bytes)));
        EXPECT_ANY_THROW(
            (apply_function<T, Backend>(is_log, a, 0, 2, result, 2, workspace.data(), bytes)));
        EXPECT_ANY_THROW(
            (apply_function<T, Backend>(is_log, a, 2, 1, result, 2, workspace.data(), bytes)));
        EXPECT_ANY_THROW(
            (apply_function<T, Backend>(is_log, a, 2, 2, result, 1, workspace.data(), bytes)));
        EXPECT_ANY_THROW((apply_function<T, Backend>(is_log, a, 2, 2, result, 2, nullptr, bytes)));
        EXPECT_ANY_THROW(
            (apply_function<T, Backend>(is_log, a, 2, 2, result, 2, workspace.data(), bytes - 1)));
        for (T invalid :
            {T(-1), std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity()})
        {
            a[0] = invalid;
            EXPECT_ANY_THROW(
                (apply_function<T, Backend>(is_log, a, 2, 2, result, 2, workspace.data(), bytes)));
        }
        a[0] = T(1);
    }
    a[0] = T(0);
    EXPECT_ANY_THROW(
        (apply_function<T, Backend>(true, a, 2, 2, result, 2, workspace.data(), log_bytes)));
    a[1] = T(1);
    a[3] = T(0);
    EXPECT_ANY_THROW(
        (apply_function<T, Backend>(false, a, 2, 2, result, 2, workspace.data(), sqrt_bytes)));
    EXPECT_ANY_THROW((linalg::logarithm<T, Backend>::workspace_size(-1)));
    // A real 25000-square matrix fits the existing Schur indexing contract;
    // its 50000-square packed logarithm system does not. Size queries allocate nothing.
    EXPECT_GT((linalg::square_root<T, Backend>::workspace_size(25000)), size_t(0));
    EXPECT_ANY_THROW((linalg::logarithm<T, Backend>::workspace_size(25000)));
    EXPECT_ANY_THROW(
        (linalg::square_root<T, Backend>::workspace_size(std::numeric_limits<int>::max())));
}

template <typename F> void cpu_backends(F&& check)
{
    check(std::integral_constant<linalg::linalg_backend, linalg::linalg_backend::NATIVE>{});
#if LINALG_HAS_BLAS
    check(std::integral_constant<linalg::linalg_backend, linalg::linalg_backend::BLAS_LAPACKE>{});
#endif
#if LINALG_HAS_MKL
    check(std::integral_constant<linalg::linalg_backend, linalg::linalg_backend::MKL>{});
#endif
}
}  // namespace

TEST(Math, MatrixFunctionsAnalytic)
{
    cpu_backends(
        [](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            analytic_cases<float, Backend>();
            analytic_cases<double, Backend>();
        });
}

TEST(Math, MatrixFunctionsRoundTrip)
{
    cpu_backends(
        [](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            round_trip<float, Backend>();
            round_trip<double, Backend>();
        });
}

TEST(Math, MatrixFunctionsValidation)
{
    cpu_backends(
        [](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            invalid_inputs<float, Backend>();
            invalid_inputs<double, Backend>();
        });
}
