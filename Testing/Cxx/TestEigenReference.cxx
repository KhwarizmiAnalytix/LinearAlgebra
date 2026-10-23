// Independent reference checks against the bundled Eigen implementation.
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/eigen.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/norm.h"
#include "include/matrix_operation/qr_decomposition.h"
#include "include/matrix_operation/svd_decomposition.h"
#include "include/matrix_operation/transpose.h"
#include "test_backend.h"

namespace
{
// Specify every parameter: MSVC mis-instantiates Eigen's dependent defaults
// in typed tests and can crash while compiling their generic backend lambdas.
template <typename T>
using Matrix = Eigen::
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor, Eigen::Dynamic, 1>;

template <typename T> T tolerance()
{
    return std::is_same_v<T, float> ? T(3e-4) : T(2e-9);
}

template <typename T> Matrix<T> random_matrix(int rows, int cols, unsigned seed = 42)
{
    std::mt19937                      rng(seed);
    std::uniform_real_distribution<T> dist(T(-1), T(1));
    Matrix<T>                         result(rows, cols);
    for (Eigen::Index i = 0; i < result.size(); ++i)
    {
        result.data()[i] = dist(rng);
    }
    return result;
}

template <typename A, typename B>
void expect_close(const Eigen::MatrixBase<A>& actual, const Eigen::MatrixBase<B>& expected)
{
    using T = typename A::RealScalar;
    ASSERT_TRUE(actual.allFinite());
    EXPECT_LE((actual - expected).norm(), tolerance<T>() * std::max(T(1), expected.norm()));
}

template <typename T> class EigenReference : public ::testing::Test
{
};
using Scalars = ::testing::Types<float, double>;
TYPED_TEST_SUITE(EigenReference, Scalars);

TYPED_TEST(EigenReference, MultiplicationWithTransposeAndPaddedStrides)
{
    using T = TypeParam;
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto B = decltype(backend)::value;
            SCOPED_TRACE(static_cast<int>(B));
            // Nonsquare shapes expose swapped dimensions/strides; 33 crosses native tiles.
            for (const auto shape : {std::array<int, 3>{1, 7, 3},
                     {5, 7, 3},
                     {33, 7, 3},
                     {7, 7, 9},
                     {33, 33, 17},
                     {1, 65, 33},
                     {65, 1, 33},
                     {5, 97, 67},
                     {97, 5, 67},
                     {65, 65, 129}})
            {
                for (const bool ta : {false, true})
                {
                    for (const bool tb : {false, true})
                    {
                        const auto [m, n, k] = shape;
                        SCOPED_TRACE(::testing::Message() << m << " ta=" << ta << " tb=" << tb);
                        const int       ar = ta ? k : m, ac = ta ? m : k;
                        const int       br = tb ? n : k, bc = tb ? k : n;
                        const Matrix<T> a              = random_matrix<T>(ar, ac);
                        const Matrix<T> b              = random_matrix<T>(br, bc, 43);
                        Matrix<T>       ap             = Matrix<T>::Constant(ar, ac + 2, T(123));
                        Matrix<T>       bp             = Matrix<T>::Constant(br, bc + 2, T(123));
                        Matrix<T>       cp             = Matrix<T>::Constant(m, n + 2, T(123));
                        ap.leftCols(ac)                = a;
                        bp.leftCols(bc)                = b;
                        Matrix<T>                  lhs = ta ? Matrix<T>(a.transpose()) : a;
                        Matrix<T>                  rhs = tb ? Matrix<T>(b.transpose()) : b;
                        const Matrix<T>            expected = lhs * rhs;
                        std::vector<unsigned char> ws(
                            linalg::multiplication<T, B>::workspace_size(m, n, k));
                        linalg::multiplication<T, B>::apply(ta,
                            tb,
                            m,
                            n,
                            k,
                            ap.data(),
                            ac + 2,
                            bp.data(),
                            bc + 2,
                            cp.data(),
                            n + 2,
                            ws.data(),
                            ws.size());
                        expect_close(cp.leftCols(n), expected);
                        EXPECT_TRUE((cp.rightCols(2).array() == T(123)).all());
                    }
                }
            }
        });
}

TYPED_TEST(EigenReference, LUFactorsAndReusedSolve)
{
    using T = TypeParam;
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto B = decltype(backend)::value;
            SCOPED_TRACE(static_cast<int>(B));
            for (const int n : {1, 7, 33, 65, 129})
            {
                SCOPED_TRACE(n);
                Matrix<T> a = random_matrix<T>(n, n);
                a.diagonal().array() += T(2 * n);
                Matrix<T>                  factors = a;
                std::vector<linalg_int>    pivot(n);
                std::vector<unsigned char> ws(linalg::lu_decomposition<T, B>::workspace_size(n));
                ASSERT_TRUE((linalg::lu_decomposition<T, B>::apply(
                    factors.data(), n, pivot.data(), ws.data(), ws.size())));
                // These dominant fixtures require no row swaps on any backend.
                Matrix<T> lower                                       = Matrix<T>::Identity(n, n);
                lower.template triangularView<Eigen::StrictlyLower>() = factors;
                Matrix<T> upper = factors.template triangularView<Eigen::Upper>();
                expect_close(Matrix<T>(lower * upper), a);
                Vector<T>       rhs          = random_matrix<T>(n, 1, 43);
                const Vector<T> expected     = a.partialPivLu().solve(rhs);
                const Vector<T> original_rhs = rhs;
                constexpr auto  mode         = linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER;
                ws.resize(linalg::linear_solver<T, B>::workspace_size(n, mode));
                linalg::linear_solver<T, B>::apply(
                    factors.data(), pivot.data(), n, rhs.data(), mode, ws.data(), ws.size());
                expect_close(rhs, expected);
                expect_close(Vector<T>(a * rhs), original_rhs);
            }
        });
}

TYPED_TEST(EigenReference, CholeskyBothTriangles)
{
    using T = TypeParam;
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto B = decltype(backend)::value;
            SCOPED_TRACE(static_cast<int>(B));
            for (const int n : {1, 7, 33, 65, 129})
            {
                SCOPED_TRACE(n);
                const Matrix<T> r = random_matrix<T>(n, n);
                Matrix<T>       a = r * r.transpose();
                a.diagonal().array() += T(n);
                const Eigen::LLT<Matrix<T>> reference(a);
                ASSERT_EQ(reference.info(), Eigen::Success);
                for (const bool lower : {false, true})
                {
                    const auto triangle =
                        lower ? linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR
                              : linalg::cholesky_decomposition_enum::UPPER_TRIANGULAR;
                    Matrix<T>                  factors = a;
                    int                        info    = -1;
                    std::vector<unsigned char> ws(
                        linalg::cholesky_decomposition<T, B>::workspace_size(n, triangle));
                    ASSERT_TRUE((linalg::cholesky_decomposition<T, B>::apply(
                        factors.data(), n, triangle, ws.data(), ws.size(), &info)));
                    EXPECT_EQ(info, 0);
                    if (lower)
                    {
                        const Matrix<T> l = factors.template triangularView<Eigen::Lower>();
                        expect_close(l, Matrix<T>(reference.matrixL()));
                        expect_close(Matrix<T>(l * l.transpose()), a);
                    }
                    else
                    {
                        const Matrix<T> u = factors.template triangularView<Eigen::Upper>();
                        expect_close(u, Matrix<T>(reference.matrixU()));
                        expect_close(Matrix<T>(u.transpose() * u), a);
                    }
                }
            }
        });
}

TYPED_TEST(EigenReference, QRReconstructionAndOrthogonality)
{
    using T = TypeParam;
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto B = decltype(backend)::value;
            SCOPED_TRACE(static_cast<int>(B));
            for (const int rows : {7, 11})
            {
                constexpr int cols = 7;
                SCOPED_TRACE(rows);
                const Matrix<T>            a = random_matrix<T>(rows, cols);
                Matrix<T>                  q = a;
                Vector<T>                  tau(cols);
                std::vector<unsigned char> ws(
                    linalg::qr_decomposition<T, B>::workspace_size(rows, cols));
                linalg::qr_decomposition<T, B>::apply_r(
                    rows, cols, q.data(), cols, tau.data(), ws.data(), ws.size());
                const Matrix<T> r = q.topRows(cols).template triangularView<Eigen::Upper>();
                linalg::qr_decomposition<T, B>::apply_q(
                    rows, cols, q.data(), cols, tau.data(), ws.data(), ws.size());
                expect_close(Matrix<T>(q * r), a);
                expect_close(Matrix<T>(q.transpose() * q), Matrix<T>::Identity(cols, cols));
                const Eigen::HouseholderQR<Matrix<T>> reference(a);
                // QR column signs may differ, but absolute R diagonals agree.
                expect_close(Vector<T>(r.diagonal().cwiseAbs()),
                    Vector<T>(reference.matrixQR().topRows(cols).diagonal().cwiseAbs()));
            }
        });
}

TYPED_TEST(EigenReference, SingularValuesAndSVDReconstruction)
{
    using T = TypeParam;
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto B = decltype(backend)::value;
            SCOPED_TRACE(static_cast<int>(B));
            for (const int rows : {5, 9})
            {
                constexpr int              cols  = 5;
                const Matrix<T>            a     = random_matrix<T>(rows, cols);
                Matrix<T>                  input = a, u(rows, cols), vt(cols, cols);
                Vector<T>                  values(cols);
                std::vector<unsigned char> ws(
                    linalg::svd_decomposition<T, B>::workspace_size(rows, cols));
                linalg::svd_decomposition<T, B>::apply(rows,
                    cols,
                    input.data(),
                    cols,
                    values.data(),
                    u.data(),
                    cols,
                    vt.data(),
                    cols,
                    ws.data(),
                    ws.size());
                const Eigen::JacobiSVD<Matrix<T>> reference(a);
                ASSERT_EQ(reference.info(), Eigen::Success);
                expect_close(Matrix<T>(u * values.asDiagonal() * vt), a);
                expect_close(Matrix<T>(u.transpose() * u), Matrix<T>::Identity(cols, cols));
                expect_close(Matrix<T>(vt * vt.transpose()), Matrix<T>::Identity(cols, cols));
                std::sort(values.data(), values.data() + cols, std::greater<T>());
                expect_close(values, reference.singularValues());
            }
        });
}

TYPED_TEST(EigenReference, SymmetricSpectrumAndEigenvectorResidual)
{
    using T = TypeParam;
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto B = decltype(backend)::value;
            SCOPED_TRACE(static_cast<int>(B));
            for (const int n : {1, 2, 7, 17, 65, 128})
            {
                const Matrix<T>                                r       = random_matrix<T>(n, n);
                const Matrix<T>                                a       = r + r.transpose();
                Matrix<T>                                      vectors = a;
                Vector<T>                                      values(n);
                const Eigen::SelfAdjointEigenSolver<Matrix<T>> reference(a);
                ASSERT_EQ(reference.info(), Eigen::Success);
                std::vector<unsigned char> ws(
                    linalg::symmetric_eigen_backend<T, B>::workspace_size(n));
                ASSERT_TRUE((linalg::symmetric_eigen_backend<T, B>::apply(
                    vectors.data(), n, n, values.data(), n, ws.data(), ws.size())));
                expect_close(values, reference.eigenvalues());
                expect_close(Matrix<T>(a * vectors), Matrix<T>(vectors * values.asDiagonal()));
                expect_close(Matrix<T>(vectors.transpose() * vectors), Matrix<T>::Identity(n, n));
            }
        });
}

TYPED_TEST(EigenReference, GeneralRealAndComplexSpectra)
{
    using T = TypeParam;
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto B = decltype(backend)::value;
            SCOPED_TRACE(static_cast<int>(B));
            for (const bool complex_spectrum : {false, true})
            {
                constexpr int n = 6;
                Matrix<T>     a = Matrix<T>::Zero(n, n);
                if (complex_spectrum)
                {
                    // Three distinct complex conjugate pairs.
                    for (int j = 0; j < n; j += 2)
                    {
                        a(j, j) = a(j + 1, j + 1) = T(j + 1);
                        a(j, j + 1)               = T(-2);
                        a(j + 1, j)               = T(2);
                    }
                }
                else
                {
                    // Dense nonsymmetric similarity transform with a real spectrum.
                    const Matrix<T> r = random_matrix<T>(n, n);
                    a                 = r + r.transpose();
                    for (int i = 0; i < n; ++i)
                    {
                        for (int j = 0; j < n; ++j)
                        {
                            a(i, j) *= T(i + 1) / T(j + 1);
                        }
                    }
                }
                const Eigen::EigenSolver<Matrix<T>> reference(a, false);
                ASSERT_EQ(reference.info(), Eigen::Success);
                Matrix<T>                  vectors = a;
                Vector<T>                  real(n), imag(n);
                const bool                 compute_vectors = !complex_spectrum;
                std::vector<unsigned char> ws(
                    linalg::general_eigen_backend<T, B>::workspace_size(n, compute_vectors));
                ASSERT_TRUE((linalg::general_eigen_backend<T, B>::apply(vectors.data(),
                    n,
                    n,
                    real.data(),
                    imag.data(),
                    compute_vectors,
                    n,
                    ws.data(),
                    ws.size())));
                // Match unordered eigenvalues one-to-one, including imaginary parts.
                std::vector<bool> used(n, false);
                for (int i = 0; i < n; ++i)
                {
                    const std::complex<T> value(real(i), imag(i));
                    int                   best = -1;
                    for (int j = 0; j < n; ++j)
                    {
                        if (!used[j] &&
                            (best < 0 || std::abs(value - reference.eigenvalues()(j)) <
                                             std::abs(value - reference.eigenvalues()(best))))
                        {
                            best = j;
                        }
                    }
                    ASSERT_GE(best, 0);
                    used[best] = true;
                    EXPECT_LE(std::abs(value - reference.eigenvalues()(best)),
                        tolerance<T>() * std::max(T(1), std::abs(value)));
                }
                if (compute_vectors)
                {
                    expect_close(Matrix<T>(a * vectors), Matrix<T>(vectors * real.asDiagonal()));
                }
            }
        });
}
TYPED_TEST(EigenReference, NativeSymmetricScaledStridedAndRepeatedSpectrum)
{
    using T                = TypeParam;
    constexpr auto backend = linalg::linalg_backend::NATIVE;
    for (const int n : {1, 2, 33, 65})
    {
        for (const T scale : {T(1e-20), T(1), T(1e20)})
        {
            SCOPED_TRACE(::testing::Message() << n << " scale=" << scale);
            const Matrix<T>                       random = random_matrix<T>(n, n);
            const Eigen::HouseholderQR<Matrix<T>> qr(random);
            const Matrix<T>                       q = qr.householderQ();
            Vector<T>                             spectrum(n);
            for (int i = 0; i < n; ++i)
            {
                const int group = i / 3;  // Repeated eigenvalues and a null space.
                spectrum[i]     = static_cast<T>(group);
            }
            const Matrix<T> a   = q * spectrum.asDiagonal() * q.transpose();
            const int       lda = n + 3, ldv = n + 5;
            std::vector<T>  storage(n * ldv, std::numeric_limits<T>::quiet_NaN());
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j <= i; ++j)
                {
                    storage[i * lda + j] = a(i, j) * scale;
                }
            }
            Vector<T>                  values(n);
            std::vector<unsigned char> workspace(
                linalg::symmetric_eigen_backend<T, backend>::workspace_size(n));
            ASSERT_TRUE((linalg::symmetric_eigen_backend<T, backend>::apply(
                storage.data(), n, lda, values.data(), ldv, workspace.data(), workspace.size())));
            const Eigen::Map<const Matrix<T>, 0, Eigen::OuterStride<>> vectors(
                storage.data(), n, n, Eigen::OuterStride<>(ldv));
            values /= scale;
            expect_close(values, spectrum);
            expect_close(Matrix<T>(a * vectors), Matrix<T>(vectors * values.asDiagonal()));
            expect_close(Matrix<T>(vectors.transpose() * vectors), Matrix<T>::Identity(n, n));
        }
    }
}

TYPED_TEST(EigenReference, NativeGeneralDenseSpectraAndStrides)
{
    using T                = TypeParam;
    constexpr auto backend = linalg::linalg_backend::NATIVE;
    for (const int n : {1, 2, 17, 65})
    {
        for (const bool compute_vectors : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << n << " vectors=" << compute_vectors);
            const Matrix<T> random = random_matrix<T>(n, n, 103);
            Matrix<T>       a      = random;
            if (compute_vectors)
            {
                a = (random + random.transpose()).eval();
                for (int i = 0; i < n; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        a(i, j) *= (T(1) + T(i) / T(n)) / (T(1) + T(j) / T(n));
                    }
                }
            }
            const Eigen::EigenSolver<Matrix<T>> reference(a, false);
            ASSERT_EQ(reference.info(), Eigen::Success);
            const int      lda = n + 2, ldv = n + 4;
            std::vector<T> storage(n * ldv, T(123));
            for (int i = 0; i < n; ++i)
            {
                std::copy_n(a.data() + i * n, n, storage.data() + i * lda);
            }
            Vector<T>                  real(n), imag(n);
            std::vector<unsigned char> workspace(
                linalg::general_eigen_backend<T, backend>::workspace_size(n, compute_vectors));
            ASSERT_TRUE((linalg::general_eigen_backend<T, backend>::apply(storage.data(),
                n,
                lda,
                real.data(),
                imag.data(),
                compute_vectors,
                ldv,
                workspace.data(),
                workspace.size())));
            std::vector<bool> used(n, false);
            for (int i = 0; i < n; ++i)
            {
                const std::complex<T> value(real[i], imag[i]);
                int                   best = -1;
                for (int j = 0; j < n; ++j)
                {
                    if (!used[j] &&
                        (best < 0 || std::abs(value - reference.eigenvalues()[j]) <
                                         std::abs(value - reference.eigenvalues()[best])))
                    {
                        best = j;
                    }
                }
                ASSERT_GE(best, 0);
                used[best] = true;
                EXPECT_LE(std::abs(value - reference.eigenvalues()[best]),
                    tolerance<T>() * std::max(T(1), std::abs(value)));
            }
            if (compute_vectors)
            {
                const Eigen::Map<const Matrix<T>, 0, Eigen::OuterStride<>> vectors(
                    storage.data(), n, n, Eigen::OuterStride<>(ldv));
                expect_close(Matrix<T>(a * vectors), Matrix<T>(vectors * real.asDiagonal()));
                for (int j = 0; j < n; ++j)
                {
                    EXPECT_NEAR(vectors.col(j).norm(), T(1), tolerance<T>());
                }
            }
        }
    }
}

TYPED_TEST(EigenReference, NormReductionsWithPaddedStrides)
{
    using T = TypeParam;
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto B = decltype(backend)::value;
            for (const auto shape : {std::array<int, 2>{1, 1}, {7, 33}, {65, 129}, {256, 65}})
            {
                const auto [rows, columns] = shape;
                const Matrix<T> input      = random_matrix<T>(rows, columns);
                Matrix<T>       padded     = Matrix<T>::Constant(rows, columns + 3, T(123));
                padded.leftCols(columns)   = input;
                for (const auto type : {linalg::norm_type::FROBENIUS,
                         linalg::norm_type::ONE,
                         linalg::norm_type::INFINITY_NORM})
                {
                    T expected;
                    if (type == linalg::norm_type::FROBENIUS)
                    {
                        expected = input.norm();
                    }
                    else if (type == linalg::norm_type::ONE)
                    {
                        expected = input.cwiseAbs().colwise().sum().maxCoeff();
                    }
                    else
                    {
                        expected = input.cwiseAbs().rowwise().sum().maxCoeff();
                    }
                    const T actual = linalg::norm<T, B>::apply(
                        padded.data(), rows, columns, columns + 3, type, nullptr, 0);
                    EXPECT_NEAR(actual, expected, tolerance<T>() * std::max(T(1), expected));
                }
            }
        });
}

TYPED_TEST(EigenReference, InPlaceTransposeTilesAndRectangles)
{
    using T = TypeParam;
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto B = decltype(backend)::value;
            for (const auto shape : {std::array<int, 2>{1, 1},
                     {1, 65},
                     {65, 1},
                     {31, 31},
                     {32, 32},
                     {33, 33},
                     {65, 65},
                     {7, 33},
                     {33, 7}})
            {
                const auto [rows, columns]        = shape;
                const Matrix<T>            input  = random_matrix<T>(rows, columns);
                Matrix<T>                  output = input;
                std::vector<unsigned char> workspace(
                    linalg::transpose<T, B>::workspace_size(rows, columns));
                linalg::transpose<T, B>::apply(
                    rows, columns, output.data(), workspace.data(), workspace.size());
                const Eigen::Map<const Matrix<T>> transposed(output.data(), columns, rows);
                expect_close(transposed, Matrix<T>(input.transpose()));
                linalg::transpose<T, B>::apply(
                    columns, rows, output.data(), workspace.data(), workspace.size());
                expect_close(output, input);
            }
        });
}

}  // namespace
