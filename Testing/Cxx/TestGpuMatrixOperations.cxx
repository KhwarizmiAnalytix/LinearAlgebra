// The linalg::gpu::* entry points under test only exist when the library was
// configured with LINALG_ENABLE_CUBLAS (see include/matrix_operation_gpu/);
// on a scalar-fallback build this file contributes no TEST cases, matching
// how backend-conditional checks elsewhere in Testing/Cxx behave (e.g.
// TestCholeskyDecomposition.cxx's #ifndef LINALG_ENABLE_MKL guard).
#ifdef LINALG_ENABLE_CUBLAS

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include <cuda_runtime_api.h>

#include "dense_matrix_test_helper.h"
#include "include/matrix_operation_gpu/cholesky_decomposition_gpu.h"
#include "include/matrix_operation_gpu/eigenvalue_decomposition_gpu.h"
#include "include/matrix_operation_gpu/least_squares_gpu.h"
#include "include/matrix_operation_gpu/linear_solver_gpu.h"
#include "include/matrix_operation_gpu/lu_decomposition_gpu.h"
#include "include/matrix_operation_gpu/matrix_inversion_gpu.h"
#include "include/matrix_operation_gpu/matrix_multiplication_batched_gpu.h"
#include "include/matrix_operation_gpu/matrix_multiplication_gpu.h"
#include "include/matrix_operation_gpu/matrix_norm_gpu.h"
#include "include/matrix_operation_gpu/matrix_rank_gpu.h"
#include "include/matrix_operation_gpu/matrix_trace_gpu.h"
#include "include/matrix_operation_gpu/matrix_transpose_gpu.h"
#include "include/matrix_operation_gpu/pseudo_inverse_gpu.h"
#include "include/matrix_operation_gpu/qr_decomposition_gpu.h"
#include "include/matrix_operation_gpu/svd_decomposition_gpu.h"
#include "gtest/gtest.h"

namespace
{
using linalg_test::dense_matrix;

template <typename> struct GpuTolerance
{
};

template <> struct GpuTolerance<float>
{
    static constexpr float value = 5.e-3f;
};

template <> struct GpuTolerance<double>
{
    static constexpr double value = 1.e-9;
};

// Minimal RAII CUDA device buffer: test-only, mirrors dense_matrix_test_helper's
// scope (not part of the library) — the GPU entry points under test take bare
// device pointers (cudaMalloc'd), so exercising them needs some minimal
// device-memory ownership here. Allocation failures are not checked, matching
// the library's own cudaMalloc call sites (e.g. linear_solver_gpu.cxx).
template <typename T> class device_buffer
{
public:
    explicit device_buffer(std::size_t count) : count_(count)
    {
        cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T));
    }
    ~device_buffer() { cudaFree(ptr_); }

    device_buffer(const device_buffer&)            = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    T* get() { return ptr_; }

    cudaError_t upload(const T* host) const
    {
        return cudaMemcpy(ptr_, host, count_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    cudaError_t download(T* host) const
    {
        return cudaMemcpy(host, ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost);
    }

private:
    std::size_t count_;
    T*          ptr_ = nullptr;
};

template <typename value_t>
dense_matrix<value_t> random_matrix(
    std::size_t rows, std::size_t cols, std::default_random_engine& gen)
{
    std::uniform_real_distribution<double> distribution(-5., 5.);
    dense_matrix<value_t>                  M(rows, cols);
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            M(i, j) = (value_t)distribution(gen);
        }
    }
    return M;
}

template <typename value_t>
dense_matrix<value_t> matmul(const dense_matrix<value_t>& a, const dense_matrix<value_t>& b)
{
    dense_matrix<value_t> c(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        for (std::size_t j = 0; j < b.cols(); ++j)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < a.cols(); ++k)
            {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }
    return c;
}

template <typename value_t> dense_matrix<value_t> transpose(const dense_matrix<value_t>& a)
{
    dense_matrix<value_t> t(a.cols(), a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        for (std::size_t j = 0; j < a.cols(); ++j)
        {
            t(j, i) = a(i, j);
        }
    }
    return t;
}

template <typename value_t>
value_t max_abs_diff(const dense_matrix<value_t>& a, const dense_matrix<value_t>& b)
{
    value_t max_error = 0;
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        for (std::size_t j = 0; j < a.cols(); ++j)
        {
            max_error = std::fmax(max_error, std::fabs(static_cast<double>(a(i, j) - b(i, j))));
        }
    }
    return max_error;
}

// GEMM: C = op(A) * op(B), compared against a naive host reference, for every
// transpose_a/transpose_b combination.
template <typename value_t>
void matrix_multiplication_gpu_test(
    int rows, int columns, int depth, bool transpose_a, bool transpose_b)
{
    constexpr auto tol = GpuTolerance<value_t>::value;

    auto a_rows = transpose_a ? depth : rows;
    auto a_cols = transpose_a ? rows : depth;
    auto b_rows = transpose_b ? columns : depth;
    auto b_cols = transpose_b ? depth : columns;

    std::default_random_engine generator;
    auto A = random_matrix<value_t>((std::size_t)a_rows, (std::size_t)a_cols, generator);
    auto B = random_matrix<value_t>((std::size_t)b_rows, (std::size_t)b_cols, generator);
    dense_matrix<value_t> C((std::size_t)rows, (std::size_t)columns);

    device_buffer<value_t> dA((std::size_t)(a_rows * a_cols));
    device_buffer<value_t> dB((std::size_t)(b_rows * b_cols));
    device_buffer<value_t> dC((std::size_t)(rows * columns));
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);
    ASSERT_EQ(dB.upload(B.data()), cudaSuccess);

    linalg::gpu::matrix_multiplication(transpose_a,
        transpose_b,
        rows,
        columns,
        depth,
        dA.get(),
        a_cols,
        dB.get(),
        b_cols,
        dC.get(),
        columns);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(dC.download(C.data()), cudaSuccess);

    dense_matrix<value_t> expected =
        matmul(transpose_a ? transpose(A) : A, transpose_b ? transpose(B) : B);

    EXPECT_LE(max_abs_diff(C, expected), tol);
}

// A (rows x rows), positive definite, built as R^T * R from a random R with
// a strictly positive diagonal, so it is well-conditioned for both the
// Cholesky and LU paths below.
template <typename value_t>
dense_matrix<value_t> spd_matrix(std::size_t n, std::default_random_engine& gen)
{
    auto R = random_matrix<value_t>(n, n, gen);
    for (std::size_t i = 0; i < n; ++i)
    {
        R(i, i) += (value_t)(2 * n);  // diagonally dominant -> well-conditioned
    }
    return matmul(transpose(R), R);
}

template <typename value_t> void linear_solver_gpu_test(linalg::linear_solver_type type)
{
    constexpr auto tol = GpuTolerance<value_t>::value;
    const int      n   = 37;

    std::default_random_engine generator;
    auto                       A      = spd_matrix<value_t>((std::size_t)n, generator);
    auto                       x_true = random_matrix<value_t>((std::size_t)n, 1, generator);
    auto                       b      = matmul(A, x_true);

    device_buffer<value_t> dA((std::size_t)(n * n));
    device_buffer<value_t> dx((std::size_t)n);
    device_buffer<int>     dinfo(1);
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);
    ASSERT_EQ(dx.upload(b.data()), cudaSuccess);

    linalg::gpu::linear_solver(dA.get(), n, dx.get(), type, dinfo.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int info = -1;
    ASSERT_EQ(dinfo.download(&info), cudaSuccess);
    ASSERT_EQ(info, 0);

    std::vector<value_t> x_solved((std::size_t)n);
    ASSERT_EQ(dx.download(x_solved.data()), cudaSuccess);

    value_t max_error = 0;
    for (int i = 0; i < n; ++i)
    {
        max_error = std::fmax(max_error,
            std::fabs(static_cast<double>(x_solved[(std::size_t)i] - x_true((std::size_t)i, 0))));
    }
    EXPECT_LE(max_error, tol * n);
}

}  // namespace

TEST(MathGpu, MatrixMultiplication)
{
    const int rows = 37, columns = 53, depth = 29;

    matrix_multiplication_gpu_test<float>(rows, columns, depth, false, false);
    matrix_multiplication_gpu_test<float>(rows, columns, depth, true, false);
    matrix_multiplication_gpu_test<float>(rows, columns, depth, false, true);
    matrix_multiplication_gpu_test<float>(rows, columns, depth, true, true);

    matrix_multiplication_gpu_test<double>(rows, columns, depth, false, false);
    matrix_multiplication_gpu_test<double>(rows, columns, depth, true, false);
    matrix_multiplication_gpu_test<double>(rows, columns, depth, false, true);
    matrix_multiplication_gpu_test<double>(rows, columns, depth, true, true);
}

TEST(MathGpu, MatrixTranspose)
{
    const std::size_t rows = 41, columns = 23;

    std::default_random_engine generator;
    auto                       A_f = random_matrix<float>(rows, columns, generator);
    auto                       A_d = random_matrix<double>(rows, columns, generator);

    device_buffer<float> dA_f(rows * columns);
    ASSERT_EQ(dA_f.upload(A_f.data()), cudaSuccess);
    linalg::gpu::matrix_transpose((long long)rows, (long long)columns, dA_f.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    dense_matrix<float> T_f(columns, rows);
    ASSERT_EQ(dA_f.download(T_f.data()), cudaSuccess);
    EXPECT_LE(max_abs_diff(T_f, transpose(A_f)), GpuTolerance<float>::value);

    device_buffer<double> dA_d(rows * columns);
    ASSERT_EQ(dA_d.upload(A_d.data()), cudaSuccess);
    linalg::gpu::matrix_transpose((long long)rows, (long long)columns, dA_d.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    dense_matrix<double> T_d(columns, rows);
    ASSERT_EQ(dA_d.download(T_d.data()), cudaSuccess);
    EXPECT_LE(max_abs_diff(T_d, transpose(A_d)), GpuTolerance<double>::value);
}

TEST(MathGpu, BatchedMatrixMultiplication)
{
    const int dim = 9, count = 6;

    std::default_random_engine             generator;
    std::vector<double>                    a((std::size_t)(dim * dim * count));
    std::vector<double>                    b((std::size_t)(dim * dim * count));
    std::vector<double>                    c((std::size_t)(dim * dim * count), 0.);
    std::uniform_real_distribution<double> distribution(-5., 5.);
    for (auto& v : a)
        v = distribution(generator);
    for (auto& v : b)
        v = distribution(generator);

    device_buffer<double> da(a.size());
    device_buffer<double> db(b.size());
    device_buffer<double> dc(c.size());
    ASSERT_EQ(da.upload(a.data()), cudaSuccess);
    ASSERT_EQ(db.upload(b.data()), cudaSuccess);

    linalg::gpu::batched_matrix_multiplication(dim, count, da.get(), db.get(), dc.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(dc.download(c.data()), cudaSuccess);

    double max_error = 0;
    for (int mat = 0; mat < count; ++mat)
    {
        const double* Ai = a.data() + (std::size_t)(mat * dim * dim);
        const double* Bi = b.data() + (std::size_t)(mat * dim * dim);
        const double* Ci = c.data() + (std::size_t)(mat * dim * dim);
        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                double sum = 0.;
                for (int k = 0; k < dim; ++k)
                {
                    sum += Ai[i * dim + k] * Bi[k * dim + j];
                }
                max_error = std::fmax(max_error, std::fabs(sum - Ci[i * dim + j]));
            }
        }
    }
    EXPECT_LE(max_error, GpuTolerance<double>::value * dim);
}

TEST(MathGpu, CholeskyDecomposition)
{
    constexpr auto    tol = GpuTolerance<double>::value;
    const std::size_t n   = 31;

    std::default_random_engine generator;
    // Known lower-triangular factor with a positive diagonal.
    dense_matrix<double>                   L(n, n);
    std::uniform_real_distribution<double> distribution(-5., 5.);
    for (std::size_t i = 0; i < n; ++i)
    {
        L(i, i) = distribution(generator) + 10.;
        for (std::size_t j = 0; j < i; ++j)
        {
            L(i, j) = distribution(generator);
        }
    }
    dense_matrix<double> A = matmul(L, transpose(L));

    device_buffer<double> dA(n * n);
    device_buffer<int>    dinfo(1);
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);

    linalg::gpu::cholesky_decomposition(
        dA.get(), (long long)n, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR, dinfo.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int info = -1;
    ASSERT_EQ(dinfo.download(&info), cudaSuccess);
    ASSERT_EQ(info, 0);

    dense_matrix<double> R(n, n);
    ASSERT_EQ(dA.download(R.data()), cudaSuccess);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = i + 1; j < n; ++j)
        {
            R(i, j) = 0.;
        }
    }

    EXPECT_LE(max_abs_diff(R, L), tol * (double)n);
}

TEST(MathGpu, LinearSolverCholesky)
{
    linear_solver_gpu_test<float>(linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);
    linear_solver_gpu_test<double>(linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);
}

TEST(MathGpu, LinearSolverLU)
{
    linear_solver_gpu_test<float>(linalg::linear_solver_type::LU_LINEAR_SOLVER);
    linear_solver_gpu_test<double>(linalg::linear_solver_type::LU_LINEAR_SOLVER);
}

// P * A = L * U reconstruction, mirroring TestLUDecomposition.cxx's CPU
// convention exactly: lu_decomposition_gpu now transposes on the device
// before and after cuSOLVER's column-major getrf (see lu_decomposition_gpu.
// cxx), so its row-major packed L\U output and 1-based pivot sequence match
// the CPU lapacke_?getrf convention bit-for-bit in semantics — unlike the
// CPU scalar fallback, cuSOLVER's getrf always partial-pivots (no
// LINALG_LU_PIVOTING/MKL guard needed here).
TEST(MathGpu, LuDecomposition)
{
    constexpr auto                         tol = GpuTolerance<double>::value;
    const std::size_t                      n   = 17;
    std::default_random_engine             generator;
    dense_matrix<double>                   A(n, n);
    std::uniform_real_distribution<double> distribution(-5., 5.);
    for (std::size_t i = 0; i < n; ++i)
    {
        A(i, i) = distribution(generator) + 10.;
        for (std::size_t j = 0; j < n; ++j)
        {
            if (i != j)
                A(i, j) = distribution(generator);
        }
    }

    device_buffer<double> dA(n * n);
    device_buffer<int>    dpivot(n);
    device_buffer<int>    dinfo(1);
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);

    linalg::gpu::lu_decomposition(dA.get(), (long long)n, dpivot.get(), dinfo.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int info = -1;
    ASSERT_EQ(dinfo.download(&info), cudaSuccess);
    ASSERT_EQ(info, 0);

    dense_matrix<double> packed(n, n);
    ASSERT_EQ(dA.download(packed.data()), cudaSuccess);
    std::vector<int> pivot(n);
    ASSERT_EQ(dpivot.download(pivot.data()), cudaSuccess);

    // Unpack L (unit lower triangular) and U (upper triangular) from the
    // row-major packed factor, exactly as TestLUDecomposition.cxx does for
    // the CPU path.
    dense_matrix<double> L(n, n);
    dense_matrix<double> U(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        U(i, i) = packed(i, i);
        L(i, i) = 1.;
        for (std::size_t j = 0; j < i; ++j)
        {
            L(i, j) = packed(i, j);
            U(j, i) = packed(j, i);
        }
    }

    // L * U == P * A; undo P by replaying the pivot sequence in reverse
    // (each entry is a transposition, so reverse order inverts the
    // composition) to recover A from L * U.
    dense_matrix<double> result = matmul(L, U);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i)
    {
        const int p = pivot[(std::size_t)i] - 1;
        if (p != i)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                const auto ii   = (std::size_t)i;
                const auto pp   = (std::size_t)p;
                const auto temp = result(ii, j);
                result(ii, j)   = result(pp, j);
                result(pp, j)   = temp;
            }
        }
    }

    EXPECT_LE(max_abs_diff(result, A), tol * (double)n);
}

// A * A^-1 == I for both LU- and Cholesky-based inversion on the same SPD
// matrix (valid input for either factorization), and the two backends'
// determinants agree with each other — mirrors TestMatrixInversion.cxx's
// CPU coverage (same test_inversion<T> structure), adapted to matrix_
// invert_gpu's in-place, no-caller-visible-pivot contract.
TEST(MathGpu, MatrixInversion)
{
    constexpr auto    tol = GpuTolerance<double>::value;
    const std::size_t n   = 23;

    std::default_random_engine generator;
    auto                       A = spd_matrix<double>(n, generator);

    dense_matrix<double> Id(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        Id(i, i) = 1.;
    }

    for (auto type : {linalg::linear_solver_type::LU_LINEAR_SOLVER,
             linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER})
    {
        device_buffer<double> dA(n * n);
        device_buffer<int>    dinfo(1);
        ASSERT_EQ(dA.upload(A.data()), cudaSuccess);

        linalg::gpu::matrix_invert(dA.get(), (long long)n, dinfo.get(), type);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        int info = -1;
        ASSERT_EQ(dinfo.download(&info), cudaSuccess);
        ASSERT_EQ(info, 0);

        dense_matrix<double> inv(n, n);
        ASSERT_EQ(dA.download(inv.data()), cudaSuccess);

        auto product = matmul(inv, A);
        EXPECT_LE(max_abs_diff(product, Id), tol * (double)n)
            << "type=" << static_cast<long long>(type);
    }

    device_buffer<double> dA_lu(n * n);
    device_buffer<double> dA_chol(n * n);
    ASSERT_EQ(dA_lu.upload(A.data()), cudaSuccess);
    ASSERT_EQ(dA_chol.upload(A.data()), cudaSuccess);

    const double det_lu = linalg::gpu::matrix_determinant(
        dA_lu.get(), (long long)n, linalg::linear_solver_type::LU_LINEAR_SOLVER);
    const double det_chol = linalg::gpu::matrix_determinant(
        dA_chol.get(), (long long)n, linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);

    EXPECT_GT(det_lu, 0.);  // A is SPD
    EXPECT_LE(std::fabs(det_lu - det_chol), tol * std::fabs(det_lu));
}

// A == U * diag(S) * VT reconstruction, mirroring TestSVDDecomposition.cxx's
// CPU test_svd<T> exactly (same ldu = min(rows, columns), ldv = columns
// packing the CPU test already uses, which is also what svd_decomposition_
// gpu.h's RESTRICTION requires), for both a tall and a wide matrix.
template <typename value_t> void svd_decomposition_gpu_test(std::size_t rows, std::size_t columns)
{
    constexpr auto tol = GpuTolerance<value_t>::value;
    const auto     k   = std::min(rows, columns);

    std::default_random_engine generator;
    auto                       A = random_matrix<value_t>(rows, columns, generator);

    device_buffer<value_t> dA(rows * columns);
    device_buffer<value_t> dS(k);
    device_buffer<value_t> dU(rows * k);
    device_buffer<value_t> dVT(k * columns);
    device_buffer<int>     dinfo(1);
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);

    linalg::gpu::svd_decomposition((long long)rows,
        (long long)columns,
        dA.get(),
        (long long)columns,
        dS.get(),
        dU.get(),
        (long long)k,
        dVT.get(),
        (long long)columns,
        dinfo.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int info = -1;
    ASSERT_EQ(dinfo.download(&info), cudaSuccess);
    ASSERT_EQ(info, 0);

    std::vector<value_t>  S(k);
    dense_matrix<value_t> U(rows, k);
    dense_matrix<value_t> VT(k, columns);
    ASSERT_EQ(dS.download(S.data()), cudaSuccess);
    ASSERT_EQ(dU.download(U.data()), cudaSuccess);
    ASSERT_EQ(dVT.download(VT.data()), cudaSuccess);

    value_t max_error = 0;
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < columns; ++j)
        {
            value_t sum = A(i, j);
            for (std::size_t kk = 0; kk < k; ++kk)
            {
                sum -= U(i, kk) * S[kk] * VT(kk, j);
            }
            max_error = std::fmax(max_error, std::fabs(static_cast<double>(sum)));
        }
    }
    EXPECT_LE(max_error, tol * (value_t)k);
}

TEST(MathGpu, SVDDecomposition)
{
    svd_decomposition_gpu_test<float>(23, 11);
    svd_decomposition_gpu_test<double>(23, 11);

    svd_decomposition_gpu_test<float>(11, 23);
    svd_decomposition_gpu_test<double>(11, 23);
}

// A == Q*R reconstruction plus Q^T*Q == I, mirroring TestQRDecomposition.cxx's
// CPU test_qr<T> and svd_decomposition_gpu_test's device-buffer plumbing.
template <typename value_t> void qr_decomposition_gpu_test(std::size_t rows, std::size_t columns)
{
    constexpr auto tol = GpuTolerance<value_t>::value;
    const auto     k   = std::min(rows, columns);

    std::default_random_engine generator;
    auto                       A = random_matrix<value_t>(rows, columns, generator);

    device_buffer<value_t> dA(rows * columns);
    device_buffer<value_t> dQ(rows * k);
    device_buffer<value_t> dR(k * columns);
    device_buffer<int>     dinfo(1);
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);

    linalg::gpu::qr_decomposition((long long)rows,
        (long long)columns,
        dA.get(),
        (long long)columns,
        dQ.get(),
        (long long)k,
        dR.get(),
        (long long)columns,
        dinfo.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int info = -1;
    ASSERT_EQ(dinfo.download(&info), cudaSuccess);
    ASSERT_EQ(info, 0);

    dense_matrix<value_t> Q(rows, k);
    dense_matrix<value_t> R(k, columns);
    ASSERT_EQ(dQ.download(Q.begin()), cudaSuccess);
    ASSERT_EQ(dR.download(R.begin()), cudaSuccess);

    value_t max_error = 0;
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < columns; ++j)
        {
            value_t sum = 0;
            for (std::size_t c = 0; c < k; ++c)
            {
                sum += Q(i, c) * R(c, j);
            }
            max_error = std::fmax(max_error, std::fabs(static_cast<double>(A(i, j) - sum)));
        }
    }
    EXPECT_LE(max_error, tol * (value_t)k);

    value_t max_orth_error = 0;
    for (std::size_t i = 0; i < k; ++i)
    {
        for (std::size_t j = 0; j < k; ++j)
        {
            value_t sum = 0;
            for (std::size_t r = 0; r < rows; ++r)
            {
                sum += Q(r, i) * Q(r, j);
            }
            const value_t expected = (i == j) ? value_t(1) : value_t(0);
            max_orth_error          = std::fmax(max_orth_error, std::fabs(static_cast<double>(sum - expected)));
        }
    }
    EXPECT_LE(max_orth_error, tol * (value_t)k);
}

TEST(MathGpu, QRDecomposition)
{
    qr_decomposition_gpu_test<float>(23, 11);
    qr_decomposition_gpu_test<double>(23, 11);

    qr_decomposition_gpu_test<float>(11, 23);
    qr_decomposition_gpu_test<double>(11, 23);
}

// A = Q * diag(known) * Q^T for a random orthogonal Q built from the GPU QR
// above; eigenvalues must come back ascending and reconstruct A.
template <typename value_t> void symmetric_eigen_gpu_test(std::size_t n)
{
    constexpr auto tol = GpuTolerance<value_t>::value;

    std::default_random_engine generator;
    auto                       R0 = random_matrix<value_t>(n, n, generator);

    device_buffer<value_t> dR0(n * n);
    device_buffer<value_t> dQ(n * n);
    device_buffer<value_t> dR(n * n);
    device_buffer<int>     dinfo(1);
    ASSERT_EQ(dR0.upload(R0.begin()), cudaSuccess);
    linalg::gpu::qr_decomposition(
        (long long)n, (long long)n, dR0.get(), (long long)n, dQ.get(), (long long)n, dR.get(), (long long)n, dinfo.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    dense_matrix<value_t> Q(n, n);
    ASSERT_EQ(dQ.download(Q.begin()), cudaSuccess);

    std::vector<value_t> known(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        known[i] = (value_t)(i + 1);
    }

    dense_matrix<value_t> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < n; ++k)
            {
                sum += Q(i, k) * known[k] * Q(j, k);
            }
            A(i, j) = sum;
        }
    }

    device_buffer<value_t> dA(n * n);
    device_buffer<value_t> devals(n);
    device_buffer<value_t> devecs(n * n);
    ASSERT_EQ(dA.upload(A.begin()), cudaSuccess);

    linalg::gpu::symmetric_eigenvalue_decomposition(
        dA.get(), (linalg_int)n, (linalg_int)n, devals.get(), devecs.get(), (linalg_int)n, dinfo.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int info = -1;
    ASSERT_EQ(dinfo.download(&info), cudaSuccess);
    ASSERT_EQ(info, 0);

    std::vector<value_t>  evals(n);
    dense_matrix<value_t> evecs(n, n);
    ASSERT_EQ(devals.download(evals.data()), cudaSuccess);
    ASSERT_EQ(devecs.download(evecs.begin()), cudaSuccess);

    for (std::size_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR((double)evals[i], (double)known[i], tol * (double)n);
    }

    // AV = A * evecs, then evecs^T * AV should reconstruct diag(evals) (two
    // matmul passes rather than one quad-nested loop, matching this
    // repository's CPU TestEigenvalueDecomposition.cxx workaround for a
    // clang-cl optimizer crash on that pattern).
    dense_matrix<value_t> AV(n, n);
    for (std::size_t p = 0; p < n; ++p)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            value_t sum = 0;
            for (std::size_t q = 0; q < n; ++q)
            {
                sum += A(p, q) * evecs(q, j);
            }
            AV(p, j) = sum;
        }
    }
    value_t max_error = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            value_t sum = 0;
            for (std::size_t p = 0; p < n; ++p)
            {
                sum += evecs(p, i) * AV(p, j);
            }
            const value_t expected = (i == j) ? evals[i] : value_t(0);
            max_error                = std::fmax(max_error, std::fabs(static_cast<double>(sum - expected)));
        }
    }
    EXPECT_LE(max_error, tol * (value_t)n);
}

TEST(MathGpu, SymmetricEigenvalueDecomposition)
{
    symmetric_eigen_gpu_test<float>(6);
    symmetric_eigen_gpu_test<double>(6);
}

// Full column-rank tall matrix: pinv(A) is a left inverse.
template <typename value_t> void pseudo_inverse_gpu_test(std::size_t rows, std::size_t columns)
{
    constexpr auto tol = GpuTolerance<value_t>::value;

    std::default_random_engine generator;
    auto                       A = random_matrix<value_t>(rows, columns, generator);

    device_buffer<value_t> dA(rows * columns);
    device_buffer<value_t> dAinv(columns * rows);
    ASSERT_EQ(dA.upload(A.begin()), cudaSuccess);

    linalg::gpu::pseudo_inverse(
        (long long)rows, (long long)columns, dA.get(), (long long)columns, dAinv.get(), (long long)rows);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    dense_matrix<value_t> Ainv(columns, rows);
    ASSERT_EQ(dAinv.download(Ainv.begin()), cudaSuccess);

    value_t max_error = 0;
    for (std::size_t i = 0; i < columns; ++i)
    {
        for (std::size_t j = 0; j < columns; ++j)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < rows; ++k)
            {
                sum += Ainv(i, k) * A(k, j);
            }
            const value_t expected = (i == j) ? value_t(1) : value_t(0);
            max_error               = std::fmax(max_error, std::fabs(static_cast<double>(sum - expected)));
        }
    }
    EXPECT_LE(max_error, tol * (value_t)rows);
}

TEST(MathGpu, PseudoInverse)
{
    pseudo_inverse_gpu_test<float>(9, 5);
    pseudo_inverse_gpu_test<double>(9, 5);
}

TEST(MathGpu, MatrixRankAndConditionNumber)
{
    const std::size_t n = 6;
    dense_matrix<double> I(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        I(i, i) = 1.0;
    }
    device_buffer<double> dI(n * n);
    ASSERT_EQ(dI.upload(I.begin()), cudaSuccess);

    const auto rank = linalg::gpu::matrix_rank(dI.get(), (long long)n, (long long)n, (long long)n);
    const auto cond = linalg::gpu::matrix_condition_number(dI.get(), (long long)n, (long long)n, (long long)n);
    EXPECT_EQ(rank, (linalg_long)n);
    EXPECT_NEAR(cond, 1.0, 1e-6);

    // Rank-1 outer product.
    std::default_random_engine generator;
    std::uniform_real_distribution<double> dist(1., 2.);
    dense_matrix<double>                   A(n, n);
    std::vector<double>                    u(n), v(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        u[i] = dist(generator);
        v[i] = dist(generator);
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            A(i, j) = u[i] * v[j];
        }
    }
    device_buffer<double> dA(n * n);
    ASSERT_EQ(dA.upload(A.begin()), cudaSuccess);
    const auto rank1 = linalg::gpu::matrix_rank(dA.get(), (long long)n, (long long)n, (long long)n);
    EXPECT_EQ(rank1, (linalg_long)1);
}

TEST(MathGpu, MatrixNorm)
{
    const std::size_t rows = 6, columns = 5;
    std::default_random_engine generator;
    auto                       A = random_matrix<double>(rows, columns, generator);

    device_buffer<double> dA(rows * columns);
    ASSERT_EQ(dA.upload(A.begin()), cudaSuccess);

    double frob_expected = 0;
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < columns; ++j)
        {
            frob_expected += A(i, j) * A(i, j);
        }
    }
    frob_expected = std::sqrt(frob_expected);

    double one_expected = 0;
    for (std::size_t j = 0; j < columns; ++j)
    {
        double s = 0;
        for (std::size_t i = 0; i < rows; ++i)
        {
            s += std::fabs(A(i, j));
        }
        one_expected = std::max(one_expected, s);
    }

    double inf_expected = 0;
    for (std::size_t i = 0; i < rows; ++i)
    {
        double s = 0;
        for (std::size_t j = 0; j < columns; ++j)
        {
            s += std::fabs(A(i, j));
        }
        inf_expected = std::max(inf_expected, s);
    }

    const auto frob = linalg::gpu::matrix_norm(
        dA.get(), (long long)rows, (long long)columns, (long long)columns, linalg::matrix_norm_type::FROBENIUS);
    const auto one = linalg::gpu::matrix_norm(
        dA.get(), (long long)rows, (long long)columns, (long long)columns, linalg::matrix_norm_type::ONE);
    const auto inf_norm = linalg::gpu::matrix_norm(
        dA.get(), (long long)rows, (long long)columns, (long long)columns, linalg::matrix_norm_type::INFINITY_NORM);

    EXPECT_NEAR(frob, frob_expected, 1e-6);
    EXPECT_NEAR(one, one_expected, 1e-6);
    EXPECT_NEAR(inf_norm, inf_expected, 1e-6);
}

TEST(MathGpu, MatrixTrace)
{
    const std::size_t n = 5;
    std::default_random_engine generator;
    auto                       A = random_matrix<double>(n, n, generator);

    double expected = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        expected += A(i, i);
    }

    device_buffer<double> dA(n * n);
    ASSERT_EQ(dA.upload(A.begin()), cudaSuccess);
    const auto trace = linalg::gpu::matrix_trace(dA.get(), (linalg_int)n, (linalg_int)n);
    EXPECT_NEAR(trace, expected, 1e-9);
}

// Overdetermined, consistent system: the least-squares solution should
// recover the exact generating X (residual is zero).
TEST(MathGpu, LeastSquares)
{
    const std::size_t rows = 6, columns = 3, nrhs = 2;

    dense_matrix<double> A(rows, columns);
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < columns; ++j)
        {
            A(i, j) = (double)(1 + (i * 3 + j * 7) % 11);
        }
    }
    dense_matrix<double> X_true(columns, nrhs);
    for (std::size_t i = 0; i < columns; ++i)
    {
        for (std::size_t j = 0; j < nrhs; ++j)
        {
            X_true(i, j) = (double)(1 + i + 2 * j);
        }
    }
    dense_matrix<double> B(rows, nrhs);
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < nrhs; ++j)
        {
            double sum = 0;
            for (std::size_t k = 0; k < columns; ++k)
            {
                sum += A(i, k) * X_true(k, j);
            }
            B(i, j) = sum;
        }
    }

    device_buffer<double> dA(rows * columns);
    device_buffer<double> dB(rows * nrhs);
    device_buffer<double> dX(columns * nrhs);
    ASSERT_EQ(dA.upload(A.begin()), cudaSuccess);
    ASSERT_EQ(dB.upload(B.begin()), cudaSuccess);

    linalg::gpu::least_squares_solve((long long)rows,
        (long long)columns,
        (long long)nrhs,
        dA.get(),
        (long long)columns,
        dB.get(),
        (long long)nrhs,
        dX.get(),
        (long long)nrhs);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    dense_matrix<double> X(columns, nrhs);
    ASSERT_EQ(dX.download(X.begin()), cudaSuccess);

    double max_error = 0;
    for (std::size_t i = 0; i < columns; ++i)
    {
        for (std::size_t j = 0; j < nrhs; ++j)
        {
            max_error = std::fmax(max_error, std::fabs(X(i, j) - X_true(i, j)));
        }
    }
    EXPECT_LT(max_error, 1e-6);
}

// Every public linalg::gpu::* entry point throws (rather than exhibiting
// undefined behavior) on a null required device pointer or a non-positive
// dimension — mirrors Math.InputValidation's CPU coverage in
// TestInputValidation.cxx. One representative case per op is enough here;
// this is a regression guard for that contract, not an exhaustive sweep.
TEST(MathGpu, InputValidation)
{
    const int n = 3;

    device_buffer<double> dA((std::size_t)(n * n));
    device_buffer<double> dx((std::size_t)n);
    device_buffer<double> dS((std::size_t)n);
    device_buffer<int>    dpivot((std::size_t)n);
    device_buffer<int>    dinfo(1);

    EXPECT_ANY_THROW(linalg::gpu::cholesky_decomposition(static_cast<double*>(nullptr),
        n,
        linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR,
        dinfo.get()));
    EXPECT_ANY_THROW(linalg::gpu::cholesky_decomposition(
        dA.get(), 0, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR, dinfo.get()));

    EXPECT_ANY_THROW(
        linalg::gpu::lu_decomposition(static_cast<double*>(nullptr), n, dpivot.get(), dinfo.get()));
    EXPECT_ANY_THROW(linalg::gpu::lu_decomposition(dA.get(), 0, dpivot.get(), dinfo.get()));

    EXPECT_ANY_THROW(linalg::gpu::linear_solver(static_cast<double*>(nullptr),
        n,
        dx.get(),
        linalg::linear_solver_type::LU_LINEAR_SOLVER,
        dinfo.get()));
    EXPECT_ANY_THROW(linalg::gpu::linear_solver(
        dA.get(), 0, dx.get(), linalg::linear_solver_type::LU_LINEAR_SOLVER, dinfo.get()));

    EXPECT_ANY_THROW(linalg::gpu::matrix_invert(static_cast<double*>(nullptr), n, dinfo.get()));
    EXPECT_ANY_THROW(linalg::gpu::matrix_invert(dA.get(), 0, dinfo.get()));
    EXPECT_ANY_THROW(linalg::gpu::matrix_determinant(static_cast<double*>(nullptr), n));

    EXPECT_ANY_THROW(linalg::gpu::matrix_multiplication(
        false, false, n, n, n, static_cast<const double*>(nullptr), n, dA.get(), n, dA.get(), n));
    EXPECT_ANY_THROW(linalg::gpu::matrix_multiplication(
        false, false, 0, n, n, dA.get(), n, dA.get(), n, dA.get(), n));

    EXPECT_ANY_THROW(
        linalg::gpu::matrix_transpose((long long)n, (long long)n, static_cast<double*>(nullptr)));
    EXPECT_ANY_THROW(linalg::gpu::matrix_transpose(0LL, (long long)n, dA.get()));

    EXPECT_ANY_THROW(linalg::gpu::svd_decomposition((long long)n,
        (long long)n,
        static_cast<const double*>(nullptr),
        (long long)n,
        dS.get(),
        dA.get(),
        (long long)n,
        dA.get(),
        (long long)n,
        dinfo.get()));
    EXPECT_ANY_THROW(linalg::gpu::svd_decomposition(0LL,
        (long long)n,
        dA.get(),
        (long long)n,
        dS.get(),
        dA.get(),
        (long long)n,
        dA.get(),
        (long long)n,
        dinfo.get()));

    EXPECT_ANY_THROW(linalg::gpu::qr_decomposition((long long)n,
        (long long)n,
        static_cast<const double*>(nullptr),
        (long long)n,
        dA.get(),
        (long long)n,
        dA.get(),
        (long long)n,
        dinfo.get()));
    EXPECT_ANY_THROW(linalg::gpu::qr_decomposition(
        0LL, (long long)n, dA.get(), (long long)n, dA.get(), (long long)n, dA.get(), (long long)n, dinfo.get()));

    EXPECT_ANY_THROW(linalg::gpu::symmetric_eigenvalue_decomposition(
        static_cast<const double*>(nullptr), n, n, dS.get(), dA.get(), n, dinfo.get()));
    EXPECT_ANY_THROW(
        linalg::gpu::symmetric_eigenvalue_decomposition(dA.get(), 0, 0, dS.get(), dA.get(), 0, dinfo.get()));

    EXPECT_ANY_THROW(linalg::gpu::pseudo_inverse(
        (long long)n, (long long)n, static_cast<const double*>(nullptr), (long long)n, dA.get(), (long long)n));
    EXPECT_ANY_THROW(
        linalg::gpu::pseudo_inverse(0LL, (long long)n, dA.get(), (long long)n, dA.get(), (long long)n));

    EXPECT_ANY_THROW(linalg::gpu::matrix_rank(
        static_cast<const double*>(nullptr), (long long)n, (long long)n, (long long)n));
    EXPECT_ANY_THROW(linalg::gpu::matrix_rank(dA.get(), 0LL, (long long)n, (long long)n));

    EXPECT_ANY_THROW(linalg::gpu::matrix_condition_number(
        static_cast<const double*>(nullptr), (long long)n, (long long)n, (long long)n));
    EXPECT_ANY_THROW(linalg::gpu::matrix_condition_number(dA.get(), 0LL, (long long)n, (long long)n));

    EXPECT_ANY_THROW(linalg::gpu::matrix_norm(static_cast<const double*>(nullptr),
        (long long)n,
        (long long)n,
        (long long)n,
        linalg::matrix_norm_type::FROBENIUS));
    EXPECT_ANY_THROW(linalg::gpu::matrix_norm(
        dA.get(), 0LL, (long long)n, (long long)n, linalg::matrix_norm_type::FROBENIUS));

    EXPECT_ANY_THROW(linalg::gpu::matrix_trace(static_cast<const double*>(nullptr), n, n));
    EXPECT_ANY_THROW(linalg::gpu::matrix_trace(dA.get(), 0, n));

    EXPECT_ANY_THROW(linalg::gpu::least_squares_solve((long long)n,
        (long long)n,
        (long long)n,
        static_cast<const double*>(nullptr),
        (long long)n,
        dA.get(),
        (long long)n,
        dA.get(),
        (long long)n));
    EXPECT_ANY_THROW(linalg::gpu::least_squares_solve(
        0LL, (long long)n, (long long)n, dA.get(), (long long)n, dA.get(), (long long)n, dA.get(), (long long)n));
}

// Two independent Cholesky solves issued on two distinct caller-created
// streams. Does not attempt to assert actual concurrency/overlap (timing-
// sensitive, unsuited to a correctness test) — the goal is that per-device
// handle reuse (include/common/cuda_handle.h) and the stream parameter
// threaded through every linalg::gpu::* entry point do not deadlock or
// corrupt either solve's result.
TEST(MathGpu, StreamParameter)
{
    constexpr auto tol = GpuTolerance<double>::value;
    const int      n   = 23;

    cudaStream_t stream_a = nullptr;
    cudaStream_t stream_b = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream_a), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream_b), cudaSuccess);

    std::default_random_engine generator;
    auto                       A_a      = spd_matrix<double>((std::size_t)n, generator);
    auto                       A_b      = spd_matrix<double>((std::size_t)n, generator);
    auto                       x_true_a = random_matrix<double>((std::size_t)n, 1, generator);
    auto                       x_true_b = random_matrix<double>((std::size_t)n, 1, generator);
    auto                       b_a      = matmul(A_a, x_true_a);
    auto                       b_b      = matmul(A_b, x_true_b);

    device_buffer<double> dA_a((std::size_t)(n * n));
    device_buffer<double> dx_a((std::size_t)n);
    device_buffer<int>    dinfo_a(1);
    device_buffer<double> dA_b((std::size_t)(n * n));
    device_buffer<double> dx_b((std::size_t)n);
    device_buffer<int>    dinfo_b(1);
    ASSERT_EQ(dA_a.upload(A_a.data()), cudaSuccess);
    ASSERT_EQ(dx_a.upload(b_a.data()), cudaSuccess);
    ASSERT_EQ(dA_b.upload(A_b.data()), cudaSuccess);
    ASSERT_EQ(dx_b.upload(b_b.data()), cudaSuccess);

    linalg::gpu::linear_solver(dA_a.get(),
        n,
        dx_a.get(),
        linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER,
        dinfo_a.get(),
        stream_a);
    linalg::gpu::linear_solver(dA_b.get(),
        n,
        dx_b.get(),
        linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER,
        dinfo_b.get(),
        stream_b);

    ASSERT_EQ(cudaStreamSynchronize(stream_a), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_b), cudaSuccess);

    int info_a = -1;
    int info_b = -1;
    ASSERT_EQ(dinfo_a.download(&info_a), cudaSuccess);
    ASSERT_EQ(dinfo_b.download(&info_b), cudaSuccess);
    EXPECT_EQ(info_a, 0);
    EXPECT_EQ(info_b, 0);

    std::vector<double> x_solved_a((std::size_t)n);
    std::vector<double> x_solved_b((std::size_t)n);
    ASSERT_EQ(dx_a.download(x_solved_a.data()), cudaSuccess);
    ASSERT_EQ(dx_b.download(x_solved_b.data()), cudaSuccess);

    double max_error = 0;
    for (int i = 0; i < n; ++i)
    {
        max_error = std::fmax(
            max_error, std::fabs(x_solved_a[(std::size_t)i] - x_true_a((std::size_t)i, 0)));
        max_error = std::fmax(
            max_error, std::fabs(x_solved_b[(std::size_t)i] - x_true_b((std::size_t)i, 0)));
    }
    EXPECT_LE(max_error, tol * n);

    ASSERT_EQ(cudaStreamDestroy(stream_a), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream_b), cudaSuccess);
}

#endif  // LINALG_ENABLE_CUBLAS
