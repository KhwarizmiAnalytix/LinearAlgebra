#include "include/common/configure.h"  // IWYU pragma: keep

// The linalg::gpu::* entry points under test only exist when the library was
// configured with LINALG_ENABLE_CUBLAS (see include/matrix_operation_gpu/);
// on a scalar-fallback build this file contributes no TEST cases, matching
// how backend-conditional checks elsewhere in Testing/Cxx behave (e.g.
// TestCholeskyDecomposition.cxx's #ifndef LINALG_ENABLE_MKL guard).
#ifdef LINALG_ENABLE_CUBLAS

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include <cuda_runtime_api.h>

#include "dense_matrix_test_helper.h"
#include "gtest/gtest.h"
#include "include/matrix_operation_gpu/cholesky_decomposition_gpu.h"
#include "include/matrix_operation_gpu/linear_solver_gpu.h"
#include "include/matrix_operation_gpu/lu_decomposition_gpu.h"
#include "include/matrix_operation_gpu/matrix_multiplication_batched_gpu.h"
#include "include/matrix_operation_gpu/matrix_multiplication_gpu.h"
#include "include/matrix_operation_gpu/matrix_transpose_gpu.h"

namespace
{
using linalg_test::dense_matrix;

template <typename>
struct GpuTolerance
{
};

template <>
struct GpuTolerance<float>
{
    static constexpr float value = 5.e-3f;
};

template <>
struct GpuTolerance<double>
{
    static constexpr double value = 1.e-9;
};

// Minimal RAII CUDA device buffer: test-only, mirrors dense_matrix_test_helper's
// scope (not part of the library) — the GPU entry points under test take bare
// device pointers (cudaMalloc'd), so exercising them needs some minimal
// device-memory ownership here. Allocation failures are not checked, matching
// the library's own cudaMalloc call sites (e.g. linear_solver_gpu.cxx).
template <typename T>
class device_buffer
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
dense_matrix<value_t> random_matrix(std::size_t rows, std::size_t cols, std::default_random_engine& gen)
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

template <typename value_t>
dense_matrix<value_t> transpose(const dense_matrix<value_t>& a)
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
            max_error = std::fmax(
                max_error, std::fabs(static_cast<double>(a(i, j) - b(i, j))));
        }
    }
    return max_error;
}

// GEMM: C = op(A) * op(B), compared against a naive host reference, for every
// transpose_a/transpose_b combination.
template <typename value_t>
void matrix_multiplication_gpu_test(int rows, int columns, int depth, bool transpose_a, bool transpose_b)
{
    constexpr auto tol = GpuTolerance<value_t>::value;

    auto a_rows = transpose_a ? depth : rows;
    auto a_cols = transpose_a ? rows : depth;
    auto b_rows = transpose_b ? columns : depth;
    auto b_cols = transpose_b ? depth : columns;

    std::default_random_engine generator;
    auto                       A = random_matrix<value_t>((std::size_t)a_rows, (std::size_t)a_cols, generator);
    auto                       B = random_matrix<value_t>((std::size_t)b_rows, (std::size_t)b_cols, generator);
    dense_matrix<value_t>      C((std::size_t)rows, (std::size_t)columns);

    device_buffer<value_t> dA((std::size_t)(a_rows * a_cols));
    device_buffer<value_t> dB((std::size_t)(b_rows * b_cols));
    device_buffer<value_t> dC((std::size_t)(rows * columns));
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);
    ASSERT_EQ(dB.upload(B.data()), cudaSuccess);

    linalg::gpu::matrix_multiplication(
        transpose_a,
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

    dense_matrix<value_t> expected = matmul(
        transpose_a ? transpose(A) : A, transpose_b ? transpose(B) : B);

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

template <typename value_t>
void linear_solver_gpu_test(linalg::linear_solver_type type)
{
    constexpr auto tol = GpuTolerance<value_t>::value;
    const int      n    = 37;

    std::default_random_engine generator;
    auto                       A       = spd_matrix<value_t>((std::size_t)n, generator);
    auto                       x_true  = random_matrix<value_t>((std::size_t)n, 1, generator);
    auto                       b       = matmul(A, x_true);

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
        max_error = std::fmax(
            max_error,
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
    auto                        A_f = random_matrix<float>(rows, columns, generator);
    auto                        A_d = random_matrix<double>(rows, columns, generator);

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

    std::default_random_engine generator;
    std::vector<double>        a((std::size_t)(dim * dim * count));
    std::vector<double>        b((std::size_t)(dim * dim * count));
    std::vector<double>        c((std::size_t)(dim * dim * count), 0.);
    std::uniform_real_distribution<double> distribution(-5., 5.);
    for (auto& v : a) v = distribution(generator);
    for (auto& v : b) v = distribution(generator);

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
    constexpr auto tol = GpuTolerance<double>::value;
    const std::size_t n = 31;

    std::default_random_engine generator;
    // Known lower-triangular factor with a positive diagonal.
    dense_matrix<double> L(n, n);
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

// lu_decomposition_gpu's row-major buffer is deliberately factored as if it
// were column-major (see lu_decomposition_gpu.h's CAVEAT): P * A^T = L' * U'
// rather than the CPU's row-major "PA = LU". That factorization is not
// meant to be compared against the CPU convention, so this only smoke-tests
// that cuSOLVER accepts the call and reports success on real device memory —
// full LU correctness on this device is already covered end-to-end via
// LinearSolverLU above (same cusolverDnXgetrf/getrs primitives).
TEST(MathGpu, LuDecompositionSmoke)
{
    const std::size_t n = 17;
    std::default_random_engine generator;
    dense_matrix<double>       A(n, n);
    std::uniform_real_distribution<double> distribution(-5., 5.);
    for (std::size_t i = 0; i < n; ++i)
    {
        A(i, i) = distribution(generator) + 10.;
        for (std::size_t j = 0; j < n; ++j)
        {
            if (i != j) A(i, j) = distribution(generator);
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
    EXPECT_EQ(info, 0);
}

#endif  // LINALG_ENABLE_CUBLAS
