// Profile every matrix operation CPU backend (native, blas_lapacke, mkl) and the
// GPU backend (cuBLAS/cuSOLVER) using KhwarizmiAnalytix/Profiler instrumentation.
//
// Each operation is wrapped in a PROFILER_SCOPE that records CPU time and (when
// LINALG_HAS_CUBLAS and a device is present) GPU activity.  Two JSON traces are
// written: one for CPU backends and one for the GPU backend.  The native profiler
// report is printed at the end of each session.
//
// Run:
//   ./bin/profile_linalg_matrix_operations [n]
//
// where n is the matrix order (default 128).

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "profiler.h"

#include "include/common/backend.h"
#include "include/common/types.h"
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/condition_number.h"
#include "include/matrix_operation/determinant.h"
#include "include/matrix_operation/eigen.h"
#include "include/matrix_operation/exponential.h"
#include "include/matrix_operation/inversion.h"
#include "include/matrix_operation/least_squares.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/norm.h"
#include "include/matrix_operation/pentadiagonal_operations.h"
#include "include/matrix_operation/power.h"
#include "include/matrix_operation/pseudo_inverse.h"
#include "include/matrix_operation/qr_decomposition.h"
#include "include/matrix_operation/rank.h"
#include "include/matrix_operation/svd_decomposition.h"
#include "include/matrix_operation/trace.h"
#include "include/matrix_operation/transpose.h"
#include "include/matrix_operation/tridiagonal_operations.h"

#if LINALG_HAS_CUBLAS
#include <cuda_runtime_api.h>

#include "include/matrix_operation_gpu/cholesky_decomposition_gpu.h"
#include "include/matrix_operation_gpu/eigen_gpu.h"
#include "include/matrix_operation_gpu/inversion_gpu.h"
#include "include/matrix_operation_gpu/least_squares_gpu.h"
#include "include/matrix_operation_gpu/linear_solver_gpu.h"
#include "include/matrix_operation_gpu/lu_decomposition_gpu.h"
#include "include/matrix_operation_gpu/multiplication_gpu.h"
#include "include/matrix_operation_gpu/norm_gpu.h"
#include "include/matrix_operation_gpu/pseudo_inverse_gpu.h"
#include "include/matrix_operation_gpu/qr_decomposition_gpu.h"
#include "include/matrix_operation_gpu/rank_gpu.h"
#include "include/matrix_operation_gpu/svd_decomposition_gpu.h"
#include "include/matrix_operation_gpu/trace_gpu.h"
#include "include/matrix_operation_gpu/transpose_gpu.h"
#endif

namespace
{

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

template <typename T>
std::vector<T> make_random_matrix(std::size_t rows, std::size_t cols, unsigned seed = 42)
{
    std::mt19937                      rng(seed);
    std::uniform_real_distribution<T> dist(T(-1), T(1));
    std::vector<T>                    m(rows * cols);
    for (auto& v : m)
    {
        v = dist(rng);
    }
    return m;
}

// Returns a symmetric positive-definite n x n matrix: B * B^T + n * I.
template <typename T>
std::vector<T> make_spd_matrix(linalg_int n, unsigned seed = 42)
{
    auto           B = make_random_matrix<T>(static_cast<std::size_t>(n), static_cast<std::size_t>(n), seed);
    std::vector<T> A(static_cast<std::size_t>(n * n), T(0));
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            T sum = T(0);
            for (linalg_int k = 0; k < n; ++k)
            {
                sum += B[static_cast<std::size_t>(i * n + k)] *
                       B[static_cast<std::size_t>(j * n + k)];
            }
            A[static_cast<std::size_t>(i * n + j)] = sum;
        }
        A[static_cast<std::size_t>(i * n + i)] += static_cast<T>(n);
    }
    return A;
}

// ---------------------------------------------------------------------------
// CPU backends
// ---------------------------------------------------------------------------

template <linalg::linalg_backend Backend>
void profile_cpu_backend(const char* scope_name, linalg_int n)
{
    PROFILER_SCOPE(scope_name);

    const linalg_long rows = static_cast<linalg_long>(n + n / 2);
    const linalg_long cols = static_cast<linalg_long>(n);

    // multiplication --------------------------------------------------------
    {
        PROFILER_SCOPE("multiplication");
        auto a = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        auto b = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n), 43);
        std::vector<double> c(static_cast<std::size_t>(n * n));
        const auto          ws =
            linalg::multiplication<double, Backend>::workspace_size(n, n, n);
        std::vector<unsigned char> workspace(ws);
        linalg::multiplication<double, Backend>::apply(
            false, false, n, n, n, a.data(), n, b.data(), n, c.data(), n, workspace.data(), ws);
    }

    // cholesky_decomposition ------------------------------------------------
    {
        PROFILER_SCOPE("cholesky_decomposition");
        auto A = make_spd_matrix<double>(n);
        linalg::cholesky_decomposition<double, Backend>::apply(
            A.data(), n, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR);
    }

    // lu_decomposition ------------------------------------------------------
    {
        PROFILER_SCOPE("lu_decomposition");
        auto                    A = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        std::vector<linalg_int> pivot(static_cast<std::size_t>(n));
        const auto              ws = linalg::lu_decomposition<double, Backend>::workspace_size(n);
        std::vector<unsigned char> workspace(ws);
        linalg::lu_decomposition<double, Backend>::apply(
            A.data(), n, pivot.data(), workspace.data(), ws);
    }

    // qr_decomposition ------------------------------------------------------
    {
        PROFILER_SCOPE("qr_decomposition");
        auto                A = make_random_matrix<double>(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        std::vector<double> tau(static_cast<std::size_t>(cols));
        const auto          ws = linalg::qr_decomposition<double, Backend>::workspace_size(rows, cols);
        std::vector<unsigned char> workspace(ws);
        linalg::qr_decomposition<double, Backend>::apply(
            rows, cols, A.data(), cols, tau.data(), workspace.data(), ws);
    }

    // svd_decomposition -----------------------------------------------------
    {
        PROFILER_SCOPE("svd_decomposition");
        auto                A = make_random_matrix<double>(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        std::vector<double> S(static_cast<std::size_t>(cols));
        std::vector<double> U(static_cast<std::size_t>(rows * rows));
        std::vector<double> Vt(static_cast<std::size_t>(cols * cols));
        const auto          ws = linalg::svd_decomposition<double, Backend>::workspace_size(rows, cols);
        std::vector<unsigned char> workspace(ws);
        linalg::svd_decomposition<double, Backend>::apply(
            rows, cols, A.data(), cols, S.data(), U.data(), rows, Vt.data(), cols, workspace.data(), ws);
    }

    // symmetric_eigen_backend -----------------------------------------------
    {
        PROFILER_SCOPE("eigen_symmetric");
        auto                A = make_spd_matrix<double>(n);
        std::vector<double> eigenvalues(static_cast<std::size_t>(n));
        std::vector<double> eigenvectors(static_cast<std::size_t>(n * n));
        const auto          ws = linalg::symmetric_eigen_backend<double, Backend>::workspace_size(n);
        std::vector<unsigned char> workspace(ws);
        linalg::symmetric_eigen_backend<double, Backend>::apply(
            A.data(), n, n, eigenvalues.data(), eigenvectors.data(), n, workspace.data(), ws);
    }

    // general_eigen_backend -------------------------------------------------
    {
        PROFILER_SCOPE("eigen_general");
        auto                A = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        std::vector<double> ev_real(static_cast<std::size_t>(n));
        std::vector<double> ev_imag(static_cast<std::size_t>(n));
        const auto          ws = linalg::general_eigen_backend<double, Backend>::workspace_size(n, false);
        std::vector<unsigned char> workspace(ws);
        linalg::general_eigen_backend<double, Backend>::apply(
            A.data(), n, n, ev_real.data(), ev_imag.data(), nullptr, n, workspace.data(), ws);
    }

    // inversion -------------------------------------------------------------
    {
        PROFILER_SCOPE("inversion");
        auto       A  = make_spd_matrix<double>(n);
        const auto ws = linalg::inversion<double, Backend>::workspace_size(
            n, linalg::linear_solver_type::LU_LINEAR_SOLVER);
        std::vector<unsigned char> workspace(ws);
        linalg::inversion<double, Backend>::apply(
            A.data(), n, linalg::linear_solver_type::LU_LINEAR_SOLVER, workspace.data(), ws);
    }

    // linear_solver ---------------------------------------------------------
    {
        PROFILER_SCOPE("linear_solver");
        auto                A = make_spd_matrix<double>(n);
        std::vector<double> b = make_random_matrix<double>(static_cast<std::size_t>(n), 1, 99);
        std::vector<linalg_int> pivot(static_cast<std::size_t>(n));
        const auto ws = linalg::linear_solver<double, Backend>::workspace_size(
            n, linalg::linear_solver_type::LU_LINEAR_SOLVER);
        std::vector<unsigned char> workspace(ws);
        linalg::linear_solver<double, Backend>::apply(
            A.data(), n, b.data(), linalg::linear_solver_type::LU_LINEAR_SOLVER, pivot.data(), workspace.data(), ws);
    }

    // least_squares ---------------------------------------------------------
    {
        PROFILER_SCOPE("least_squares");
        auto                A = make_random_matrix<double>(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        std::vector<double> b = make_random_matrix<double>(static_cast<std::size_t>(rows), 1, 77);
        const auto          ws = linalg::least_squares<double, Backend>::workspace_size(rows, cols);
        std::vector<unsigned char> workspace(ws);
        linalg::least_squares<double, Backend>::apply(
            rows, cols, A.data(), cols, b.data(), 1, workspace.data(), ws);
    }

    // pseudo_inverse --------------------------------------------------------
    {
        PROFILER_SCOPE("pseudo_inverse");
        auto                A      = make_random_matrix<double>(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        std::vector<double> result(static_cast<std::size_t>(cols * rows));
        const auto          ws = linalg::pseudo_inverse<double, Backend>::workspace_size(rows, cols);
        std::vector<unsigned char> workspace(ws);
        linalg::pseudo_inverse<double, Backend>::apply(
            rows, cols, A.data(), cols, result.data(), rows, workspace.data(), ws);
    }

    // norm ------------------------------------------------------------------
    {
        PROFILER_SCOPE("norm_frobenius");
        auto       A  = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        const auto ws = linalg::norm<double, Backend>::workspace_size(n, n, linalg::norm_type::FROBENIUS);
        std::vector<unsigned char> workspace(ws);
        linalg::norm<double, Backend>::apply(A.data(), n, n, n, linalg::norm_type::FROBENIUS, workspace.data(), ws);
    }

    // trace -----------------------------------------------------------------
    {
        PROFILER_SCOPE("trace");
        auto A = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        linalg::trace<double, Backend>::apply(A.data(), n, n);
    }

    // transpose -------------------------------------------------------------
    {
        PROFILER_SCOPE("transpose");
        auto       A  = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        const auto ws = linalg::transpose<double, Backend>::workspace_size(n, n);
        std::vector<unsigned char> workspace(ws);
        linalg::transpose<double, Backend>::apply(n, n, A.data(), workspace.data(), ws);
    }

    // rank ------------------------------------------------------------------
    {
        PROFILER_SCOPE("rank");
        auto       A  = make_random_matrix<double>(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        const auto ws = linalg::rank<double, Backend>::workspace_size(rows, cols);
        std::vector<unsigned char> workspace(ws);
        linalg::rank<double, Backend>::apply(A.data(), rows, cols, cols, T(0), workspace.data(), ws);
    }

    // condition_number -------------------------------------------------------
    {
        PROFILER_SCOPE("condition_number");
        auto       A  = make_spd_matrix<double>(n);
        const auto ws = linalg::condition_number<double, Backend>::workspace_size(n, n);
        std::vector<unsigned char> workspace(ws);
        linalg::condition_number<double, Backend>::apply(A.data(), n, n, n, workspace.data(), ws);
    }

    // determinant -----------------------------------------------------------
    {
        PROFILER_SCOPE("determinant");
        auto                    A      = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        std::vector<linalg_int> pivot(static_cast<std::size_t>(n));
        const auto              ws = linalg::determinant<double, Backend>::workspace_size(
            n, linalg::linear_solver_type::LU_LINEAR_SOLVER);
        std::vector<unsigned char> workspace(ws);
        linalg::determinant<double, Backend>::apply(
            A.data(), n, pivot.data(), linalg::linear_solver_type::LU_LINEAR_SOLVER, workspace.data(), ws);
    }

    // power -----------------------------------------------------------------
    {
        PROFILER_SCOPE("power");
        auto                A      = make_spd_matrix<double>(n);
        std::vector<double> result(static_cast<std::size_t>(n * n));
        const auto          ws     = linalg::power<double, Backend>::workspace_size(n);
        std::vector<unsigned char> workspace(ws);
        linalg::power<double, Backend>::apply(A.data(), n, n, 3, result.data(), n, workspace.data(), ws);
    }

    // exponential (native only, scalar backend) ----------------------------
    {
        PROFILER_SCOPE("exponential");
        auto                A      = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        std::vector<double> result(static_cast<std::size_t>(n * n));
        const auto          ws     = linalg::exponential<double, Backend>::workspace_size(n);
        std::vector<unsigned char> workspace(ws);
        linalg::exponential<double, Backend>::apply(A.data(), n, n, result.data(), n, workspace.data(), ws);
    }
}

void profile_scalar_only_ops(linalg_int n)
{
    PROFILER_SCOPE("scalar_only_ops");

    const linalg_int tri_n = n;

    // tridiagonal_operations ------------------------------------------------
    {
        PROFILER_SCOPE("tridiagonal/decomposition");
        std::vector<double> lower(static_cast<std::size_t>(tri_n - 1), -1.0);
        std::vector<double> diag(static_cast<std::size_t>(tri_n), 2.0);
        std::vector<double> upper(static_cast<std::size_t>(tri_n - 1), -1.0);
        std::vector<double> output;
        linalg::tridiagonal::decomposition(output, lower, diag, upper);
    }

    {
        PROFILER_SCOPE("tridiagonal/solve_decomposed");
        std::vector<double> lower(static_cast<std::size_t>(tri_n - 1), -1.0);
        std::vector<double> diag(static_cast<std::size_t>(tri_n), 2.0);
        std::vector<double> upper(static_cast<std::size_t>(tri_n - 1), -1.0);
        std::vector<double> decomposed;
        linalg::tridiagonal::decomposition(decomposed, lower, diag, upper);
        std::vector<double> rhs(static_cast<std::size_t>(tri_n), 1.0);
        std::vector<double> x;
        linalg::tridiagonal::solve_decomposed(x, decomposed, rhs);
    }

    {
        PROFILER_SCOPE("tridiagonal/multiply");
        std::vector<double> lower(static_cast<std::size_t>(tri_n - 1), -1.0);
        std::vector<double> diag(static_cast<std::size_t>(tri_n), 2.0);
        std::vector<double> upper(static_cast<std::size_t>(tri_n - 1), -1.0);
        std::vector<double> vec(static_cast<std::size_t>(tri_n), 1.0);
        std::vector<double> result;
        linalg::tridiagonal::multiply(result, lower, diag, upper, vec);
    }

    // pentadiagonal_operations ----------------------------------------------
    {
        PROFILER_SCOPE("pentadiagonal/decomposition");
        std::vector<double> ll(static_cast<std::size_t>(tri_n - 2), -0.5);
        std::vector<double> l(static_cast<std::size_t>(tri_n - 1), -1.0);
        std::vector<double> d(static_cast<std::size_t>(tri_n), 4.0);
        std::vector<double> u(static_cast<std::size_t>(tri_n - 1), -1.0);
        std::vector<double> uu(static_cast<std::size_t>(tri_n - 2), -0.5);
        std::vector<double> output;
        linalg::pentadiagonal::decomposition(output, ll, l, d, u, uu);
    }

    {
        PROFILER_SCOPE("pentadiagonal/solve_decomposed");
        std::vector<double> ll(static_cast<std::size_t>(tri_n - 2), -0.5);
        std::vector<double> l(static_cast<std::size_t>(tri_n - 1), -1.0);
        std::vector<double> d(static_cast<std::size_t>(tri_n), 4.0);
        std::vector<double> u(static_cast<std::size_t>(tri_n - 1), -1.0);
        std::vector<double> uu(static_cast<std::size_t>(tri_n - 2), -0.5);
        std::vector<double> decomposed;
        linalg::pentadiagonal::decomposition(decomposed, ll, l, d, u, uu);
        std::vector<double> rhs(static_cast<std::size_t>(tri_n), 1.0);
        std::vector<double> x;
        linalg::pentadiagonal::solve_decomposed(x, decomposed, rhs);
    }

    {
        PROFILER_SCOPE("pentadiagonal/multiply");
        std::vector<double> ll(static_cast<std::size_t>(tri_n - 2), -0.5);
        std::vector<double> l(static_cast<std::size_t>(tri_n - 1), -1.0);
        std::vector<double> d(static_cast<std::size_t>(tri_n), 4.0);
        std::vector<double> u(static_cast<std::size_t>(tri_n - 1), -1.0);
        std::vector<double> uu(static_cast<std::size_t>(tri_n - 2), -0.5);
        std::vector<double> vec(static_cast<std::size_t>(tri_n), 1.0);
        std::vector<double> result;
        linalg::pentadiagonal::multiply(result, ll, l, d, u, uu, vec);
    }
}

// ---------------------------------------------------------------------------
// GPU backend
// ---------------------------------------------------------------------------

#if LINALG_HAS_CUBLAS

// Allocate a device buffer, copy host data into it, and return the device ptr.
// The caller owns the buffer and must cudaFree it.
template <typename T>
T* cuda_upload(const std::vector<T>& host)
{
    T*           ptr = nullptr;
    const size_t sz  = host.size() * sizeof(T);
    if (cudaMalloc(reinterpret_cast<void**>(&ptr), sz) != cudaSuccess)
    {
        return nullptr;
    }
    if (cudaMemcpy(ptr, host.data(), sz, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(ptr);
        return nullptr;
    }
    return ptr;
}

// Allocate an uninitialised device buffer of `count` elements.
template <typename T>
T* cuda_alloc(std::size_t count)
{
    T* ptr = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T));
    return ptr;
}

void profile_gpu_ops(linalg_int n)
{
    PROFILER_SCOPE("gpu_cublas");

    const linalg_long rows = static_cast<linalg_long>(n + n / 2);
    const linalg_long cols = static_cast<linalg_long>(n);

    // multiplication --------------------------------------------------------
    {
        PROFILER_SCOPE("gpu/multiplication");
        auto   ha = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        auto   hb = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n), 43);
        double* da  = cuda_upload(ha);
        double* db  = cuda_upload(hb);
        double* dc  = cuda_alloc<double>(static_cast<std::size_t>(n * n));
        if (da && db && dc)
        {
            linalg::gpu::multiplication(false, false, n, n, n, da, n, db, n, dc, n);
            cudaDeviceSynchronize();
        }
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
    }

    // cholesky_decomposition ------------------------------------------------
    {
        PROFILER_SCOPE("gpu/cholesky_decomposition");
        auto    hA = make_spd_matrix<double>(n);
        double* dA = cuda_upload(hA);
        int*    d_info = cuda_alloc<int>(1);
        const auto ws_bytes = linalg::gpu::cholesky_decomposition_workspace_size(
            static_cast<const double*>(nullptr), n, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR);
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA && d_info)
        {
            linalg::gpu::cholesky_decomposition(
                dA, n, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR, d_info, d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(d_info);
        cudaFree(d_ws);
    }

    // lu_decomposition ------------------------------------------------------
    {
        PROFILER_SCOPE("gpu/lu_decomposition");
        auto    hA = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        double* dA = cuda_upload(hA);
        int*    d_pivot = cuda_alloc<int>(static_cast<std::size_t>(n));
        int*    d_info  = cuda_alloc<int>(1);
        const auto ws_bytes =
            linalg::gpu::lu_decomposition_workspace_size(static_cast<const double*>(nullptr), n);
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA && d_pivot && d_info)
        {
            linalg::gpu::lu_decomposition(dA, n, d_pivot, d_info, d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(d_pivot);
        cudaFree(d_info);
        cudaFree(d_ws);
    }

    // qr_decomposition ------------------------------------------------------
    {
        PROFILER_SCOPE("gpu/qr_decomposition");
        auto    hA   = make_random_matrix<double>(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        double* dA   = cuda_upload(hA);
        double* d_tau = cuda_alloc<double>(static_cast<std::size_t>(cols));
        int*    d_info = cuda_alloc<int>(1);
        const auto ws_bytes =
            linalg::gpu::qr_decomposition_workspace_size(static_cast<const double*>(nullptr), rows, cols);
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA && d_tau && d_info)
        {
            linalg::gpu::qr_decomposition(rows, cols, dA, cols, d_tau, d_info, d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(d_tau);
        cudaFree(d_info);
        cudaFree(d_ws);
    }

    // svd_decomposition -----------------------------------------------------
    {
        PROFILER_SCOPE("gpu/svd_decomposition");
        auto    hA  = make_random_matrix<double>(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        double* dA  = cuda_upload(hA);
        double* d_S = cuda_alloc<double>(static_cast<std::size_t>(cols));
        double* d_U = cuda_alloc<double>(static_cast<std::size_t>(rows * rows));
        double* d_Vt = cuda_alloc<double>(static_cast<std::size_t>(cols * cols));
        int*    d_info = cuda_alloc<int>(1);
        const auto ws_bytes =
            linalg::gpu::svd_decomposition_workspace_size(static_cast<const double*>(nullptr), rows, cols);
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA && d_S && d_U && d_Vt && d_info)
        {
            linalg::gpu::svd_decomposition(rows, cols, dA, cols, d_S, d_U, rows, d_Vt, cols, d_info, d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(d_S);
        cudaFree(d_U);
        cudaFree(d_Vt);
        cudaFree(d_info);
        cudaFree(d_ws);
    }

    // symmetric_eigenvalue_decomposition ------------------------------------
    {
        PROFILER_SCOPE("gpu/eigen_symmetric");
        auto    hA  = make_spd_matrix<double>(n);
        double* dA  = cuda_upload(hA);
        double* d_ev = cuda_alloc<double>(static_cast<std::size_t>(n));
        double* d_vecs = cuda_alloc<double>(static_cast<std::size_t>(n * n));
        int*    d_info = cuda_alloc<int>(1);
        const auto ws_bytes = linalg::gpu::symmetric_eigenvalue_decomposition_workspace_size(
            static_cast<const double*>(nullptr), n);
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA && d_ev && d_vecs && d_info)
        {
            linalg::gpu::symmetric_eigenvalue_decomposition(
                dA, n, n, d_ev, d_vecs, n, d_info, d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(d_ev);
        cudaFree(d_vecs);
        cudaFree(d_info);
        cudaFree(d_ws);
    }

    // matrix_invert ---------------------------------------------------------
    {
        PROFILER_SCOPE("gpu/inversion");
        auto    hA = make_spd_matrix<double>(n);
        double* dA = cuda_upload(hA);
        int*    d_info = cuda_alloc<int>(1);
        const auto ws_bytes =
            linalg::gpu::matrix_invert_workspace_size(static_cast<const double*>(nullptr), n);
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA && d_info)
        {
            linalg::gpu::matrix_invert(dA, n, d_info, d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(d_info);
        cudaFree(d_ws);
    }

    // linear_solver ---------------------------------------------------------
    {
        PROFILER_SCOPE("gpu/linear_solver");
        auto    hA = make_spd_matrix<double>(n);
        auto    hb = make_random_matrix<double>(static_cast<std::size_t>(n), 1, 77);
        double* dA = cuda_upload(hA);
        double* db = cuda_upload(hb);
        int*    d_info = cuda_alloc<int>(1);
        const auto ws_bytes = linalg::gpu::linear_solver_workspace_size(
            static_cast<const double*>(nullptr), n, linalg::linear_solver_type::LU_LINEAR_SOLVER);
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA && db && d_info)
        {
            linalg::gpu::linear_solver(dA, n, db, linalg::linear_solver_type::LU_LINEAR_SOLVER, d_info, d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(db);
        cudaFree(d_info);
        cudaFree(d_ws);
    }

    // least_squares ---------------------------------------------------------
    {
        PROFILER_SCOPE("gpu/least_squares");
        auto    hA = make_random_matrix<double>(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        auto    hb = make_random_matrix<double>(static_cast<std::size_t>(rows), 1, 55);
        double* dA = cuda_upload(hA);
        double* db = cuda_upload(hb);
        int*    d_info = cuda_alloc<int>(1);
        const auto ws_bytes = linalg::gpu::least_squares_solve_workspace_size(
            static_cast<const double*>(nullptr), rows, cols);
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA && db && d_info)
        {
            linalg::gpu::least_squares_solve(rows, cols, dA, cols, db, 1, d_info, d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(db);
        cudaFree(d_info);
        cudaFree(d_ws);
    }

    // pseudo_inverse --------------------------------------------------------
    {
        PROFILER_SCOPE("gpu/pseudo_inverse");
        auto    hA      = make_random_matrix<double>(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        double* dA      = cuda_upload(hA);
        double* d_result = cuda_alloc<double>(static_cast<std::size_t>(cols * rows));
        int*    d_info  = cuda_alloc<int>(1);
        const auto ws_bytes =
            linalg::gpu::pseudo_inverse_workspace_size(static_cast<const double*>(nullptr), rows, cols);
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA && d_result && d_info)
        {
            linalg::gpu::pseudo_inverse(rows, cols, dA, cols, d_result, rows, d_info, d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(d_result);
        cudaFree(d_info);
        cudaFree(d_ws);
    }

    // norm ------------------------------------------------------------------
    {
        PROFILER_SCOPE("gpu/norm_frobenius");
        auto    hA = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        double* dA = cuda_upload(hA);
        const auto ws_bytes = linalg::gpu::matrix_norm_workspace_size(
            static_cast<const double*>(nullptr), n, n, linalg::norm_type::FROBENIUS);
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA)
        {
            linalg::gpu::norm(dA, n, n, n, linalg::norm_type::FROBENIUS, d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(d_ws);
    }

    // trace -----------------------------------------------------------------
    {
        PROFILER_SCOPE("gpu/trace");
        auto    hA = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        double* dA = cuda_upload(hA);
        const auto ws_bytes =
            linalg::gpu::matrix_trace_workspace_size(static_cast<const double*>(nullptr));
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA)
        {
            linalg::gpu::trace(dA, n, n, d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(d_ws);
    }

    // transpose -------------------------------------------------------------
    {
        PROFILER_SCOPE("gpu/transpose");
        auto    hA = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        double* dA = cuda_upload(hA);
        const auto ws_bytes = linalg::gpu::matrix_transpose_workspace_size(
            static_cast<const double*>(nullptr), n, n);
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA)
        {
            linalg::gpu::transpose(n, n, dA, d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(d_ws);
    }

    // rank ------------------------------------------------------------------
    {
        PROFILER_SCOPE("gpu/rank");
        auto    hA = make_random_matrix<double>(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        double* dA = cuda_upload(hA);
        const auto ws_bytes = linalg::gpu::matrix_rank_workspace_size(
            static_cast<const double*>(nullptr), rows, cols);
        void* d_ws = nullptr;
        if (ws_bytes > 0)
        {
            cudaMalloc(&d_ws, ws_bytes);
        }
        if (dA)
        {
            linalg::gpu::rank(dA, rows, cols, cols, double(0), d_ws, ws_bytes);
            cudaDeviceSynchronize();
        }
        cudaFree(dA);
        cudaFree(d_ws);
    }
}
#endif  // LINALG_HAS_CUBLAS

}  // namespace

int main(int argc, char** argv)
{
    linalg_int n = 128;
    if (argc > 1)
    {
        char*      end    = nullptr;
        const long parsed = std::strtol(argv[1], &end, 10);
        if (end != argv[1] && parsed > 4 && parsed <= 4096)
        {
            n = static_cast<linalg_int>(parsed);
        }
    }

    std::printf("=== LinearAlgebra matrix operation profiler ===\n");
    std::printf("Matrix order n=%d  (rectangular rows=%d, cols=%d)\n\n", static_cast<int>(n),
        static_cast<int>(n + n / 2), static_cast<int>(n));

    // --- CPU session -------------------------------------------------------
    {
        std::printf("--- CPU backends ---\n");

        profiler::session_options opts;
        opts.activities         = {profiler::activity::cpu};
        opts.statistical_analysis = true;

        profiler::session session(opts);
        if (!session.start())
        {
            std::fprintf(stderr, "warning: CPU profiler session failed to start\n");
        }

        profile_cpu_backend<linalg::linalg_backend::native>("cpu_native", n);

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
        profile_cpu_backend<linalg::linalg_backend::blas_lapacke>("cpu_blas_lapacke", n);
#endif

#if LINALG_HAS_MKL
        profile_cpu_backend<linalg::linalg_backend::mkl>("cpu_mkl", n);
#endif

        profile_scalar_only_ops(n);

        session.stop();

        const char* cpu_trace = "profile_linalg_cpu.json";
        if (session.write_trace(cpu_trace))
        {
            std::printf("CPU trace written to %s\n", cpu_trace);
        }

        auto report = session.generate_report();
        if (report)
        {
            std::printf("\n--- CPU profiler report ---\n");
            report->print_detailed_report();
        }
    }

    // --- GPU session -------------------------------------------------------
#if LINALG_HAS_CUBLAS
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        {
            std::printf("\n--- GPU backend ---\nNOTE: no CUDA device detected; skipping GPU profile.\n");
        }
        else
        {
            std::printf("\n--- GPU backend (%d device(s)) ---\n", device_count);

            profiler::session_options gpu_opts;
            gpu_opts.activities  = {profiler::activity::cpu, profiler::activity::cuda};
            gpu_opts.gpu_tracing = true;
            gpu_opts.statistical_analysis = true;

            profiler::session gpu_session(gpu_opts);
            if (!gpu_session.start())
            {
                std::fprintf(stderr, "warning: GPU profiler session failed to start\n");
            }

            profile_gpu_ops(n);

            gpu_session.stop();

            const char* gpu_trace = "profile_linalg_gpu.json";
            if (gpu_session.write_trace(gpu_trace))
            {
                std::printf("GPU trace written to %s\n", gpu_trace);
            }

            auto gpu_report = gpu_session.generate_report();
            if (gpu_report)
            {
                std::printf("\n--- GPU profiler report ---\n");
                gpu_report->print_detailed_report();
            }
        }
    }
#else
    std::printf("\nGPU backend: not compiled in (LINALG_HAS_CUBLAS=0)\n");
#endif

    std::printf("\nVisualization: open JSON traces in chrome://tracing or https://ui.perfetto.dev\n");
    return 0;
}
