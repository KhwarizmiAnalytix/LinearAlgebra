// Benchmarks linalg::symmetric_eigen_backend<T, Backend>::apply and
// linalg::general_eigen_backend<T, Backend>::apply for every backend this
// build actually compiled in.
//
// The library never defaults to a backend — every linalg::*<T, Backend> call
// requires Backend to be named explicitly. This benchmark mirrors that:
// BM_SymmetricEigen / BM_GeneralEigen are templated on Backend too, and
// registered once per backend gated on LINALG_HAS_MKL / LINALG_HAS_BLAS /
// LINALG_HAS_LAPACKE (native is unconditional, the scalar fallback always
// builds) — see Testing/Cxx/test_backend.h for the same per-backend coverage
// on the unit test side, and BenchmarkQRDecomposition.cxx for the sibling
// benchmark this one is modeled on. A single run now reports all backends
// present in this build, each row labeled with the backend that produced it.
//
// Eigenvalue iteration counts depend on the spectrum, so report time rather
// than FLOPS. The general registrations below distinguish eigenvalues only
// from eigenvalues with real eigenvectors.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "include/common/backend.h"
#include "include/common/types.h"
#include "include/matrix_operation/eigen.h"

namespace
{

const char* backend_name(linalg::linalg_backend backend)
{
    switch (backend)
    {
    case linalg::linalg_backend::MKL:
        return "mkl";
    case linalg::linalg_backend::BLAS_LAPACKE:
        return "blas_lapacke";
    default:
        return "native";
    }
}

// Random row-major n x n symmetric matrix: A = M + M^T, so only the lower
// triangle symmetric_eigen_backend::apply reads is ever exercised by
// construction, rather than by also mirroring it after the fact.
template <typename T> std::vector<T> make_symmetric_matrix(std::size_t n, unsigned seed)
{
    std::mt19937                      rng(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(-1), static_cast<T>(1));

    std::vector<T> a(n * n, T{});
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const T v    = dist(rng);
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    return a;
}

// Dense nonsymmetric input. Diagonal dominance does not guarantee real
// eigenvalues, so the general benchmark requests eigenvalues only.
template <typename T> std::vector<T> make_well_conditioned_matrix(std::size_t n, unsigned seed)
{
    std::mt19937                      rng(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(-1), static_cast<T>(1));

    std::vector<T> a(n * n);
    for (auto& v : a)
    {
        v = dist(rng);
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        a[i * n + i] += static_cast<T>(2 * n);
    }
    return a;
}

template <typename T, linalg::linalg_backend Backend>
void BM_SymmetricEigen(benchmark::State& state)
{
    const std::size_t    n = static_cast<std::size_t>(state.range(0));
    const std::vector<T> a = make_symmetric_matrix<T>(n, /*seed=*/42);

    const size_t ws_size =
        linalg::symmetric_eigen_backend<T, Backend>::workspace_size(static_cast<linalg_int>(n));
    std::vector<unsigned char> workspace(ws_size);

    std::vector<T> eigenvalues(n);
    auto input = a;

    for (auto _ : state)
    {
        state.PauseTiming();
        std::copy(a.begin(), a.end(), input.begin());
        state.ResumeTiming();
        bool ok = linalg::symmetric_eigen_backend<T, Backend>::apply(input.data(),
            static_cast<linalg_int>(n),
            static_cast<linalg_int>(n),
            eigenvalues.data(),
            static_cast<linalg_int>(n),
            workspace.data(),
            ws_size);
        if (!ok)
        {
            state.SkipWithError("Eigenvalue iteration failed to converge");
            break;
        }
        benchmark::DoNotOptimize(ok);
    }

    state.SetLabel(backend_name(Backend));
    state.SetComplexityN(static_cast<std::int64_t>(n));
}

template <typename T, linalg::linalg_backend Backend, bool ComputeEigenvectors = false>
void BM_GeneralEigen(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<T>    a = make_well_conditioned_matrix<T>(n, /*seed=*/42);
    if constexpr (ComputeEigenvectors)
    {
        // A = D S D^-1 for symmetric S and positive diagonal D: a dense,
        // nonsymmetric matrix with a real spectrum, supported by native vectors.
        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = 0; j < i; ++j)
            {
                const T ratio = (T(1) + T(i) / T(n)) / (T(1) + T(j) / T(n));
                const T value = a[i * n + j];
                a[i * n + j]  = value * ratio;
                a[j * n + i]  = value / ratio;
            }
        }
    }

    const size_t ws_size = linalg::general_eigen_backend<T, Backend>::workspace_size(
        static_cast<linalg_int>(n), ComputeEigenvectors);
    std::vector<unsigned char> workspace(ws_size);

    std::vector<T> eigenvalues_real(n);
    std::vector<T> eigenvalues_imag(n);
    auto input = a;

    for (auto _ : state)
    {
        state.PauseTiming();
        std::copy(a.begin(), a.end(), input.begin());
        state.ResumeTiming();
        bool ok = linalg::general_eigen_backend<T, Backend>::apply(input.data(),
            static_cast<linalg_int>(n),
            static_cast<linalg_int>(n),
            eigenvalues_real.data(),
            eigenvalues_imag.data(),
            ComputeEigenvectors,
            static_cast<linalg_int>(n),
            workspace.data(),
            ws_size);
        if (!ok)
        {
            state.SkipWithError("Eigenvalue iteration failed to converge");
            break;
        }
        benchmark::DoNotOptimize(ok);
    }

    state.SetLabel(backend_name(Backend));
    state.SetComplexityN(static_cast<std::int64_t>(n));
}

}  // namespace

BENCHMARK_TEMPLATE(BM_GeneralEigen, float, linalg::linalg_backend::NATIVE, true)
    ->Arg(8)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_GeneralEigen, double, linalg::linalg_backend::NATIVE, true)
    ->Arg(8)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_SymmetricEigen, float, linalg::linalg_backend::NATIVE)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(96)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_SymmetricEigen, double, linalg::linalg_backend::NATIVE)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(96)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_GeneralEigen, float, linalg::linalg_backend::NATIVE)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(96)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_GeneralEigen, double, linalg::linalg_backend::NATIVE)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(96)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
BENCHMARK_TEMPLATE(BM_SymmetricEigen, float, linalg::linalg_backend::BLAS_LAPACKE)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(96)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_SymmetricEigen, double, linalg::linalg_backend::BLAS_LAPACKE)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(96)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_GeneralEigen, float, linalg::linalg_backend::BLAS_LAPACKE)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(96)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_GeneralEigen, double, linalg::linalg_backend::BLAS_LAPACKE)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(96)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
#endif


#if LINALG_HAS_MKL
BENCHMARK_TEMPLATE(BM_SymmetricEigen, float, linalg::linalg_backend::MKL)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(96)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_SymmetricEigen, double, linalg::linalg_backend::MKL)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(96)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_GeneralEigen, float, linalg::linalg_backend::MKL)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(96)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_GeneralEigen, double, linalg::linalg_backend::MKL)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(96)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
#endif
