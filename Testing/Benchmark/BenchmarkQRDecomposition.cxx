// Benchmarks linalg::qr_decomposition<T, Backend>::apply_r/apply_q for every backend
// this build actually compiled in.
//
// The library never defaults to a backend — every linalg::*<T, Backend> call
// requires Backend to be named explicitly. This benchmark mirrors that:
// BM_QRDecomposition is templated on Backend too, and registered once per
// backend gated on LINALG_HAS_MKL / LINALG_HAS_BLAS / LINALG_HAS_LAPACKE
// (native is unconditional, the scalar fallback always builds) — see
// Testing/Cxx/test_backend.h for the same per-backend coverage on the unit
// test side, and BenchmarkCholeskyDecomposition.cxx for the sibling
// benchmark this one is modeled on. A single run now reports all backends
// present in this build, each row labeled with the backend that produced it.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "include/common/backend.h"
#include "include/common/types.h"
#include "include/matrix_operation/qr_decomposition.h"

#include <Eigen/QR>
#include <type_traits>

namespace
{

const char* backend_name(linalg::linalg_backend backend)
{
    switch (backend)
    {
    case linalg::linalg_backend::MKL:
        return "mkl";
    case linalg::linalg_backend::BLAS_LAPACKE:
#if LINALG_BENCHMARK_BLIS
        return "blis";
#else
        return "blas_lapacke";
#endif
    default:
        return "native";
    }
}

template <typename T>
std::vector<T> make_random_matrix(std::size_t rows, std::size_t columns, unsigned seed)
{
    std::mt19937                      rng(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(-1), static_cast<T>(1));

    std::vector<T> a(rows * columns);
    for (auto& v : a)
    {
        v = dist(rng);
    }
    return a;
}

template <typename T, linalg::linalg_backend Backend>
void BM_QRDecomposition(benchmark::State& state)
{
    const std::size_t    n = static_cast<std::size_t>(state.range(0));
    const std::vector<T> a = make_random_matrix<T>(n, n, /*seed=*/42);

    const size_t ws_size = linalg::qr_decomposition<T, Backend>::workspace_size(
        static_cast<linalg_long>(n), static_cast<linalg_long>(n));
    std::vector<std::byte> workspace(ws_size);

    auto           input = a;
    std::vector<T> tau(n);
    for (auto _ : state)
    {
        state.PauseTiming();
        std::copy(a.begin(), a.end(), input.begin());
        state.ResumeTiming();
        linalg::qr_decomposition<T, Backend>::apply_r(
            n, n, input.data(), n, tau.data(), workspace.data(), ws_size);
        linalg::qr_decomposition<T, Backend>::apply_q(
            n, n, input.data(), n, tau.data(), workspace.data(), ws_size);
        benchmark::DoNotOptimize(input.data());
        benchmark::DoNotOptimize(tau.data());
    }

    state.SetLabel(backend_name(Backend));
    state.SetComplexityN(static_cast<std::int64_t>(n));

    // Economy QR factorization of a square n x n matrix costs ~2n^3 flops.
    const double flops_per_iter =
        2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    state.counters["FLOPS"] = benchmark::Counter(flops_per_iter,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
}

template <typename T> void BM_EigenQRDecomposition(benchmark::State& state)
{
    const auto n = state.range(0);
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const auto                               input = make_random_matrix<T>(n, n, 42);
    const Eigen::Map<const Matrix>           original(input.data(), n, n);
    Matrix                                   work = original;
    Matrix                                   q(n, n);
    Eigen::HouseholderQR<Eigen::Ref<Matrix>> qr(work);
    for (auto _ : state)
    {
        state.PauseTiming();
        work = original;
        state.ResumeTiming();
        qr.compute(work);
        q = qr.householderQ();
        benchmark::DoNotOptimize(q.data());
        benchmark::ClobberMemory();
    }
    state.SetLabel("eigen");
    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(2.0 * n * n * n,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
}

}  // namespace

BENCHMARK_TEMPLATE(BM_QRDecomposition, float, linalg::linalg_backend::NATIVE)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_QRDecomposition, double, linalg::linalg_backend::NATIVE)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
BENCHMARK_TEMPLATE(BM_QRDecomposition, float, linalg::linalg_backend::BLAS_LAPACKE)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_QRDecomposition, double, linalg::linalg_backend::BLAS_LAPACKE)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
#endif

#if LINALG_HAS_MKL
BENCHMARK_TEMPLATE(BM_QRDecomposition, float, linalg::linalg_backend::MKL)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_QRDecomposition, double, linalg::linalg_backend::MKL)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
#endif

BENCHMARK_TEMPLATE(BM_EigenQRDecomposition, float)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
BENCHMARK_TEMPLATE(BM_EigenQRDecomposition, double)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
