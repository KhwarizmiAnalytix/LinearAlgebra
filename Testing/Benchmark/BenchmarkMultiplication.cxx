// Benchmarks linalg::multiplication<T, Backend>::apply for every backend
// this build actually compiled in.
//
// The library never defaults to a backend — every linalg::*<T, Backend> call
// requires Backend to be named explicitly. This benchmark mirrors that:
// BM_Multiplication is templated on Backend too, and registered once per
// backend gated on LINALG_HAS_MKL / LINALG_HAS_BLAS / LINALG_HAS_LAPACKE
// (native is unconditional, the scalar fallback always builds) — see
// Testing/Cxx/test_backend.h for the same per-backend coverage on the unit
// test side. A single run now reports all backends present in this build,
// each row labeled with the backend that produced it.
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "include/common/backend.h"
#include "include/common/types.h"
#include "include/matrix_operation/multiplication.h"

namespace
{

const char* backend_name(linalg::linalg_backend backend)
{
    switch (backend)
    {
    case linalg::linalg_backend::mkl:
        return "mkl";
    case linalg::linalg_backend::blas_lapacke:
        return "blas_lapacke";
    default:
        return "native";
    }
}

template <typename T> std::vector<T> make_random_matrix(std::size_t n, unsigned seed)
{
    std::mt19937                      rng(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(-1), static_cast<T>(1));

    std::vector<T> m(n * n);
    for (auto& v : m)
    {
        v = dist(rng);
    }
    return m;
}

template <typename T, linalg::linalg_backend Backend>
void BM_Multiplication(benchmark::State& state)
{
    const std::size_t    n = static_cast<std::size_t>(state.range(0));
    const std::vector<T> a = make_random_matrix<T>(n, /*seed=*/42);
    const std::vector<T> b = make_random_matrix<T>(n, /*seed=*/43);
    std::vector<T>       c(n * n);

    const auto ws_bytes = linalg::multiplication<T, Backend>::workspace_size(
        static_cast<linalg_int>(n), static_cast<linalg_int>(n), static_cast<linalg_int>(n));
    std::vector<unsigned char> workspace(ws_bytes);

    for (auto _ : state)
    {
        linalg::multiplication<T, Backend>::apply(false,
            false,
            static_cast<linalg_int>(n),
            static_cast<linalg_int>(n),
            static_cast<linalg_int>(n),
            a.data(),
            static_cast<linalg_int>(n),
            b.data(),
            static_cast<linalg_int>(n),
            c.data(),
            static_cast<linalg_int>(n),
            workspace.data(),
            ws_bytes);
        benchmark::DoNotOptimize(c.data());
    }

    state.SetLabel(backend_name(Backend));
    state.SetComplexityN(static_cast<std::int64_t>(n));

    // n x n * n x n matrix multiplication costs 2*n^3 flops (multiply-add pairs).
    const double flops_per_iter =
        2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    state.counters["FLOPS"] = benchmark::Counter(flops_per_iter,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
}

}  // namespace

BENCHMARK_TEMPLATE(BM_Multiplication, float, linalg::linalg_backend::native)
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

BENCHMARK_TEMPLATE(BM_Multiplication, double, linalg::linalg_backend::native)
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
BENCHMARK_TEMPLATE(BM_Multiplication, float, linalg::linalg_backend::blas_lapacke)
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

BENCHMARK_TEMPLATE(BM_Multiplication, double, linalg::linalg_backend::blas_lapacke)
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
BENCHMARK_TEMPLATE(BM_Multiplication, float, linalg::linalg_backend::mkl)
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

BENCHMARK_TEMPLATE(BM_Multiplication, double, linalg::linalg_backend::mkl)
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
