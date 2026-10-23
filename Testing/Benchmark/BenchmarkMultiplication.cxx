// Benchmarks linalg::multiplication<T, Backend>::apply for every backend
// this build actually compiled in.
//
// The library never defaults to a backend — every linalg::*<T, Backend> call
// requires Backend to be named explicitly. This benchmark mirrors that:
// BM_Multiplication is templated on Backend and transpose flags, and registered
// once per backend gated on LINALG_HAS_MKL / LINALG_HAS_BLAS (native is
// unconditional, the scalar fallback always builds).
//
// Parameters: {M, N, K} — C(M×N) = A(M×K) * B(K×N).
// Square runs use M=N=K; rectangular runs exercise fat (M<<N) and tall (M>>N)
// shapes that stress cache tiling differently.
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "include/common/backend.h"
#include "include/common/types.h"
#include "include/matrix_operation/multiplication.h"

#include <Eigen/Core>
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

template <typename T> std::vector<T> make_random(std::size_t n, unsigned seed)
{
    std::mt19937                      rng(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(-1), static_cast<T>(1));
    std::vector<T>                    v(n);
    for (auto& x : v)
    {
        x = dist(rng);
    }
    return v;
}

// BM_Multiplication — square and rectangular, no transpose.
// state.range(0)=M, state.range(1)=N, state.range(2)=K.
template <typename T, linalg::linalg_backend Backend>
void BM_Multiplication(benchmark::State& state)
{
    const auto m = static_cast<std::size_t>(state.range(0));
    const auto n = static_cast<std::size_t>(state.range(1));
    const auto k = static_cast<std::size_t>(state.range(2));

    const auto a = make_random<T>(m * k, /*seed=*/42);
    const auto b = make_random<T>(k * n, /*seed=*/43);
    auto       c = std::vector<T>(m * n);

    const auto ws_bytes = linalg::multiplication<T, Backend>::workspace_size(
        static_cast<linalg_int>(m), static_cast<linalg_int>(n), static_cast<linalg_int>(k));
    std::vector<unsigned char> workspace(ws_bytes);

    for (auto _ : state)
    {
        linalg::multiplication<T, Backend>::apply(false,
            false,
            static_cast<linalg_int>(m),
            static_cast<linalg_int>(n),
            static_cast<linalg_int>(k),
            a.data(),
            static_cast<linalg_int>(k),
            b.data(),
            static_cast<linalg_int>(n),
            c.data(),
            static_cast<linalg_int>(n),
            workspace.data(),
            ws_bytes);
        benchmark::DoNotOptimize(c.data());
    }

    state.SetLabel(backend_name(Backend));
    state.SetComplexityN(
        static_cast<std::int64_t>(m) * static_cast<std::int64_t>(n) * static_cast<std::int64_t>(k));

    const double flops =
        2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    state.counters["FLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
}

// BM_MultiplicationTranspose — square matrices with all four transpose combos.
// state.range(0)=N (square dimension).
// TransposeA / TransposeB are compile-time booleans so each variant gets its
// own benchmark row.
template <typename T, linalg::linalg_backend Backend, bool TransposeA, bool TransposeB>
void BM_MultiplicationTranspose(benchmark::State& state)
{
    const auto n = static_cast<std::size_t>(state.range(0));

    // A is (rows × depth): if TransposeA, the stored matrix is depth×rows = n×n (square).
    const auto a = make_random<T>(n * n, /*seed=*/42);
    const auto b = make_random<T>(n * n, /*seed=*/43);
    auto       c = std::vector<T>(n * n);

    const auto ws_bytes = linalg::multiplication<T, Backend>::workspace_size(
        static_cast<linalg_int>(n), static_cast<linalg_int>(n), static_cast<linalg_int>(n));
    std::vector<unsigned char> workspace(ws_bytes);

    for (auto _ : state)
    {
        linalg::multiplication<T, Backend>::apply(TransposeA,
            TransposeB,
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

    char label[64];
    std::snprintf(label,
        sizeof(label),
        "%s tA=%d tB=%d",
        backend_name(Backend),
        static_cast<int>(TransposeA),
        static_cast<int>(TransposeB));
    state.SetLabel(label);

    const double flops =
        2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    state.counters["FLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);
}

// clang-format off
// ---- Helper macro so each backend block stays DRY -------------------------
// Square: 32, 64, 128, 256, 512, 1024.
// Rectangular: fat M<<N (64×1024×256), tall M>>N (1024×64×256).
#define LINALG_BENCH_MATMUL(T, Backend)                                                             \
    BENCHMARK_TEMPLATE(BM_Multiplication, T, linalg::linalg_backend::Backend)                      \
        ->Args({32,   32,   32})                                                                    \
        ->Args({64,   64,   64})                                                                    \
        ->Args({128,  128,  128})                                                                   \
        ->Args({256,  256,  256})                                                                   \
        ->Args({512,  512,  512})                                                                   \
        ->Args({1024, 1024, 1024})                                                                  \
        ->Args({64,   1024, 256})                                                                   \
        ->Args({1024, 64,   256})                                                                   \
        ->Unit(benchmark::kMicrosecond)                                                             \
        ->MinTime(0.1)                                                                              \
        ->Repetitions(3)                                                                            \
        ->ReportAggregatesOnly(true)                                                                \
        ->Threads(1);                                                                               \
    BENCHMARK_TEMPLATE(BM_MultiplicationTranspose, T, linalg::linalg_backend::Backend, false, true) \
        ->Arg(64)->Arg(256)->Arg(512)->Arg(1024)                                                    \
        ->Unit(benchmark::kMicrosecond)->MinTime(0.1)->Repetitions(3)                              \
        ->ReportAggregatesOnly(true)->Threads(1);                                                   \
    BENCHMARK_TEMPLATE(BM_MultiplicationTranspose, T, linalg::linalg_backend::Backend, true, false) \
        ->Arg(64)->Arg(256)->Arg(512)->Arg(1024)                                                    \
        ->Unit(benchmark::kMicrosecond)->MinTime(0.1)->Repetitions(3)                              \
        ->ReportAggregatesOnly(true)->Threads(1);                                                   \
    BENCHMARK_TEMPLATE(BM_MultiplicationTranspose, T, linalg::linalg_backend::Backend, true, true) \
        ->Arg(64)->Arg(256)->Arg(512)->Arg(1024)                                                    \
        ->Unit(benchmark::kMicrosecond)->MinTime(0.1)->Repetitions(3)                              \
        ->ReportAggregatesOnly(true)->Threads(1)
// clang-format on

template <typename T, bool TransposeA = false, bool TransposeB = false>
void BM_EigenMultiplication(benchmark::State& state)
{
    const auto m = state.range(0);
    const auto n = (TransposeA || TransposeB) ? m : state.range(1);
    const auto k = (TransposeA || TransposeB) ? m : state.range(2);
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const auto                     a = make_random<T>(m * k, 42);
    const auto                     b = make_random<T>(k * n, 43);
    const Eigen::Map<const Matrix> lhs(a.data(), TransposeA ? k : m, TransposeA ? m : k);
    const Eigen::Map<const Matrix> rhs(b.data(), TransposeB ? n : k, TransposeB ? k : n);
    Matrix                         output(m, n);
    for (auto _ : state)
    {
        if constexpr (TransposeA && TransposeB)
            output.noalias() = lhs.transpose() * rhs.transpose();
        else if constexpr (TransposeA)
            output.noalias() = lhs.transpose() * rhs;
        else if constexpr (TransposeB)
            output.noalias() = lhs * rhs.transpose();
        else
            output.noalias() = lhs * rhs;
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }
    state.SetLabel("eigen");
    state.SetComplexityN(m * n * k);
    state.counters["FLOPS"] = benchmark::Counter(2.0 * m * n * k,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
}

}  // namespace

LINALG_BENCH_MATMUL(float, NATIVE);
LINALG_BENCH_MATMUL(double, NATIVE);

#if LINALG_HAS_BLAS
LINALG_BENCH_MATMUL(float, BLAS_LAPACKE);
LINALG_BENCH_MATMUL(double, BLAS_LAPACKE);
#endif

#if LINALG_HAS_MKL
LINALG_BENCH_MATMUL(float, MKL);
LINALG_BENCH_MATMUL(double, MKL);
#endif

BENCHMARK_TEMPLATE(BM_EigenMultiplication, float)
    ->Args({32, 32, 32})
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({64, 1024, 256})
    ->Args({1024, 64, 256})
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
BENCHMARK_TEMPLATE(BM_EigenMultiplication, float, false, true)
    ->Arg(64)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
BENCHMARK_TEMPLATE(BM_EigenMultiplication, float, true, false)
    ->Arg(64)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
BENCHMARK_TEMPLATE(BM_EigenMultiplication, float, true, true)
    ->Arg(64)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
BENCHMARK_TEMPLATE(BM_EigenMultiplication, double)
    ->Args({32, 32, 32})
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({64, 1024, 256})
    ->Args({1024, 64, 256})
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
BENCHMARK_TEMPLATE(BM_EigenMultiplication, double, false, true)
    ->Arg(64)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
BENCHMARK_TEMPLATE(BM_EigenMultiplication, double, true, false)
    ->Arg(64)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
BENCHMARK_TEMPLATE(BM_EigenMultiplication, double, true, true)
    ->Arg(64)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
