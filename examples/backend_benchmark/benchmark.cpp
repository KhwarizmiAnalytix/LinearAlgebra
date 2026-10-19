#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime_api.h>

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/eigen.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/qr_decomposition.h"
#include "include/matrix_operation/svd_decomposition.h"

using backend        = linalg::linalg_backend;
using clock_type     = std::chrono::steady_clock;
bool rerun_mode      = false;
bool steady_gpu_mode = false;

std::set<std::string>& selected_cases()
{
    static std::set<std::string> cases;
    return cases;
}

void cuda_check(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename T> class device_buffer
{
    T* data_ = nullptr;

public:
    explicit device_buffer(std::size_t n)
    {
        if (n)
        {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T)));
        }
    }
    ~device_buffer()
    {
        if (data_)
        {
            cudaFree(data_);
        }
    }
    device_buffer(const device_buffer&)            = delete;
    device_buffer& operator=(const device_buffer&) = delete;
    T*             get() const { return data_; }
};

class stream_owner
{
    cudaStream_t stream_ = nullptr;

public:
    stream_owner() { cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~stream_owner() { cudaStreamDestroy(stream_); }
    cudaStream_t get() const { return stream_; }
    void         wait() const { cuda_check(cudaStreamSynchronize(stream_)); }
};

template <typename T> std::vector<T> random_matrix(std::size_t n, unsigned seed)
{
    std::mt19937                      rng(seed);
    std::uniform_real_distribution<T> distribution(T(-1), T(1));
    std::vector<T>                    result(n);
    for (auto& value : result)
    {
        value = distribution(rng);
    }
    return result;
}

enum class operation : std::uint8_t
{
    gemm,
    chol,
    lu,
    qr,
    svd,
    symmetric_eigen,
    general_eigen
};

const char* backend_name(backend b)
{
    switch (b)
    {
    case backend::native:
        return "native";
    case backend::blas_lapacke:
        return "openblas";
    case backend::mkl:
        return "mkl";
    default:
        return "gpu";
    }
}

const char* operation_name(operation op)
{
    switch (op)
    {
    case operation::gemm:
        return "gemm";
    case operation::chol:
        return "cholesky";
    case operation::lu:
        return "lu";
    case operation::qr:
        return "qr";
    case operation::svd:
        return "svd";
    case operation::symmetric_eigen:
        return "symmetric_eigen";
    default:
        return "general_eigen_values";
    }
}

template <typename T, backend B, operation Op>
void run_case(std::ofstream& csv, int m, int n, int k, bool ta = false, bool tb = false)
{
    const char*        precision = std::is_same_v<T, float> ? "float" : "double";
    std::ostringstream key;
    key << operation_name(Op) << ',' << precision << ',' << backend_name(B) << ',' << m << ',' << n
        << ',' << k << ',' << ta << ',' << tb;
    if (rerun_mode && selected_cases().count(key.str()) == 0)
    {
        return;
    }
    constexpr bool gpu = B == backend::cuda_blas;
    auto           a   = random_matrix<T>(static_cast<std::size_t>(m) * k, 42);
    auto           b   = random_matrix<T>(static_cast<std::size_t>(k) * n, 43);
    if constexpr (Op == operation::chol || Op == operation::symmetric_eigen)
    {
        // Fixed input makes the operation comparable across all providers.
        // NOLINTNEXTLINE(bugprone-random-generator-seed)
        std::mt19937                      rng(42);
        std::uniform_real_distribution<T> distribution(T(-1), T(1));
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                a[i * n + j] = a[j * n + i] = distribution(rng);
            }
        }
        if constexpr (Op == operation::chol)
        {
            for (int i = 0; i < n; ++i)
            {
                a[i * n + i] += T(2 * n);
            }
        }
    }
    if constexpr (Op == operation::lu || Op == operation::general_eigen)
    {
        for (int i = 0; i < n; ++i)
        {
            a[i * n + i] += T(2 * n);
        }
    }

    std::size_t    bytes = 0;
    constexpr auto lower = linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR;
    if constexpr (Op == operation::gemm)
    {
        bytes = linalg::multiplication<T, B>::workspace_size(m, n, k);
    }
    if constexpr (Op == operation::chol && gpu)
    {
        bytes = linalg::cholesky_decomposition<T, B>::workspace_size(n, lower);
    }
    if constexpr (Op == operation::lu)
    {
        bytes = linalg::lu_decomposition<T, B>::workspace_size(n);
    }
    if constexpr (Op == operation::qr)
    {
        bytes = linalg::qr_decomposition<T, B>::workspace_size(n, n);
    }
    if constexpr (Op == operation::svd)
    {
        bytes = linalg::svd_decomposition<T, B>::workspace_size(n, n);
    }
    if constexpr (Op == operation::symmetric_eigen)
    {
        bytes = linalg::symmetric_eigen_backend<T, B>::workspace_size(n);
    }
    if constexpr (Op == operation::general_eigen)
    {
        bytes = linalg::general_eigen_backend<T, B>::workspace_size(n, false);
    }

    std::vector<T> work(a.size()), result(static_cast<std::size_t>(m) * n), other(result.size()),
        values(n), imaginary(n);
    std::vector<linalg_int> pivot(n);
    std::vector<std::byte>  workspace(bytes);
    device_buffer<T> da(gpu ? a.size() : 0), db(gpu ? b.size() : 0), original(gpu ? a.size() : 0);
    device_buffer<T> dc(gpu ? result.size() : 0), dout(gpu ? result.size() : 0), ds(gpu ? n : 0);
    device_buffer<std::byte> dws(gpu ? bytes : 0);
    device_buffer<int>       dinfo(gpu ? 1 : 0), dpivot(gpu ? n : 0);
    stream_owner             stream;
    T*                       ap   = work.data();
    T*                       bp   = b.data();
    T*                       cp   = result.data();
    T*                       outp = other.data();
    T*                       sp   = values.data();
    void*                    wsp  = workspace.data();
    if constexpr (gpu)
    {
        cuda_check(
            cudaMemcpy(original.get(), a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(db.get(), b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
        ap   = da.get();
        bp   = db.get();
        cp   = dc.get();
        outp = dout.get();
        sp   = ds.get();
        wsp  = dws.get();
    }
    bool input_ready = false;
    auto reset       = [&]
    {
        if constexpr (gpu)
        {
            if (steady_gpu_mode && (Op == operation::gemm || Op == operation::svd) && input_ready)
            {
                return;
            }
            cuda_check(cudaMemcpyAsync(
                ap, original.get(), a.size() * sizeof(T), cudaMemcpyDeviceToDevice, stream.get()));
            stream.wait();
            input_ready = true;
        }
        else
        {
            work = a;
        }
    };
    auto invoke = [&]
    {
        bool ok = true;
        if constexpr (Op == operation::gemm)
        {
            linalg::multiplication<T, B>::apply(
                ta, tb, m, n, k, ap, ta ? m : k, bp, tb ? k : n, cp, n, wsp, bytes, stream.get());
        }
        if constexpr (Op == operation::chol)
        {
            if constexpr (gpu)
            {
                linalg::cholesky_decomposition<T, B>::apply(
                    ap, n, lower, wsp, bytes, dinfo.get(), stream.get());
            }
            else
            {
                ok = linalg::cholesky_decomposition<T, B>::apply(ap, n, lower);
            }
        }
        if constexpr (Op == operation::lu)
        {
            if constexpr (gpu)
            {
                linalg::lu_decomposition<T, B>::apply(
                    ap, n, dpivot.get(), wsp, bytes, dinfo.get(), stream.get());
            }
            else
            {
                ok = linalg::lu_decomposition<T, B>::apply(ap, n, pivot.data(), wsp, bytes);
            }
        }
        if constexpr (Op == operation::qr)
        {
            linalg::qr_decomposition<T, B>::apply_r(n, n, ap, n, sp, wsp, bytes, stream.get());
            linalg::qr_decomposition<T, B>::apply_q(n, n, ap, n, sp, wsp, bytes, stream.get());
        }
        if constexpr (Op == operation::svd)
        {
            linalg::svd_decomposition<T, B>::apply(
                n, n, ap, n, sp, cp, n, outp, n, wsp, bytes, stream.get());
        }
        if constexpr (Op == operation::symmetric_eigen)
        {
            if constexpr (gpu)
            {
                linalg::symmetric_eigen_backend<T, B>::apply(
                    ap, n, n, sp, n, wsp, bytes, dinfo.get(), stream.get());
            }
            else
            {
                ok = linalg::symmetric_eigen_backend<T, B>::apply(ap, n, n, sp, n, wsp, bytes);
            }
        }
        if constexpr (Op == operation::general_eigen)
        {
            ok = linalg::general_eigen_backend<T, B>::apply(
                ap, n, n, sp, imaginary.data(), false, n, wsp, bytes);
        }
        if (!ok)
        {
            throw std::runtime_error("Numerical operation failed");
        }
        if constexpr (gpu)
        {
            stream.wait();
        }
    };
    auto check_info = [&]
    {
        if constexpr (gpu && Op != operation::gemm)
        {
            int         info = -999;
            const void* p    = (Op == operation::qr || Op == operation::svd) ? wsp : dinfo.get();
            cuda_check(cudaMemcpy(&info, p, sizeof(info), cudaMemcpyDeviceToHost));
            if (info != 0)
            {
                throw std::runtime_error("cuSOLVER info=" + std::to_string(info));
            }
        }
    };
    reset();
    invoke();
    check_info();
    if (rerun_mode)
    {
        const auto warmup_start = clock_type::now();
        do
        {
            reset();
            invoke();
        } while (std::chrono::duration<double>(clock_type::now() - warmup_start).count() < 0.2);
        check_info();
    }
    std::vector<double> samples;
    std::size_t         total_iterations = 0;
    for (int repeat = 0; repeat < (rerun_mode ? 7 : 5); ++repeat)
    {
        double      elapsed    = 0;
        std::size_t iterations = 0;
        do
        {
            reset();
            auto start = clock_type::now();
            invoke();
            elapsed += std::chrono::duration<double>(clock_type::now() - start).count();
            ++iterations;
        } while (elapsed < (steady_gpu_mode ? 0.1
                               : rerun_mode ? 0.2
                                            : 0.05) &&
                 iterations < 500000);
        check_info();
        samples.push_back(elapsed * 1.e6 / static_cast<double>(iterations));
        total_iterations += iterations;
    }
    const auto mean     = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double     variance = 0;
    for (auto value : samples)
    {
        variance += (value - mean) * (value - mean);
    }
    const auto sd = std::sqrt(variance / (samples.size() - 1));
    std::sort(samples.begin(), samples.end());
    csv << operation_name(Op) << ',' << precision << ',' << backend_name(B) << ',' << m << ',' << n
        << ',' << k << ',' << ta << ',' << tb << ',' << mean << ',' << samples[samples.size() / 2]
        << ',' << sd << ',' << sd / mean * 100.0 << ',' << total_iterations << ',' << bytes << '\n'
        << std::flush;
    std::cout << operation_name(Op) << ' ' << precision << ' ' << backend_name(B) << ' ' << m << 'x'
              << n << 'x' << k << " ta=" << ta << " tb=" << tb << " mean_us=" << mean
              << " cv=" << sd / mean * 100.0 << '%' << '\n';
}

template <typename T, operation Op>
void all_backends(std::ofstream& csv,
    int                          m,
    int                          n,
    int                          k,
    bool                         ta          = false,
    bool                         tb          = false,
    bool                         skip_native = false)
{
    if (!skip_native)
    {
        run_case<T, backend::native, Op>(csv, m, n, k, ta, tb);
    }
    run_case<T, backend::blas_lapacke, Op>(csv, m, n, k, ta, tb);
    run_case<T, backend::mkl, Op>(csv, m, n, k, ta, tb);
    if constexpr (Op != operation::general_eigen)
    {
        run_case<T, backend::cuda_blas, Op>(csv, m, n, k, ta, tb);
    }
}

template <typename T> void suite(std::ofstream& csv)
{
    for (int n : {32, 64, 128, 256, 512, 1024, 2048})
    {
        all_backends<T, operation::gemm>(csv, n, n, n);
    }
    all_backends<T, operation::gemm>(csv, 64, 1024, 256);
    all_backends<T, operation::gemm>(csv, 1024, 64, 256);
    for (int n : {64, 256, 512, 1024})
    {
        for (auto flags : {std::pair<bool, bool>{false, true}, {true, false}, {true, true}})
        {
            all_backends<T, operation::gemm>(csv, n, n, n, flags.first, flags.second);
        }
    }
    for (int n : {16, 32, 64, 128, 256, 512, 1024})
    {
        all_backends<T, operation::chol>(csv, n, n, n);
        all_backends<T, operation::lu>(csv, n, n, n);
        all_backends<T, operation::qr>(csv, n, n, n);
    }
    for (int n : {16, 32, 64, 128, 256, 512, 1024})
    {
        all_backends<T, operation::svd>(csv, n, n, n, false, false, n > 128);
    }
    for (int n : {8, 16, 32, 64, 96, 128, 256, 512, 1024})
    {
        all_backends<T, operation::symmetric_eigen>(csv, n, n, n, false, false, n > 128);
    }
    for (int n : {8, 16, 32, 64, 96, 128})
    {
        all_backends<T, operation::general_eigen>(csv, n, n, n);
    }
}

int main(int argc, char** argv)
{
    try
    {
        if (argc > 2)
        {
            rerun_mode = true;
            std::ifstream selections(argv[2]);
            if (!selections)
            {
                throw std::runtime_error("Cannot open selection file");
            }
            std::string key;
            while (std::getline(selections, key))
            {
                if (!key.empty())
                {
                    selected_cases().insert(key);
                }
            }
        }
        steady_gpu_mode = argc > 3 && std::string(argv[3]) == "steady_gpu";
        std::ofstream csv(argc > 1 ? argv[1] : "results.csv");
        if (!csv)
        {
            throw std::runtime_error("Cannot open results CSV");
        }
        csv << std::setprecision(10);
        csv << "operation,precision,backend,m,n,k,transpose_a,transpose_b,mean_us,median_us,stdev_"
               "us,cv_percent,iterations,workspace_bytes\n";
        suite<float>(csv);
        suite<double>(csv);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "Benchmark failed: %s\n", error.what());
        return 1;
    }
    catch (...)
    {
        std::fputs("Benchmark failed with an unknown exception\n", stderr);
        return 1;
    }
}
