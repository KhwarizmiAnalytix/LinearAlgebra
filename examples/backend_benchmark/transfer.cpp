#include "include/matrix_operation/multiplication.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

// This file is written against CUDA's runtime API names; LINALG_ENABLE_CUBLAS
// and LINALG_ENABLE_ROCBLAS are mutually exclusive (CMakeLists.txt), and under
// the latter those names are aliased to their HIP equivalents here (HIP
// mirrors CUDA's runtime API 1:1 for everything this file uses), so the rest
// of the file is backend-agnostic.
#if LINALG_HAS_CUBLAS
#include <cuda_runtime_api.h>
constexpr auto GPU_BACKEND = linalg::linalg_backend::CUDA_BLAS;
#elif LINALG_HAS_ROCBLAS
#include <hip/hip_runtime_api.h>
using cudaError_t  = hipError_t;
using cudaStream_t = hipStream_t;
#define cudaSuccess hipSuccess
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemset hipMemset
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamDestroy hipStreamDestroy
#define cudaGetErrorString hipGetErrorString
constexpr auto GPU_BACKEND = linalg::linalg_backend::ROCM_BLAS;
#else
#error "transfer_analysis requires LINALG_ENABLE_CUBLAS or LINALG_ENABLE_ROCBLAS"
#endif

void check(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

template <typename T>
void run(std::ofstream& csv, int n, bool pinned, int uploads, bool compute = false)
{
    const std::size_t bytes = static_cast<std::size_t>(n) * n * sizeof(T);
    std::vector<T>    pageable(pinned ? 0 : static_cast<std::size_t>(n) * n * 3, T(1));
    T*                host   = pageable.data();
    T*                device = nullptr;
    cudaStream_t      stream = nullptr;
    if (pinned)
    {
        check(cudaMallocHost(reinterpret_cast<void**>(&host), bytes * 3));
    }
    std::unique_ptr<T, decltype(&cudaFreeHost)> pinned_host(pinned ? host : nullptr, cudaFreeHost);
    for (int operand = 0; operand < 2; ++operand)
    {
        std::mt19937                      rng(42 + operand);
        std::uniform_real_distribution<T> distribution(T(-1), T(1));
        for (int i = 0; i < n * n; ++i)
        {
            host[operand * n * n + i] = distribution(rng);
        }
    }
    check(cudaMalloc(reinterpret_cast<void**>(&device), bytes * 3));
    std::unique_ptr<T, decltype(&cudaFree)> device_owner(device, cudaFree);
    check(cudaMemset(device, 0, bytes * 3));
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> stream_owner(
        stream, cudaStreamDestroy);
    auto copy = [&]
    {
        check(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream));
        if (uploads == 2)
        {
            check(cudaMemcpyAsync(
                device + n * n, host + n * n, bytes, cudaMemcpyHostToDevice, stream));
        }
        if (compute)
        {
            linalg::multiplication<T, GPU_BACKEND>::apply(false,
                false,
                n,
                n,
                n,
                device,
                n,
                device + n * n,
                n,
                device + n * n * 2,
                n,
                nullptr,
                0,
                stream);
        }
        check(cudaMemcpyAsync(
            host + n * n * 2, device + n * n * 2, bytes, cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
    };
    copy();
    double      total      = 0;
    std::size_t iterations = 0;
    for (int repeat = 0; repeat < 5; ++repeat)
    {
        auto        begin = std::chrono::steady_clock::now();
        std::size_t count = 0;
        do
        {
            copy();
            ++count;
        } while (
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() < 0.05);
        total += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        iterations += count;
    }
    csv << (compute ? "gemm_end_to_end" : "transfer_only") << ','
        << (std::is_same_v<T, float> ? "float" : "double") << ',' << n << ','
        << (pinned ? "pinned" : "pageable") << ',' << uploads << ',' << total * 1.e6 / iterations
        << '\n'
        << std::flush;
}

int main(int argc, char** argv)
{
    try
    {
        std::ofstream csv(argc > 1 ? argv[1] : "transfers.csv");
        if (!csv)
        {
            throw std::runtime_error("Cannot open transfer CSV");
        }
        csv << "measurement,precision,n,host_memory,uploads,mean_us\n";
        for (int n : {16, 32, 64, 128, 256, 512, 1024, 2048})
        {
            for (bool pinned : {false, true})
            {
                for (int uploads : {1, 2})
                {
                    run<float>(csv, n, pinned, uploads);
                    run<double>(csv, n, pinned, uploads);
                    if (uploads == 2)
                    {
                        run<float>(csv, n, pinned, uploads, true);
                        run<double>(csv, n, pinned, uploads, true);
                    }
                }
            }
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "Transfer benchmark failed: %s\n", error.what());
        return 1;
    }
    catch (...)
    {
        std::fputs("Transfer benchmark failed with an unknown exception\n", stderr);
        return 1;
    }
}
