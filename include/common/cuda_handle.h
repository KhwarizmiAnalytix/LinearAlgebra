/*
 * LinearAlgebra — shared cuBLAS/cuSOLVER handle accessors.
 *
 * One handle per thread per device per library, created lazily (cuBLAS/
 * cuSOLVER handles are not safe to share across concurrently-executing
 * threads, and creating one is comparatively expensive). Keyed by the
 * currently active CUDA device (cudaGetDevice), so a thread that switches
 * devices between two library calls (cudaSetDevice) gets that device's own
 * handle instead of silently reusing whichever device's handle happened to
 * be created first.
 */
#pragma once

#ifdef LINALG_ENABLE_CUBLAS

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <unordered_map>

#include "ThirdParty/Logging/include/logging.h"

namespace linalg
{
namespace detail
{

inline int current_cuda_device()
{
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        LOGGING_THROW("cudaGetDevice failed");
    }
    return device;
}

inline cublasHandle_t cublas_handle_for_current_device()
{
    thread_local std::unordered_map<int, cublasHandle_t> handles;
    const int                                            device = current_cuda_device();
    auto                                                 it     = handles.find(device);
    if (it != handles.end())
    {
        return it->second;
    }
    cublasHandle_t handle{};
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        LOGGING_THROW("cublasCreate failed");
    }
    handles.emplace(device, handle);
    return handle;
}

inline cusolverDnHandle_t cusolver_handle_for_current_device()
{
    thread_local std::unordered_map<int, cusolverDnHandle_t> handles;
    const int                                                device = current_cuda_device();
    auto                                                     it     = handles.find(device);
    if (it != handles.end())
    {
        return it->second;
    }
    cusolverDnHandle_t handle{};
    if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS)
    {
        LOGGING_THROW("cusolverDnCreate failed");
    }
    handles.emplace(device, handle);
    return handle;
}

// Binds `stream` (nullptr = the legacy default stream) to `handle` before
// the caller issues work on it. Every linalg::gpu::* entry point takes an
// explicit trailing cudaStream_t parameter and routes it through here.
inline void set_stream(cublasHandle_t handle, cudaStream_t stream)
{
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS)
    {
        LOGGING_THROW("cublasSetStream failed");
    }
}

inline void set_stream(cusolverDnHandle_t handle, cudaStream_t stream)
{
    if (cusolverDnSetStream(handle, stream) != CUSOLVER_STATUS_SUCCESS)
    {
        LOGGING_THROW("cusolverDnSetStream failed");
    }
}

// Synchronizes `stream` and frees `ptr` with plain cudaFree. Plain cudaFree
// is not stream-ordered against work still in flight on a non-default
// stream (that is exactly what cudaFreeAsync / a stream-ordered allocator
// exist to fix — out of scope here, see docs/gpu_backend_design.md §2/§9);
// every op that allocates its own workspace against a caller-supplied
// stream uses this instead of a bare cudaFree so a kernel still reading
// `ptr` is never raced. `sync_status`, if non-null, receives
// cudaStreamSynchronize's own result so the caller can report it without
// masking an earlier, more specific failure.
inline void synchronize_and_free(void* ptr, cudaStream_t stream, cudaError_t* sync_status = nullptr)
{
    const auto status = cudaStreamSynchronize(stream);
    if (sync_status != nullptr)
    {
        *sync_status = status;
    }
    cudaFree(ptr);
}

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
