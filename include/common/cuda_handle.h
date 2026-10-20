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

#if LINALG_HAS_CUBLAS

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <unordered_map>

#include "include/common/linalg_checks.h"
#include "include/common/types.h"

namespace linalg
{
namespace detail
{

inline int current_cuda_device()
{
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        LINALG_THROW("cudaGetDevice failed");
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
        LINALG_THROW("cublasCreate failed");
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
        LINALG_THROW("cusolverDnCreate failed");
    }
    handles.emplace(device, handle);
    return handle;
}

// Binds `stream` (nullptr = the legacy default stream) to `handle` before
// the caller issues work on it. Every CUDA_BLAS struct apply() overload takes
// an explicit trailing stream parameter and routes it through here.
inline void set_stream(cublasHandle_t handle, cudaStream_t stream)
{
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS)
    {
        LINALG_THROW("cublasSetStream failed");
    }
}

inline void set_stream(cusolverDnHandle_t handle, cudaStream_t stream)
{
    if (cusolverDnSetStream(handle, stream) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnSetStream failed");
    }
}

// Scoped device-result mode; restore the previous mode for later BLAS calls.
class device_pointer_mode
{
public:
    explicit device_pointer_mode(cublasHandle_t handle) : handle_(handle)
    {
        LINALG_CHECK(cublasGetPointerMode(handle_, &previous_) == CUBLAS_STATUS_SUCCESS,
            "cublasGetPointerMode failed");
        LINALG_CHECK(
            cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE) == CUBLAS_STATUS_SUCCESS,
            "cublasSetPointerMode failed");
    }
    ~device_pointer_mode() { cublasSetPointerMode(handle_, previous_); }
    device_pointer_mode(const device_pointer_mode&)            = delete;
    device_pointer_mode& operator=(const device_pointer_mode&) = delete;

private:
    cublasHandle_t      handle_;
    cublasPointerMode_t previous_{};
};

// Forward declarations for the device-side in-place transpose helper defined in
// src/matrix_operation/gpu/cublas/transpose_cublas.cpp. Every GPU translation unit
// that needs detail::transpose_device includes this header (already a dependency
// for the cuBLAS/cuSOLVER handle accessors above), so this is the natural single
// declaration point — no extra header is required.
void transpose_device(linalg_long rows, linalg_long columns,
    float* m, float* scratch, cudaStream_t stream);
void transpose_device(linalg_long rows, linalg_long columns,
    double* m, double* scratch, cudaStream_t stream);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
