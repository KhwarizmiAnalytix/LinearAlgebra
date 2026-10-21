/*
 * LinearAlgebra — shared hipBLAS/hipSOLVER handle accessors.
 *
 * One handle per thread per device per library, created lazily (hipBLAS/
 * hipSOLVER handles are not safe to share across concurrently-executing
 * threads, and creating one is comparatively expensive). Keyed by the
 * currently active HIP device (hipGetDevice), so a thread that switches
 * devices between two library calls (hipSetDevice) gets that device's own
 * handle instead of silently reusing whichever device's handle happened to
 * be created first. Direct port of cuda_handle.h; see that file for the
 * CUDA_BLAS counterpart.
 */
#pragma once

#if LINALG_HAS_ROCBLAS

#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

#include <unordered_map>

#include "include/common/linalg_checks.h"
#include "include/common/types.h"

namespace linalg
{
namespace detail
{

inline int current_hip_device()
{
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess)
    {
        LINALG_THROW("hipGetDevice failed");
    }
    return device;
}

inline hipblasHandle_t hipblas_handle_for_current_device()
{
    thread_local std::unordered_map<int, hipblasHandle_t> handles;
    const int                                             device = current_hip_device();
    auto                                                  it     = handles.find(device);
    if (it != handles.end())
    {
        return it->second;
    }
    hipblasHandle_t handle{};
    if (hipblasCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
    {
        LINALG_THROW("hipblasCreate failed");
    }
    handles.emplace(device, handle);
    return handle;
}

inline hipsolverDnHandle_t hipsolver_handle_for_current_device()
{
    thread_local std::unordered_map<int, hipsolverDnHandle_t> handles;
    const int                                                 device = current_hip_device();
    auto                                                      it     = handles.find(device);
    if (it != handles.end())
    {
        return it->second;
    }
    hipsolverDnHandle_t handle{};
    if (hipsolverDnCreate(&handle) != HIPSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("hipsolverDnCreate failed");
    }
    handles.emplace(device, handle);
    return handle;
}

// Binds `stream` (nullptr = the legacy default stream) to `handle` before
// the caller issues work on it. Every ROCM_BLAS struct apply() overload takes
// an explicit trailing stream parameter and routes it through here.
inline void set_stream(hipblasHandle_t handle, hipStream_t stream)
{
    if (hipblasSetStream(handle, stream) != HIPBLAS_STATUS_SUCCESS)
    {
        LINALG_THROW("hipblasSetStream failed");
    }
}

inline void set_stream(hipsolverDnHandle_t handle, hipStream_t stream)
{
    if (hipsolverDnSetStream(handle, stream) != HIPSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("hipsolverDnSetStream failed");
    }
}

// Scoped device-result mode; restore the previous mode for later BLAS calls.
class device_pointer_mode
{
public:
    explicit device_pointer_mode(hipblasHandle_t handle) : handle_(handle)
    {
        LINALG_CHECK(hipblasGetPointerMode(handle_, &previous_) == HIPBLAS_STATUS_SUCCESS,
            "hipblasGetPointerMode failed");
        LINALG_CHECK(
            hipblasSetPointerMode(handle_, HIPBLAS_POINTER_MODE_DEVICE) == HIPBLAS_STATUS_SUCCESS,
            "hipblasSetPointerMode failed");
    }
    ~device_pointer_mode() { hipblasSetPointerMode(handle_, previous_); }
    device_pointer_mode(const device_pointer_mode&)            = delete;
    device_pointer_mode& operator=(const device_pointer_mode&) = delete;

private:
    hipblasHandle_t      handle_;
    hipblasPointerMode_t previous_{};
};

// Forward declarations for the device-side in-place transpose helper defined in
// src/matrix_operation/gpu/hipblas/transpose_hip.cpp. Every GPU translation unit
// that needs detail::transpose_device includes this header (already a dependency
// for the hipBLAS/hipSOLVER handle accessors above), so this is the natural single
// declaration point — no extra header is required.
void transpose_device(linalg_long rows, linalg_long columns,
    float* m, float* scratch, hipStream_t stream);
void transpose_device(linalg_long rows, linalg_long columns,
    double* m, double* scratch, hipStream_t stream);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_HAS_ROCBLAS
