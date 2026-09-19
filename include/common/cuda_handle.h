/*
 * LinearAlgebra — shared cuBLAS/cuSOLVER handle accessors.
 *
 * One handle per thread per library, created lazily (cuBLAS/cuSOLVER handles
 * are not safe to share across concurrently-executing threads, and creating
 * one is comparatively expensive). These are `inline` functions so every
 * *_cublas.cxx translation unit that includes this header shares the same
 * thread_local instance (guaranteed by the One Definition Rule for inline
 * functions), rather than each getting its own handle.
 */
#pragma once

#ifdef LINALG_ENABLE_CUBLAS

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "include/util/exception.h"

namespace linalg
{
namespace detail
{

inline cublasHandle_t cublas_handle()
{
    thread_local cublasHandle_t handle = [] {
        cublasHandle_t h{};
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
        {
            LINALG_THROW("cublasCreate failed");
        }
        return h;
    }();
    return handle;
}

inline cusolverDnHandle_t cusolver_handle()
{
    thread_local cusolverDnHandle_t handle = [] {
        cusolverDnHandle_t h{};
        if (cusolverDnCreate(&h) != CUSOLVER_STATUS_SUCCESS)
        {
            LINALG_THROW("cusolverDnCreate failed");
        }
        return h;
    }();
    return handle;
}

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
