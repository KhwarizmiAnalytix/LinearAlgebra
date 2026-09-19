/*
 * LinearAlgebra — per-op backend registry.
 *
 * PyTorch reference point: aten/src/ATen/native/DispatchStub.h. ATen's own
 * CPU/CUDA kernel dispatch for library-style ops (elementwise, reductions,
 * linear algebra helpers) goes through DispatchStub rather than the full
 * Tensor/DispatchKeySet machinery: a small function-pointer table per op,
 * populated by REGISTER_DISPATCH calls in each backend's translation unit,
 * resolved lazily. This header is that pattern, adapted to a library with
 * raw pointers instead of Tensor (so device_type is passed explicitly,
 * see include/common/device.h) and a fixed, small backend set (see
 * include/common/backend.h) instead of ATen's open DispatchKey registry.
 */
#pragma once

#include <atomic>

#include "include/common/backend.h"
#include "include/common/context.h"
#include "include/common/device.h"
#include "include/util/exception.h"

namespace linalg
{

template <typename FnPtr>
struct dispatch_stub
{
    std::atomic<FnPtr> scalar_fn{nullptr};
    std::atomic<FnPtr> blas_fn{nullptr};
    std::atomic<FnPtr> mkl_fn{nullptr};
    std::atomic<FnPtr> cuda_fn{nullptr};

    FnPtr slot_for(backend b) const noexcept
    {
        switch (b)
        {
            case backend::scalar:
                return scalar_fn.load(std::memory_order_relaxed);
            case backend::blas_lapack:
                return blas_fn.load(std::memory_order_relaxed);
            case backend::mkl:
                return mkl_fn.load(std::memory_order_relaxed);
            case backend::cublas:
                return cuda_fn.load(std::memory_order_relaxed);
        }
        return nullptr;
    }

    // Picks the implementation to call for `device`, honoring the process-wide
    // Context preference order for CPU backends. Throws if nothing usable is
    // registered (e.g. device_type::cuda requested but the library was built
    // without LINALG_ENABLE_CUBLAS, or with it but no GPU is present).
    FnPtr resolve(device_type device) const
    {
        if (device == device_type::cuda)
        {
            if (FnPtr fn = cuda_fn.load(std::memory_order_relaxed))
            {
                return fn;
            }
            LINALG_THROW(
                "device_type::cuda requested but no cuBLAS/cuSOLVER implementation is "
                "registered for this operation (build with LINALG_ENABLE_CUBLAS and ensure "
                "a CUDA device is available)");
        }

        for (backend b : globalContext().cpu_preference_order())
        {
            if (FnPtr fn = slot_for(b))
            {
                return fn;
            }
        }
        LINALG_THROW("no CPU implementation registered for this operation");
    }
};

}  // namespace linalg

// Declares `extern linalg::dispatch_stub<fn_type> name;` — put in the op's
// internal dispatch header, included by the dispatcher .cxx and every
// backend .cxx/.cu that registers into it.
#define LINALG_DECLARE_DISPATCH(fn_type, name) \
    extern LINALG_API ::linalg::dispatch_stub<fn_type> name

// Defines the stub storage — exactly once, in the op's dispatcher .cxx.
#define LINALG_DEFINE_DISPATCH(fn_type, name) ::linalg::dispatch_stub<fn_type> name

// Registers `impl` into `name`'s `slot` (one of scalar/blas/mkl/cuda) at
// static-init time. Used once per backend .cxx/.cu that implements the op.
#define LINALG_REGISTER_DISPATCH(name, slot, impl)                                       \
    namespace                                                                            \
    {                                                                                     \
    struct name##_##slot##_registrar                                                     \
    {                                                                                     \
        name##_##slot##_registrar() { name.slot##_fn.store(impl, std::memory_order_relaxed); } \
    };                                                                                    \
    static name##_##slot##_registrar name##_##slot##_registrar_instance;                 \
    }
