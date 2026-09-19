/*
 * LinearAlgebra — global backend context.
 *
 * PyTorch reference point: at::Context / at::globalContext() tracks which
 * backends (MKL, MKL-DNN, CUDA, ...) are compiled in and available at
 * runtime, and exposes toggles (at::globalContext().userEnabledMkldnn(),
 * torch.backends.cuda.*) that steer dispatch without the caller having to
 * pass a backend explicitly on every call. This Context plays the same
 * role for LinearAlgebra, scoped to the four backends in include/common/
 * backend.h: it is what include/common/dispatch_stub.h consults to turn
 * "cpu, no explicit backend requested" into an actual function pointer.
 */
#pragma once

#include <vector>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"

namespace linalg
{

class LINALG_API Context
{
public:
    // Compiled-in AND usable right now (for cuda: a device is actually present).
    bool has_mkl() const noexcept;
    bool has_blas() const noexcept;
    bool has_cuda() const noexcept;

    // CPU backend preference order used when no explicit backend is passed to
    // a dispatch_stub::resolve(device_type::cpu) call. Only backends that are
    // compiled in appear here; `scalar` is always present as the final entry.
    std::vector<backend> cpu_preference_order() const;

    // Pins a specific CPU backend as the sole preference (throws LINALG_THROW
    // if that backend was not compiled in). Passing `backend::scalar` is
    // always valid. This does not affect device_type::cuda dispatch, which
    // always resolves to the cublas/cusolver slot.
    void set_backend(backend b);

    // Clears any override set via set_backend(), reverting to the default
    // mkl > blas_lapack > scalar preference order.
    void clear_backend_override();

private:
    bool has_override_ = false;
    backend override_backend_ = backend::scalar;
};

LINALG_API Context& globalContext();

}  // namespace linalg
