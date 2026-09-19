#pragma once

namespace linalg
{

// Where the raw pointers passed to a matrix_operation entry point live.
// cpu: ordinary host memory (the only option today).
// cuda: a CUDA device pointer (cudaMalloc'd) — only meaningful when the
// library was built with LINALG_ENABLE_CUBLAS.
//
// This cannot be inferred from a raw pointer, unlike backend choice among
// CPU implementations, so every public op takes it explicitly.
enum class device_type
{
    cpu,
    cuda
};

}  // namespace linalg
