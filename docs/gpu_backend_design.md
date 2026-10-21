# GPU backend design

## Contract

GPU implementations live in `src/matrix_operation/gpu/cublas/` (NVIDIA) and
`src/matrix_operation/gpu/hipblas/` (AMD). CPU implementations live in
`src/matrix_operation/cpu/`, with `native/`, `blas/` and `mkl/`
subdirectories. CMake selects optional backend sources; Bazel selects CPU
sources only.

`linalg_backend::CUDA_BLAS` and `linalg_backend::ROCM_BLAS` are the two GPU
backends. `CUDA_BLAS`'s name covers cuBLAS, cuSOLVER and custom CUDA
kernels; `ROCM_BLAS`'s name covers hipBLAS, hipSOLVER and the same custom
kernels ported to HIP (hipBLAS/hipSOLVER mirror cuBLAS/cuSOLVER's API
closely enough that every op is a 1:1 port — see `backend_roadmap.md` §4
and §6). `LINALG_ENABLE_CUBLAS` and `LINALG_ENABLE_ROCBLAS` are mutually
exclusive (CMakeLists.txt): `multiplication_batched`'s free functions are
not backend-templated, so only one GPU vendor backend may be compiled into
a given binary. `linalg_backend::NATIVE` is the CPU implementation; there
is no separate native CUDA/HIP backend.

All matrix data, intermediate values, scalar results, pivots, numerical
status and scratch buffers stay in device memory. GPU entry points do not
copy data to or from host memory, explicitly synchronize streams, or
allocate/free scratch. The caller allocates buffers, queries required
workspace bytes, and keeps buffers alive until the requested stream has
completed. Device-to-device copies for layout conversion are allowed.

Dimensions, enums, tolerance arguments, stream handles and workspace byte
counts are ordinary host-side launch metadata. CPU-side API dispatch and
CUDA/cuBLAS/cuSOLVER status checks still occur: this is a host-launched
CUDA library. The residency guarantee concerns application data and
numerical results, not eliminating the host's role in launching work.
CUDA libraries may perform their own initialization or synchronization;
the library does not promise that every vendor API is nonblocking or
CUDA-graph-capture compatible.

## Implementations

| Operation | Computation |
|---|---|
| Matrix multiplication | cuBLAS GEMM |
| Transpose | cuBLAS GEAM plus device-to-device copy |
| LU, Cholesky, linear solve | cuSOLVER, with device layout conversion where needed |
| SVD, symmetric eigen | cuSOLVER, with device layout conversion |
| QR | cuSOLVER plus a CUDA kernel to clear the lower triangle of R |
| Inversion | cuSOLVER factor/solve, CUDA identity fill, device layout conversion |
| Determinant | cuSOLVER factorization plus CUDA diagonal/pivot reduction |
| Frobenius norm | cuBLAS NRM2 in host pointer mode (synchronizes internally) |
| One/infinity norm | CUDA row/column sums and device maximum reduction, synced back to host |
| Spectral norm, rank, condition number | cuSOLVER SVD plus device reductions, synced back to host |
| Pseudo-inverse | cuSOLVER SVD plus CUDA combine/threshold kernel |
| Least squares | GPU pseudo-inverse followed by cuBLAS GEMM |
| Batched square multiplication | Custom CUDA kernel |
| Trace | Strided device-to-host copy of the diagonal, summed on host |

“Pure cuBLAS” refers to computation expressed using cuBLAS primitives;
“native CUDA” refers to custom `__global__` kernels. Both execute on the
GPU. A `.cu` suffix indicates CUDA compilation, not which library performs
the mathematical operation. Operations may combine these approaches. The
table above applies identically to `ROCM_BLAS`, substituting hipBLAS for
cuBLAS, hipSOLVER for cuSOLVER, and a `.hip` suffix for `.cu`.

## Device result API

Scalar-returning operations use the *same* signature on every backend —
GPU specializations are full specializations of the primary
(`Backend`-templated) struct's `apply`/`workspace_size`, not a
separately-declared struct with a different interface. There is no
`rank<T, CUDA_BLAS>` (etc.) type in the header; the GPU implementation
files (`src/matrix_operation/gpu/{cublas,hipblas}/*`) just define
`template <> linalg_long rank<float, linalg_backend::CUDA_BLAS>::apply(...)`
directly against `rank<T, Backend>`.

- `rank::apply(A, rows, columns, lda, tol, workspace, bytes, stream) -> linalg_long`
- `condition_number::apply(A, rows, columns, lda, workspace, bytes, stream) -> T`
- `norm::apply(A, rows, columns, lda, type, workspace, bytes, stream) -> T`
- `determinant::apply(m, pivot, lda, type, workspace, bytes, stream) -> T`
- `trace::apply(A, n, lda, stream) -> T`
- `symmetric_eigen_backend::apply(eigenvectors, n, lda, eigenvalues, ldv, workspace, bytes, stream) -> bool`
- `lu_decomposition::apply(m, lda, pivot, workspace, bytes, stream) -> bool`

Each of these synchronizes the bound stream internally before returning,
so the call blocks until the result is known on the host — unlike the
other GPU operations below, which stay fully async. `stream` defaults to
`nullptr` (the legacy default stream). SVD-derived rank returns the
maximum `linalg_long` value if SVD fails; condition number and spectral
norm produce NaN on that failure. Determinant returns zero on
factorization failure. CUDA/cuBLAS/cuSOLVER API errors throw on the host.

`lu_decomposition`'s device `pivot` is `linalg_int*`, matching the CPU
API's index width exactly (unlike the other scalar ops here, this one
carries an output array rather than a single value). cuSOLVER/hipSOLVER's
own pivot and info are 32-bit `int`, so the GPU implementation widens the
pivot and checks the info host-side during the same synchronizing round
trip, then writes the widened pivot back to the caller's device buffer.
It returns row-major packed factors for `P A = L U`; `false` means
numerical failure (a singular factor), matching the CPU backends'
`bool` convention.

`cholesky_decomposition` and `inversion` are the exceptions that keep
device output parameters and stay asynchronous, since their result is
matrix-shaped rather than a single scalar or pivot vector cheap enough to
justify a blocking round trip:

- `cholesky_decomposition::apply(A, n, triangle, workspace, bytes, int* info, stream)`

`info == 0` means numerical success; nonzero values use the cuSOLVER
routine's status convention. Cholesky exposes a `workspace_size(n,
triangle)` query and takes caller-owned scratch.

Because the scalar-returning operations above block on their own stream
sync, chaining their outputs into another GPU call now goes through a
host round trip (re-upload the returned value) rather than staying
device-resident:

```cpp
using backend = linalg::linalg_backend;
const double norm_value = linalg::norm<double, backend::CUDA_BLAS>::apply(
    dA, rows, cols, cols, linalg::norm_type::FROBENIUS, nullptr, 0, stream);
// Re-upload to feed it into the next device-resident step, if needed.
cudaMemcpyAsync(dnorm, &norm_value, sizeof(double), cudaMemcpyHostToDevice, stream);
```

cuBLAS/hipBLAS handles are cached per thread and device. Each call binds
its requested stream. Frobenius norm relies on cuBLAS/hipBLAS's default
host pointer mode, under which NRM2 synchronizes internally and writes
straight to a host scalar.

## Restrictions and remaining work

- Float and double only; real symmetric eigen decomposition only.
- QR/SVD and their composed operations require tightly packed strides.
- Existing inversion/solve UPFRONT modes currently refactor on GPU.
- Rank/condition/pseudo-inverse compute economy singular vectors even
  when only singular values are needed; this is a performance opportunity.
- Small scalar reductions use serial device loops. Large reductions and
  the custom batched multiplication kernel are candidates for tuning.
- Multi-RHS linear solves, batched factorizations and measured selection
  of cuBLAS strided-batched GEMM remain future work.

## Verification

`Testing/Cxx/CheckGpuResidency.cmake` runs in CPU-only CTest configurations
too. It rejects host-directed/default-direction copies, host vector/matrix
accessors, explicit synchronization and managed-memory migration in GPU
sources, and GPU implementation code in the CPU tree. This source check
is a regression guard, not proof of vendor-library runtime behavior.

GPU tests explicitly upload inputs and download final observations at the
test boundary. Tests cover device scalar pipelines on non-default
streams, rank/condition/spectral norm/pseudo-inverse/determinant outputs,
cuBLAS pointer-mode restoration, LU pivot reconstruction and device
factorization failure status, alongside existing numerical tests.

Enable `LINALG_ENABLE_CUBLAS=ON` together with `LINALG_ENABLE_TESTING=ON`
to compile and register `LinearAlgebraCudaTests`. Run them with
`ctest --test-dir <build> -C Release -L CUDA --output-on-failure`, or run
`LinearAlgebraCxxTests --gtest_filter=MathGpu.*` directly. Enable
`LINALG_ENABLE_ROCBLAS=ON` instead (the two are mutually exclusive) to
compile and register the same `MathGpu.*` suite as `LinearAlgebraRocmTests`
against `ROCM_BLAS`; run it with `ctest --test-dir <build> -C Release -L
ROCM --output-on-failure`. `TestGpuMatrixOperations.cxx` is written against
CUDA's runtime API names and aliases them to HIP's under
`LINALG_HAS_ROCBLAS` (HIP mirrors CUDA's runtime API 1:1 for everything
that file uses), so the same test bodies exercise either backend. The CPU
backend loop intentionally uses host buffers; GPU tests allocate device
data and workspace separately. `MathGpu.DeviceMemoryAllocation` checks
that both allocations are device memory and uses them for a transpose
round trip. Allocation failures report the GPU error. These small
allocations are released at the end of each test and may be too brief to
see in a GPU monitor.

CUDA/HIP compilation and GPU execution require a CUDA- or ROCm-capable
environment respectively. A successful CPU build does not validate CUDA or
HIP kernels or runtime behavior.
