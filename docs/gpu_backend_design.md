# GPU backend design

## Contract

GPU implementations live in `src/matrix_operation/gpu/cublas/`. CPU
implementations live in `src/matrix_operation/cpu/`, with `native/`,
`blas/` and `mkl/` subdirectories. CMake selects optional backend sources;
Bazel selects CPU sources only.

`linalg_backend::CUDA_BLAS` is the GPU backend. Its name covers cuBLAS,
cuSOLVER and custom CUDA kernels. `linalg_backend::NATIVE` is the CPU
implementation; there is no separate native CUDA backend.

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
| Frobenius norm | cuBLAS NRM2 in device pointer mode |
| One/infinity norm | CUDA row/column sums and device maximum reduction |
| Spectral norm, rank, condition number | cuSOLVER SVD plus device reductions |
| Pseudo-inverse | cuSOLVER SVD plus CUDA combine/threshold kernel |
| Least squares | GPU pseudo-inverse followed by cuBLAS GEMM |
| Batched square multiplication | Custom CUDA kernel |
| Trace | Custom CUDA diagonal sum |

“Pure cuBLAS” refers to computation expressed using cuBLAS primitives;
“native CUDA” refers to custom `__global__` kernels. Both execute on the
GPU. A `.cu` suffix indicates CUDA compilation, not which library performs
the mathematical operation. Operations may combine these approaches.

## Device result API

CPU signatures are unchanged. GPU specializations deliberately use device
output parameters instead of returning scalar values or a host `bool`.
Old GPU calls returning host values must migrate; there is no implicit
host-download compatibility wrapper.

- `rank::apply(..., workspace, bytes, linalg_long* result, stream)`
- `condition_number::apply(..., workspace, bytes, T* result, stream)`
- `norm::apply(..., type, workspace, bytes, T* result, stream)`
- `determinant::apply(..., type, workspace, bytes, T* result, stream)`
- `trace::apply(A, n, lda, T* result, stream)`
- `lu_decomposition::apply(A, n, int* pivot, workspace, bytes, int* info, stream)`
- `cholesky_decomposition::apply(A, n, triangle, workspace, bytes, int* info, stream)`
- `symmetric_eigen_backend::apply(..., workspace, bytes, int* info, stream)`

`stream` defaults to `nullptr` (the legacy default stream). Device pivot
entries are cuSOLVER's 1-based `int` row-interchange sequence, independent
of the CPU API's index width. LU returns row-major packed factors for
`P A = L U`. `info == 0` means numerical success; nonzero values use the
cuSOLVER routine's status convention. Cholesky exposes a
`workspace_size(n, triangle)` query and takes caller-owned scratch.

The other GPU operations retain device status in their workspace. SVD
places its status at the start of its workspace. SVD-derived rank returns
the maximum `linalg_long` value if SVD fails; condition number, spectral
norm and pseudo-inverse produce NaN on that failure. Determinant returns
zero on factorization failure. CUDA/cuBLAS/cuSOLVER API errors throw on
the host; numerical failure is not checked through a hidden download.

For example, scalar outputs can feed another GPU operation directly:

```cpp
using backend = linalg::linalg_backend;
// dA, dnorm and dtrace are caller-owned device pointers.
linalg::norm<double, backend::CUDA_BLAS>::apply(
    dA, rows, cols, cols, linalg::norm_type::FROBENIUS,
    nullptr, 0, dnorm, stream);
linalg::trace<double, backend::CUDA_BLAS>::apply(
    dnorm, 1, 1, dtrace, stream);
```

cuBLAS handles are cached per thread and device. Each call binds its
requested stream. Frobenius norm temporarily switches cuBLAS to device
pointer mode and restores the previous mode, so later GEMM/GEAM calls
continue accepting host launch coefficients.

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
`LinearAlgebraCxxTests --gtest_filter=MathGpu.*` directly. The CPU backend
loop intentionally uses host buffers; GPU tests allocate device data and
workspace separately. `MathGpu.DeviceMemoryAllocation` checks that both
allocations are device memory and uses them for a transpose round trip.
Allocation failures report the CUDA error. These small allocations are
released at the end of each test and may be too brief to see in a GPU monitor.

CUDA compilation and GPU execution require a CUDA-capable environment.
A successful CPU build does not validate CUDA kernels or runtime behavior.
