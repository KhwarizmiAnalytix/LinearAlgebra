**Three CPU backends compared with CUDA — 20 October 2026**

Fresh measurements using the existing [benchmark.cpp](../../benchmark.cpp) and [transfer.cpp](../../transfer.cpp), built against library commit `ccbdc42e8357e555eef8f3e1496943a73bcdd9ee`. All four backends run in the same executable. [Comparison CSV](comparison.csv) · [Final per-backend samples](final_results.csv) · [Environment and quality](metadata.json) · [Validation log](tests.log).

**Host and policy**

| Setting | Value |
| --- | --- |
| cpu | Intel(R) Core(TM) i9-14900K |
| logical processors | 32 |
| gpu | NVIDIA GeForce RTX 4060 Ti, 591.86, 8188 MiB, WDDM |
| os | Windows-11-10.0.26200-SP0 |
| compiler | Clang 22.1.2; C++17; Release -O3 -DNDEBUG; AVX2/FMA; LTO off |
| cuda compiler | nvcc 13.2.78; MSVC host 19.44.35229.0 |
| openblas | OpenBLAS 0.3.34 DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64 |
| mkl | Intel(R) oneAPI Math Kernel Library Version 2026.0-Product Build 20260401 for Intel(R) 64 architecture applications |
| mkl interface | ILP64 |

CPU vendor thread counts are both 1 (`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`); the benchmark process and its children use affinity mask `0x1`. Native code uses AVX2/FMA. This measures one CPU thread against the RTX 4060 Ti, and does not measure the CPU’s full multicore capacity. Native LU pivoting is OFF; OpenBLAS, MKL and CUDA LU use pivoted factorization. There are no locked CPU/GPU clocks; Windows uses WDDM.

**Large matrices: n=1024, mean milliseconds**

**float**

| Operation | Native | OpenBLAS | MKL | CUDA | CUDA speedup / best CPU |
| --- | --- | --- | --- | --- | --- |
| GEMM | 16.037 | 12.786 | 13.870 | 0.199 | 64.40x |
| Cholesky | 12.915 | 6.296 | 3.734 | 0.836 | 4.47x |
| LU | 14.685 | 12.224 | 6.845 | 1.851 | 3.70x |
| QR: factor + explicit Q | 104.155 | 45.827 | 28.368 | 4.038 | 7.03x |
| SVD | Not sampled | 2,939.965 | 817.590 | 111.611 | 7.33x |
| Symmetric eigen | Not sampled | 67.451 | 54.057 | 10.681 | 5.06x |

**double**

| Operation | Native | OpenBLAS | MKL | CUDA | CUDA speedup / best CPU |
| --- | --- | --- | --- | --- | --- |
| GEMM | 33.647 | 25.873 | 25.855 | 6.872 | 3.76x |
| Cholesky | 34.464 | 11.179 | 8.274 | 3.321 | 2.49x |
| LU | 25.301 | 17.826 | 14.578 | 5.447 | 2.68x |
| QR: factor + explicit Q | 221.388 | 73.190 | 56.037 | 23.145 | 2.42x |
| SVD | Not sampled | 3,744.976 | 475.614 | 281.817 | 1.69x |
| Symmetric eigen | Not sampled | 117.058 | 106.937 | 42.040 | 2.54x |

A speedup above 1 means CUDA was faster. Native SVD and symmetric eigen stop at n=128 because the existing suite caps their expensive native paths. Missing rows are not zero times. QR measures in-place factorization followed by construction of explicit Q; R is overwritten and is not separately extracted. SVD includes U/S/VT; symmetric eigen includes vectors.

**Smaller matrices: n=128, mean microseconds**

**float**

| Operation | Native | OpenBLAS | MKL | CUDA | CUDA speedup / best CPU |
| --- | --- | --- | --- | --- | --- |
| GEMM | 37.151 | 32.013 | 27.662 | 12.101 | 2.29x |
| Cholesky | 43.350 | 27.254 | 36.103 | 80.656 | 0.34x |
| LU | 36.513 | 43.771 | 30.049 | 152.988 | 0.20x |
| QR: factor + explicit Q | 236.333 | 253.507 | 131.248 | 732.123 | 0.18x |
| SVD | 5,215.180 | 2,985.525 | 1,441.195 | 10,030.855 | 0.14x |
| Symmetric eigen | 24,030.107 | 524.048 | 379.971 | 1,794.176 | 0.21x |
| General eigenvalues | 20,532.407 | 1,500.248 | 1,362.772 | Unsupported | Unsupported |

**double**

| Operation | Native | OpenBLAS | MKL | CUDA | CUDA speedup / best CPU |
| --- | --- | --- | --- | --- | --- |
| GEMM | 81.477 | 57.023 | 51.523 | 36.824 | 1.40x |
| Cholesky | 53.166 | 46.298 | 31.730 | 173.152 | 0.18x |
| LU | 54.351 | 65.594 | 43.754 | 200.432 | 0.22x |
| QR: factor + explicit Q | 351.383 | 369.582 | 225.170 | 1,489.605 | 0.15x |
| SVD | 7,990.409 | 4,956.098 | 2,464.875 | 12,669.672 | 0.19x |
| Symmetric eigen | 44,987.020 | 730.696 | 563.487 | 4,297.299 | 0.13x |
| General eigenvalues | 39,734.640 | 2,393.755 | 2,210.093 | Unsupported | Unsupported |

**Observed GPU crossover**

| Operation | First sustained CUDA win, float n | First sustained CUDA win, double n |
| --- | --- | --- |
| GEMM | 128 | 128 |
| Cholesky | 512 | 512 |
| LU | 512 | 512 |
| QR: factor + explicit Q | 512 | 512 |
| SVD | 512 | 1024 |
| Symmetric eigen | 256 | 512 |

These are sampled crossovers for resident square matrices: CUDA first wins and stays ahead at all larger sampled sizes. They are not exact size thresholds or transfer-inclusive decomposition results. General nonsymmetric eigenvalues have no CUDA implementation in this library.

**Transfers: directly measured upload + GEMM + download, microseconds**

| Precision | n | Fastest CPU | CPU us | Resident CUDA us | CUDA E2E pageable us | CUDA E2E pinned us |
| --- | --- | --- | --- | --- | --- | --- |
| float | 32 | mkl | 0.498 | 14.531 | 37.513 | 35.459 |
| float | 64 | mkl | 3.188 | 13.980 | 39.511 | 44.069 |
| float | 128 | mkl | 27.662 | 12.101 | 55.605 | 65.049 |
| float | 256 | mkl | 204.694 | 13.875 | 126.920 | 104.614 |
| float | 512 | mkl | 1,615.619 | 33.950 | 408.506 | 315.471 |
| float | 1024 | openblas | 12,786.485 | 198.553 | 1,353.050 | 1,216.760 |
| float | 2048 | mkl | 100,304.440 | 1,451.064 | 5,612.610 | 5,293.130 |
| double | 32 | mkl | 0.909 | 15.753 | 41.978 | 44.061 |
| double | 64 | mkl | 6.238 | 23.074 | 54.617 | 71.022 |
| double | 128 | mkl | 51.523 | 36.824 | 107.698 | 104.504 |
| double | 256 | mkl | 410.005 | 117.928 | 340.990 | 279.451 |
| double | 512 | mkl | 3,226.330 | 872.058 | 1,516.770 | 1,402.290 |
| double | 1024 | mkl | 25,854.520 | 6,872.354 | 9,121.650 | 8,900.580 |
| double | 2048 | mkl | 206,365.580 | 54,747.836 | 63,438.700 | 62,624.700 |

The end-to-end flow times two uploads, GEMM, one result download and a stream wait, using preallocated buffers. Pinned-memory allocation is excluded. The [96 transfer samples](transfers.csv) also include transfer-only flows. No overlap or batching is used. These end-to-end samples have five accumulated timing blocks but the existing transfer harness reports only their combined mean, so transfer variance is unavailable.

**Method, stability and limits**

488 initial operation cases use five repetition blocks with at least 50 ms of measured calls each. 8 CPU cases whose initial CV exceeded 10% were repeated with 200 ms warm-up and seven blocks of at least 200 ms. All 116 GPU cases were repeated with 200 ms warm-up and seven blocks of at least 100 ms. The one GPU case whose steady-run CV exceeded 10% was repeated once with the same settings, and that last sample is used. The final CSV selects the longer CPU reruns and steady GPU reruns regardless of whether a new sample was faster. Median CV is 1.40%; maximum is 9.97%; 0 final cases exceed 10%. CV describes variability between repetition averages, not a confidence interval.

Each timed GPU call includes host dispatch and `cudaStreamSynchronize`; allocation, public workspace queries, input restoration, initial upload and numerical status downloads are outside operation timing. Internal library work remains included. Steady GPU GEMM/SVD reuse inputs; Cholesky/LU/QR/eigen restore consumed inputs before each call. Fixed mt19937 seeds 42/43 provide matching random inputs. Cholesky inputs are symmetric positive definite; LU/general eigen diagonals are shifted by 2n. Backend order is fixed native/OpenBLAS/MKL/CUDA. This is one host with warm reused buffers, and includes no cold-start, ill-conditioned-input, power or full multicore study.

The [unit test and GPU residency check log](tests.log) records 93 passing unit tests and a passing GPU source-residency check for the rebuilt library. All timed programs exited successfully and their cuSOLVER status checks passed. This does not establish per-case numerical residuals across all 488 timings. Native LU’s unpivoted contract differs from the vendor LU paths. [Initial log](run.log) · [GPU repeat log](gpu_steady.log) | [GPU retry log](gpu_retry.log) · [CPU repeat selection](rerun_selection.txt) · [GPU repeat selection](gpu_steady_selection.txt).

Preparatory limitation: unpinned report-generation attempts failed, while generation pinned to CPU 0 succeeded. All recorded tests and benchmark processes were pinned throughout and exited successfully. No diagnosis of the unpinned failures is established by these measurements.

**Reproduce**

Follow the [build instructions](../../README.md) with all vendor backends and LAPACKE enabled, `LINALG_BUILD_EXAMPLES=ON`, Release, AVX2 and LTO off. Supply matching OpenBLAS/MKL/CUDA runtime paths. From the repository root, with vendor thread counts set to one and the process pinned to one logical CPU, run:

```powershell
cmake --build build_cpu_cuda_20261020 --target backend_analysis transfer_analysis LinearAlgebraCxxTests --parallel 1
ctest --test-dir build_cpu_cuda_20261020 -C Release -R "^(LinearAlgebraCxxTests|GpuResidencySourceCheck)$" --verbose
./build_cpu_cuda_20261020/bin/backend_analysis.exe results.csv
./build_cpu_cuda_20261020/bin/backend_analysis.exe gpu_steady_results.csv examples/backend_benchmark/results/2026-10-20/gpu_steady_selection.txt steady_gpu
./build_cpu_cuda_20261020/bin/backend_analysis.exe rerun_results.csv examples/backend_benchmark/results/2026-10-20/rerun_selection.txt
./build_cpu_cuda_20261020/bin/backend_analysis.exe gpu_retry_results.csv examples/backend_benchmark/results/2026-10-20/gpu_retry_selection.txt steady_gpu
./build_cpu_cuda_20261020/bin/transfer_analysis.exe transfers.csv
```

The CPU repeat selection belongs to this measurement set; choose cases from a new run’s CV values for a new comparison. The older `report.py` contains historical cohort counts and narrative; it was not used to label this new dataset.

**Complete comparison, microseconds**

**GEMM**

| Precision | M x N x K | TA,TB | Native us | OpenBLAS us | MKL us | CUDA us | Best CPU time / CUDA time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| double | 32x32x32 | 0,0 | 1.273 | 1.219 | 0.909 | 15.753 | 0.06x |
| double | 64x64x64 | 0,0 | 8.937 | 8.297 | 6.238 | 23.074 | 0.27x |
| double | 64x64x64 | 0,1 | 9.062 | 8.129 | 7.335 | 22.666 | 0.32x |
| double | 64x64x64 | 1,0 | 9.180 | 8.992 | 6.434 | 22.594 | 0.28x |
| double | 64x64x64 | 1,1 | 8.939 | 8.734 | 7.313 | 22.470 | 0.33x |
| double | 64x1024x256 | 0,0 | 567.785 | 485.937 | 502.381 | 127.191 | 3.82x |
| double | 128x128x128 | 0,0 | 81.477 | 57.023 | 51.523 | 36.824 | 1.40x |
| double | 256x256x256 | 0,0 | 571.422 | 422.660 | 410.005 | 117.928 | 3.48x |
| double | 256x256x256 | 0,1 | 563.545 | 425.764 | 412.817 | 119.975 | 3.44x |
| double | 256x256x256 | 1,0 | 574.088 | 431.821 | 417.051 | 118.611 | 3.52x |
| double | 256x256x256 | 1,1 | 562.093 | 444.870 | 413.746 | 119.813 | 3.45x |
| double | 512x512x512 | 0,0 | 4,016.632 | 3,349.498 | 3,226.330 | 872.058 | 3.70x |
| double | 512x512x512 | 0,1 | 4,048.033 | 3,321.167 | 3,224.326 | 871.698 | 3.70x |
| double | 512x512x512 | 1,0 | 4,115.417 | 3,381.974 | 3,367.087 | 873.437 | 3.85x |
| double | 512x512x512 | 1,1 | 4,032.186 | 3,427.497 | 3,362.325 | 873.092 | 3.85x |
| double | 1024x64x256 | 0,0 | 551.772 | 522.628 | 441.470 | 117.892 | 3.74x |
| double | 1024x1024x1024 | 0,0 | 33,647.490 | 25,872.930 | 25,854.520 | 6,872.354 | 3.76x |
| double | 1024x1024x1024 | 0,1 | 32,197.920 | 26,585.830 | 26,089.350 | 6,693.293 | 3.90x |
| double | 1024x1024x1024 | 1,0 | 32,005.660 | 27,449.200 | 25,700.250 | 6,858.422 | 3.75x |
| double | 1024x1024x1024 | 1,1 | 30,524.880 | 27,503.190 | 26,330.150 | 6,853.952 | 3.84x |
| double | 2048x2048x2048 | 0,0 | 295,420.580 | 215,358.140 | 206,365.580 | 54,747.836 | 3.77x |
| float | 32x32x32 | 0,0 | 0.858 | 0.787 | 0.498 | 14.531 | 0.03x |
| float | 64x64x64 | 0,0 | 4.984 | 4.768 | 3.188 | 13.980 | 0.23x |
| float | 64x64x64 | 0,1 | 5.223 | 4.765 | 4.070 | 14.125 | 0.29x |
| float | 64x64x64 | 1,0 | 4.775 | 4.843 | 3.251 | 13.613 | 0.24x |
| float | 64x64x64 | 1,1 | 4.860 | 4.837 | 3.756 | 13.865 | 0.27x |
| float | 64x1024x256 | 0,0 | 288.277 | 265.766 | 196.021 | 14.547 | 13.48x |
| float | 128x128x128 | 0,0 | 37.151 | 32.013 | 27.662 | 12.101 | 2.29x |
| float | 256x256x256 | 0,0 | 332.233 | 219.245 | 204.694 | 13.875 | 14.75x |
| float | 256x256x256 | 0,1 | 333.370 | 215.327 | 203.381 | 14.132 | 14.39x |
| float | 256x256x256 | 1,0 | 329.840 | 222.378 | 200.828 | 15.622 | 12.86x |
| float | 256x256x256 | 1,1 | 328.371 | 222.364 | 209.516 | 13.979 | 14.99x |
| float | 512x512x512 | 0,0 | 2,256.112 | 1,645.654 | 1,615.619 | 33.950 | 47.59x |
| float | 512x512x512 | 0,1 | 2,247.170 | 1,662.283 | 1,598.496 | 36.805 | 43.43x |
| float | 512x512x512 | 1,0 | 2,252.310 | 1,685.656 | 1,635.827 | 36.128 | 45.28x |
| float | 512x512x512 | 1,1 | 2,246.037 | 1,666.360 | 1,640.562 | 33.881 | 48.42x |
| float | 1024x64x256 | 0,0 | 260.399 | 226.053 | 194.992 | 14.419 | 13.52x |
| float | 1024x1024x1024 | 0,0 | 16,037.295 | 12,786.485 | 13,870.445 | 198.553 | 64.40x |
| float | 1024x1024x1024 | 0,1 | 15,669.100 | 13,321.905 | 12,937.855 | 214.638 | 60.28x |
| float | 1024x1024x1024 | 1,0 | 16,156.837 | 13,150.270 | 13,402.420 | 191.977 | 68.50x |
| float | 1024x1024x1024 | 1,1 | 15,731.445 | 12,946.415 | 13,123.285 | 203.893 | 63.50x |
| float | 2048x2048x2048 | 0,0 | 131,296.040 | 101,739.900 | 100,304.440 | 1,451.064 | 69.12x |

**Cholesky**

| Precision | M x N x K | TA,TB | Native us | OpenBLAS us | MKL us | CUDA us | Best CPU time / CUDA time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| double | 16x16x16 | 0,0 | 0.588 | 0.579 | 0.563 | 23.062 | 0.02x |
| double | 32x32x32 | 0,0 | 2.554 | 1.685 | 1.168 | 33.536 | 0.03x |
| double | 64x64x64 | 0,0 | 11.187 | 6.641 | 4.747 | 80.113 | 0.06x |
| double | 128x128x128 | 0,0 | 53.166 | 46.298 | 31.730 | 173.152 | 0.18x |
| double | 256x256x256 | 0,0 | 300.922 | 242.051 | 140.306 | 368.532 | 0.38x |
| double | 512x512x512 | 0,0 | 2,003.173 | 1,714.248 | 1,249.054 | 991.484 | 1.26x |
| double | 1024x1024x1024 | 0,0 | 34,464.130 | 11,179.428 | 8,274.466 | 3,321.324 | 2.49x |
| float | 16x16x16 | 0,0 | 0.425 | 0.566 | 0.547 | 18.938 | 0.02x |
| float | 32x32x32 | 0,0 | 1.950 | 1.659 | 1.127 | 23.658 | 0.05x |
| float | 64x64x64 | 0,0 | 8.910 | 5.371 | 5.759 | 41.500 | 0.13x |
| float | 128x128x128 | 0,0 | 43.350 | 27.254 | 36.103 | 80.656 | 0.34x |
| float | 256x256x256 | 0,0 | 219.610 | 178.742 | 76.734 | 156.574 | 0.49x |
| float | 512x512x512 | 0,0 | 1,352.839 | 1,086.103 | 620.759 | 349.758 | 1.77x |
| float | 1024x1024x1024 | 0,0 | 12,914.885 | 6,295.544 | 3,734.061 | 835.978 | 4.47x |

**LU**

| Precision | M x N x K | TA,TB | Native us | OpenBLAS us | MKL us | CUDA us | Best CPU time / CUDA time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| double | 16x16x16 | 0,0 | 0.275 | 1.117 | 0.665 | 64.426 | 0.00x |
| double | 32x32x32 | 0,0 | 1.402 | 4.424 | 3.715 | 80.340 | 0.02x |
| double | 64x64x64 | 0,0 | 8.154 | 14.753 | 10.719 | 106.263 | 0.08x |
| double | 128x128x128 | 0,0 | 54.351 | 65.594 | 43.754 | 200.432 | 0.22x |
| double | 256x256x256 | 0,0 | 407.313 | 320.314 | 223.646 | 507.020 | 0.44x |
| double | 512x512x512 | 0,0 | 3,500.692 | 2,492.062 | 2,077.912 | 1,676.463 | 1.24x |
| double | 1024x1024x1024 | 0,0 | 25,300.673 | 17,825.533 | 14,577.887 | 5,446.637 | 2.68x |
| float | 16x16x16 | 0,0 | 0.241 | 1.084 | 0.695 | 60.575 | 0.00x |
| float | 32x32x32 | 0,0 | 1.228 | 3.033 | 3.253 | 73.348 | 0.02x |
| float | 64x64x64 | 0,0 | 6.445 | 10.415 | 8.803 | 90.438 | 0.07x |
| float | 128x128x128 | 0,0 | 36.513 | 43.771 | 30.049 | 152.988 | 0.20x |
| float | 256x256x256 | 0,0 | 229.048 | 217.946 | 139.707 | 272.495 | 0.51x |
| float | 512x512x512 | 0,0 | 1,816.826 | 1,343.125 | 1,066.134 | 956.752 | 1.11x |
| float | 1024x1024x1024 | 0,0 | 14,685.245 | 12,223.541 | 6,844.645 | 1,851.496 | 3.70x |

**QR: factor + explicit Q**

| Precision | M x N x K | TA,TB | Native us | OpenBLAS us | MKL us | CUDA us | Best CPU time / CUDA time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| double | 16x16x16 | 0,0 | 3.346 | 3.667 | 2.208 | 175.321 | 0.01x |
| double | 32x32x32 | 0,0 | 9.787 | 12.376 | 6.068 | 276.736 | 0.02x |
| double | 64x64x64 | 0,0 | 49.461 | 50.069 | 35.683 | 586.945 | 0.06x |
| double | 128x128x128 | 0,0 | 351.383 | 369.582 | 225.170 | 1,489.605 | 0.15x |
| double | 256x256x256 | 0,0 | 2,718.393 | 1,417.069 | 1,172.310 | 2,455.565 | 0.48x |
| double | 512x512x512 | 0,0 | 19,171.680 | 9,890.029 | 8,620.942 | 6,544.534 | 1.32x |
| double | 1024x1024x1024 | 0,0 | 221,388.440 | 73,189.560 | 56,037.440 | 23,144.929 | 2.42x |
| float | 16x16x16 | 0,0 | 2.217 | 3.650 | 2.374 | 89.605 | 0.02x |
| float | 32x32x32 | 0,0 | 9.657 | 13.649 | 5.687 | 128.302 | 0.04x |
| float | 64x64x64 | 0,0 | 43.387 | 51.822 | 22.869 | 255.718 | 0.09x |
| float | 128x128x128 | 0,0 | 236.333 | 253.507 | 131.248 | 732.123 | 0.18x |
| float | 256x256x256 | 0,0 | 1,548.895 | 1,024.643 | 663.604 | 699.327 | 0.95x |
| float | 512x512x512 | 0,0 | 10,419.256 | 5,298.210 | 4,237.225 | 1,643.699 | 2.58x |
| float | 1024x1024x1024 | 0,0 | 104,154.900 | 45,827.470 | 28,367.820 | 4,037.832 | 7.03x |

**SVD**

| Precision | M x N x K | TA,TB | Native us | OpenBLAS us | MKL us | CUDA us | Best CPU time / CUDA time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| double | 16x16x16 | 0,0 | 13.349 | 20.064 | 18.212 | 1,062.956 | 0.01x |
| double | 32x32x32 | 0,0 | 64.285 | 87.492 | 62.900 | 1,920.104 | 0.03x |
| double | 64x64x64 | 0,0 | 956.485 | 447.300 | 412.699 | 4,759.845 | 0.09x |
| double | 128x128x128 | 0,0 | 7,990.409 | 4,956.098 | 2,464.875 | 12,669.672 | 0.19x |
| double | 256x256x256 | 0,0 | Not sampled | 39,142.870 | 8,229.309 | 27,368.443 | 0.30x |
| double | 512x512x512 | 0,0 | Not sampled | 336,396.240 | 54,254.860 | 81,626.021 | 0.66x |
| double | 1024x1024x1024 | 0,0 | Not sampled | 3,744,975.860 | 475,614.080 | 281,817.286 | 1.69x |
| float | 16x16x16 | 0,0 | 9.195 | 16.958 | 14.934 | 806.201 | 0.01x |
| float | 32x32x32 | 0,0 | 52.609 | 76.254 | 46.374 | 1,530.832 | 0.03x |
| float | 64x64x64 | 0,0 | 397.034 | 392.417 | 324.367 | 3,386.769 | 0.10x |
| float | 128x128x128 | 0,0 | 5,215.180 | 2,985.525 | 1,441.195 | 10,030.855 | 0.14x |
| float | 256x256x256 | 0,0 | Not sampled | 27,345.640 | 16,472.440 | 21,392.746 | 0.77x |
| float | 512x512x512 | 0,0 | Not sampled | 230,736.180 | 107,857.060 | 49,642.579 | 2.17x |
| float | 1024x1024x1024 | 0,0 | Not sampled | 2,939,965.400 | 817,589.760 | 111,611.114 | 7.33x |

**Symmetric eigen**

| Precision | M x N x K | TA,TB | Native us | OpenBLAS us | MKL us | CUDA us | Best CPU time / CUDA time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| double | 8x8x8 | 0,0 | 2.884 | 3.602 | 4.871 | 189.424 | 0.02x |
| double | 16x16x16 | 0,0 | 20.825 | 12.821 | 12.822 | 300.760 | 0.04x |
| double | 32x32x32 | 0,0 | 134.267 | 47.017 | 37.871 | 504.915 | 0.08x |
| double | 64x64x64 | 0,0 | 1,103.521 | 182.914 | 134.857 | 1,609.700 | 0.08x |
| double | 96x96x96 | 0,0 | 4,607.822 | 400.285 | 299.258 | 3,336.939 | 0.09x |
| double | 128x128x128 | 0,0 | 44,987.020 | 730.696 | 563.487 | 4,297.299 | 0.13x |
| double | 256x256x256 | 0,0 | Not sampled | 3,390.340 | 2,882.143 | 5,724.400 | 0.50x |
| double | 512x512x512 | 0,0 | Not sampled | 17,689.953 | 16,704.512 | 12,857.343 | 1.30x |
| double | 1024x1024x1024 | 0,0 | Not sampled | 117,057.560 | 106,937.320 | 42,040.067 | 2.54x |
| float | 8x8x8 | 0,0 | 2.312 | 3.403 | 3.269 | 109.063 | 0.02x |
| float | 16x16x16 | 0,0 | 15.399 | 11.424 | 8.636 | 145.735 | 0.06x |
| float | 32x32x32 | 0,0 | 93.153 | 41.738 | 28.602 | 197.799 | 0.14x |
| float | 64x64x64 | 0,0 | 779.119 | 132.368 | 90.917 | 848.186 | 0.11x |
| float | 96x96x96 | 0,0 | 2,662.440 | 292.928 | 198.301 | 1,407.101 | 0.14x |
| float | 128x128x128 | 0,0 | 24,030.107 | 524.048 | 379.971 | 1,794.176 | 0.21x |
| float | 256x256x256 | 0,0 | Not sampled | 2,120.347 | 1,586.293 | 1,445.201 | 1.10x |
| float | 512x512x512 | 0,0 | Not sampled | 10,780.856 | 8,673.167 | 3,427.747 | 2.53x |
| float | 1024x1024x1024 | 0,0 | Not sampled | 67,450.580 | 54,057.300 | 10,681.124 | 5.06x |

**General eigenvalues**

| Precision | M x N x K | TA,TB | Native us | OpenBLAS us | MKL us | CUDA us | Best CPU time / CUDA time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| double | 8x8x8 | 0,0 | 9.860 | 5.429 | 3.561 | Not sampled | Unsupported |
| double | 16x16x16 | 0,0 | 45.987 | 17.329 | 11.482 | Not sampled | Unsupported |
| double | 32x32x32 | 0,0 | 390.839 | 74.894 | 52.986 | Not sampled | Unsupported |
| double | 64x64x64 | 0,0 | 3,455.151 | 346.218 | 267.214 | Not sampled | Unsupported |
| double | 96x96x96 | 0,0 | 13,621.190 | 1,251.663 | 733.762 | Not sampled | Unsupported |
| double | 128x128x128 | 0,0 | 39,734.640 | 2,393.755 | 2,210.093 | Not sampled | Unsupported |
| float | 8x8x8 | 0,0 | 11.245 | 4.563 | 5.333 | Not sampled | Unsupported |
| float | 16x16x16 | 0,0 | 34.044 | 14.850 | 14.379 | Not sampled | Unsupported |
| float | 32x32x32 | 0,0 | 240.486 | 54.773 | 45.270 | Not sampled | Unsupported |
| float | 64x64x64 | 0,0 | 2,534.072 | 254.575 | 232.761 | Not sampled | Unsupported |
| float | 96x96x96 | 0,0 | 14,285.445 | 861.190 | 824.040 | Not sampled | Unsupported |
| float | 128x128x128 | 0,0 | 20,532.407 | 1,500.248 | 1,362.772 | Not sampled | Unsupported |
