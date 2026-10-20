# CPU and GPU backend benchmark example

Fresh comparison of native C++, OpenBLAS and MKL against CUDA:
[20 October 2026 results](results/2026-10-20/comparison.md). The new run includes
488 operation cases, 96 transfer cases and the current QR API.

Read the [full matrix operation analysis](analysis.md) for measurements of
native, OpenBLAS, MKL and CUDA backends on the recorded Windows host.
The report includes float/double results, size crossovers, transfer costs,
workspace requirements and measurement limits.

The C++ programs call the public library API. Sources and recorded results
live here alongside the repository's `include` directory. Compiled programs,
CMake caches and fresh runtime CSV output belong in a build directory.

- `benchmark.cpp`: same-executable comparison of all four backends.
- `transfer.cpp`: host/device transfer and upload–GEMM–download measurements.
- `report.py`: generates Markdown tables and charts from recorded CSV data.
- `results/2026-10-19/`: raw cohorts, merged results, metadata and test logs.
- `selections/`: case lists used for longer-sample and steady-input reruns.
- `assets/`: charts embedded in the Markdown report.

## Build and run

Configure from the repository root using its normal compiler and dependency
settings. All vendor backends, including LAPACKE, are required:

```sh
cmake -S . -B build_examples -DCMAKE_BUILD_TYPE=Release -DLINALG_BUILD_EXAMPLES=ON -DLINALG_ENABLE_BLAS=ON -DLINALG_ENABLE_MKL=ON -DLINALG_ENABLE_CUBLAS=ON
cmake --build build_examples --config Release --target backend_analysis transfer_analysis
ctest --test-dir build_examples -C Release -R '^example_' --output-on-failure
```

Supply your BLAS include/runtime paths and MKL/CUDA discovery settings as
described in the [repository README](../../README.md). The example links the
`LinearAlgebra::LinearAlgebra` target and inherits its backend definitions and
dependencies. It does not link a hard-coded library path.

The example uses the current `linalg::linalg_backend` enum values. CSV labels
stay lowercase so the recorded selections and report script remain compatible:

| Backend enum | CSV label |
| --- | --- |
| `NATIVE` | `native` |
| `BLAS_LAPACKE` | `openblas` by default; `blis` with `LINALG_ENABLE_BLIS=ON` |
| `MKL` | `mkl` |
| `CUDA_BLAS` | `gpu` |

To build the Google Benchmark executables alongside the examples, also set
`-DLINALG_ENABLE_BENCHMARK=ON`. Their registered names contain the uppercase
enum values. For example, compare double-precision 64-by-64 GEMM on both CPU
vendor backends with:

```powershell
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'
./build_examples/bin/benchmark_linalg_multiplication.exe --benchmark_filter='double.*(BLAS_LAPACKE|MKL)>/64/64/64'
```

For a BLIS comparison, configure a fresh build tree with
`-DLINALG_ENABLE_BLIS=ON` in place of `-DLINALG_ENABLE_BLAS=ON`. Both use
`linalg_backend::BLAS_LAPACKE`; BLIS is a provider rather than a separate enum.
Supply matching BLIS headers and runtime plus a LAPACKE implementation such as
AOCL libflame or netlib LAPACKE. When MKL is also enabled, set
`LINALG_BLAS_RUNTIME_LIBRARY` to the BLIS shared library and
`LINALG_LAPACKE_RUNTIME_LIBRARY` to the separate LAPACKE shared library.
Do not reuse cached OpenBLAS runtime paths for a BLIS-labeled run.
The stock benchmarks label these cases `blis`; their registered template
names still contain `BLAS_LAPACKE`.

CTest pins each process to one CPU and sets MKL, OpenBLAS and BLIS thread counts to one.
Use a Release build for timings. The recorded configuration used Clang/Ninja
on Windows; other toolchains and platforms were not measured. This CUDA example
uses CMake; the repository's Bazel configuration currently supports CPU sources.

Current QR timings call `apply_r` followed by `apply_q` in place, without extracting
R. QR and eigen inputs are restored outside the timed region on every iteration,
including steady GPU runs, because these APIs consume their input matrices.
The dated result files describe the earlier recorded revision and are unchanged.

The four CTest entries are:

| Test | Measurement |
| --- | --- |
| `example_backend_analysis` | Initial comparison of 488 operation cases |
| `example_backend_analysis_rerun` | Longer samples of the 19 recorded noisy cases |
| `example_backend_analysis_gpu_steady` | 88 GPU cases with resident input reuse |
| `example_transfer_analysis` | 64 transfer-only and 32 end-to-end GEMM cases |

New CSV output is written under
`build_examples/examples/backend_benchmark/results/`. CTest logs stay in the
build tree. Running the example does not overwrite the dated recorded data.
The selection lists reproduce the recorded run; new measurements may have
different noisy cases and should be reviewed before drawing conclusions.

## Regenerate the Markdown report

Use Python 3.12 or newer and install the plotting requirements in your Python
environment:

```sh
python -m pip install -r examples/backend_benchmark/requirements.txt
python examples/backend_benchmark/report.py
```

The default command reads the recorded data and regenerates `analysis.md` and
the PNG charts in `assets/`. No GPU is required to regenerate a report.
To place a regenerated report elsewhere, pass `--output-dir <directory>`.

For a new measurement set, pass `--data-dir <directory>` as well. That directory
must contain `results.csv`, `rerun_results.csv`, `gpu_steady_results.csv`,
`transfers.csv` and a `metadata.json` recording the new host, library commit,
compiler, runtime versions and thread policy. The script also writes
`final_results.csv` into that data directory. Do not reuse the historical host
metadata for measurements from a different machine or configuration.

The report reads provider labels from the CSV files. BLIS data produces BLIS
columns and chart legends; OpenBLAS data retains OpenBLAS labels, and a merged
dataset can contain both. Case and rerun counts come from the input files.
For BLIS runs, record the BLIS and LAPACKE companion versions in `metadata.json`
along with `BLIS_NUM_THREADS` in `cpu_environment`. Selection files must use
the same provider label as the current executable (`blis` for a BLIS build).
Changing a label on an old CSV does not constitute a new provider measurement.

Run report regression tests after installing the plotting requirements:

```sh
python -m unittest discover -s examples/backend_benchmark -p test_report.py -v
```
