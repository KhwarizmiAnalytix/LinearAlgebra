# CPU and GPU backend benchmark example

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

CTest pins each process to one CPU and sets both vendor thread counts to one.
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
