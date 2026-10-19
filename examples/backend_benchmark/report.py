"""Build the analysis artifacts from measured CSV data; no benchmarks are run here."""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, default=ROOT / "results/2026-10-19")
parser.add_argument("--output-dir", type=Path, default=ROOT)
args = parser.parse_args()
DATA = args.data_dir.resolve()
OUTPUT = args.output_dir.resolve()
ASSETS = OUTPUT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

KEYS = [
    "operation",
    "precision",
    "backend",
    "m",
    "n",
    "k",
    "transpose_a",
    "transpose_b",
]
BACKENDS = ["native", "openblas", "mkl", "gpu"]
NAMES = {
    "gemm": "GEMM",
    "cholesky": "Cholesky",
    "lu": "LU",
    "qr": "QR (Q and R)",
    "svd": "SVD (U, S, VT)",
    "symmetric_eigen": "Symmetric eigen (values + vectors)",
    "general_eigen_values": "General eigen (values only)",
}
COLORS = {
    "native": "#637a90",
    "openblas": "#d58a21",
    "mkl": "#1c9185",
    "gpu": "#7052c1",
}


def read(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return tuple(row[field] for field in KEYS)


initial = read("results.csv")
reruns = {key(row): row for row in read("rerun_results.csv")}
steady = {key(row): row for row in read("gpu_steady_results.csv")}
assert len(initial) == 488 and len(reruns) == 19 and len(steady) == 88
rows = []
for original in initial:
    identity = key(original)
    source, repetitions, duration = "initial", 5, 0.05
    row = original
    if identity in reruns:
        row = reruns[identity]
        source, repetitions, duration = "longer_sample", 7, 0.2
    if identity in steady:
        row = steady[identity]
        source, repetitions, duration = "gpu_steady_input", 7, 0.1
    row = dict(
        row, measurement_source=source, repetitions=repetitions, target_seconds=duration
    )
    rows.append(row)
with (DATA / "final_results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

transfers = read("transfers.csv")
assert len(transfers) == 96


def number(row, field="mean_us"):
    return float(row[field])


def square(op, precision, n):
    return {
        row["backend"]: row
        for row in rows
        if row["operation"] == op
        and row["precision"] == precision
        and row["m"] == row["n"] == row["k"] == str(n)
        and row["transpose_a"] == row["transpose_b"] == "0"
    }


def best_cpu(group):
    return min((row for backend, row in group.items() if backend != "gpu"), key=number)


def table(headers, records):
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    result.extend("| " + " | ".join(map(str, record)) + " |" for record in records)
    return "\n".join(result)


def format_us(value):
    return f"{value:,.3f}" if value < 10 else f"{value:,.2f}"


def comparison_table(precision, n, unit=1.0):
    records = []
    for op in NAMES:
        group = square(op, precision, n)
        if not group:
            continue
        cpu = best_cpu(group)
        speedup = (
            f"{number(cpu) / number(group['gpu']):.2f}×"
            if "gpu" in group
            else "Unsupported"
        )
        record = [NAMES[op]]
        record.extend(
            format_us(number(group[b]) / unit) if b in group else "Not measured"
            for b in BACKENDS
        )
        record.append(speedup)
        records.append(record)
    return table(
        ["Operation", "Native", "OpenBLAS", "MKL", "GPU", "GPU speedup over best CPU"],
        records,
    )


def crossover(op, precision):
    sizes = sorted(
        {
            int(row["n"])
            for row in rows
            if row["operation"] == op
            and row["precision"] == precision
            and row["backend"] == "gpu"
            and row["m"] == row["n"] == row["k"]
            and row["transpose_a"] == row["transpose_b"] == "0"
        }
    )
    for index, n in enumerate(sizes):
        if all(
            number(square(op, precision, size)["gpu"])
            < number(best_cpu(square(op, precision, size)))
            for size in sizes[index:]
        ):
            return str(n)
    return "No crossover measured"


cv = [number(row, "cv_percent") for row in rows]
quality = {
    "cases": len(rows),
    "transfer_and_end_to_end_cases": len(transfers),
    "median_cv_percent": statistics.median(cv),
    "max_cv_percent": max(cv),
    "cases_above_10_percent_cv": sum(value > 10 for value in cv),
}
metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
metadata["quality"] = quality

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "savefig.facecolor": "white",
    }
)
figure, axes = plt.subplots(3, 2, figsize=(13, 12))
figure.subplots_adjust(
    top=0.89, bottom=0.055, left=0.075, right=0.98, hspace=0.34, wspace=0.25
)
for ax, op in zip(axes.flat, list(NAMES)[:6]):
    for backend in BACKENDS:
        for precision, style in (("float", "-"), ("double", "--")):
            selected = [
                row
                for row in rows
                if row["operation"] == op
                and row["backend"] == backend
                and row["precision"] == precision
                and row["m"] == row["n"] == row["k"]
                and row["transpose_a"] == row["transpose_b"] == "0"
            ]
            selected.sort(key=lambda row: int(row["n"]))
            ax.plot(
                [int(row["n"]) for row in selected],
                [number(row) / 1000 for row in selected],
                color=COLORS[backend],
                linestyle=style,
                marker="o",
                markersize=3,
                linewidth=1.7,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 128, 512, 2048] if op == "gemm" else [16, 64, 256, 1024])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_title(NAMES[op], loc="left", fontweight="bold")
    ax.set_xlabel("Square matrix dimension n")
    ax.set_ylabel("Completed-call latency (ms; lower is better)")
handles = [
    plt.Line2D(
        [],
        [],
        color=COLORS[b],
        label=b.upper() if b in ("gpu", "mkl") else b.capitalize(),
        linewidth=2,
    )
    for b in BACKENDS
]
handles += [
    plt.Line2D([], [], color="#333333", linestyle=style, label=precision)
    for precision, style in (("float", "-"), ("double", "--"))
]
figure.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.94),
    ncol=6,
    frameon=False,
)
figure.suptitle(
    "Matrix operation scaling · one CPU thread vs RTX 4060 Ti\nGPU data resident; allocation and input restoration excluded",
    fontsize=14,
    y=0.995,
)
figure.savefig(ASSETS / "scaling.png", dpi=160)
plt.close(figure)

figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
for ax, precision in zip(axes, ("float", "double")):
    sizes = [32, 64, 128, 256, 512, 1024, 2048]
    ax.plot(
        sizes,
        [number(best_cpu(square("gemm", precision, n))) for n in sizes],
        label="Fastest CPU backend",
        color="#1c9185",
        marker="o",
    )
    ax.plot(
        sizes,
        [number(square("gemm", precision, n)["gpu"]) for n in sizes],
        label="GPU: data resident",
        color="#7052c1",
        marker="o",
    )
    for memory, style, color in (
        ("pageable", "-", "#d58a21"),
        ("pinned", "--", "#c25c4d"),
    ):
        selected = {
            int(row["n"]): number(row)
            for row in transfers
            if row["measurement"] == "gemm_end_to_end"
            and row["precision"] == precision
            and row["host_memory"] == memory
        }
        ax.plot(
            sizes,
            [selected[n] for n in sizes],
            label=f"GPU: upload + GEMM + download ({memory})",
            color=color,
            linestyle=style,
            marker="o",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_title(precision, loc="left", fontweight="bold")
    ax.set_xlabel("Square matrix dimension n")
    ax.set_ylabel("Latency (µs; lower is better)")
axes[0].legend(fontsize=8, loc="upper left")
figure.suptitle("GEMM: GPU residency changes the crossover", fontsize=14)
figure.savefig(ASSETS / "gemm_transfers.png", dpi=160)
plt.close(figure)

sections = []


def paragraph(text):
    sections.append(text)


paragraph("**Matrix operation benchmark analysis — 19 October 2026**")
paragraph(
    f"[Reproduction instructions]({Path(os.path.relpath(ROOT / 'README.md', OUTPUT)).as_posix()}) · "
    f"[Recorded results]({Path(os.path.relpath(DATA / 'final_results.csv', OUTPUT)).as_posix()}) · "
    f"[Environment metadata]({Path(os.path.relpath(DATA / 'metadata.json', OUTPUT)).as_posix()})"
)
paragraph(
    "The measured backends are the AVX2 native implementation, independently loaded OpenBLAS, Intel MKL and CUDA (cuBLAS/cuSOLVER). All four are called from the same analysis executable linked to the tested library. Large matrices generally favor the GPU under this single-thread CPU policy; small decompositions frequently favor a CPU backend. Precision and data transfers materially change the crossover."
)
paragraph("**Measurement environment and scope**")
paragraph(
    table(
        ["Component", "Measured configuration"],
        [
            [k.replace("_", " ").capitalize(), v]
            for k, v in metadata.items()
            if k not in ("quality", "cpu_environment")
        ],
    )
)
paragraph(
    "CTest applies PROCESSOR_AFFINITY=TRUE, PROCESSORS=1, MKL_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. The harness affinity was observed as 0x1. This is a comparison against one CPU thread, not the full multicore capacity of the i9. The native backend is vectorized with AVX2/FMA; describing it as a purely scalar implementation would be inaccurate for this build."
)
paragraph(
    "The stock six benchmark executables currently register CPU cases only, even in a cuBLAS-enabled build. GPU results in this report come from the attached analysis harness, not from those stock registrations. The harness sources, this Markdown report, charts and recorded data live in examples/backend_benchmark alongside the repository include directory. Production library code is unchanged."
)
paragraph(
    table(
        ["Operation", "Shapes / dimensions", "Backend coverage"],
        [
            [
                "GEMM, NN",
                "Square 32–2048; 64×1024×256 and 1024×64×256 (M,N,K)",
                "All four; float and double",
            ],
            [
                "GEMM, NT/TN/TT",
                "Square 64, 256, 512, 1024",
                "All four; float and double",
            ],
            [
                "Cholesky, LU, QR",
                "Square 16,32,64,128,256,512,1024",
                "All four; float and double",
            ],
            [
                "SVD",
                "Square 16,32,64,128,256,512,1024",
                "All four through 128; vendor CPU + GPU above 128",
            ],
            [
                "Symmetric eigen",
                "Square 8,16,32,64,96,128,256,512,1024",
                "All four through 128; vendor CPU + GPU above 128",
            ],
            [
                "General eigenvalues",
                "Square 8,16,32,64,96,128",
                "Three CPU backends; no GPU implementation exposed",
            ],
        ],
    )
)
paragraph(
    "This covers the operation families in the existing dense benchmark suite. It does not measure every library API: solves, inverse, determinant, norms, reductions, derivatives and batched operations are outside this run. General-eigenvector comparisons are also outside this harness; the stock suite's additional native eigenvector rows use a different constructed input and should not be mixed with eigenvalues-only results."
)
paragraph("**Timing and inputs**")
paragraph(
    "Completed-call wall time is measured with std::chrono::steady_clock. GPU calls run on a non-default, nonblocking stream and are followed by cudaStreamSynchronize inside the measured interval. Allocations, workspace queries, initial uploads and input restoration are outside that interval. Internal library copies, layout conversions, workspace queries performed by apply(), host API dispatch and completion waits remain included. Numerical status downloads occur after warm-up and timed repetition blocks, outside measured calls."
)
paragraph(
    "CPU inputs are restored before each call, outside timing. GPU Cholesky and LU restore their overwritten inputs with an untimed device-to-device copy and completion wait. The final read-only GPU measurements reuse resident inputs without unnecessary per-call restoration. All paths reuse allocated buffers. These are warm repeated-input measurements, not cold-cache or first-call/JIT measurements."
)
paragraph(
    "Matrices use fixed mt19937 seeds 42 (A) and 43 (B). GEMM/QR/SVD use uniform random values in [-1,1]. Cholesky uses a random symmetric, strictly diagonally dominant SPD matrix (diagonal shifted by 2n); LU and general eigen use random matrices with the same diagonal shift. Symmetric eigen uses a random symmetric matrix. Shapes and inputs match between backends within this harness. The Cholesky construction and untimed SVD restoration differ from the stock benchmarks, so the raw timings from the two harnesses should not be combined."
)
paragraph(
    "Initial cases used five repetitions, accumulating at least 50 ms of measured calls per repetition. Nineteen cases with >10% coefficient of variation (CV) were repeated with 200 ms warm-up and seven repetitions of at least 200 ms. The 88 read-only GPU cases were then measured with resident input reuse, 200 ms warm-up and seven repetitions of at least 100 ms. final_results.csv records the selected cohort, target duration and repetition count for every row; all earlier measurements are preserved. Each reported mean is the mean of per-repetition latency averages; median and sample standard deviation are also supplied. These CV values describe repetition stability, not confidence intervals for hardware-independent speedups."
)
paragraph(
    f"There are **488 operation cases**, **64 transfer-only cases** and **32 directly measured upload–GEMM–download cases**. Final operation CV: median **{statistics.median(cv):.2f}%**, maximum **{max(cv):.2f}%**, cases over 10%: **{sum(value > 10 for value in cv)}**. No numerical status failure or crash occurred. Success status and prior unit tests are not an exhaustive accuracy/conditioning study for every matrix size."
)
paragraph("**Large matrices: mean latency in milliseconds, n=1024**")
for precision in ("float", "double"):
    paragraph(f"**{precision}**\n\n" + comparison_table(precision, 1024, 1000.0))
paragraph(
    "Native SVD and symmetric eigen were intentionally capped at 128 in this analysis. The stock SVD benchmark documents a pre-existing convergence/runtime issue at larger native sizes; the eigen cap follows its stock benchmark range. Missing native rows above that cap are not failures or zero-time results. General eigen is reported separately at 128 because larger cases were not in this comparison's scope."
)
paragraph("**Smaller matrices: mean latency in microseconds, n=128**")
for precision in ("float", "double"):
    paragraph(f"**{precision}**\n\n" + comparison_table(precision, 128))
paragraph("**Observed crossover and per-operation interpretation**")
paragraph(
    table(
        [
            "Operation",
            "First sustained GPU win: float n",
            "First sustained GPU win: double n",
        ],
        [
            [NAMES[op], crossover(op, "float"), crossover(op, "double")]
            for op in list(NAMES)[:6]
        ],
    )
)
paragraph(
    "A crossover is the first sampled square size where GPU latency beats the fastest measured CPU backend and stays ahead at all larger sampled sizes. It is not an exact threshold between samples, does not apply to every input or shape, and is for data-resident GPU calls. Comparisons near parity should be rechecked on the intended workload."
)
paragraph(
    "GEMM has the strongest large-matrix GPU scaling, especially in float. MKL and OpenBLAS become close for large CPU GEMM, so preserving independent providers is useful even when one wins by only a few percent. GPU transpose flags are passed through cuBLAS with swapped row-major operands, without the library explicitly materializing transposes. The complete rectangular and transpose rows are included below rather than folded into the square results."
)
paragraph(
    "Cholesky and LU have much larger GPU overhead relative to tiny CPU factorizations. Native wins several tiny LU cases, but native LU was built with optional pivoting OFF while OpenBLAS, MKL and cuSOLVER use pivoted GETRF. Those small-LU results are not identical algorithm contracts. The strongly diagonally dominant input also avoids difficult pivoting behavior. MKL generally leads the larger CPU decomposition cases, with some size-specific OpenBLAS wins in Cholesky."
)
paragraph(
    "QR returns explicit Q and R. Its GPU path includes layout conversion, GEQRF, ORGQR and clearing R's lower triangle, so these timings should not be compared with a factorization-only GEQRF result. GPU QR's substantial workspace is a separate cost, shown below. Native QR becomes expensive with size; large vendor CPU and GPU paths offer substantially better latency in this run."
)
paragraph(
    "SVD computes singular vectors as well as singular values. Small cases favor CPU. MKL is substantially faster than OpenBLAS on larger SVD cases in this installation, and float/double timings do not follow a simple universal ratio. Both CPU adapters call GESVD, so this result is not evidence of a GESDD-versus-GESVD API mismatch. Backend implementations and convergence behavior differ; assigning a precise cause needs profiling. GPU's double-precision crossover is later than its float crossover."
)
paragraph(
    "Symmetric eigen includes eigenvectors. The native implementation is much slower than vendor CPU libraries near the upper end of its measured range. GPU becomes attractive at larger sizes; general nonsymmetric eigen remains a CPU choice because this library does not expose that GPU specialization."
)
paragraph(
    "The GPU's float and double GEMM performance differ far more than the CPU's in this run. Treat precision as part of backend selection. NVIDIA's [native arithmetic throughput table](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#arithmetic-instructions) documents architecture-dependent FP32/FP64 throughput; the measured result here should not be generalized to a GPU designed for high FP64 throughput. GPU clocks, driver scheduling and vendor algorithm selection can also make small-size curves nonmonotonic; this run does not isolate each cause."
)
paragraph("![Scaling across all CPU and GPU backends](assets/scaling.png)")
paragraph("**Transfers and directly measured end-to-end GEMM**")
transfer_records = []
for precision in ("float", "double"):
    for n in (64, 128, 256, 512, 1024, 2048):
        group = square("gemm", precision, n)
        lookup = {
            row["host_memory"]: number(row)
            for row in transfers
            if row["measurement"] == "gemm_end_to_end"
            and row["precision"] == precision
            and row["n"] == str(n)
        }
        transfer_records.append(
            [
                precision,
                n,
                format_us(number(best_cpu(group))),
                format_us(number(group["gpu"])),
                format_us(lookup["pageable"]),
                format_us(lookup["pinned"]),
            ]
        )
paragraph(
    table(
        [
            "Precision",
            "n",
            "Best CPU µs",
            "GPU resident µs",
            "GPU E2E pageable µs",
            "GPU E2E pinned µs",
        ],
        transfer_records,
    )
)
paragraph(
    "End-to-end GEMM directly times two host-to-device uploads, the library GEMM, one device-to-host result download and a final stream wait. Input/output and device buffers are preallocated. It is a separate measured flow, not a sum of independent transfer and compute timings. Pageable and cudaMallocHost-pinned host memory are both measured; pinned allocation cost is excluded. Transfers are serial in one stream, with no overlap or batching. Small-case driver behavior can differ between this sequence and an isolated resident call, so costs are not strictly additive."
)
paragraph(
    "Moving a result to the CPU after every operation can erase the advantage of a small GPU call. Keeping a chain of operations on-device avoids repeated transfers. NVIDIA recommends reducing host/device traffic and explains pinned-memory and overlap tradeoffs in its [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device). The full transfer-only measurements, including one-upload/one-download and two-upload/one-download flows, are in transfers.csv. End-to-end decomposition transfers were not measured, so no transfer-inclusive decomposition crossover is claimed."
)
paragraph("![Measured GEMM transfer impact](assets/gemm_transfers.png)")
paragraph("**Caller-owned workspace at n=1024, MiB**")
workspace_records = []
for precision in ("float", "double"):
    for op in list(NAMES)[:6]:
        group = square(op, precision, 1024)
        workspace_records.append(
            [precision, NAMES[op]]
            + [
                f"{int(group[b]['workspace_bytes']) / 1048576:.3f}"
                if b in group
                else "Not measured"
                for b in BACKENDS
            ]
        )
paragraph(
    table(
        ["Precision", "Operation", "Native", "OpenBLAS", "MKL", "GPU"],
        workspace_records,
    )
)
paragraph(
    "These are public workspace_size() results only. Input/output buffers, retained handles, vendor internal allocations and total peak VRAM are not included. In particular, GPU QR requests roughly 210 MiB in float and 419 MiB in double for a 1024 square matrix, despite an input of only 4 or 8 MiB. Plan memory capacity using the actual query, not just matrix storage. A small or zero caller workspace does not prove zero internal vendor memory use."
)
paragraph("**Practical choices supported by this run**")
paragraph(
    "Use native when its small-case latency and algorithm contract fit; tiny LU is an example, subject to the pivoting difference. Use a vendor CPU backend for small and medium factorizations and for general eigen. MKL is a strong CPU choice in this installation, but retain OpenBLAS as a distinct selectable provider and use measured operation/size data. Use the GPU for sufficiently large resident matrices, with separate float/double policies. For host-resident GEMM, consult the directly measured transfer-inclusive rows. Avoid one unconditional threshold for every operation."
)
paragraph(
    "The next performance work should target real application sizes and residency patterns, then CPU thread scaling, batched small matrices and GPU memory reuse. Repeated workspace/layout queries and conversions visible in the current GPU adapters are profiling candidates; these measurements alone do not establish how much speedup removing them would deliver. Any multicore CPU comparison needs a deliberate new affinity/thread policy and separate stability validation because this report uses the validated one-core workaround."
)
paragraph("**Validation, limits and reproduction**")
paragraph(
    "At library commit 15534ab, python setup.py build.test.blas.cublas.mkl.benchmark passed all eight CTest entries: 92 unit tests (including 21 GPU tests and two provider-isolation tests), the GPU source-residency check and six stock CPU benchmark executables. QR and SVD then each passed 10/10 additional full CTest runs. The analysis harness's 488 cases, focused longer-sample runs, steady-input GPU runs and 96 transfer/E2E cases also completed successfully. This supports the current stability workaround on this host; it does not identify or permanently repair the underlying unpinned crash."
)
paragraph(
    "This is one host, one compiler/library set, fixed random seeds, warm inputs, one CPU thread, no locked CPU/GPU clocks and Windows WDDM scheduling. Cases run in a fixed native/OpenBLAS/MKL/GPU order, not randomized across backend processes. Five or seven repetition averages do not establish universal statistical significance. No power/energy, cold-start, concurrent-stream, batched, full multicore, ill-conditioned-input or exhaustive numerical-accuracy study was performed."
)
paragraph(
    "Clang itself access-violated once while building the harness, and Python access-violated once while preparing plot dependencies. Both preparatory operations succeeded when pinned to one CPU. No measured benchmark failure followed. This is further reason to describe affinity as a validated workaround rather than a proven diagnosis of an MKL-specific bug."
)
paragraph(
    "Build the example from the repository root with LINALG_BUILD_EXAMPLES=ON and the BLAS, LAPACKE, MKL and cuBLAS backends enabled. CMake links LinearAlgebra::LinearAlgebra directly; no path to a prebuilt import library is embedded in the example. The README describes configuration, CTest commands and Markdown regeneration. New runtime CSV files go to the example build directory; the dated recorded data remain under results/2026-10-19. This four-backend CUDA example currently uses CMake; the repository Bazel integration supports CPU sources only."
)
paragraph("**Complete operation measurements**")
paragraph(
    "All times below are mean completed-call latency in microseconds. Shapes are M×N×K; transpose flags are (A,B). The CSV also contains medians, sample standard deviations, measured iteration counts and sampling provenance."
)
for op in NAMES:
    records = []
    for precision in ("float", "double"):
        selected = [
            row
            for row in rows
            if row["operation"] == op and row["precision"] == precision
        ]
        shapes = sorted(
            {
                (
                    int(row["m"]),
                    int(row["n"]),
                    int(row["k"]),
                    row["transpose_a"],
                    row["transpose_b"],
                )
                for row in selected
            }
        )
        for m, n, k, ta, tb in shapes:
            group = {
                row["backend"]: row
                for row in selected
                if (
                    int(row["m"]),
                    int(row["n"]),
                    int(row["k"]),
                    row["transpose_a"],
                    row["transpose_b"],
                )
                == (m, n, k, ta, tb)
            }
            records.append(
                [precision, f"{m}×{n}×{k}", f"{ta},{tb}"]
                + [format_us(number(group[b])) if b in group else "—" for b in BACKENDS]
            )
    paragraph(
        "**"
        + NAMES[op]
        + "**\n\n"
        + table(
            [
                "Precision",
                "Shape",
                "Transpose",
                "Native µs",
                "OpenBLAS µs",
                "MKL µs",
                "GPU µs",
            ],
            records,
        )
    )

markdown = "\n\n".join(sections) + "\n"
(OUTPUT / "analysis.md").write_text(markdown, encoding="utf-8", newline="\n")
print(json.dumps(quality, indent=2))
print(f"Generated {OUTPUT / 'analysis.md'} and charts in {ASSETS}")
