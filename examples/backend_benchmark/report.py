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
BACKEND_NAMES = {
    "native": "Native",
    "openblas": "OpenBLAS",
    "blis": "BLIS",
    "mkl": "MKL",
    "gpu": "GPU",
}
NAMES = {
    "gemm": "GEMM",
    "cholesky": "Cholesky",
    "lu": "LU",
    "qr": "QR",
    "svd": "SVD (U, S, VT)",
    "symmetric_eigen": "Symmetric eigen (values + vectors)",
    "general_eigen_values": "General eigen (values only)",
}
COLORS = {
    "native": "#637a90",
    "openblas": "#d58a21",
    "blis": "#ba5835",
    "mkl": "#1c9185",
    "gpu": "#7052c1",
}


def read(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return tuple(row[field] for field in KEYS)


def indexed(name):
    result = {}
    for row in read(name):
        identity = key(row)
        if identity in result:
            raise ValueError(f"Duplicate case in {name}: {identity}")
        if row["backend"] not in BACKEND_NAMES:
            raise ValueError(f"Unknown backend in {name}: {row['backend']}")
        result[identity] = row
    return result


initial_cases = indexed("results.csv")
if not initial_cases:
    raise ValueError("results.csv contains no measurements")
initial = list(initial_cases.values())
BACKENDS = [
    backend
    for backend in BACKEND_NAMES
    if any(row["backend"] == backend for row in initial)
]
reruns = indexed("rerun_results.csv")
steady = indexed("gpu_steady_results.csv")
if (
    not reruns.keys() <= initial_cases.keys()
    or not steady.keys() <= initial_cases.keys()
):
    raise ValueError("Rerun and steady-input cases must match initial measurements")
if any(row["backend"] != "gpu" for row in steady.values()):
    raise ValueError("Steady-input measurements must use the GPU backend")
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
if not transfers:
    raise ValueError("transfers.csv contains no measurements")


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
        ["Operation"]
        + [BACKEND_NAMES[b] for b in BACKENDS]
        + ["GPU speedup over best CPU"],
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
        label=BACKEND_NAMES[b],
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
    ncol=len(handles),
    frameon=False,
)
figure.suptitle(
    "Matrix operation scaling\nGPU data resident; allocation and input restoration excluded",
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


paragraph(
    f"**Matrix operation benchmark analysis — {metadata.get('date', DATA.name)}**"
)
paragraph(
    f"[Reproduction instructions]({Path(os.path.relpath(ROOT / 'README.md', OUTPUT)).as_posix()}) · "
    f"[Recorded results]({Path(os.path.relpath(DATA / 'final_results.csv', OUTPUT)).as_posix()}) · "
    f"[Environment metadata]({Path(os.path.relpath(DATA / 'metadata.json', OUTPUT)).as_posix()})"
)
paragraph(
    f"The measured backends are {', '.join(BACKEND_NAMES[b] for b in BACKENDS)}. They are called from the same analysis executable linked to the tested library. BLIS supplies CBLAS operations; decompositions use its paired LAPACKE implementation. Provider names come from the recorded CSV labels, not assumptions about the host running this report."
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
    f"Current CTest runs pin each process to one CPU and set MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and BLIS_NUM_THREADS=1. The recorded CPU environment is {json.dumps(metadata.get('cpu_environment', dict()), sort_keys=True)}. Consult the recorded compiler and CPU configuration before comparing vectorization or thread counts."
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
                "All measured backends; float and double",
            ],
            [
                "GEMM, NT/TN/TT",
                "Square 64, 256, 512, 1024",
                "All measured backends; float and double",
            ],
            [
                "Cholesky, LU, QR",
                "Square 16,32,64,128,256,512,1024",
                "All measured backends; float and double",
            ],
            [
                "SVD",
                "Square 16,32,64,128,256,512,1024",
                "All measured backends through 128; vendor CPU + GPU above 128",
            ],
            [
                "Symmetric eigen",
                "Square 8,16,32,64,96,128,256,512,1024",
                "All measured backends through 128; vendor CPU + GPU above 128",
            ],
            [
                "General eigenvalues",
                "Square 8,16,32,64,96,128",
                "Available CPU backends; no GPU implementation exposed",
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
    "CPU inputs are restored before each call, outside timing. Overwritten GPU inputs are restored with an untimed device-to-device copy and completion wait. Current steady-input GPU runs reuse GEMM and SVD inputs while restoring QR and eigen inputs. All paths reuse allocated buffers. Historical revisions may have different restoration contracts; consult their recorded metadata."
)
paragraph(
    "Matrices use fixed mt19937 seeds 42 (A) and 43 (B). GEMM/QR/SVD use uniform random values in [-1,1]. Cholesky uses a random symmetric, strictly diagonally dominant SPD matrix (diagonal shifted by 2n); LU and general eigen use random matrices with the same diagonal shift. Symmetric eigen uses a random symmetric matrix. Shapes and inputs match between backends within this harness. The Cholesky construction and untimed SVD restoration differ from the stock benchmarks, so the raw timings from the two harnesses should not be combined."
)
paragraph(
    f"Initial cases use five repetitions of at least 50 ms each. {len(reruns)} cases have longer-sample measurements with 200 ms warm-up and seven repetitions of at least 200 ms. {len(steady)} GPU cases have steady-input measurements with seven repetitions of at least 100 ms. final_results.csv records the selected cohort, target duration and repetition count. Means, medians and sample standard deviations describe this run, not hardware-independent speedups."
)
paragraph(
    f"There are **{len(rows)} operation cases**, **{sum(row['measurement'] == 'transfer_only' for row in transfers)} transfer-only cases** and **{sum(row['measurement'] == 'gemm_end_to_end' for row in transfers)} directly measured upload–GEMM–download cases**. Final operation CV: median **{statistics.median(cv):.2f}%**, maximum **{max(cv):.2f}%**, cases over 10%: **{sum(value > 10 for value in cv)}**. These timing statistics are not an exhaustive accuracy or conditioning study."
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
    "Compare GEMM providers at the application's actual precision and dimensions. The tables include rectangular and transpose cases as well as square matrices. GPU transpose flags pass through cuBLAS with swapped row-major operands, without explicitly materializing transposes in the library."
)
paragraph(
    "For Cholesky and LU, compare the measured provider rows and check pivoting contracts. Native LU can be built without pivoting, while vendor GETRF paths pivot. BLIS itself supplies BLAS; its paired LAPACKE library determines factorization behavior and performance."
)
paragraph(
    "The current QR harness times apply_r followed by apply_q on the same packed input and tau, restoring the input outside timing. It does not extract a separate R. Historical data may use an earlier QR contract; consult the recorded library revision. A factorization-only GEQRF timing is a different workload."
)
paragraph(
    "SVD computes singular vectors as well as singular values. CPU adapters call GESVD; the BLIS row measures the selected LAPACKE companion as well as BLIS. Compare each precision and matrix size directly rather than inferring a universal provider ranking."
)
paragraph(
    "Symmetric eigen includes eigenvectors. Compare its size-dependent rows separately from general nonsymmetric eigen, which remains a CPU operation because this library does not expose that GPU specialization."
)
paragraph(
    "Treat precision as part of backend selection. GPU FP32/FP64 throughput depends on the architecture; compare measured float and double rows independently. Clocks, driver scheduling and vendor algorithm selection can also affect size-dependent results; this harness does not isolate each cause."
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
        ["Precision", "Operation"] + [BACKEND_NAMES[b] for b in BACKENDS],
        workspace_records,
    )
)
paragraph(
    "These are public workspace_size() results only. Input/output buffers, retained handles, vendor internal allocations and total peak VRAM are excluded. Plan capacity using the actual workspace query. A small or zero caller workspace does not prove zero internal vendor memory use."
)
paragraph("**Practical choices supported by this run**")
paragraph(
    "Choose a backend using the measured operation, precision, size and algorithm contract. For host-resident GEMM, consult the directly measured transfer-inclusive rows. BLIS and OpenBLAS are alternative providers for BLAS_LAPACKE; record their LAPACKE companions too. Avoid one unconditional threshold for every operation."
)
paragraph(
    "The next performance work should target real application sizes and residency patterns, then CPU thread scaling, batched small matrices and GPU memory reuse. Repeated workspace/layout queries and conversions visible in the current GPU adapters are profiling candidates; these measurements alone do not establish how much speedup removing them would deliver. Any multicore CPU comparison needs a deliberate new affinity/thread policy and separate stability validation because this report uses the validated one-core workaround."
)
paragraph("**Validation, limits and reproduction**")
paragraph(
    "Report generation reads CSV files and does not rerun numerical validation. Record the library revision, provider versions and build/test results with each measurement set. Successful timing samples alone do not establish numerical residuals, convergence on other inputs or correctness at every matrix size."
)
paragraph(
    f"This is one recorded host and configuration with fixed random seeds and warm inputs. Cases use the provider order {'/'.join(BACKEND_NAMES[b] for b in BACKENDS)}. Repetition averages do not establish universal statistical significance. Power, cold-start, concurrent-stream, batched, full multicore and exhaustive numerical-accuracy studies are outside this harness."
)
paragraph(
    "Consult the recorded runtime and test logs for failures and workarounds on the measured host. This report does not infer the cause of machine-specific failures."
)
paragraph(
    "Build with LINALG_BUILD_EXAMPLES=ON, BLAS or BLIS with LAPACKE, MKL and cuBLAS enabled. CMake links LinearAlgebra::LinearAlgebra directly. The README describes configuration, CTest and report regeneration. Fresh CSV files belong in the build tree. The CUDA example uses CMake; Bazel supports the CPU benchmark sources only."
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
            ]
            + [f"{BACKEND_NAMES[b]} µs" for b in BACKENDS],
            records,
        )
    )

markdown = "\n\n".join(sections) + "\n"
(OUTPUT / "analysis.md").write_text(markdown, encoding="utf-8", newline="\n")
print(json.dumps(quality, indent=2))
print(f"Generated {OUTPUT / 'analysis.md'} and charts in {ASSETS}")
