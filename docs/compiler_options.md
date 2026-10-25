# Compiler and platform options

Investigation date: 20 October 2026. Source revision: `fa2b776`.

This report describes the compiler selection, compile flags, linker flags, and
platform branches implemented by this repository. CMake is the primary build
system; Bazel differences are recorded separately. A configured branch is not a
claim that the corresponding platform works.

Evidence comes from source inspection, the checked-in CI workflow, existing local
compile commands, four fresh macOS configurations, and the compiler documentation
linked alongside the relevant explanations. No build configuration or source code
was changed for this investigation. No full project builds or runtime tests were
run; CMake performed its normal configure-time compiler probes.

## Compiler selection and version gates

The standalone project enables both C and C++. An embedding parent must enable
the languages its helpers require. Direct CMake configuration selects compilers
through `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER`, or through `CC`/`CXX` when a
new build tree is initialized. Use separate build directories for toolchains.

`Scripts/setup.py` explicitly defaults to `clang`/`clang++`, Ninja, and Release.
Those explicit compiler arguments take precedence over relying on `CC`/`CXX`.
On macOS, the executable resolved from `PATH` determines whether this means
AppleClang or upstream/Homebrew Clang.

| Compiler family | CMake ID | First version gate | Additional gate on a fresh configure | Effective numeric floor |
|---|---|---|---|---|
| GCC | `GNU` | 8.0 | 7.0 | 8.0 |
| LLVM Clang, including clang-cl | `Clang` | 7.0 | 5.0 | 7.0 |
| AppleClang | `AppleClang` | 11.0 | 9.0 | 11.0 |
| Microsoft C/C++ | `MSVC` | 19.10 | 19.14 | 19.14 |
| Intel classic ICC | `Intel` | 19.0 | 18.0 | 19.0 |
| IntelLLVM, PGI/NVHPC, other IDs | No dedicated complete policy | No matching numeric gate | Unknown-ID warning | Not established |

These are project checks, not demonstrated compatibility guarantees. Newer
language modes, dependencies, CUDA releases, and analysis tools can require newer
compilers. No equivalent explicit C compiler version floor is enforced. The
additional validation module caches its entry marker before completing its
checks, so a failed/repeated configure may not repeat the same validation.

Sources: [initial checks](../Cmake/compiler_checks.cmake),
[additional checks](../Cmake/checks.cmake),
[setup selection](../Scripts/setup.py), [project setup](../CMakeLists.txt).

### Setup script compiler and generator tokens

Run setup commands from `Scripts/`, as the helper derives the source directory
from its working directory.

| Token | Selection |
|---|---|
| `clang`, `clang-18`, or a path to the Clang C driver | Sets the C compiler and derives the C++ name by replacing `clang` with `clang++`. |
| `gcc`, `gcc-13`, `g++`, or a GCC driver path | Derives the matching C/C++ driver pair by string replacement. |
| `clang-cl` with Ninja | Sets both compilers to `clang-cl`. |
| `vs22.clang-cl` | Selects Visual Studio 2022 and the `ClangCL` generator toolset. Order matters. |
| `vs17`, `vs19`, `vs22`, `vs26` | On Windows, selects the corresponding Visual Studio generator. Without an earlier explicit compiler token, compiler arguments are cleared so the generator selects its compiler. |
| `ninja` | Selects Ninja; also selects a build directory incorporating feature suffixes. |
| `xcode` | Selects Xcode on macOS when available; otherwise falls back to Ninja. |

Compiler detection in this script is based on token strings. Supply `clang`, not
`clang++`, to its Clang setter: supplying a C++ driver can generate a malformed
C++ executable name. Prefer direct CMake arguments for unusual driver names,
cross toolchains, or independently selected C and C++ compilers. There is no
dedicated Intel, Emscripten, or CUDA-host-compiler selector in this helper.

Visual Studio generator availability depends on the installed CMake and Visual
Studio versions. Listing a token does not guarantee that its generator exists.

## How the final command is assembled

The effective compile command combines several independently configured layers:

1. CMake/toolchain defaults and initial environment flags.
2. Directory flags such as `CMAKE_CXX_FLAGS`.
3. Configuration flags such as `CMAKE_CXX_FLAGS_RELEASE`.
4. Target properties and options, including the language standard and SIMD flags.
5. Usage requirements inherited from linked targets and per-source definitions.

CMake puts language-wide flags before configuration flags, and those before
`target_compile_options`. Consequently, the order of competing optimization
options matters. A cache entry alone does not expose target options, inherited
options, or every directory-scope modification. Inspect generated commands as
well as the printed summary. See [CMake flag ordering](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS.html).

The `LinearAlgebra` target requires the selected standard and disables language
extensions. Test targets inherit `CMAKE_CXX_STANDARD` but do not explicitly disable
extensions: the inspected local library command uses `-std=c++17`, while its unit
test command uses `-std=gnu++17`. The exported library target does not publish an
`INTERFACE_COMPILE_FEATURES` requirement for the selected C++ standard.

Sources: [target configuration](../CMakeLists.txt),
[unit test configuration](../Testing/Cxx/CMakeLists.txt).

## Platform-specific compile and link flags

### Linux and other non-Apple Unix with GCC or Clang

| Scope or condition | Implemented options | Purpose |
|---|---|---|
| C and C++, when individual probes pass | `-fPIC`, `-pipe` | Position-independent code; compiler-stage pipes. |
| Clang, when the probe passes | `-fcolor-diagnostics` | Colored diagnostics. |
| GCC, when the probe passes | `-fdiagnostics-color=always` | Colored diagnostics. |
| Library target, non-MSVC-style branch | `-include cstdlib` | Force-includes the C++ standard library header. |
| Enabled x86 packet implementation | ISA options listed below | Enables the selected instruction set. |

The C and C++ probes are independent. The helper does not enable `-Wall`,
`-Wextra`, `-Wpedantic`, `-Werror`, `-march=native`, or `-ffast-math` as a general
CMake policy. Bazel has a different warning policy. The force-include option is
PRIVATE to the library; SIMD options are PUBLIC and propagate to consumers.

Source: [platform helpers](../Cmake/platform.cmake).

### macOS: AppleClang and upstream LLVM Clang

The library receives `-include cstdlib`, the selected language mode, and the SIMD
selection. Shared-library PIC and architecture flags can be supplied by CMake;
local arm64 commands contain `-fPIC` and `-arch arm64`.

Every Apple-platform configuration also adds these directory-level link options,
including when AppleClang is selected:

- `-L<prefix>/lib/c++` and `-L<prefix>/lib`.
- `-Wl,-rpath,<prefix>/lib/c++` and `-Wl,-rpath,<prefix>/lib`.

The prefix comes from `PROJECT_LLVM_INSTALL_PREFIX`, then an existing Homebrew
LLVM installation under `/opt/homebrew` or `/usr/local`. If neither exists, the
helper still falls back to `/opt/homebrew/opt/llvm`. This is not evidence that the
selected compiler uses that LLVM installation or its C++ runtime. The helper does
not explicitly select `-stdlib=libc++`.

`CMAKE_OSX_ARCHITECTURES`, `CMAKE_OSX_SYSROOT`, and
`CMAKE_OSX_DEPLOYMENT_TARGET` remain CMake/toolchain controls. The SIMD default
uses one `CMAKE_SYSTEM_PROCESSOR` value; it does not model both slices of a
universal x86_64/arm64 build.

Source: [Apple platform branch](../Cmake/platform.cmake).

### Windows: MSVC, clang-cl, and GNU-style Clang

Compiler identity, target ABI, and accepted command-line syntax are separate
properties. CMake's `MSVC` condition can also apply to Clang targeting the Microsoft
ABI. This repository therefore checks `CMAKE_CXX_COMPILER_FRONTEND_VARIANT` in
several branches. Its `GNU`/`MSVC` values describe the expected command-line
syntax. See [CMake frontend variants](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_FRONTEND_VARIANT.html).

| Condition | Options and definitions configured by this project |
|---|---|
| MSVC or MSVC-style Clang, excluding GNU-style Clang | C++: `/Zc:__cplusplus`. Both C and C++ flag strings: `/wd4244 /wd4267 /wd4715 /wd4018 /bigobj /Zi /utf-8 /EHsc`. |
| Same branch with `CMAKE_CXX_MP_FLAG=ON` | `/MP<N>` for C and C++; `<N>` comes from `CMAKE_CXX_MP_NUM_PROCESSORS`, initially `NUMBER_OF_PROCESSORS`. Skipped for Clang with Ninja. |
| Same branch, library target | `/WX`, excluding CUDA translation units. |
| Windows excluding MinGW | `_CRT_SECURE_NO_DEPRECATE`, `_CRT_NONSTDC_NO_DEPRECATE`, `_CRT_SECURE_NO_WARNINGS`, `_SCL_SECURE_NO_DEPRECATE`, `_SCL_SECURE_NO_WARNINGS`. |
| Any Windows compiler with CMake ID `Clang` | C/C++: `-fcolor-diagnostics -Wno-unused-command-line-argument -Wno-deprecated-declarations`; C additionally gets `-Wno-strict-prototypes`. |
| Windows Clang with Ninja and `NOT MSVC` | Adds `-gcodeview`. This precise condition may exclude GNU-style Clang using the Microsoft ABI. |
| GNU-style Clang excluded from the MSVC-style branch | Receives the library's `-include cstdlib` path instead of `/WX`. |
| MSVC-style non-coverage link | Executable/shared linker flags append `/DEBUG:FULL`. |
| GNU-style Clang with `MSVC` true, non-coverage link | Uses `-Wl,/DEBUG:FULL`, avoiding a bare MSVC argument in the GNU driver. |

`/Zc:__cplusplus` corrects the reported language-version macro; it does not select
C++20. `/EHsc` selects C++ exception handling assumptions. `/WX` treats enabled
warnings as errors; the commented-out `/W4` code is not active.

`LINALG_DISPLAY_WIN32_WARNINGS` controls a C4244 suppression pragma around the
native multiplication implementation. The platform helper still globally adds
`/wd4244`; enabling this switch does not remove that global suppression.

The dependency wrapper temporarily sets `MSVC=FALSE` while configuring selected
dependencies for Windows GNU-style Clang. This avoids their MSVC argument paths;
it does not change the compiler ABI or CUDA host compiler.

Sources: [platform flags](../Cmake/platform.cmake),
[link/debug and BLAS source definitions](../CMakeLists.txt),
[dependency wrapper](../Cmake/LinearAlgebraDependencies.cmake),
[native multiplication pragmas](../src/matrix_operation/cpu/native/multiplication_native.cpp).

### MinGW, Cygwin, WebAssembly, and legacy branches

These branches exist in source but have no dedicated job in the current CI file.
Treat them as unverified implementation paths.

| Platform/compiler condition | Implemented behavior |
|---|---|
| GCC and `WIN32` | Adds `-mwin32` to C/C++; links `gdi32`. The source comment calls this Cygwin, but the actual condition is broader. |
| `MINGW`, within the GCC branch | Adds `-mthreads` to C/C++ and executable/shared/module linker flags. |
| Emscripten | Adds `-fwasm-exceptions` to C/C++ and link flags; link flags also include `-sEXCEPTION_STACK_TRACES=1`. |
| Emscripten with `PROJECT_WEBASSEMBLY_THREADS` | Adds `-pthread -Wno-pthreads-mem-growth` to C/C++, and `-pthread` to link flags. |
| Emscripten with `PROJECT_WEBASSEMBLY_64_BIT` | Adds `-sMEMORY64=1` to compile and link flags. |
| SunOS, non-GCC | Searches for and links `Crun` and `Cstd`. |
| GCC/SunOS with `DART_ROOT` and `BUILD_TESTING` | Adds `-Wno-unknown-pragmas`. |
| Non-GCC with `CMAKE_ANSI_CFLAGS` | Appends that value to C flags. |
| Non-GCC on OSF1 | C++: `-timplicit_local -no_implicit_include`. |
| Non-GCC on AIX | C++: `-qrtti=all`; executable/shared/module link flags: `-bhalt:5`. |
| Non-GCC on HP-UX | C/C++: `+W2111 +W2236 +W4276`. |
| Unix compiler ID matching `Intel` | Includes `TestNO_ICC_IDYNAMIC_NEEDED.cmake`, probes the need for `-i_dynamic`, and conditionally appends it to C++. The included file is absent from `Cmake/`. |
| Exact compiler ID `PGI` | C++: `--diag_suppress=236 --diag_suppress=381`. This does not explicitly match the `NVHPC` ID. |

When a target called `XSigmaPlatform` exists, the WebAssembly branch also attaches
corresponding interface options to it. Standalone LinearAlgebra does not create
that target. Other generic Unix probes can still run on these platforms, and the
compiler/standard/dependency requirements above still apply.

Source: [all platform branches](../Cmake/platform.cmake).

## SIMD and CPU architecture

`LINALG_ENABLE_VECTORIZATION` defaults to ON. The default
`LINALG_VECTORIZATION_ISA` is `avx2` for x86_64/AMD64/i386/i686, `neon` for
arm64/aarch64, and `scalar` otherwise. There is no runtime ISA selection here.

The bundled `Scalars::Scalars` target supplies the selected backend's definitions
and compiler flags. `sse2` maps to Scalars' `sse`; `avx`, `avx2`, `avx512`, and
`neon` map directly. `scalar` and disabled vectorization select Scalars' `no`
backend and the scalar algorithm loops. The selected ISA is checked against
the target architecture during configuration.

ISA options propagate to consumers through `Scalars::Scalars`; deploy binaries
on CPUs that support that ISA. There is no runtime ISA dispatch.

Sources: [SIMD selection](../Cmake/platform.cmake),
[Scalars SIMD interface](../ThirdParty/Scalars/include/backend/simd.h).

## Build modes, LTO, linkers, and instrumentation

### Language and optimization modes

| Control | Default and effect |
|---|---|
| `LINALG_CXX_STANDARD` | `20`; accepts `20` or `23`. Applies to the library's required C++ standard. |
| `CMAKE_BUILD_TYPE` | Defaults to `Release` if empty. `build_type.cmake` does not add its own optimization flags or normalize existing names. |
| `CMAKE_CXX_FLAGS_<CONFIG>` and C equivalents | Toolchain/CMake defaults plus user overrides. Local Clang values are Debug `-g`, Release `-O3 -DNDEBUG`, RelWithDebInfo `-O2 -g -DNDEBUG`, and MinSizeRel `-Os -DNDEBUG`. These are local observations, not universal defaults. |
| `CMAKE_CONFIGURATION_TYPES` / `--config` | Visual Studio and Xcode select their actual build configuration at build time; a printed `CMAKE_BUILD_TYPE` alone does not establish the selected configuration. |

The setup helper supports `release`, `debug`, and `relwithdebinfo` and forces Debug
when sanitizer, Valgrind, or coverage is selected. Direct CMake does not enforce
that choice. The helper uses Python `capitalize()`, producing `Relwithdebinfo`
instead of canonical `RelWithDebInfo`; this deserves verification with
case-sensitive multi-config builds.

Sources: [language settings](../CMakeLists.txt),
[build type helper](../Cmake/build_type.cmake), [setup](../Scripts/setup.py).
See [single-config build types](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)
and [configuration flags](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS_CONFIG.html).

### Link-time optimization and linker selection

`LINALG_LTO_MODE` defaults to `off`, because this project requests the helper's
`SUPPORT` profile. This remains the default for Release.

| Mode | Implementation |
|---|---|
| `off` | Adds no project LTO options. |
| `auto` | Selects ThinLTO for Clang with GNU-style linking, otherwise IPO. |
| `thin` | Probes `-flto=thin`, then `/clang:-flto=thin` for the MSVC-link case; on failure, attempts IPO. |
| `full` | Native MSVC: `/GL` for compilation and `/LTCG` for linking. Otherwise probes `-flto`, then falls back to IPO. |
| `ipo` | Checks C++ IPO support, then sets `INTERPROCEDURAL_OPTIMIZATION` on the target. |

LTO is skipped when coverage or sanitizers are enabled. Unsupported modes and
failed probes can warn or fall back; the requested cache mode does not always
equal the effective mode. ThinLTO also needs an appropriate link toolchain, as
explained by the [Clang ThinLTO documentation](https://clang.llvm.org/docs/ThinLTO.html).

`LINALG_LINKER_CHOICE` accepts `default`, `lld`, `mold`, `gold`, and `lld-link`.
Missing explicit choices warn and fall back. Automatic Linux Clang search order
is `ld.lld`, `mold`, `ld.gold` when LTO is requested, and `mold`, `ld.gold`,
`ld.lld` otherwise. Linux GCC uses the latter order. macOS first searches for
`ld64.lld`, then keeps the system linker. Windows GNU-style paths search
`ld.lld`/`lld`; MSVC-style paths generally retain their existing linker unless
handled explicitly.

Selected GNU-style linker options are added as PUBLIC `-fuse-ld=<name>` usage
requirements. The discovered absolute path is reduced to a name. MSVC's explicit
`lld-link` handling sets `CMAKE_LINKER`. For MSVC-style Clang the helper returns
without applying a discovered linker in `_xsigma_apply_fuse`; do not assume that
its printed choice changed the actual linker.

Source: [LTO and linker helper](../Cmake/lto.cmake).

### Coverage, sanitizers, and developer tools

| Control / compiler | Options or behavior |
|---|---|
| `LINALG_ENABLE_COVERAGE`, GCC | C/C++: `--coverage -g -O0 -fprofile-arcs -ftest-coverage`; executable/shared/module link flags: `--coverage`. |
| Coverage, GNU-style Clang/AppleClang | C/C++: `-g -O0 -fprofile-instr-generate -fcoverage-mapping`; executable/shared/module link flags: `-fprofile-instr-generate`. |
| Coverage, MSVC-style Clang | C/C++: `/Zi /Od -fprofile-instr-generate -fcoverage-mapping`; matching profile-generation link flag. |
| Coverage, native MSVC | Release compile flags append `/Zi`; Release executable/shared links append `/DEBUG /OPT:REF /OPT:ICF`; global links append `/DEBUG:FULL /INCREMENTAL:NO`. Uses PDBs for external coverage tooling. |
| `LINALG_ENABLE_SANITIZER`, GCC/Clang | C/C++: `-O1 -g -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize=<type>`; executable/shared links add `-fsanitize=<type>`. |
| Sanitizer, Clang/AppleClang | Also adds `-fsanitize-blacklist=<generated-file>`. This helper uses GNU-style flags even for the `Clang` ID with an MSVC frontend. |
| Sanitizer, native MSVC or unknown compiler | Warns and adds no sanitizer flags. |
| `LINALG_SANITIZER_TYPE` | Defaults to `address`; advertised choices: `address`, `undefined`, `thread`, `memory`, `leak`. Direct CMake also accepts unvalidated strings, including the CI value `address,undefined`. |
| `LINALG_ENABLE_CACHE` / `LINALG_CACHE_BACKEND` | ON / `none`. Selects a per-target compiler launcher: `ccache`, `sccache`, or `buildcache`; includes CUDA when configured. `none` launches no cache. Program overrides use `XSIGMA_<BACKEND>_PROGRAM`. |
| `LINALG_ENABLE_ICECC` | Selects `XSIGMA_ICECC_EXECUTABLE` as the C/C++ launcher. Called after the cache helper, so it replaces those launcher properties when found. |
| `LINALG_ENABLE_CLANGTIDY` / `LINALG_ENABLE_FIX` | Configures target clang-tidy with `-warnings-as-errors=*`, header filtering, and optional `-fix-errors -fix`. Exclude-header filtering is version-gated at LLVM 19. |
| `LINALG_ENABLE_IWYU` | Adds the project's include-what-you-use wrapper and mapping arguments to C/C++ targets. |
| Spell, cppcheck, and Valgrind | Separate analysis/test commands; they do not establish a C/C++ optimization or instruction-set policy. `cppcheck` has no CMake flag in setup.py. |

An advertised sanitizer name does not establish that every compiler/OS supports
it. GCC has no MemorySanitizer equivalent to Clang's mode, and instrumented
runtimes/dependencies matter. See [GCC instrumentation options](https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html)
and [Clang MemorySanitizer requirements](https://clang.llvm.org/docs/MemorySanitizer.html).

Fresh local generated commands confirm that coverage flags also reach loguru.
The sanitizer option is applied after Logging/loguru are configured, and loguru's
fresh generated command is not sanitizer-instrumented. A direct Release ASan
configuration contains `-O1` followed by `-O3 -DNDEBUG`; do not describe that
configuration as reliably compiling at `-O1`. These observations concern generated
commands, not successful execution of instrumented binaries.

Sources: [coverage](../Cmake/coverage.cmake), [sanitizers](../Cmake/sanitize.cmake),
[cache](../Cmake/cache.cmake), [Icecream](../Cmake/helper_macros.cmake),
[clang-tidy](../Cmake/clang_tidy.cmake), [IWYU](../Cmake/iwyu.cmake).

## CUDA and numerical backends

`LINALG_ENABLE_CUBLAS=ON` finds CUDAToolkit, enables the CUDA language, and adds
CUDA sources. The intended CUDA standard follows `LINALG_CXX_STANDARD` if unset.
The code attempts to default architectures to `native`, but does so after
`enable_language(CUDA)`, which can initialize the variable first. Set
`CMAKE_CUDA_ARCHITECTURES` explicitly when reproducible output is required.

`CMAKE_CUDA_COMPILER`, `CMAKE_CUDA_HOST_COMPILER`, and CUDA architecture settings
must be selected before CUDA language initialization. The toolkit's supported
host-compiler range is separate from this project's compiler floors. The CI
compile-only job uses CUDA 12.6.2 and architecture `75`, avoiding GPU-dependent
`native` discovery. Consult the matching
[NVIDIA host compiler policy](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-installation-guide-linux/index.html#host-compiler-support-policy)
and [CMake CUDA architecture behavior](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html).
The `native` architecture keyword requires CMake 3.24 or later; see
[CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html).

For that pinned Linux CUDA 12.6.2 release, NVIDIA lists GCC 6.x–13.2 and Clang
7.x–18.0 for x86_64 and Arm64 SBSA, with C++ dialects through C++20. This project's
own compiler floors still apply. The library's C++23 selector does not establish
C++23 support in that CUDA toolkit. These are version-specific requirements;
use the documentation matching the toolkit actually installed. See the
[CUDA 12.6.2 compiler and dialect tables](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-installation-guide-linux/index.html#host-compiler-support-policy).

The project excludes `/WX` and x86 SIMD compile options from CUDA translation
units. Thin/full LTO options are not similarly filtered, so mixed CUDA/LTO builds
need separate validation. Windows GNU-style Clang has additional CUDA runtime
library path handling and a `dbghelp` link-name adjustment for Logging.

BLAS, BLIS, MKL, and Accelerate are library/backend choices, not compiler
families. `LINALG_ENABLE_BLIS` enables the vendor-neutral BLAS backend.
`LINALG_BLAS_HAS_LAPACKE` records header availability and is not a comprehensive
ABI/link test. MKL configuration forces `MKL_INTERFACE=ilp64`; it does not require
the Intel compiler. Combined MKL/BLAS builds load an independent shared BLAS
provider, and the Linux deep-binding path explicitly rejects sanitizer builds.
On MSVC, BLAS source files get `lapack_complex_float=std::complex<float>` and
`lapack_complex_double=std::complex<double>` definitions.

The library publicly exports 0/1 capability macros for MKL, BLAS, LAPACKE,
Accelerate, cuBLAS, cpuinfo, Profiler, LU pivoting, vectorization, and Windows
warnings. A shared build also exports `LINALG_SHARED_DEFINE`, with
`LINALG_BUILDING_DLL` private to the library; a static build exports
`LINALG_STATIC_DEFINE`.

Sources: [backend and target setup](../CMakeLists.txt),
[isolated BLAS](../Cmake/isolated_blas.cmake).

## Bazel differs from CMake

| Area | Bazel implementation |
|---|---|
| Standard selection | `.bazelrc` sets C++20 globally and for host tools; `linear_algebra_copts()` also requests C++20 for the library. Inspect resolved actions to establish the effective option order. |
| Unix warnings | `-Wall -Wextra -Wpedantic`, plus `-include cstdlib`. |
| Windows options | `/std:<mode> /Zc:__cplusplus /EHsc /bigobj /utf-8 /wd4244 /wd4267 /wd4715 /wd4018 /WX`. Selection is by OS, rather than compiler frontend. |
| Other global include option | Adds the vendored Logging include root using `-I` or `/I`. |
| Link options | macOS: `-undefined dynamic_lookup`; other non-Windows platforms: `-lpthread -ldl`; Windows: no shared-helper additions. |
| Setup default | Clang, Debug (`-c dbg`), unlike the CMake setup helper's Release default. |
| Setup release modes | `release` maps to `-c opt`; `relwithdebinfo` adds `--copt=-g`. |
| Setup compiler selection | Passes `--repo_env=CC=...` and `--repo_env=CXX=...` to trigger toolchain reconfiguration. |
| Setup standard override | Adds GNU-style `--cxxopt=-std=c++N` and `--host_cxxopt=-std=c++N`, including on Windows. This requires frontend-specific review. |
| Backend parity | MKL/BLAS/LU defines exist, but package resolution is not equivalent to CMake. No corresponding complete CUDA/SIMD option path exists. |

Do not transfer CMake cache switches directly into Bazel commands or assume a
Windows GNU-style compiler can consume the OS-selected MSVC flags. This report
inspected Bazel source configuration; it did not resolve or compile Bazel actions.

Sources: [.bazelrc](../.bazelrc), [shared Bazel helpers](../bazel/xsigma.bzl),
[LinearAlgebra helpers](../bazel/linear_algebra.bzl),
[Bazel setup](../Scripts/setup_bazel.py), [Bazel target](../BUILD.bazel).

## CI coverage and local evidence

The checked-in workflow defines the following coverage. This records configured
jobs, not their current remote pass/fail status.

| Configuration | Evidence in CI source |
|---|---|
| Linux GCC and Clang, native CPU | Explicit compiler matrix, build and test. |
| macOS `clang`, native CPU | Build and test; the ordinary job does not pin a Homebrew LLVM compiler path. |
| Linux MKL; Linux/macOS BLAS and LU pivoting | Build and test jobs. Installing Clang in a job does not itself select Clang when `CC`/`CXX` are unset. |
| Linux/macOS ASan+UBSan | Debug configure with `address,undefined`, build and test. |
| SIMD | Disabled/scalar/SSE2/AVX/AVX2/AVX512 on Linux, NEON on macOS. Runtime tests are skipped when required CPU capabilities are unavailable. |
| CUDA | Linux 12.6.2 container, architecture `75`, compile-only job. Runtime job is conditional on a repository variable and a self-hosted GPU runner. |
| Coverage | Linux GCC, macOS Homebrew LLVM Clang, and Windows native MSVC/OpenCppCoverage. |
| Installed shared/static packages and Bazel | Linux/macOS jobs. |
| Windows clang-cl/GNU-style Clang/MinGW, other legacy platforms | No dedicated matrix job. |

Fresh local evidence at this revision: Darwin arm64, CMake 4.1.2, Ninja,
Homebrew Clang 22.1.2 and AppleClang 21.0.0. All four configurations completed
configure/generate with testing disabled, LTO off, cache launchers disabled, and
compile-command export enabled:

| Local configuration | Generated command observation |
|---|---|
| LLVM Clang Release | `-O3 -DNDEBUG -std=c++17 -fPIC -include cstdlib`, `LINALG_SIMD_NEON`. |
| AppleClang Release | Same selected library flags and NEON definition. |
| LLVM Clang Debug + coverage | Profile-generation/mapping flags appear in library and loguru compile commands. |
| LLVM Clang Release + ASan | `-O1 -g -fsanitize=address` precedes `-O3 -DNDEBUG`; loguru lacks sanitizer instrumentation. |

The existing `build_ninja` commands additionally show that the current local BLAS
configuration uses OpenBLAS headers with `LINALG_HAS_BLAS=1`,
`LINALG_HAS_LAPACKE=1`, and `LINALG_HAS_ACCELERATE=0`. These describe that build
artifact, not a universal platform default.

Source: [CI workflow](../.github/workflows/ci.yml).

## Configuration gaps requiring follow-up

These are investigation findings; no corrective changes are included.

1. **Compiler policy is duplicated.** Numeric floors disagree between modules;
   cached validation can skip work after a failed configure.
2. **SIMD portability is incomplete.** x86 defaults require AVX2/FMA, native MSVC
   does not receive `/arch:`, and invalid/cross-architecture choices are not
   validated at configuration time. PUBLIC ISA flags constrain consumers.
3. **Legacy Intel configuration references a missing file.** The branch matches
   Unix compiler IDs containing `Intel`, including IntelLLVM.
4. **Windows frontend handling needs broader validation.** Some paths distinguish
   GNU/MSVC syntax while sanitizer and Bazel paths do not. clang-cl and MinGW lack
   dedicated CI jobs.
5. **Requested instrumentation and actual commands can differ.** Release flags can
   override sanitizer optimization; coverage and sanitizer dependency scope
   differs. The LTO/sccache check examines `RULE_LAUNCH_COMPILE` before the cache
   helper sets `CXX_COMPILER_LAUNCHER`, so it does not inspect the final launcher.
6. **LTO/linker requests need effective-state reporting.** Skipped LTO still
   influences linker search, mixed CUDA options are not fully filtered, and
   exported linker preferences can constrain consumers. The printed linker
   choice is not definitive evidence of the invoked linker.
7. **macOS toolchain paths are unconditional.** Homebrew LLVM search paths/rpaths
   are added even for AppleClang or a missing fallback prefix.
8. **Bazel and CMake have different standards and warning policies.** Bazel also
   contains competing global/target language-standard selections.
9. **The declared CMake minimum is insufficient for every branch.** The project
   declares 3.16, while the helper uses `find_package(... GLOBAL)`, and CUDA's
   `native` keyword requires 3.24. The CUDA CI job already installs CMake >=3.24.
   The [CMake find_package reference](https://cmake.org/cmake/help/latest/command/find_package.html)
   also dates the `GLOBAL` keyword to 3.24.
10. **The summary is an inventory, not the full compiler command.** It does not
    expand all target/inherited options or every undeclared platform input; for
    example `PROJECT_WEBASSEMBLY_THREADS`, `PROJECT_WEBASSEMBLY_64_BIT`, and
    `CMAKE_ANSI_CFLAGS` are read by platform helpers but not listed in the summary.

The earlier [CMake diagnostic](cmake_diagnostic.md) contains related findings at
its own recorded revision. Its historical test results were not rerun here.

## Inspecting or selecting a configuration

Use a new build directory and explicit drivers for reproducible investigations.
The following are selection examples, not additional configurations tested by
this report. Commands run from the repository root unless stated otherwise.

```sh
# Linux GCC, explicit CPU ISA
cmake -S . -B build-gcc -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DLINALG_VECTORIZATION_ISA=sse2 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# macOS system AppleClang
cmake -S . -B build-apple -G Ninja -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Linux CUDA: choose a host compiler supported by the selected toolkit
cmake -S . -B build-cuda -G Ninja -DLINALG_ENABLE_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release

# Commands supported by every generator
cmake --build build-gcc --verbose
cmake -LAH -N build-gcc

# Setup helper: run from Scripts/
cd Scripts
python3 setup.py config.build.test.gcc.release.sse2
```

For Windows, run from a developer environment with the selected compiler, SDK,
and generator installed:

```powershell
cmake -S . -B build-msvc -G "Visual Studio 17 2022" -A x64
cmake --build build-msvc --config Release --verbose
cmake -S . -B build-clang-cl -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

`compile_commands.json` is useful with Ninja/Makefile generators; use verbose
build output for Visual Studio/Xcode. For link options, inspect the actual link
command as well. In particular, neither a raw `CMAKE_CXX_FLAGS` cache value nor
`CMAKE_LINKER` alone captures every target-level toolchain decision.
