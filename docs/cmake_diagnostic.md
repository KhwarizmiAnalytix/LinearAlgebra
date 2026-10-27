# CMake diagnostic — LinearAlgebra

Review date: 19 October 2026. Source revision: `c52ba9fb6d1ade8d7e97f0cc294b01eec2318c9f`.

## Assessment

The default build works on this machine, but several advertised configurations do not. The most consequential project-owned defects affect embedding, BLAS provider selection, test switches, and static sanitizer package requirements. Optional developer tools also have failure paths that normal CI does not exercise.

The review found **20 prioritized issues**. Findings distinguish reproduced behavior from conclusions based on source inspection. A passing native build does not validate CUDA, MKL, Linux, or Windows behavior.

This is a diagnostic only. This review changed no build scripts or implementation files. All experimental projects, builds, installations, and logs were created under:

```text
/private/tmp/linalg-cmake-review-tw0n6fa2
```

## Scope and method

The review is limited to project-owned files. **Every repository under `ThirdParty/` is excluded**, including Logging and all nested dependencies. Dependency names appear only where needed to explain this project's own configuration, target interfaces, or observed integration behavior. Internal dependency defects, standalone dependency checks, vendor comparisons, and vendor lint results are excluded from the findings and counts.

The scoped source inventory contains **29 files / 4,075 lines**:

| Kind | Files |
|---|---:|
| `CMakeLists.txt` | 5 |
| Other `.cmake` files | 18 |
| Other `.txt` files | 4 |
| `.cmake.in` / `.txt.in` templates | 2 |

The appendix lists every in-scope file. Additional project-owned context includes generated-header templates, `.gitmodules`, CMake lint settings, README build instructions, Python build helpers, and `.github/workflows/ci.yml`. The helper `Cmake/third_party_helpers.cmake` remains in scope because it belongs to this project; its filename does not make it a vendored repository.

Generated build trees, Bazel output links, local Python/MKL environments, and tool caches are excluded from the source inventory. Fresh generated caches, compile commands, exported LinearAlgebra targets, and CTest/install files were inspected as evidence for project-owned behavior. Builds necessarily resolve dependencies, but successful dependency compilation is not presented as a review of their internals.

The build/test observations below come from the original audit. This scope revision reran lint only on project-owned CMake scripts and did not repeat the build matrix.

Local environment: Darwin arm64; CMake 4.1.2; Ninja; Homebrew Clang 22.1.2; AppleClang also tested. The default search found external fmt 12.1.0, cpuinfo, and Profiler. A separate build disabled cpuinfo/Profiler discovery and external third-party packages to check the vendored path.

At the end of the audit, eight BLAS/MKL implementation files had working-tree edits that appeared during the review; these were preserved. All 29 in-scope CMake/text/template files still matched their original content hashes. Build/test results describe the source compiled at each check, rather than certifying concurrently edited implementation files.

## Build architecture

The root project enables C and C++ only when built standalone. It declares `LINALG_*` options, loads shared helpers, resolves optional CPU/GPU packages, and unconditionally adds Logging. It discovers library sources through recursive globs, creates `LinearAlgebra` and its namespaced alias, attaches flags and dependencies, then defines installation/export rules and optional test, benchmark, and profiling subdirectories.

The root project disables Logging's tests, benchmarks, and examples at its integration boundary. GoogleTest is configured separately for LinearAlgebra tests. Google Benchmark is resolved only when benchmarks are enabled. The internal build setup of those dependencies is outside this review.

The installed library exports `LinearAlgebra::LinearAlgebra` and resolves Logging separately through `find_dependency(Logging)`. Optional dependencies found during the build can also become requirements of the installed package. Logging is excluded from the parent install traversal, so installation intentionally requires a separate Logging installation.

## Execution results

| Check | Result | Evidence / limitation |
|---|---|---|
| Default configure/build/test | Pass | Both CTest entries pass; GoogleTest lists 69 cases |
| Static configure/build/test | Pass | Default LOGURU backend |
| AppleClang configure/build | Pass | Default CPU configuration |
| Ninja Multi-Config Debug build/test | Pass | `ctest -C Debug`; both entries pass |
| ASan + UBSan shared build/test | Pass | Debug; `address,undefined` |
| LLVM coverage build/test | Pass | Instrumentation checked; no coverage percentage audit |
| OpenBLAS build/test | Pass | Homebrew OpenBLAS headers and library |
| Benchmarks enabled | Build passes | Unit/source-check tests pass; performance tests excluded from this run |
| Vendored build, optional packages disabled | Pass | `XSIGMA_ENABLE_EXTERNAL=OFF`; cpuinfo/Profiler disabled |
| Installed shared consumer | Pass | Logging installed separately first |
| Installed static consumer | Pass | Default LOGURU backend; separate Logging installation |
| C++-only parent project | Configure fails | C-based header probes without C enabled |
| C/C++ parent project | Configure passes | Embedded GPU source check then fails |
| Parent already defines Logging | Configure fails | Duplicate target `Logging` |
| Embedded sanitizer build | Configure fails | Suppression template looked up in parent source tree |
| Explicit Apple BLAS vendor | Build fails | Accelerate libraries mixed with OpenBLAS/LAPACKE headers |
| Installed static ASan LinearAlgebra | Consumer link fails | Missing `__asan_*` runtime symbols |
| Missing compiler-cache executable | Build fails | Launcher becomes a literal `*-NOTFOUND` command |
| Benchmark FetchContent fallback | Configure fails | Folder helper queries wrong binary directory |
| Valgrind wiring probe | Broken | CTest cannot find its checker; options absent from generated dashboard config |
| Invalid SIMD ISA | Configure passes; build fails | `typo` accepted, then `packet.h` emits `#error` |
| Reconfigure after failed validation | Incorrectly succeeds | Cached early-return marker suppresses failed required check |
| Reconfigure after removing generated sanitizer list | File not recreated | Persistent cache marker prevents regeneration |

The Valgrind probe deliberately supplied `/usr/bin/true` as a harmless executable to inspect configuration plumbing. **No actual Valgrind memory check was performed.** The benchmark FetchContent probe used `FETCHCONTENT_SOURCE_DIR_BENCHMARK` with the existing vendored source, so it exercised the fallback without downloading anything.

Logs are named after checks, such as `default-test.log`, `blas-apple-build.log`, `embedded-cxx-configure.log`, and `asan-static-consumer-build.log`. Command/result manifests are `results.json`, `install-results.json`, `probe-results.json`, `extra-results.json`, and `last-results.json`; supplementary checks have individual logs.

## Prioritized findings

Priority meanings: **P1** = fix before relying on the affected supported configuration; **P2** = functional or portability defect to schedule next; **P3** = maintenance or diagnostic quality. Priorities do not imply that the default build is broken.

### F01 — P1 — Declared CMake minimum does not cover the commands used

**References:** `CMakeLists.txt:22`; `Cmake/third_party_helpers.cmake:480–481`; `CMakeLists.txt:431–438`; `CMakeLists.txt:166–169`.

LinearAlgebra declares 3.16, but its external dependency helper uses `find_package(... GLOBAL)`, introduced in 3.24. Older CMake can interpret an unrecognized keyword as a component, changing dependency behavior instead of reliably providing the intended global imports. The CUDA branch additionally uses FindCUDAToolkit, introduced in 3.17, and `native` CUDA architectures, introduced in 3.24. C++23 target-standard support was added in 3.20.

**Evidence:** source plus versioned CMake documentation; no old CMake executable was installed for this audit. The CUDA CI job already upgrades CMake because of `GLOBAL`, acknowledging the mismatch.

**Fix:** raise the supported baseline consistently to at least 3.24, or explicitly gate newer syntax and features. Test the advertised minimum in CI. Supporting C++23 still requires a compiler/toolkit that implements it.

Documentation: [find_package GLOBAL](https://cmake.org/cmake/help/v3.24/command/find_package.html), [FindCUDAToolkit](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html), [CUDA architecture versions](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html), [CXX_STANDARD versions](https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html).

### F02 — P1 — BLAS library selection and header/capability detection can disagree

**References:** `CMakeLists.txt:271–319`, `324–338`, `352–407`.

The build finds BLAS/LAPACK first, then independently searches for `cblas.h` and `lapacke.h`. It derives Accelerate use from the header path and enables LAPACKE from header presence. Neither operation proves that those headers and symbols belong to the linked provider.

**Reproduced:** `-DLINALG_ENABLE_BLAS=ON -DBLA_VENDOR=Apple` selects Accelerate libraries, but automatic Homebrew prefix discovery selects OpenBLAS headers. `LINALG_BLAS_HAS_LAPACKE` becomes ON although no LAPACKE library is linked. Configuration succeeds, then the library fails to link with undefined `LAPACKE_*` symbols. The ordinary OpenBLAS configuration passes.

**Fix:** resolve a coherent provider target containing matching headers and libraries. Determine capabilities using compile-and-link probes against that target. Respect an explicit `BLA_VENDOR`; do not infer provider identity from an independently discovered header. Validate integer widths when selecting LP64 versus ILP64 providers. Retest explicit Accelerate on a machine that also has OpenBLAS installed.

### F03 — P2 — Testing switches do not consistently control test creation

**References:** `CMakeLists.txt:36–37`, `110–113`, `707–709`; `Testing/Cxx/CMakeLists.txt:7`, `19–25`.

LinearAlgebra ignores `BUILD_TESTING` for its own tests, unconditionally calls `enable_testing()` standalone, and unconditionally resolves GoogleTest inside its test subtree. The separately exposed `LINALG_ENABLE_GTEST` switch does not affect that behavior.

**Reproduced:** configuring with `BUILD_TESTING=OFF` still creates the suite. `LINALG_ENABLE_GTEST=OFF` still creates and links GoogleTest and defines `LINALG_HAS_GTEST=1`. Dependency-owned writes to generic cache options are excluded from this finding.

**Fix:** define how the project-specific testing option relates to `BUILD_TESTING`, preferably using standalone status and the host's setting as its default. Remove the unused GTest switch or define a coherent behavior when it is OFF. Keep dependency-specific options scoped at the project's integration boundary.

### F04 — P1 — Embedding fails in an ordinary C++-only parent

**References:** `CMakeLists.txt:26–44`; `Cmake/checks.cmake:176–178`, `277–297`.

The embedded branch omits `project()` and never enables C, but the checks use `check_include_file`, which compiles C source. A parent using `project(Host LANGUAGES CXX)` fails before creating the library.

**Reproduced:** the isolated C++-only host fails with `Unknown extension ".c" ... try_compile() works only for enabled languages`. The same host with `LANGUAGES CXX C` configures successfully.

**Fix:** use C++ header/symbol probes for this C++ library, or explicitly enable and document the additional language requirement. Add a minimal C++-only `add_subdirectory` consumer to CI.

### F05 — P1 — Existing Logging targets cannot be reused

**References:** `CMakeLists.txt:498–514`.

LinearAlgebra always configures its Logging subdirectory, even when a parent already provides `Logging::Logging`. The missing target-existence guard is in this project's own dependency integration.

**Reproduced:** a host adding Logging before LinearAlgebra fails because target `Logging` already exists. An imported Logging target is also unsupported by assumptions in the Windows-specific target-mutation block.

**Fix:** reuse `Logging::Logging` when it exists, support an installed package when requested, and add the bundled subtree only when no suitable target is available. Apply bundled-target adjustments only when this project owns that target.

### F06 — P2 — Several paths refer to the host project instead of this repository

**References:** `Testing/Cxx/CMakeLists.txt:42–44`; `Cmake/sanitize.cmake:59–68`; `Cmake/clang_tidy.cmake:46`; `Cmake/iwyu.cmake:38`, `119`, `156`; `Cmake/spell.cmake:56`; `Cmake/valgrind.cmake:94–96`.

`PROJECT_SOURCE_DIR`, `PROJECT_BINARY_DIR`, and `CMAKE_SOURCE_DIR` identify the host when embedded. They are used for project-owned assets and source scans.

**Reproduced:** the embedded GPU source test fails with `No GPU sources found`, because `SOURCE_ROOT` is the host. Enabling sanitizers in that host fails to find `Scripts/suppressions/sanitizer_ignore.txt.in` under the host source directory. The other listed paths are analogous source-inspection findings: they can select host suppressions or mappings, miss this library's headers, or write wrappers into a shared host location.

**Fix:** capture the library's source/binary roots explicitly and pass them to helpers. For module-owned assets, use a captured module directory or `CMAKE_CURRENT_FUNCTION_LIST_DIR` after settling the minimum version. Run embedded tests and analysis tools from a host with no matching `Scripts/` directory.

### F07 — P2 — Valgrind configuration does not reach the CTest dashboard

**References:** `CMakeLists.txt:36`, `194–195`, `723–724`; `Cmake/valgrind.cmake:74–108`, `131–145`.

The module populates `CMAKE_MEMORYCHECK_COMMAND` and `CTEST_MEMORYCHECK_*`, while the included CTest module generates `DartConfiguration.tcl` from `MEMORYCHECK_COMMAND`, `MEMORYCHECK_COMMAND_OPTIONS`, and `MEMORYCHECK_SUPPRESSIONS_FILE`. CTest is also included before this configuration is applied. The timeout helper reads tests only in the root directory; the actual tests live in child directories.

**Reproduced:** after both initial configure and reconfigure, the generated dashboard file still has a missing checker, empty options/suppressions, and the original global timeout. The unit test timeout remains 120 seconds, not 2,400. `ctest -T memcheck` exits with `Memory checker ... not set`.

**Fix:** choose one supported CTest memory-check workflow and configure its actual inputs before dashboard generation. Apply timeout changes in test-owning directories or recursively traverse them. Verify the generated dashboard values, then run a real memory check on Linux. Ordinary `ctest` does not become a Valgrind run just because this option is ON.

### F08 — P2 — A missing cache executable is accepted as a launcher

**References:** `Cmake/cache.cmake:43–52`.

The helper tests whether a `find_program` result is defined and nonempty. CMake's `*-NOTFOUND` result satisfies both conditions, so it is assigned to `CXX_COMPILER_LAUNCHER` and related properties. An existing NOTFOUND result also bypasses the intended retry.

**Reproduced:** a helper fixture with `XSIGMA_CCACHE_PROGRAM=XSIGMA_CCACHE_PROGRAM-NOTFOUND` configures successfully, then the build exits 127 trying to execute that literal name.

**Fix:** test the result's CMake truth value, and optionally check that an explicitly supplied path exists. Either fail clearly at configure time or leave the launcher unset with a clear message. Test an unavailable backend. The current default `ENABLE_CACHE=ON` plus backend `none` performs no caching; the summary should say so directly.

### F09 — P2 — Benchmark FetchContent fallback queries a nonexistent directory

**References:** `Cmake/LinearAlgebraDependencies.cmake:113–133`; `Cmake/third_party_helpers.cmake:124–149`.

The vendored branch uses `${CMAKE_BINARY_DIR}/ThirdParty/benchmark_build`. The FetchContent branch normally uses `_deps/benchmark-build` or an override. Both then call the folder helper with the vendored directory.

**Reproduced:** a network-free fixture forcing the FetchContent branch successfully configures the benchmark dependency, then fails with `get_property DIRECTORY scope ... requested directory was not found`.

**Fix:** use the actual benchmark binary directory or obtain it from the created target; apply folder organization only to directories that exist in the configured graph. Test both vendored and FetchContent branches. Use `CMAKE_CURRENT_BINARY_DIR` for project-owned vendored build directories to reduce host collisions.



### F10 — P2 — Static sanitizer packages do not propagate runtime link requirements

**References:** `Cmake/sanitize.cmake:74–82`; `CMakeLists.txt:653–655`; analogous coverage flag handling in `Cmake/coverage.cmake`.

Sanitizer runtime flags are directory-level `CMAKE_*_FLAGS`, not target usage requirements. They work for tests built in the configured tree, but disappear when an installed static library is consumed elsewhere.

**Reproduced:** a static ASan library builds and installs; the normal installed consumer then fails with unresolved `__asan_*` symbols. The same consumer works with the ordinary static package. Coverage has the analogous export design risk, although an installed static coverage consumer was not separately tested.

**Fix:** model instrumentation as target compile/link requirements, including the necessary runtime requirement for static consumers, or explicitly disallow installation of instrumented development packages. Test consumption from a separate process/build tree.

### F11 — P2 — Cached validation can turn a failed configure into a successful one

**References:** `Cmake/checks.cmake:13–29`, `142–149`.

`TEMP_CHECKS_CONFIGURED` is stored in the persistent cache before any validation completes. Future configure runs return immediately, even if the previous run failed. The computed validation key cannot help because the early return occurs first, and several cached probe names are generic across projects.

**Reproduced:** a fixture disabling the required Threads package fails its first configure. Running the identical command again succeeds because the validation module is skipped. This fixture isolates the helper; other dependencies in a full build may independently catch the missing package.

**Fix:** use a configure-run include guard, not a persistent marker. Let CMake cache individual probes with sufficiently specific names and inputs. Never cache successful completion before the checks succeed. Test repeated configure after an intentionally failed dependency check.

### F12 — P2 — Generated sanitizer assets are guarded by stale cache state

**References:** `Cmake/sanitize.cmake:58–68`, `86–103`.

The generated suppression list is protected by `_xsigma_sanitizer_blacklist_configured` in the cache. If that output is removed, configure does not recreate it. The GTest link-propagation marker is similarly persistent even though CMake targets and their properties are recreated on each configure. In this project's current ordering, GTest is not yet created when that propagation loop first runs.

**Reproduced:** removing only the generated sanitizer list from the audit build, then reconfiguring, returns success without restoring the file. This happened after the successful ASan build/test recorded above.

**Fix:** call `configure_file` on every relevant configure; it already avoids unnecessary output changes. Attach target requirements after the corresponding targets exist, and use per-run properties rather than cache booleans for target initialization.

### F13 — P2 — SIMD selection is incompletely validated and can overconstrain consumers

**References:** `Cmake/platform.cmake:336–360`, `383–414`; `src/common/packet.h:39–42`.

The ISA cache property offers choices but never validates the actual value. Defaults select AVX2 for all listed x86 processors and NEON for arm64. Processor-family detection does not establish runtime CPU capability. The x86 flags are PUBLIC, so they also constrain consumers and are exported. Native MSVC receives the selection macro but no corresponding `/arch:` option. macOS universal or alternate-architecture builds are not represented by one host-family choice.

**Reproduced:** an unrecognized ISA used to configure successfully and fail later in the packet header. `LINALG_CPU_BACKEND` is now rejected at configure unless it is `no`, `sse`, `avx`, `avx2`, `avx512`, `neon`, or `sve`. Default arm64 NEON builds pass. x86 runtime portability, native MSVC ISA code generation, and universal binaries were not executed locally.

**Fix:** validate names and target-architecture compatibility at configure time. Decide and document the supported minimum CPU ISA; use runtime dispatch or a portable default for broadly distributed binaries. Scope ISA flags to implementation files where possible, implement native MSVC flags, and handle multi-architecture outputs explicitly.

### F14 — P2 — Generic helper lookup can load another project's implementation

**References:** `CMakeLists.txt:49–54`, `179–187`; project-owned `Cmake/` modules.

LinearAlgebra appends its module directory and includes generic names such as `platform`, `checks`, `lto`, and `sanitize`. An embedding host can already have a same-name module earlier in `CMAKE_MODULE_PATH`; the project's own implementation is then not guaranteed to load. Generic function names and `TEMP_*` / `XSIGMA_*` state add order dependence across an embedding build.

**Evidence:** source inspection of the project's append order and name-based includes. This finding does not assess duplicate helper implementations inside dependencies, and no deliberately conflicting host-module fixture was executed.

**Fix:** load owned helpers by absolute path and prefix project-owned module/function/cache names. Keep explicit host extension points instead of relying on accidental module shadowing.

### F15 — P2 — Installation supports the tested layout but has portability gaps

**References:** `CMakeLists.txt:608`, `658–703`; `Cmake/isolated_blas.cmake:72–75`; `Cmake/isolated_blas_config.h.in`.

LinearAlgebra hardcodes `lib`, `bin`, and `include` in its own install destinations, package-config location, and installed include interface. A caller requesting `CMAKE_INSTALL_LIBDIR=lib64` does not obtain the requested library/package layout from these rules. Custom include and binary destinations are likewise not modeled through GNUInstallDirs variables.

Combined MKL/BLAS builds embed absolute provider paths in a private generated header and do not install those runtime providers. README documents that those paths must remain available. This is an explicit deployment constraint, not a newly discovered undocumented failure. Relocating such an installation requires retaining the original providers or changing the loading design.

**Evidence:** both default shared/static installed consumers pass with a separate Logging install. Custom install directories, relocation, and combined-backend deployment were not executed.

**Fix:** consistently use GNUInstallDirs and matching build/install include roots. Document the two-package installation clearly. Add custom-directory and moved-prefix consumer checks; decide whether combined-provider paths should be configurable at runtime. Introduce library VERSION/SOVERSION if binary ABI/version management is intended; the current 1.0.0 config version does not version the shared-library filename.

### F16 — P2 — The classic FindMKL fallback is incomplete

**References:** `CMakeLists.txt:220–251`; `Cmake/LinearAlgebraConfig.cmake.in:23–39`.

The fallback accepts `MKL_FOUND`, but the build attaches libraries only if target `MKL::MKL` exists. It does not translate classic variables such as `MKL_LIBRARIES` or `MKL_INCLUDE_DIRS` into target requirements. A traditional finder returning variables without that target therefore passes the found check but supplies no usable dependency. The package config also cannot recreate that build-time interface.

**Evidence:** static inspection; real MKL configuration/linking was not run on this arm64 host.

**Fix:** require the supported oneMKL imported-target contract with a clear error, or normalize the variable-based fallback into a consistent imported/interface target and export-compatible dependency resolution. Verify ILP64 compile definitions and actual libraries together. Test standalone and installed MKL consumers, and combined MKL/BLAS separately.

### F17 — P2 — LTO/linker handling has target-interface and mixed-language gaps

**References:** `Cmake/lto.cmake:106–115`, `127–174`, `185–215`, `271–285`; `CMakeLists.txt:574–580`.

The sccache check examines `RULE_LAUNCH_COMPILE`, but caching is assigned later through `CXX_COMPILER_LAUNCHER`, so the intended incompatibility check does not inspect the configured launcher. Thin/full LTO compile flags are not language-filtered on a CUDA-containing target. Their link flags are PRIVATE, which warrants a separate static archive/consumer check. The linker picker considers the requested LTO mode even when instrumentation caused LTO to be skipped.

Linker selection exports `-fuse-ld=...` PUBLIC, imposing a developer-machine linker choice on consumers. It discovers a full executable path but reduces it to a name, potentially losing the searched location. Invalid modes and unavailable explicit linkers generally warn/fall back instead of strictly honoring a requested configuration.

**Evidence:** source inspection. Default LTO is OFF and passed. Thin/full/IPO, alternate linkers, and CUDA/LTO combinations were not built in this audit.

**Fix:** compute one effective LTO state, validate mode/linker inputs, query the actual launcher property after configuration, scope flags by language/configuration, and keep implementation linker choices PRIVATE unless consumers truly require them. Validate static consumers and CUDA combinations before claiming them supported.

The CUDA architecture default is also assigned after `enable_language(CUDA)`. That command can initialize `CMAKE_CUDA_ARCHITECTURES` itself, so the later `NOT DEFINED` guard need not produce the intended `native` default. Set an intended default before enabling the language, preserve user overrides, and verify GPU-less configuration with an explicit architecture.

### F18 — P2 — Build requirements and platform flags are not consistently modeled on targets

**References:** `CMakeLists.txt:571`, `597–600`; `Cmake/platform.cmake:132–135`, `158–210`, `252–275`; `Cmake/coverage.cmake`; `Cmake/sanitize.cmake`.

Setting `CXX_STANDARD` compiles the library in that dialect but does not publish a language-level requirement. The exported LinearAlgebra target has no `INTERFACE_COMPILE_FEATURES`. A small C++11 trace consumer actually compiled locally; this observation should not be misreported as proof that all public headers or supported configurations accept C++11. The documented minimum and exported contract should agree.

macOS adds Homebrew LLVM library search paths and rpaths even with AppleClang and even if the chosen default prefix does not exist. These directory-level link options are inherited by nested dependencies, extending the scope of this project's toolchain policy. Coverage is likewise applied before dependency configuration: compile commands show Logging, loguru, and GoogleTest instrumented despite comments describing module-only coverage. Sanitizer ordering differs: Logging/loguru are already configured and are not instrumented by the parent's sanitizer option.

The Intel compiler branch includes `TestNO_ICC_IDYNAMIC_NEEDED.cmake`, which is absent from the inventoried source tree. That branch is a definite missing-source dependency, although no Intel compiler was available to execute it. Minimum compiler rules are duplicated and disagree between `compiler_checks.cmake` and `checks.cmake`.

**Fix:** publish the actual public requirement using target compile features; document any additional consumer requirements introduced by enabled dependencies. Make platform/link/rpath adjustments opt-in and tied to the selected toolchain. Model instrumentation with explicit targets and a documented dependency policy. Remove or restore the missing Intel helper and consolidate compiler validation.

Documentation: [CMake interface compile features](https://cmake.org/cmake/help/latest/prop_tgt/INTERFACE_COMPILE_FEATURES.html).

### F19 — P3 — Source discovery and tooling carry unnecessary work and stale assumptions

**References:** `CMakeLists.txt:118`, `532–568`, `593–595`; `Cmake/helper_macros.cmake:11–38`; `Cmake/iwyu.cmake:118–149`, `243`; `Cmake/spell.cmake:73–75`; `Testing/Cxx/CMakeLists.txt:9`.

Library/test/benchmark source globs do not use `CONFIGURE_DEPENDS`; adding a source is not guaranteed to trigger regeneration. The project-owned IDE grouping macro recursively scans the whole repository, including vendored and existing build trees. It constructs an unused include-directory list. This has avoidable configure cost and weakens the intended source boundary.

IWYU still references absent XSigma-specific scripts and a `Library/` layout; the active wrapper has improved error propagation, but its auxiliary configure analysis cannot run in this repository. Spell skips exactly named `build` directories but lacks a consistent strategy for this workspace's `build_*`, `build-*`, and local environments. The benchmark option comment says no benchmark targets exist even though the benchmark subtree creates them. Several comments still reference the moved `include/common/packet.h` instead of `src/common/packet.h`.

**Fix:** use explicit sources or deliberate regeneration-aware globs, group only the actual target source list, remove dead helper work and stale XSigma hooks, and align tool exclusions/docs with current paths. These are maintenance changes; no evidence suggests they explain a failure in the passing default build.

### F20 — P3 — Third-party helper arguments do not match their documented semantics

**References:** `Cmake/third_party_helpers.cmake:349–369`, `419–475`, `480–547`, `593–595`.

`AUTO_ADD_TO_DEPENDENCIES` is parsed as a boolean presence option, although the helper documents a default-ON behavior that can be disabled. Passing `AUTO_ADD_TO_DEPENDENCIES OFF` therefore sets the option true and leaves `OFF` unparsed. Header-only and external branches also append dependencies without consistently checking that option. This is a source-level finding about the project-owned helper API; dependency-owned call sites are excluded.

`CONDITION` is checked after the header-only return and external package resolution, so it is not a universal guard. The warning-filter regex `-W.*` is greedy across flag strings and can remove subsequent non-warning flags when that helper branch is used. Not every latent helper branch is exercised by the default integration.

**Fix:** parse the dependency switch as an explicit boolean value or introduce an unambiguous `NO_AUTO_ADD_TO_DEPENDENCIES` option; reject unparsed arguments. Check conditions before all resolution branches and filter flags as tokens. Keep the helper API smaller where unused generality provides no project benefit.

## File/module conclusions

| Project-owned module or subtree | Conclusion |
|---|---|
| Root `CMakeLists.txt` | Functional default target; central integration is too broad and mixes detection, flags, dependency policy, packaging, and test control. F01–F06, F13–F19 |
| `LinearAlgebraConfig.cmake.in` | Resolves ordinary installed dependencies; default consumers pass. MKL fallback, instrumentation, and runtime-provider constraints need attention |
| `LinearAlgebraDependencies.cmake` | Vendored GTest/benchmark paths work; GTest switch ignored, fixed host build directories, broken FetchContent folder path |
| `build_type.cmake` | Defaults Release even for multi-config generators; does not actually validate/normalize values as its description suggests. Debug multi-config still passes |
| `cache.cmake` | Per-target approach is appropriate; NOTFOUND acceptance is a functional bug |
| `checks.cmake` / `compiler_checks.cmake` | Duplicated policy, generic cache names, failed-check bypass, C language assumption |
| `platform.cmake` | Largest portability surface: unconditional macOS paths, missing Intel helper, global flags, incomplete ISA handling |
| `lto.cmake` | Default OFF works; explicit modes need launcher, language, static-consumer, and linker validation |
| `coverage.cmake` | LLVM build/test passes; flags include descendants; report generation actually lives in external tooling/Python workflow |
| `sanitize.cmake` | Shared in-tree ASan/UBSan works; embedded paths, persistent generated-file state, static exports fail |
| `isolated_blas.cmake` and header template | Explicitly rejects Linux sanitizer/deepbind combination and requires a shared provider. No local combined-MKL runtime validation |
| `clang_tidy.cmake` | Good version guard for exclude-filter support; embedded header root is wrong. No tidy run was used as build correctness evidence |
| `iwyu.cmake` | Active root wrapper serializes logs and propagates analysis failure; stale host-root assumptions and absent auxiliary scripts remain |
| `spell.cmake` | Check-only behavior is appropriate; root/exclusion assumptions need cleanup |
| `valgrind.cmake` | Checker/options and timeout integration are broken |
| `compile_definitions.cmake` | Small 0/1 mapping helper; not used by the main target's manual mapping; no independent functional defect identified |
| `helper_macros.cmake` | Icecream overwrites cache launchers if both enabled; source-group scan is unnecessarily broad; string helper is generic legacy code |
| `Testing/Cxx` | Tests execute standalone; embedding root and option semantics fail; one aggregate GoogleTest entry limits per-case CTest selection |
| `Testing/Benchmark` | Targets build; benchmarks register as performance tests, so default CTest runs can include substantial work when enabled |
| `Testing/Profile` | Requires detected Profiler target; otherwise enabling profiling only warns and produces no profiling program. Real profiling run not audited |
| `Testing/InstallConsumer` | Useful external smoke test, but only exercises native trace. It masks language-contract omissions by explicitly requesting C++17 |

## Project-owned text files and templates

| File | Role / conclusion |
|---|---|
| `Scripts/suppressions/cppcheck_suppressions.txt` | cppcheck data; not executable CMake. Entries only affect scans that explicitly load this file |
| `Scripts/suppressions/spell_suppressions.txt` | codespell ignore words, consumed through the project's spell helper |
| `Scripts/suppressions/lsan_suppressions.txt` | Runtime LSan suppression data; not automatically activated merely by being present |
| `Scripts/suppressions/valgrind_suppression.txt` | Empty suppression file; Valgrind wiring defects are covered by F07 |
| `Scripts/suppressions/sanitizer_ignore.txt.in` | Clang sanitizer input template, including exclusion patterns for dependency paths; its host-root and regeneration problems are covered above |
| `Cmake/LinearAlgebraConfig.cmake.in` | Installed-package configuration; optional dependency resolution and the MKL fallback were reviewed |

Generated-header templates outside the counted filename patterns were also used to check project-owned configuration paths. No text files or templates inside excluded dependency repositories contribute to this review.

## Reproducibility and CI gaps

Existing CI is materially useful: native Linux/macOS, BLAS, MKL, pivoting, sanitizers, SIMD, installed shared/static consumers, coverage including Windows, and CUDA compilation are represented. GPU runtime execution is conditional on repository settings and a suitable runner. This audit does not establish the current remote status of those jobs.

The important missing scenarios are minimum-version CMake, C++-only embedding, pre-existing dependency targets, failed/repeated configure, unavailable cache programs, FetchContent fallback, real Valgrind dashboard wiring, explicit Accelerate with another provider installed, instrumented static consumers, and custom install layouts/relocation. Thin/full/IPO LTO and alternate linkers also lack the execution evidence collected here.

Optional cpuinfo/Profiler autodetection makes two otherwise identical builds acquire different public package dependencies depending on the machine. Existing `CMAKE_DISABLE_FIND_PACKAGE_*` flags can suppress the optional packages, but explicit project-level AUTO/ON/OFF controls and selected package/version summaries would make the contract clearer. Host Homebrew/Conda/vcpkg prefix discovery also deserves toolchain-aware handling for cross-compiles.

There is no source CMake preset file in the reviewed tree. Presets would make supported combinations and a reproducible minimal build easier to discover. They would not fix the functional defects above.

## Suggested repair sequence and acceptance checks

1. **Stabilize configuration contracts:** settle the minimum version, define test/GTest behavior, support C++-only embedding and pre-existing Logging. Acceptance: configure/run tiny standalone and embedded consumers with testing both ON and OFF; parent cache choices remain intact.
2. **Repair dependency correctness:** bind BLAS headers/capabilities to the selected libraries; complete or remove the MKL fallback; retain static sanitizer runtime dependencies. Acceptance: coherent OpenBLAS/Accelerate configurations plus external shared/static backend consumers; unsupported combinations fail early with actionable messages.
3. **Repair optional tooling:** fix Valgrind inputs and recursive timeouts, missing-cache handling, benchmark fallback directories, and generated-file regeneration. Acceptance: the reproduced failing fixtures change to the intended success or clear configure-time failure; real Linux Valgrind execution uses the expected options.
4. **Reduce scope and portability surprises:** use owned root paths, namespaced/absolute helper includes, target-based requirements, validated SIMD/LTO choices, and consistent install directories. Acceptance: repeated configure, multi-config Debug/Release, chosen ISA/compiler combinations, custom install roots, and moved-prefix consumers.
5. **Simplify maintenance:** replace broad globs/source grouping, remove obsolete XSigma hooks/comments, simplify project-owned helper arguments and naming. Add focused regression coverage for the actual failure modes.

## Lint result

`cmake-lint` was rerun over the **23 project-owned executable CMake files**. It returns 1, with 1,455 indentation, 139 naming, 43 line-length, 12 missing-documentation, one escaped-variable-string, three branch-complexity, and three statement-count messages. There are no `E`-category diagnostics in this scoped run.

These messages reflect style and helper complexity, not a count of build failures. The escaped-string warning is in generated IWYU wrapper text and requires manual interpretation. Functional conclusions rely on source review and the recorded configuration/build checks. Raw scoped output is `project-cmake-lint.log`; counts are in `project-lint-summary.json` in the evidence directory.

## Complete source inventory

The following list contains every in-scope CMake/text/template file. Line counts identify the reviewed snapshot. Content hashes are retained in the evidence directory’s `project-inventory.json`. Files under `ThirdParty/` do not contribute to this inventory.

| File | Lines | Role |
|---|---:|---|
| [CMakeLists.txt](../CMakeLists.txt) | 771 | CMake |
| [Cmake/LinearAlgebraConfig.cmake.in](../Cmake/LinearAlgebraConfig.cmake.in) | 76 | template |
| [Cmake/LinearAlgebraDependencies.cmake](../Cmake/LinearAlgebraDependencies.cmake) | 134 | CMake |
| [Cmake/build_type.cmake](../Cmake/build_type.cmake) | 22 | CMake |
| [Cmake/cache.cmake](../Cmake/cache.cmake) | 57 | CMake |
| [Cmake/checks.cmake](../Cmake/checks.cmake) | 323 | CMake |
| [Cmake/clang_tidy.cmake](../Cmake/clang_tidy.cmake) | 68 | CMake |
| [Cmake/compile_definitions.cmake](../Cmake/compile_definitions.cmake) | 24 | CMake |
| [Cmake/compiler_checks.cmake](../Cmake/compiler_checks.cmake) | 24 | CMake |
| [Cmake/coverage.cmake](../Cmake/coverage.cmake) | 61 | CMake |
| [Cmake/helper_macros.cmake](../Cmake/helper_macros.cmake) | 74 | CMake |
| [Cmake/isolated_blas.cmake](../Cmake/isolated_blas.cmake) | 82 | CMake |
| [Cmake/iwyu.cmake](../Cmake/iwyu.cmake) | 265 | CMake |
| [Cmake/lto.cmake](../Cmake/lto.cmake) | 286 | CMake |
| [Cmake/platform.cmake](../Cmake/platform.cmake) | 415 | CMake |
| [Cmake/sanitize.cmake](../Cmake/sanitize.cmake) | 105 | CMake |
| [Cmake/spell.cmake](../Cmake/spell.cmake) | 111 | CMake |
| [Cmake/third_party_helpers.cmake](../Cmake/third_party_helpers.cmake) | 767 | CMake |
| [Cmake/valgrind.cmake](../Cmake/valgrind.cmake) | 148 | CMake |
| [Scripts/suppressions/cppcheck_suppressions.txt](../Scripts/suppressions/cppcheck_suppressions.txt) | 22 | text/data |
| [Scripts/suppressions/lsan_suppressions.txt](../Scripts/suppressions/lsan_suppressions.txt) | 12 | text/data |
| [Scripts/suppressions/sanitizer_ignore.txt.in](../Scripts/suppressions/sanitizer_ignore.txt.in) | 11 | template |
| [Scripts/suppressions/spell_suppressions.txt](../Scripts/suppressions/spell_suppressions.txt) | 2 | text/data |
| [Scripts/suppressions/valgrind_suppression.txt](../Scripts/suppressions/valgrind_suppression.txt) | 0 | text/data |
| [Testing/Benchmark/CMakeLists.txt](../Testing/Benchmark/CMakeLists.txt) | 45 | CMake |
| [Testing/Cxx/CMakeLists.txt](../Testing/Cxx/CMakeLists.txt) | 45 | CMake |
| [Testing/Cxx/CheckGpuResidency.cmake](../Testing/Cxx/CheckGpuResidency.cmake) | 29 | CMake |
| [Testing/InstallConsumer/CMakeLists.txt](../Testing/InstallConsumer/CMakeLists.txt) | 11 | CMake |
| [Testing/Profile/CMakeLists.txt](../Testing/Profile/CMakeLists.txt) | 85 | CMake |
