"""Regression checks for CMake configuration contracts; no network downloads."""

import argparse
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]


class ConfigurationChecks(unittest.TestCase):
    """Exercise embedding, failure recovery, tooling, and dependency boundaries."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="linalg-cmake-checks-")
        cls.work = Path(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def configure(self, name, body, options=(), error=None):
        """Configure a small host and check its result and expected diagnostic."""
        source = self.work / name
        source.mkdir(exist_ok=True)
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(ConfigurationProbe LANGUAGES CXX)\n"
            f'set(LINALG_SOURCE_DIR "{SOURCE.as_posix()}")\n'
            'set(LINALG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")\n' + body,
            encoding="utf-8",
        )
        build = source / "build"
        command = [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_DISABLE_FIND_PACKAGE_cpuinfo=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_Profiler=ON",
            *options,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        output = result.stdout + result.stderr
        if error:
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn(error, output)
        else:
            self.assertEqual(result.returncode, 0, output)
        return build

    def host(self, before="", after=""):
        """Supply an imported Logging target to exercise target reuse."""
        return (
            "enable_testing()\n"
            "add_library(Logging::Logging INTERFACE IMPORTED GLOBAL)\n"
            + before
            + f'add_subdirectory("{SOURCE.as_posix()}" linalg)\n'
            + after
        )

    def test_embedding_and_test_switches(self):
        for name, options, has_gtest, count in [
            ("host-default", [], False, 0),
            ("host-testing-off", ["-DBUILD_TESTING=OFF"], False, 0),
            (
                "host-gtest-off",
                ["-DBUILD_TESTING=ON", "-DLINALG_ENABLE_GTEST=OFF"],
                False,
                1,
            ),
            ("host-gtest-on", ["-DBUILD_TESTING=ON"], True, 2),
        ]:
            with self.subTest(name=name):
                assertion = "if(NOT TARGET gtest)" if has_gtest else "if(TARGET gtest)"
                build = self.configure(
                    name,
                    self.host(
                        after=assertion
                        + '\nmessage(FATAL_ERROR "GTest switch ignored")\nendif()\n'
                    ),
                    options,
                )
                result = subprocess.run(
                    ["ctest", "--test-dir", str(build), "--show-only=json-v1"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                tests = json.loads(result.stdout)["tests"]
                self.assertEqual(len(tests), count)
                if not has_gtest and count:
                    subprocess.run(
                        ["ctest", "--test-dir", str(build), "--output-on-failure"],
                        check=True,
                    )

    def test_bundled_logging_preserves_parent_cache(self):
        build = self.configure(
            "bundled-logging",
            'set(BUILD_TESTING ON CACHE BOOL "Host testing")\n'
            'set(LOGGING_ENABLE_TESTING ON CACHE BOOL "Host logging choice")\n'
            "set(LINALG_ENABLE_TESTING OFF)\n"
            f'add_subdirectory("{SOURCE.as_posix()}" linalg)\n'
            "if(NOT BUILD_TESTING OR NOT LOGGING_ENABLE_TESTING OR TARGET LoggingCxxTests)\n"
            'message(FATAL_ERROR "Dependency configuration changed host choices")\nendif()\n'
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/consumer.cpp" '
            '"#include \\"include/matrix_operation/trace.h\\"\\n'
            "int main() { double a[] = {1, 2, 3, 4}; "
            'return linalg::trace<double, linalg::linalg_backend::NATIVE>::apply(a, 2, 2) == 5 ? 0 : 1; }")\n'
            'add_executable(consumer "${CMAKE_CURRENT_BINARY_DIR}/consumer.cpp")\n'
            "target_link_libraries(consumer PRIVATE LinearAlgebra::LinearAlgebra)\n"
            'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/consumer-path.txt" '
            'CONTENT "$<TARGET_FILE:consumer>")\n',
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--target", "consumer", "--parallel", "4"],
            check=True,
        )
        subprocess.run([(build / "consumer-path.txt").read_text()], check=True)

    def test_legacy_msvc_logging_overlay(self):
        # Exercise the compatibility headers on every host; the MSVC minimum
        # job additionally verifies them with the compiler that needs them.
        source = self.work / "logging-compat-probe.cpp"
        source.write_text(
            '#include <fmt/format.h>\n'
            '#include "include/logger/structured.h"\n'
            'int main() {\n'
            '  logging::structured_event event("probe");\n'
            '  event.add("value", "ok");\n'
            '  if (fmt::format("{} {} {:.1f}", "ok", 42, 1.5) != "ok 42 1.5") return 1;\n'
            '  try { fmt::format(fmt::runtime("{:d}"), "bad"); }\n'
            '  catch (const fmt::format_error&) { return 0; }\n'
            '  return 2;\n'
            '}\n'
        )
        build = self.configure(
            "logging-msvc-compat",
            'set(LINALG_ENABLE_TESTING OFF)\n'
            f'add_subdirectory("{SOURCE.as_posix()}" linalg)\n'
            'include("${LINALG_SOURCE_DIR}/Cmake/LoggingMsvcCompatibility.cmake")\n'
            'set(LINALG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/linalg")\n'
            'linalg_apply_logging_msvc_compatibility()\n'
            f'add_executable(probe "{source.as_posix()}")\n'
            'target_link_libraries(probe PRIVATE Logging::Logging)\n'
            'target_compile_definitions(probe PRIVATE FMT_HEADER_ONLY=1)\n'
            # fmt requires UTF-8 on MSVC; the parent probe does not inherit
            # the library's private compiler options.
            'if(MSVC)\n'
            'target_compile_options(probe PRIVATE /utf-8)\n'
            'endif()\n'
            'target_include_directories(probe PRIVATE '
            '"${LINALG_BINARY_DIR}/ThirdParty/logging-msvc-compat/fmt-include")\n'
            'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/probe-path.txt" '
            'CONTENT "$<TARGET_FILE:probe>")\n',
            options=[
                "-DLINALG_ENABLE_CACHE=OFF", "-DLOGGING_ENABLE_CACHE=OFF",
                "-DLOGGING_CXX_STANDARD=17", "-DCMAKE_CXX_STANDARD=17",
            ],
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--target", "probe", "--parallel", "4"],
            check=True,
        )
        subprocess.run([(build / "probe-path.txt").read_text()], check=True)

    def test_hip_stream_helpers_with_opaque_handles(self):
        # ROCm uses void* for BOTH handles. Keep this compile/routing check
        # available on CPU-only hosts, where no HIP SDK is installed.
        sdk = self.work / "hip-stubs"
        for header, contents in {
            "hip/hip_runtime_api.h": (
                "#pragma once\nusing hipStream_t = void*;\n"
                "constexpr int hipSuccess = 0;\n"
                "int hipGetDevice(int*);\n"
            ),
            "hipblas/hipblas.h": (
                "#pragma once\n#include <hip/hip_runtime_api.h>\n"
                "using hipblasHandle_t = void*;\nusing hipblasPointerMode_t = int;\n"
                "constexpr int HIPBLAS_STATUS_SUCCESS = 0, HIPBLAS_POINTER_MODE_DEVICE = 1;\n"
                "int hipblasCreate(hipblasHandle_t*);\n"
                "int hipblasSetStream(hipblasHandle_t, hipStream_t);\n"
                "int hipblasGetPointerMode(hipblasHandle_t, hipblasPointerMode_t*);\n"
                "int hipblasSetPointerMode(hipblasHandle_t, hipblasPointerMode_t);\n"
            ),
            "hipsolver/hipsolver.h": (
                "#pragma once\n#include <hip/hip_runtime_api.h>\n"
                "using hipsolverDnHandle_t = void*;\n"
                "constexpr int HIPSOLVER_STATUS_SUCCESS = 0;\n"
                "int hipsolverDnCreate(hipsolverDnHandle_t*);\n"
                "int hipsolverDnSetStream(hipsolverDnHandle_t, hipStream_t);\n"
            ),
        }.items():
            path = sdk / header
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        source = self.work / "hip-stream-probe.cpp"
        source.write_text(
            '#include "include/common/hip_handle.h"\n'
            'int blas_calls = 0, solver_calls = 0;\n'
            'int hipblasSetStream(void* h, void* s) {\n'
            '  if (h != s) return 1; ++blas_calls; return 0; }\n'
            'int hipsolverDnSetStream(void* h, void* s) {\n'
            '  if (h != s) return 1; ++solver_calls; return 0; }\n'
            'int main() {\n'
            '  int token = 0;\n'
            '  linalg::detail::set_hipblas_stream(&token, &token);\n'
            '  linalg::detail::set_hipsolver_stream(&token, &token);\n'
            '  return blas_calls == 1 && solver_calls == 1 ? 0 : 1;\n'
            '}\n'
        )
        build = self.configure(
            "hip-stream-helpers",
            'set(LINALG_ENABLE_TESTING OFF)\n'
            f'add_subdirectory("{SOURCE.as_posix()}" linalg)\n'
            f'add_executable(probe "{source.as_posix()}")\n'
            'target_link_libraries(probe PRIVATE Logging::Logging)\n'
            'target_compile_features(probe PRIVATE cxx_std_17)\n'
            'target_compile_definitions(probe PRIVATE LINALG_HAS_ROCBLAS=1)\n'
            f'target_include_directories(probe PRIVATE "{sdk.as_posix()}" "{SOURCE.as_posix()}")\n'
            'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/probe-path.txt" '
            'CONTENT "$<TARGET_FILE:probe>")\n',
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--target", "probe", "--parallel", "4"],
            check=True,
        )
        subprocess.run([(build / "probe-path.txt").read_text()], check=True)

    def test_installed_hip_dependencies(self):
        prefix = self.work / "hip-packages"
        for package, targets in {
            "Logging": ["Logging::Logging"],
            "hip": ["hip::host"],
            "hipblas": ["roc::hipblas"],
            "hipsolver": ["roc::hipsolver"],
        }.items():
            package_dir = prefix / package
            package_dir.mkdir(parents=True)
            (package_dir / f"{package}Config.cmake").write_text(
                "".join(f"add_library({t} INTERFACE IMPORTED)\n" for t in targets)
            )
        self.configure(
            "installed-hip",
            'set(LINALG_ENABLE_ROCBLAS ON)\n'
            'include(CMakePackageConfigHelpers)\n'
            'configure_package_config_file('
            '"${LINALG_SOURCE_DIR}/Cmake/LinearAlgebraConfig.cmake.in" '
            '"${CMAKE_CURRENT_BINARY_DIR}/LinearAlgebraConfig.cmake" INSTALL_DESTINATION .)\n'
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/LinearAlgebraTargets.cmake" '
            '"add_library(LinearAlgebra::LinearAlgebra INTERFACE IMPORTED)\\n" '
            '"set_property(TARGET LinearAlgebra::LinearAlgebra PROPERTY INTERFACE_LINK_LIBRARIES '
            'hip::host roc::hipblas roc::hipsolver)\\n")\n'
            'include("${CMAKE_CURRENT_BINARY_DIR}/LinearAlgebraConfig.cmake")\n'
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp" "int main() { return 0; }")\n'
            'add_executable(probe "${CMAKE_CURRENT_BINARY_DIR}/main.cpp")\n'
            'target_link_libraries(probe PRIVATE LinearAlgebra::LinearAlgebra)\n',
            options=[f"-DCMAKE_PREFIX_PATH={prefix.as_posix()}"],
        )

    def test_owned_modules_cannot_be_shadowed(self):
        shadow = self.work / "shadow"
        shadow.mkdir()
        for module in (SOURCE / "Cmake").glob("*.cmake"):
            (shadow / module.name).write_text(
                'message(FATAL_ERROR "Loaded host module")\n'
            )
        self.configure(
            "module-shadow",
            self.host(
                before=f'list(PREPEND CMAKE_MODULE_PATH "{shadow.as_posix()}")\n'
            ),
        )

    def test_required_validation_repeats_after_failure(self):
        body = 'include("${LINALG_SOURCE_DIR}/Cmake/checks.cmake")\n'
        for _ in range(2):
            self.configure(
                "failed-validation",
                body,
                ["-DCMAKE_DISABLE_FIND_PACKAGE_Threads=ON"],
                error="REQUIRED",
            )

    def test_invalid_isa_and_lto_fail_at_configure(self):
        for option, error in [
            ("-DLINALG_VECTORIZATION_ISA=typo", "Invalid LINALG_VECTORIZATION_ISA"),
            ("-DLINALG_LTO_MODE=typo", "Invalid LINALG_LTO_MODE"),
            ("-DLINALG_LINKER_CHOICE=typo", "Invalid linker choice"),
        ]:
            with self.subTest(option=option):
                self.configure(
                    error.replace(" ", "-"), self.host(), [option], error=error
                )

    def test_missing_cache_is_rejected(self):
        body = (
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp" "int answer() { return 42; }")\n'
            'add_library(probe ${PROBE_LIBRARY_TYPE} "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp")\n'
            "set_target_properties(probe PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)\n"
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp" '
            '"int answer(); int main() { return answer() != 42; }")\n'
            'add_executable(consumer "${CMAKE_CURRENT_BINARY_DIR}/main.cpp")\n'
            "target_link_libraries(consumer PRIVATE probe)\n"
            'include("${LINALG_SOURCE_DIR}/Cmake/cache.cmake")\n'
            "set(ENABLE_CACHE ON)\nset(CACHE_BACKEND ccache)\n"
            'set(LINALG_CCACHE_PROGRAM "${CMAKE_CURRENT_BINARY_DIR}/missing-ccache")\n'
            "linalg_target_apply_cache(probe ENABLE_CACHE CACHE_BACKEND)\n"
        )
        self.configure("missing-cache", body, error="requested cache ccache not found")
        unavailable = body.replace(
            'set(LINALG_CCACHE_PROGRAM "${CMAKE_CURRENT_BINARY_DIR}/missing-ccache")',
            "set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH OFF)\n"
            "set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH OFF)\n"
            "set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH OFF)\n"
            "set(CMAKE_FIND_USE_CMAKE_PATH OFF)\n"
            "set(CMAKE_FIND_USE_INSTALL_PREFIX OFF)\n"
            "set(LINALG_CCACHE_PROGRAM LINALG_CCACHE_PROGRAM-NOTFOUND)",
        )
        self.configure(
            "notfound-cache", unavailable, error="requested cache ccache not found"
        )

    @unittest.skipUnless(os.name == "posix", "Icecream uses POSIX compiler wrappers")
    def test_icecream_and_compiler_caches_compose(self):
        for backend in ("sccache", "ccache"):
            with self.subTest(backend=backend):
                tools = self.work / f"{backend} tools' directory"
                tools.mkdir()
                log = tools / "commands.jsonl"
                cache = tools / backend
                icecc = tools / "icecc"
                script = (
                    "#!/usr/bin/env python3\n"
                    "import json, os, subprocess, sys\n"
                    f"with open({str(log)!r}, 'a') as log:\n"
                    "    log.write(json.dumps([os.path.basename(sys.argv[0]), *sys.argv[1:]]) + '\\n')\n"
                )
                for executable in (cache, icecc):
                    executable.write_text(
                        script
                        + "sys.exit(subprocess.run(sys.argv[1:], check=False).returncode)\n"
                    )
                cache.chmod(0o755)
                icecc.chmod(0o755)
                body = (
                    "enable_language(C)\n"
                    'include("${LINALG_SOURCE_DIR}/Cmake/cache.cmake")\n'
                    'include("${LINALG_SOURCE_DIR}/Cmake/helper_macros.cmake")\n'
                    'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/answer.c" "int answer(void) { return 42; }")\n'
                    'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp" '
                    '"extern \\"C\\" int answer(void); int main() { return answer() == 42 ? 0 : 1; }")\n'
                    'add_executable(probe "${CMAKE_CURRENT_BINARY_DIR}/answer.c" "${CMAKE_CURRENT_BINARY_DIR}/main.cpp")\n'
                    "set(ENABLE_CACHE ON)\nset(ENABLE_ICECC ON)\n"
                    f"set(CACHE_BACKEND {backend})\n"
                    f'set(LINALG_{backend.upper()}_PROGRAM "{cache.as_posix()}")\n'
                    f'set(LINALG_ICECC_EXECUTABLE "{icecc.as_posix()}")\n'
                    "linalg_target_apply_cache(probe ENABLE_CACHE CACHE_BACKEND)\n"
                    "linalg_target_optional_icecc(probe ENABLE_ICECC)\n"
                    'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/probe-path.txt" CONTENT "$<TARGET_FILE:probe>")\n'
                )
                build = self.configure(f"icecc-{backend}", body)
                result = subprocess.run(
                    ["cmake", "--build", str(build), "--parallel", "2"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                subprocess.run([(build / "probe-path.txt").read_text()], check=True)
                commands = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(sum(command[0] == backend for command in commands), 2)
                self.assertEqual(sum(command[0] == "icecc" for command in commands), 2)
                wrapper = build / "CMakeFiles/probe-launchers/icecc-CXX-compiler"
                previous_wrapper = wrapper.read_text()
                icecc.write_text(
                    icecc.read_text() + "\n# Updated Icecream executable\n"
                )
                regenerated = subprocess.run(
                    ["cmake", "-S", str(build.parent), "-B", str(build)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    regenerated.returncode, 0, regenerated.stdout + regenerated.stderr
                )
                self.assertNotEqual(previous_wrapper, wrapper.read_text())
                (build / "main.cpp").write_text(
                    "#error compiler failure must propagate\n"
                )
                result = subprocess.run(
                    ["cmake", "--build", str(build), "--parallel", "2"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "compiler failure must propagate", result.stdout + result.stderr
                )
                if backend == "sccache":
                    self.configure(
                        "icecc-sccache-lto",
                        body
                        + 'include("${LINALG_SOURCE_DIR}/Cmake/lto.cmake")\n'
                        + "set(LINALG_LTO_MODE thin)\n"
                        + "linalg_target_apply_lto(probe LINALG)\n",
                        error="LTO incompatible with sccache",
                    )

    @unittest.skipIf(
        os.name == "nt" and os.environ.get("CXX", "").lower() in ("cl", "cl.exe"),
        "Native MSVC does not support the GCC/Clang sanitizer helper",
    )
    def test_sanitizer_regenerates_assets_and_exports_runtime(self):
        body = (
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/plain.cpp" '
            '"#include <string>\\n#include <vector>\\n'
            "int plain() { return static_cast<int>(std::string(8, 'x').size() + "
            'std::vector<int>(4, 1).size()); }")\n'
            'add_library(plain STATIC "${CMAKE_CURRENT_BINARY_DIR}/plain.cpp")\n'
            "set_target_properties(plain PROPERTIES POSITION_INDEPENDENT_CODE ON)\n"
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp" '
            '"#include <string>\\n#include <vector>\\n'
            "int plain(); int answer() { return plain() + "
            "static_cast<int>(std::string(8, 'x').size() + "
            'std::vector<int>(4, 1).size()); }")\n'
            'add_library(probe ${PROBE_LIBRARY_TYPE} "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp")\n'
            "set_target_properties(probe PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)\n"
            "target_link_libraries(probe PRIVATE plain)\n"
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp" '
            '"#include <string>\\n#include <vector>\\n'
            "int answer(); int main() { return answer() != "
            "static_cast<int>(std::string(16, 'x').size() + "
            'std::vector<int>(8, 1).size()); }")\n'
            'add_executable(consumer "${CMAKE_CURRENT_BINARY_DIR}/main.cpp")\n'
            "target_link_libraries(consumer PRIVATE probe)\n"
            'include("${LINALG_SOURCE_DIR}/Cmake/sanitize.cmake")\n'
            "set(LINALG_SANITIZER_TYPE address)\n"
            "linalg_configure_sanitizer(probe LINALG)\n"
            "get_target_property(runtime probe INTERFACE_LINK_OPTIONS)\n"
            'if(NOT runtime MATCHES "fsanitize=address")\n'
            'message(FATAL_ERROR "Missing sanitizer runtime requirement")\nendif()\n'
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/compiler.txt" "${CMAKE_CXX_COMPILER_ID}")\n'
        )
        build = self.configure(
            "sanitizer-regeneration", body, ["-DPROBE_LIBRARY_TYPE=STATIC"]
        )
        if "Clang" in (build / "compiler.txt").read_text():
            ignore = build / "sanitizer_ignore.txt"
            self.assertTrue(ignore.is_file())
            ignore.unlink()
            self.configure(
                "sanitizer-regeneration", body, ["-DPROBE_LIBRARY_TYPE=STATIC"]
            )
            self.assertTrue(ignore.is_file())
        # The consumer must select the same ASan runtime as the instrumented
        # library. Exercise both archive and DLL links on Windows as well.
        for library_type in ("STATIC", "SHARED"):
            with self.subTest(library_type=library_type):
                build = self.configure(
                    "sanitizer-link-" + library_type.lower(),
                    body,
                    ["-DPROBE_LIBRARY_TYPE=" + library_type],
                )
                result = subprocess.run(
                    ["cmake", "--build", str(build)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_msvc_debug_information_does_not_require_a_shared_pdb(self):
        # Exercise MSVC flag selection even when this suite runs on another host.
        # Keep compiler overrides inside a function so CMake still generates with
        # the real compiler after recording the selected options.
        build = self.configure(
            "msvc-debug-information",
            "function(record_msvc_flags)\n"
            "set(MSVC TRUE)\n"
            "set(CMAKE_CXX_COMPILER_ID MSVC)\n"
            "set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT MSVC)\n"
            "set(CMAKE_CONFIGURATION_TYPES Debug Release RelWithDebInfo MinSizeRel Custom)\n"
            "foreach(language C CXX)\n"
            "foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL CUSTOM)\n"
            'set(CMAKE_${language}_FLAGS_${config} "/Zi /DKEEP_${config}")\n'
            "endforeach()\n"
            "endforeach()\n"
            'include("${LINALG_SOURCE_DIR}/Cmake/platform.cmake")\n'
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/debug-format.txt" '
            '"${CMAKE_MSVC_DEBUG_INFORMATION_FORMAT}")\n'
            "foreach(language C CXX)\n"
            "foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL CUSTOM)\n"
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${language}-${config}.txt" '
            '"${CMAKE_${language}_FLAGS} ${CMAKE_${language}_FLAGS_${config}}")\n'
            "endforeach()\n"
            "endforeach()\n"
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp" "int value;")\n'
            'add_library(probe STATIC "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp")\n'
            'include("${LINALG_SOURCE_DIR}/Cmake/coverage.cmake")\n'
            "linalg_configure_coverage(probe)\n"
            "get_target_property(options probe COMPILE_OPTIONS)\n"
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/coverage-options.txt" "${options}")\n'
            "endfunction()\n"
            "record_msvc_flags()\n",
        )
        self.assertEqual((build / "debug-format.txt").read_text(), "Embedded")
        for language in ("C", "CXX"):
            for config in (
                "DEBUG",
                "RELEASE",
                "RELWITHDEBINFO",
                "MINSIZEREL",
                "CUSTOM",
            ):
                flags = (build / f"{language}-{config}.txt").read_text()
                self.assertIn("/Z7", flags)
                self.assertNotRegex(flags, r"[-/]Z[iI]\b")
                self.assertIn(f"/DKEEP_{config}", flags)
        coverage = (build / "coverage-options.txt").read_text()
        self.assertIn("/Z7", coverage)
        self.assertNotIn("/Zi", coverage)

    def test_benchmark_fetchcontent_and_vendored_paths(self):
        body = (
            'include("${LINALG_SOURCE_DIR}/Cmake/LinearAlgebraDependencies.cmake")\n'
            "linalg_setup_benchmark()\n"
        )
        self.configure("benchmark-vendored", body)
        self.configure(
            "benchmark-fetchcontent",
            body,
            [f"-DFETCHCONTENT_SOURCE_DIR_BENCHMARK={SOURCE / 'ThirdParty/benchmark'}"],
        )

    def test_dependency_helper_guards_and_flag_filter(self):
        self.configure(
            "dependency-helper",
            'include("${LINALG_SOURCE_DIR}/Cmake/third_party_helpers.cmake")\n'
            "linalg_add_third_party_library(skipped CONDITION OFF LIBRARY_TYPE HEADER_ONLY)\n"
            'if(TARGET skipped)\nmessage(FATAL_ERROR "Condition ignored")\nendif()\n'
            "linalg_add_third_party_library(headers LIBRARY_TYPE HEADER_ONLY "
            'INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" AUTO_ADD_TO_DEPENDENCIES OFF)\n'
            "get_property(deps GLOBAL PROPERTY _LINALG_DEPENDENCY_LIBS)\n"
            'if(deps)\nmessage(FATAL_ERROR "Auto dependency switch ignored")\nendif()\n'
            'linalg_filter_flags_internal("-Wall -O2 -g" "-W.*" filtered)\n'
            'if(NOT filtered STREQUAL "-O2 -g")\n'
            'message(FATAL_ERROR "Flag filter removed non-warning flags")\nendif()\n',
        )

    def test_valgrind_dashboard_and_recursive_timeouts(self):
        child = self.work / "valgrind-child"
        child.mkdir()
        (child / "CMakeLists.txt").write_text(
            'add_test(NAME probe COMMAND "${CMAKE_COMMAND}" -E true)\n'
            "set_tests_properties(probe PROPERTIES TIMEOUT 2)\n"
        )
        build = self.configure(
            "valgrind-dashboard",
            "set(BUILD_TESTING ON)\n"
            'set(MEMORYCHECK_COMMAND "${CMAKE_COMMAND}")\n'
            'include("${LINALG_SOURCE_DIR}/Cmake/valgrind.cmake")\n'
            "include(CTest)\n"
            f'add_subdirectory("{child.as_posix()}" child)\n'
            'linalg_apply_valgrind_timeouts("${CMAKE_CURRENT_SOURCE_DIR}")\n',
        )
        dashboard = (build / "DartConfiguration.tcl").read_text()
        self.assertIn("--error-exitcode=1", dashboard)
        self.assertIn("valgrind_suppression.txt", dashboard)
        result = subprocess.run(
            ["ctest", "--test-dir", str(build), "--show-only=json-v1"],
            capture_output=True,
            text=True,
            check=True,
        )
        properties = json.loads(result.stdout)["tests"][0]["properties"]
        self.assertIn({"name": "TIMEOUT", "value": 40.0}, properties)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    unittest.main(argv=[__file__], verbosity=2)
