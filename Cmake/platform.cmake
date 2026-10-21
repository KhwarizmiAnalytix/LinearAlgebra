if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    message(STATUS "Emscripten: enabling WebAssembly exception handling (-fwasm-exceptions)")
    # Enable exceptions because XSigma and third party code rely on C++ exceptions. Allow C++ to
    # catch exceptions. Emscripten disables it by default due to high overhead. Generate helper
    # functions to get stack traces for uncaught exceptions
    string(APPEND CMAKE_CXX_FLAGS " -fwasm-exceptions")
    string(APPEND CMAKE_C_FLAGS " -fwasm-exceptions")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -fwasm-exceptions -sEXCEPTION_STACK_TRACES=1")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -fwasm-exceptions -sEXCEPTION_STACK_TRACES=1")
    string(APPEND CMAKE_MODULE_LINKER_FLAGS " -fwasm-exceptions -sEXCEPTION_STACK_TRACES=1")
    # Consumers linking to XSigma also need to add the exception flag.
    if(TARGET XSigmaPlatform)
        target_link_options(
            XSigmaPlatform INTERFACE "-fwasm-exceptions" "-sEXCEPTION_STACK_TRACES=1"
        )
    endif()
    if(PROJECT_WEBASSEMBLY_THREADS)
        message(STATUS "Emscripten: enabling pthreads (-pthread -Wno-pthreads-mem-growth)")
        # Remove after https://github.com/WebAssembly/design/issues/1271 is closed Set Wno flag globally
        # because even though the flag is added in XSigmaCompilerWarningFlags.cmake, wrapping tools do
        # not link with `XSigmaPlatform`
        string(APPEND CMAKE_CXX_FLAGS " -pthread -Wno-pthreads-mem-growth")
        string(APPEND CMAKE_C_FLAGS " -pthread -Wno-pthreads-mem-growth")
        string(APPEND CMAKE_EXE_LINKER_FLAGS " -pthread")
        string(APPEND CMAKE_SHARED_LINKER_FLAGS " -pthread")
        string(APPEND CMAKE_MODULE_LINKER_FLAGS " -pthread")
        # Consumers linking to XSigma also need to add the pthread flag.
        if(TARGET XSigmaPlatform)
            target_compile_options(XSigmaPlatform INTERFACE "-pthread" "-Wno-pthreads-mem-growth")
            target_link_options(XSigmaPlatform INTERFACE "-pthread")
        endif()
    endif()
    if(PROJECT_WEBASSEMBLY_64_BIT)
        message(STATUS "Emscripten: enabling 64-bit memory (-sMEMORY64=1)")
        string(APPEND CMAKE_CXX_FLAGS " -sMEMORY64=1")
        string(APPEND CMAKE_C_FLAGS " -sMEMORY64=1")
        string(APPEND CMAKE_EXE_LINKER_FLAGS " -sMEMORY64=1")
        string(APPEND CMAKE_SHARED_LINKER_FLAGS " -sMEMORY64=1")
        string(APPEND CMAKE_MODULE_LINKER_FLAGS " -sMEMORY64=1")
        # Consumers linking to XSigma also need to add the memory64 flag.
        if(TARGET XSigmaPlatform)
            target_compile_options(XSigmaPlatform INTERFACE "-sMEMORY64=1")
            target_link_options(XSigmaPlatform INTERFACE "-sMEMORY64=1")
        endif()
    endif()
endif()

# MinGW GCC threading support.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND MINGW)
    message(STATUS "MinGW: adding -mthreads")
    string(APPEND CMAKE_CXX_FLAGS " -mthreads")
    string(APPEND CMAKE_C_FLAGS " -mthreads")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -mthreads")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -mthreads")
    string(APPEND CMAKE_MODULE_LINKER_FLAGS " -mthreads")
endif()

if(MSVC AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT
                                                            STREQUAL "GNU")
)
    message(STATUS "MSVC compiler detected: applying MSVC-specific flags")
    # Use the highest warning level for visual c++ compiler. set(CMAKE_CXX_WARNING_LEVEL 4)
    # if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]") string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS
    # "${CMAKE_CXX_FLAGS}") else() set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4") endif() Enable C++20
    # support: /Zc:__cplusplus
    message(STATUS "  MSVC: enabling C++20 __cplusplus macro (/Zc:__cplusplus)")
    string(APPEND CMAKE_CXX_FLAGS " /Zc:__cplusplus")
    # Treat warnings as errors set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX") Disable C4244: conversion
    # warnings
    message(
        STATUS
            "  MSVC: disabling conversion/truncation/signed warnings (/wd4244 /wd4267 /wd4715 /wd4018)"
    )
    string(APPEND CMAKE_CXX_FLAGS " /wd4244 /wd4267 /wd4715 /wd4018")
    string(APPEND CMAKE_C_FLAGS " /wd4244 /wd4267 /wd4715 /wd4018")

    # Enable /MP flag for Visual Studio
    set(CMAKE_CXX_MP_FLAG ON CACHE BOOL "Build with /MP flag enabled")
    set(PROCESSOR_COUNT "$ENV{NUMBER_OF_PROCESSORS}")
    set(CMAKE_CXX_MP_NUM_PROCESSORS ${PROCESSOR_COUNT}
        CACHE STRING "The maximum number of processes for the /MP flag"
    )
    if(CMAKE_CXX_MP_FLAG AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_GENERATOR MATCHES
                                                                             "Ninja")
    )
        message(STATUS "  MSVC: parallel compilation enabled (/MP${CMAKE_CXX_MP_NUM_PROCESSORS})")
        string(APPEND CMAKE_CXX_FLAGS " /MP${CMAKE_CXX_MP_NUM_PROCESSORS}")
        string(APPEND CMAKE_C_FLAGS " /MP${CMAKE_CXX_MP_NUM_PROCESSORS}")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_GENERATOR MATCHES "Ninja")
        message(STATUS "  Clang/Ninja: skipping /MP — Ninja handles parallelism")
    endif()

    # Enable /bigobj for MSVC to allow larger symbol tables
    message(STATUS "  MSVC: enabling large object files (/bigobj)")
    string(APPEND CMAKE_CXX_FLAGS " /bigobj")
    string(APPEND CMAKE_C_FLAGS " /bigobj")

    # Ninja launches independent compiler processes that share the target PDB.
    message(STATUS "  MSVC: enabling synchronized PDB debug info (/Zi /FS)")
    string(APPEND CMAKE_CXX_FLAGS " /Zi /FS")
    string(APPEND CMAKE_C_FLAGS " /Zi /FS")

    # Use /utf-8 so that MSVC uses utf-8 in source files and object files
    message(STATUS "  MSVC: setting source/output encoding to UTF-8 (/utf-8)")
    string(APPEND CMAKE_CXX_FLAGS " /utf-8")
    string(APPEND CMAKE_C_FLAGS " /utf-8")

    # use /EHsc for exception handling
    message(STATUS "  MSVC: enabling structured exception handling (/EHsc)")
    string(APPEND CMAKE_CXX_FLAGS " /EHsc")
    string(APPEND CMAKE_C_FLAGS " /EHsc")
endif()

# Suppress MSVC CRT deprecation warnings for all compilers that target the
# Windows CRT (cl.exe, clang-cl, clang++ with MSVC CRT).  MinGW ships its own
# CRT that does not use these macros, so it is excluded.
if(WIN32 AND NOT MINGW)
    add_definitions(
        -D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS
    )
    add_definitions(-D_SCL_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)
endif()

if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Clang on Windows: applying Clang-specific flags")

    # Colored diagnostics — Ninja relays Clang's ANSI codes correctly; MSVC color codes are not
    string(APPEND CMAKE_CXX_FLAGS " -fcolor-diagnostics")
    string(APPEND CMAKE_C_FLAGS " -fcolor-diagnostics")

    # Suppress spurious warnings that fire when MSVC-style flags are mixed with Clang
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-command-line-argument")
    string(APPEND CMAKE_C_FLAGS " -Wno-unused-command-line-argument")

    # MSVC CRT marks getenv/strcpy/etc. with __declspec(deprecated) under Clang;
    # _CRT_SECURE_NO_WARNINGS removes the #pragma form but not the __declspec form, so Clang still
    # fires -Wdeprecated-declarations
    string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated-declarations")
    string(APPEND CMAKE_C_FLAGS " -Wno-deprecated-declarations")

    # Third-party C code (e.g. mimalloc) uses old-style K&R function declarations without prototypes
    string(APPEND CMAKE_C_FLAGS " -Wno-strict-prototypes")

    if(CMAKE_GENERATOR MATCHES "Ninja")
        message(STATUS "  Clang/Ninja/Windows: applying Ninja-specific adjustments")

        if(NOT MSVC)
            # Plain clang++ driver: emit CodeView debug info so WinDbg / VS can read symbols
            string(APPEND CMAKE_CXX_FLAGS " -gcodeview")
            string(APPEND CMAKE_C_FLAGS " -gcodeview")
        endif()
    endif()
endif()

if(UNIX AND NOT APPLE)
    include(CheckCXXCompilerFlag)
    include(CheckCCompilerFlag)
    include(CheckCXXSourceCompiles)

    # Append flag to CMAKE_CXX_FLAGS and CMAKE_C_FLAGS if both compilers accept it.
    macro(_linalg_add_compile_flag _flag)
        string(MAKE_C_IDENTIFIER "${_flag}" _id)
        check_cxx_compiler_flag("${_flag}" _CXX_HAS${_id})
        if(CMAKE_C_COMPILER_LOADED)
            check_c_compiler_flag("${_flag}" _C_HAS${_id})
        endif()
        if(_CXX_HAS${_id})
            string(APPEND CMAKE_CXX_FLAGS " ${_flag}")
        endif()
        if(_C_HAS${_id})
            string(APPEND CMAKE_C_FLAGS " ${_flag}")
        endif()
    endmacro()

    # Append flag to all linker flag variables if the linker accepts it. Uses
    # CMAKE_REQUIRED_LINK_OPTIONS (CMake >= 3.14) since check_linker_flag needs 3.18.
    macro(_linalg_add_linker_flag _flag)
        string(MAKE_C_IDENTIFIER "${_flag}" _id)
        set(CMAKE_REQUIRED_LINK_OPTIONS "${_flag}")
        check_cxx_source_compiles("int main(){}" _LINKER_HAS${_id})
        unset(CMAKE_REQUIRED_LINK_OPTIONS)
        if(_LINKER_HAS${_id})
            string(APPEND CMAKE_EXE_LINKER_FLAGS " ${_flag}")
            string(APPEND CMAKE_SHARED_LINKER_FLAGS " ${_flag}")
            string(APPEND CMAKE_MODULE_LINKER_FLAGS " ${_flag}")
        endif()
    endmacro()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        _linalg_add_compile_flag("-fPIC")
        _linalg_add_compile_flag("-pipe")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        _linalg_add_compile_flag("-fcolor-diagnostics")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        _linalg_add_compile_flag("-fdiagnostics-color=always")
    endif()
endif()

# --- LINALG vectorization ISA option ---
# Which linalg::packet<T> specialization backs LINALG_ENABLE_VECTORIZATION.
# There is no "auto" choice: the ISA is always explicit, and CMake computes a
# platform-appropriate default when the cache variable is not overridden on the
# command line:
#   scalar - force linalg::packet<T>'s scalar (width-1) primary template.
#            Default for platforms other than x86_64/aarch64.
#   sse2/avx/avx2/avx512/neon - force that specialization and (x86 targets
#            only) add the matching -m<isa> compiler flag so it is actually
#            enabled; a target that cannot assemble the requested ISA fails
#            to build rather than silently falling back to a narrower one.
#            SSE2 is the x86_64 baseline; neon is the
#            default on aarch64/arm64.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _linalg_system_processor_lower)
if(CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES _linalg_arch_count)
    if(_linalg_arch_count GREATER 1)
        set(_linalg_system_processor_lower universal)
    else()
        string(TOLOWER "${CMAKE_OSX_ARCHITECTURES}" _linalg_system_processor_lower)
    endif()
endif()
if(_linalg_system_processor_lower MATCHES "^(aarch64|arm64)$")
    set(_linalg_vectorization_isa_default "neon")
elseif(_linalg_system_processor_lower MATCHES "^(x86_64|amd64)$")
    set(_linalg_vectorization_isa_default "sse2")
else()
    set(_linalg_vectorization_isa_default "scalar")
endif()
set(LINALG_VECTORIZATION_ISA
    "${_linalg_vectorization_isa_default}"
    CACHE
        STRING
        "SIMD ISA for LinearAlgebra's vectorized scalar GEMM: scalar, sse2, avx, avx2, avx512, neon"
)
unset(_linalg_vectorization_isa_default)
set_property(
    CACHE LINALG_VECTORIZATION_ISA
    PROPERTY STRINGS
             scalar
             sse2
             avx
             avx2
             avx512
             neon
)

if(NOT LINALG_VECTORIZATION_ISA MATCHES "^(scalar|sse2|avx|avx2|avx512|neon)$")
    message(FATAL_ERROR "Invalid LINALG_VECTORIZATION_ISA: ${LINALG_VECTORIZATION_ISA}")
endif()
if(LINALG_ENABLE_VECTORIZATION AND NOT LINALG_VECTORIZATION_ISA STREQUAL "scalar")
    if(_linalg_system_processor_lower STREQUAL "universal")
        message(FATAL_ERROR "Universal builds require LINALG_VECTORIZATION_ISA=scalar")
    elseif(LINALG_VECTORIZATION_ISA STREQUAL "neon")
        if(NOT _linalg_system_processor_lower MATCHES "^(aarch64|arm64)$")
            message(FATAL_ERROR "NEON requires an arm64 target")
        endif()
    elseif(NOT _linalg_system_processor_lower MATCHES "^(x86_64|amd64|i386|i686|x86)$")
        message(FATAL_ERROR "${LINALG_VECTORIZATION_ISA} requires an x86 target")
    endif()
endif()

# Apply per-target compilation flags: /WX on MSVC (excluding CUDA TUs and the
# GNU-frontend clang-cl variant that rejects /WX as a path), -include cstdlib
# everywhere else.
function(linalg_apply_platform_compile_flags target)
    if(MSVC AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT
                                                                STREQUAL "GNU")
    )
        # Excludes CUDA: nvcc's frontend does not recognize the bare MSVC-style
        # /WX flag and misparses it as a second input file when
        # LINALG_ENABLE_CUBLAS compiles .cu sources.
        target_compile_options(${target} PRIVATE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/WX>)
    else()
        target_compile_options(${target} PRIVATE -include cstdlib)
    endif()
endfunction()

# Apply SIMD vectorization compile options to a target.  Emits the
# LINALG_SIMD_<ISA> compile definition and, on non-MSVC toolchains, the
# matching -m<isa> / -mfma flags wrapped in a $<NOT:CUDA> generator expression
# so nvcc's cl.exe host compiler is never passed GCC-style flags.
# Does nothing when LINALG_ENABLE_VECTORIZATION is OFF.
function(linalg_apply_vectorization_flags target)
    if(NOT LINALG_ENABLE_VECTORIZATION)
        return()
    endif()

    string(TOUPPER "${LINALG_VECTORIZATION_ISA}" _linalg_simd_isa_upper)
    target_compile_definitions(${target} PRIVATE "LINALG_SIMD_${_linalg_simd_isa_upper}")
    unset(_linalg_simd_isa_upper)

    # x86 ISAs need the matching -m<isa> flag to actually be assembled;
    # NEON is the aarch64 baseline (no single standard x86-style flag), so
    # "neon" adds none.  cl.exe (MSVC, non-Clang) has no -mXXX flags;
    # clang-cl sets MSVC=1 but accepts GCC-style -m flags via its Clang
    # backend, so it is explicitly included in this path.
    if(NOT MSVC OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(LINALG_VECTORIZATION_ISA STREQUAL "sse2")
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-msse2>)
        elseif(LINALG_VECTORIZATION_ISA STREQUAL "avx")
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx>)
        elseif(LINALG_VECTORIZATION_ISA STREQUAL "avx2")
            target_compile_options(
                ${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
                                  $<$<COMPILE_LANGUAGE:CXX>:-mfma>
            )
        elseif(LINALG_VECTORIZATION_ISA STREQUAL "avx512")
            # Half-packet operations also use the 128/256-bit FMA intrinsics.
            target_compile_options(
                ${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx512f>
                                  $<$<COMPILE_LANGUAGE:CXX>:-mfma>
            )
        endif()
    else()
        if(LINALG_VECTORIZATION_ISA MATCHES "^(avx|avx2|avx512)$")
            string(TOUPPER "${LINALG_VECTORIZATION_ISA}" _arch)
            target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/arch:${_arch}>")
        elseif(LINALG_VECTORIZATION_ISA STREQUAL "sse2" AND CMAKE_SIZEOF_VOID_P EQUAL 4)
            target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/arch:SSE2>")
        endif()
    endif()
endfunction()
