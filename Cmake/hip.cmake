# =============================================================================
# hip.cmake — HIP toolchain, architecture detection, and per-target flags
# for LinearAlgebra.
#
# Included by CMakeLists.txt when LINALG_GPU_BACKEND=hip. After inclusion:
#   LINALG_HIP_ARCHITECTURES    — gfxXXX semicolon list (e.g. "gfx908;gfx90a;gfx1100")
#   LINALG_HIP_VERSION          — integer major*100+minor (matches PyTorch's
#                                  TORCH_HIP_VERSION / Scalars' SCALARS_HIP_VERSION)
#   LINALG_ROCM_PREFIX          — resolved ROCm SDK root (set by rocm_prefix.cmake)
#
# Callers must include before add_library() so that CMAKE_HIP_ARCHITECTURES
# and CMAKE_HIP_STANDARD are already set when enable_language(HIP) configures
# the ROCm toolchain.
#
# Function exported after inclusion:
#   linalg_apply_hip_flags(<target>)
#       Sets HIP_STANDARD / HIP_ARCHITECTURES target properties and drops
#       host-incompatible hip::device CXX flags from <target>. Call after
#       add_library().
#
# Design follows Scalars' cmake/gpu_backend.cmake (HIP block) and PyTorch's
# cmake/Hip.cmake (TORCH_HIP_VERSION integer encoding, ROCM_PATH discovery,
# hip::device CXX-flag stripping, and architecture default list).
# =============================================================================
include_guard(GLOBAL)

if(WIN32)
    message(
        FATAL_ERROR
            "LINALG_GPU_BACKEND=hip is not supported on Windows (HIP/ROCm is Unix-only). "
            "Use LINALG_GPU_BACKEND=cuda on Windows, or build on Linux for HIP."
    )
endif()

# ---------------------------------------------------------------------------
# ROCm prefix — must run before find_package(hip) so hip-config.cmake is
# findable even when /opt/rocm is not in CMAKE_PREFIX_PATH by default.
# ---------------------------------------------------------------------------
include("${CMAKE_CURRENT_LIST_DIR}/rocm_prefix.cmake")

# ---------------------------------------------------------------------------
# LINALG_HIP_ARCHITECTURES
#
# Semicolon-separated gfxXXX list. Default covers the MI series and RDNA 2/3
# families most commonly targeted by production HIP workloads, matching
# Scalars' SCALARS_GPU_ARCHITECTURES default. Set LINALG_HIP_ARCHITECTURES or
# CMAKE_HIP_ARCHITECTURES on the command line to override; CMAKE_HIP_ARCHITECTURES
# takes precedence when already set to a non-"native" value, mirroring the
# LINALG_CUDA_ARCHITECTURES / CMAKE_CUDA_ARCHITECTURES parallel.
# ---------------------------------------------------------------------------
if(NOT DEFINED LINALG_HIP_ARCHITECTURES)
    if(DEFINED CMAKE_HIP_ARCHITECTURES
       AND CMAKE_HIP_ARCHITECTURES
       AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "native"
    )
        set(LINALG_HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
            CACHE STRING "HIP GPU architecture list for LinearAlgebra (gfxXXX)"
        )
    else()
        set(LINALG_HIP_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100"
            CACHE STRING "HIP GPU architecture list for LinearAlgebra (gfxXXX)"
        )
    endif()
endif()

# Resolve architectures before package discovery and compiler detection. This
# also avoids requiring a physical GPU on headless build machines.
if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${LINALG_HIP_ARCHITECTURES}")
endif()
linalg_setup_rocm_prefix()

# CMake's HIP language requires ROCm Clang directly, never the hipcc wrapper.
# Respect an explicit compiler or HIPCXX before probing the discovered SDK.
if(NOT CMAKE_HIP_COMPILER AND NOT DEFINED ENV{HIPCXX} AND LINALG_ROCM_PREFIX)
    find_program(_linalg_hip_clang NAMES clang++
        PATHS "${LINALG_ROCM_PREFIX}/llvm/bin" "${LINALG_ROCM_PREFIX}/bin"
        NO_DEFAULT_PATH NO_CACHE)
    if(_linalg_hip_clang)
        set(CMAKE_HIP_COMPILER "${_linalg_hip_clang}" CACHE FILEPATH "HIP compiler")
    endif()
    unset(_linalg_hip_clang)
endif()

# ---------------------------------------------------------------------------
# CMAKE_HIP_STANDARD — inherit from the project's CXX standard. Must be set
# before enable_language(HIP) so the ROCm toolchain is configured correctly
# at first-run detection.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_HIP_STANDARD)
    set(CMAKE_HIP_STANDARD ${LINALG_CXX_STANDARD})
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
endif()

# ---------------------------------------------------------------------------
# find_package: hip (runtime + device), hipblas, hipsolver.
# ---------------------------------------------------------------------------
find_package(hip CONFIG REQUIRED)
find_package(hipblas CONFIG REQUIRED)
find_package(hipsolver CONFIG REQUIRED)

enable_language(HIP)

# ---------------------------------------------------------------------------
# _linalg_drop_hip_cxx_device_flags(<target>)
#
# hip::device injects "-x hip", "--offload-arch", and "--hip-link" into the
# INTERFACE compile/link options so that consumers compiled as plain CXX TUs
# get device-enabled compilation. This is the "compile HIP as CXX" legacy
# workflow; CMake's HIP language supplies those flags for .hip files instead.
# On a host g++ the CXX-path flags cause compile errors, so strip them.
#
# Ported from Scalars' cmake/gpu_backend.cmake (_scalars_drop_hip_cxx_device_flags).
# ---------------------------------------------------------------------------
function(_linalg_drop_hip_cxx_device_flags target)
    if(NOT TARGET "${target}")
        return()
    endif()
    foreach(_prop IN ITEMS INTERFACE_COMPILE_OPTIONS INTERFACE_LINK_LIBRARIES INTERFACE_LINK_OPTIONS)
        get_property(_prop_set TARGET "${target}" PROPERTY "${_prop}" SET)
        if(NOT _prop_set)
            continue()
        endif()
        get_property(_opts TARGET "${target}" PROPERTY "${_prop}")
        set(_kept "")
        set(_dropped 0)
        set(_i 0)
        list(LENGTH _opts _n)
        while(_i LESS _n)
            list(GET _opts ${_i} _opt)
            set(_consume 1)
            set(_poison 0)
            if(_opt STREQUAL "-x" OR _opt MATCHES "(^|SHELL:)-x$")
                math(EXPR _j "${_i} + 1")
                if(_j LESS _n)
                    list(GET _opts ${_j} _next)
                    if(_next STREQUAL "hip" OR _next MATCHES "^hip>")
                        set(_poison 1)
                        set(_consume 2)
                    endif()
                endif()
            endif()
            if(NOT _poison AND _opt MATCHES "--offload-arch|--hip-link|-xhip|-x[ ;]hip|SHELL:-x;hip"
            )
                set(_poison 1)
            endif()
            if(_poison)
                math(EXPR _dropped "${_dropped} + ${_consume}")
            else()
                list(APPEND _kept "${_opt}")
            endif()
            math(EXPR _i "${_i} + ${_consume}")
        endwhile()
        if(_dropped GREATER 0)
            list(LENGTH _kept _kept_len)
            if(_kept_len EQUAL 0)
                set_property(TARGET "${target}" PROPERTY "${_prop}" "")
            else()
                set_property(TARGET "${target}" PROPERTY "${_prop}" ${_kept})
            endif()
            message(
                STATUS
                    "LinearAlgebra: dropped ${_dropped} host-incompatible ${_prop} from ${target}"
            )
        endif()
    endforeach()
endfunction()

_linalg_drop_hip_cxx_device_flags(hip::device)
_linalg_drop_hip_cxx_device_flags(hip::host)

# ---------------------------------------------------------------------------
# LINALG_HIP_VERSION — integer major*100+minor, matching PyTorch's
# TORCH_HIP_VERSION and Scalars' SCALARS_HIP_VERSION encoding. Used in
# include/common/backend.h and any other header that needs to branch on the
# ROCm API version at compile time.
# ---------------------------------------------------------------------------
set(LINALG_HIP_VERSION 0)
if(hip_FOUND AND DEFINED hip_VERSION)
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _linalg_hip_ver_match "${hip_VERSION}")
    if(_linalg_hip_ver_match)
        math(EXPR LINALG_HIP_VERSION "${CMAKE_MATCH_1} * 100 + ${CMAKE_MATCH_2}")
    endif()
    unset(_linalg_hip_ver_match)
endif()
message(
    STATUS
        "LinearAlgebra: HIP GPU backend enabled (archs: ${LINALG_HIP_ARCHITECTURES}, LINALG_HIP_VERSION=${LINALG_HIP_VERSION})"
)

# ---------------------------------------------------------------------------
# linalg_apply_hip_flags(<target>)
#
# Sets HIP_STANDARD and HIP_ARCHITECTURES as target properties, applies
# -ffast-math for HIP TUs, and links hip::host. Call after
# add_library().
#
# The HIP language handles device linking without linking hip::device.
#
# Design parallels scalars_apply_gpu_backend() in Scalars' cmake/gpu_backend.cmake.
# ---------------------------------------------------------------------------
function(linalg_apply_hip_flags target)
    set_target_properties(
        ${target}
        PROPERTIES HIP_STANDARD ${LINALG_CXX_STANDARD}
                   HIP_STANDARD_REQUIRED ON
                   HIP_ARCHITECTURES "${LINALG_HIP_ARCHITECTURES}"
    )
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>)
    if(TARGET hip::host)
        target_link_libraries(${target} PUBLIC hip::host)
    endif()
endfunction()
