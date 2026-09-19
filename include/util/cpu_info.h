/*
 * LinearAlgebra — CPU cache-size introspection.
 *
 * Reconstructed minimal stand-in for the (orphaned) util/cpu_info.h this
 * code used to depend on. The only call site (matrix_multiplication.cxx's
 * gemm blocking-size heuristic) is itself gated behind the never-defined
 * LINALG_VECTORIZED macro (see include/matrix_operation/
 * matrix_multiplication.cxx), so this header only needs to exist and be
 * self-consistent for the unconditional #include to compile — it is not on
 * any currently-live code path.
 *
 * Real hardware cache detection (CPUID leaf 0x4 / 0x8000001D on x86,
 * sysctlbyname("hw.l1dcachesize", ...) on macOS, /sys/devices/system/cpu/
 * cpu0/cache/ on Linux) is a documented follow-up, not lost functionality:
 * the values below are conservative, commonly-seen defaults for a modern
 * x86-64 desktop/server core.
 */
#pragma once

#include <cstddef>

namespace linalg
{
namespace cpu_info
{

inline void cpuinfo_cach(
    std::ptrdiff_t& l1, std::ptrdiff_t& l2, std::ptrdiff_t& l3, std::ptrdiff_t& l3_count)
{
    l1       = 32 * 1024;         // 32 KiB L1 data cache (conservative default)
    l2       = 256 * 1024;        // 256 KiB L2 cache (conservative default)
    l3       = 8 * 1024 * 1024;   // 8 MiB L3 cache (conservative default)
    l3_count = 1;                 // one L3 domain (conservative default)
}

}  // namespace cpu_info
}  // namespace linalg
