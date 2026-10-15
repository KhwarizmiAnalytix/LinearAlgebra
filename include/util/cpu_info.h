/*
 * LinearAlgebra — CPU cache-size introspection.
 *
 * Reconstructed minimal stand-in for the (orphaned) util/cpu_info.h this
 * code used to depend on. Its only call site is the scalar backend's
 * blocked-GEMM blocking-size heuristic (src/matrix_operation/
 * multiplication.cpp's LINALG_HAS_VECTORIZATION block, on by default — see
 * LINALG_ENABLE_VECTORIZATION in CMakeLists.txt).
 *
 * When LINALG_HAS_CPUINFO (CMakeLists.txt's find_package(cpuinfo)) is 1, real
 * cache sizes come from pytorch/cpuinfo's cpuinfo_get_l1d_cache/l2_cache/
 * l3_cache. Otherwise this falls back to conservative, commonly-seen defaults
 * for a modern x86-64 desktop/server core.
 */
#pragma once

#include <cstddef>

#if LINALG_HAS_CPUINFO
#include <cpuinfo.h>
#endif

namespace linalg
{
namespace cpu_info
{

inline void cpuinfo_cach(
    std::ptrdiff_t& l1, std::ptrdiff_t& l2, std::ptrdiff_t& l3, std::ptrdiff_t& l3_count)
{
#if LINALG_HAS_CPUINFO
    cpuinfo_initialize();

    const struct cpuinfo_cache* l1d_cache = cpuinfo_get_l1d_cache(0);
    const struct cpuinfo_cache* l2_cache  = cpuinfo_get_l2_cache(0);
    const struct cpuinfo_cache* l3_cache  = cpuinfo_get_l3_cache(0);

    l1 = l1d_cache ? static_cast<std::ptrdiff_t>(l1d_cache->size) : std::ptrdiff_t{32} * 1024;
    l2 = l2_cache ? static_cast<std::ptrdiff_t>(l2_cache->size) : std::ptrdiff_t{256} * 1024;
    l3 = l3_cache ? static_cast<std::ptrdiff_t>(l3_cache->size)
                  : std::ptrdiff_t{8} * 1024 * 1024;
    l3_count = static_cast<std::ptrdiff_t>(cpuinfo_get_l3_caches_count());
    if (l3_count == 0)
    {
        l3_count = 1;  // no L3 domain reported (e.g. some ARM/mobile parts)
    }
#else
    l1       = std::ptrdiff_t{32} * 1024;         // 32 KiB L1 data cache (conservative default)
    l2       = std::ptrdiff_t{256} * 1024;        // 256 KiB L2 cache (conservative default)
    l3       = std::ptrdiff_t{8} * 1024 * 1024;   // 8 MiB L3 cache (conservative default)
    l3_count = 1;                                 // one L3 domain (conservative default)
#endif
}

}  // namespace cpu_info
}  // namespace linalg
