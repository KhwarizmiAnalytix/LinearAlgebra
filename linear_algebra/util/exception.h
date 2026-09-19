/*
 * LinearAlgebra — error-reporting helper.
 *
 * Reconstructed minimal stand-in for the (orphaned) util/exception.h this
 * code used to depend on. matrix_operation only ever calls LINALG_THROW(msg)
 * or LINALG_THROW(msg, extra) to report an unsupported enum value / invalid
 * configuration; there is no AAD/logging framework left in this repo to
 * integrate with, so this throws std::runtime_error directly (the
 * historical "THROW" exception mode).
 */
#pragma once

#include <sstream>
#include <stdexcept>
#include <utility>

namespace linalg
{
namespace util
{

template <typename... Args>
[[noreturn]] inline void throw_runtime_error(const char* file, int line, Args&&... args)
{
    std::ostringstream oss;
    oss << file << ":" << line << ": ";
    (oss << ... << args);
    throw std::runtime_error(oss.str());
}

}  // namespace util
}  // namespace linalg

#define LINALG_THROW(...) ::linalg::util::throw_runtime_error(__FILE__, __LINE__, __VA_ARGS__)
