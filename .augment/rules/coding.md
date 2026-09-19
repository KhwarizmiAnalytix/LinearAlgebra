---
type: "always_apply"
---

# C++ coding standards

These rules apply to first-party C++. Repository-specific contracts in
`CLAUDE.md`, public headers, and nearby code take precedence over defaults.

## Naming and formatting

- Use `snake_case` for new classes, functions, namespaces, and enum values;
  use `snake_case_` for members, never an `m_` prefix. Prefer `kConstantName`
  for constants unless the surrounding API has an established convention.
- Preserve public API names and match the surrounding file. Avoid unrelated
  renames while applying this guidance.
- Follow the existing `.clang-format` and `.clang-tidy` configuration when
  present. Keep code compatible with the configured C++ language version.

## Errors and interfaces

- Prefer return values (`bool`, optional values, or existing result types)
  for new expected-failure paths. Preserve documented exception APIs and
  boundary code that handles throwing dependencies; consult `CLAUDE.md` for
  repository-specific exceptions. Do not rewrite those contracts to match
  a general no-exceptions preference.
- Match the repository's public include paths. Use angle brackets for
  external dependencies and quotes for project headers. Follow configured
  include ordering, with standard, third-party, and project groups.
- Use existing export, visibility, and unused-parameter macros from the
  owning library, in the placement its headers use. Do not import macros
  from another library or invent them for header-only templates.

## Ownership, concurrency, and robustness

- Use RAII and smart pointers for ownership; raw pointers are non-owning
  unless an existing low-level API explicitly documents otherwise.
- Protect shared mutable state with scoped locks or suitable atomics;
  document thread-safety and lifetime guarantees. Prefer condition variables
  to busy waiting and explicit ownership of worker-thread lifetimes.
- Check lengths, ranges, and return values at input boundaries. Use safe
  standard-library operations and preserve existing resource limits.
- Keep interfaces focused and dependencies explicit. Document intent,
  ownership, and non-obvious behavior rather than restating the code.

For behavioral changes, follow `testing.md`; for static analysis, use the
`clang-tidy` skill. Coverage targets and merge gates come from this
repository's actual configuration, not an assumed organization-wide number.
