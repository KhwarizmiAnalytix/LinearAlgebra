---
type: "always_apply"
---

# C++ builder conventions

Apply these conventions when adding or editing a C++ builder; they do not
require introducing builders into unrelated code or Python packages.

- Name new builder classes with an `_builder` suffix.
- Use `with_<field>` setters with one parameter and a fluent return type.
- Use the repository's own namespace and established ownership type.
  Do not introduce another project's pointer aliases or namespace.
- Keep member names in `snake_case_` and document the builder's purpose,
  setter behavior, and ownership transfer.
- Preserve existing public interfaces unless the requested change includes
  changing them.
