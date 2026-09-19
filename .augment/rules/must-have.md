---
type: "always_apply"
---

# Cross-platform compatibility

Preserve the project's supported Linux, macOS, and Windows behavior unless
the feature is explicitly platform-scoped. Keep paths portable and relative
to the project or supplied configuration; avoid machine-specific paths and
assumptions. Use standard or existing cross-platform libraries.

Guard platform-specific code and document its requirements or fallback.
Keep scripts and tests independent of the developer's local environment.
Use the repository's existing CI matrix as the source of truth for tested
platforms; do not claim validation on a platform that was not exercised.
