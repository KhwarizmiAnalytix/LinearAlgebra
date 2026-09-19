---
type: "always_apply"
---

# Third-party dependencies

Keep vendored source, submodules, third-party build files, and downloaded
dependencies unchanged. Consult `.gitmodules` and the dependency layout in
`CLAUDE.md`; directory spelling and capitalization vary by repository.

Resolve integration issues in first-party wrappers, compatibility headers,
compile definitions, or build configuration. If that cannot solve the issue,
explain the blocker before proposing a dependency update or upstream fix.
Do not patch a vendored library in place to make a build or lint check pass.
