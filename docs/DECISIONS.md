# Decisions log

Approved change requests and resolved [OPEN] items, newest last.

## CR-001: add mypy (approved 2026-11-19)

```
CHANGE REQUEST
What I want to change: add mypy (dev-only) and run it in strict mode on qsim/
Why: owner wants TypeScript-style type safety; type hints alone are never checked
What it affects: pyproject.toml (dev deps + config), one extra check command; no runtime change
Cost/risk: small; NumPy/CuPy types are sometimes loose, so a few `# type: ignore` lines may be needed
Alternative that keeps the current scope: write hints but never check them
Owner decision: approved   Date: 2026-11-19
```

mypy added to the dev/test dependency list.
