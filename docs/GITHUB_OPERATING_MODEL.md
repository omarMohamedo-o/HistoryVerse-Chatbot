# GitHub Operating Model

This document outlines the version control, branching, pull request, and deployment guidelines for the HistoryVerse repository.

## 1. Branch Hierarchy

```
main (Production)
  ▲
  │ (Staged PR when 100% complete)
microservices (Integration Branch)
  ▲
  ├── feat/<short-name>
  └── fix/<short-name>
```

- Direct pushes to `main` and `microservices` are prohibited.
- All active development branches are prefixed with `feat/` or `fix/` and branched directly from `microservices`.

## 2. Commit Policy
- Commit subject format: `type: summary`
  - Valid types: `feat`, `fix`, `docs`, `test`, `ci`, `refactor`, `chore`, `perf`, `build`, `revert`.
- Commits must stage explicit paths only:
  ```bash
  git add src/historyverse/services/llm/factory.py tests/unit/test_llm_factory.py
  ```
  Never use `git add .` or `git add -A`.
- **Strict Prohibition on AI Attribution**:
  Commits, pull requests, code comments, and documentation must not contain co-author trailers, tool signatures, or assistant mentions.

## 3. Pull Request Guidelines
Each PR targeting `microservices` must include:
1. Clear explanation of changes and rationale.
2. Verification steps and test evidence.
3. Updated documentation (`PLAN.md`, `CODE_MAP.md`, `RUNBOOKS.md`).
4. A changelog entry created via `python scripts/docs_log.py new changelog "<title>"`.
5. Passing CI checks for linting, typing, and testing.
