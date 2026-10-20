# Contributing

The operator (`@omarMohamedo-o`) is the maintainer and the only author. Read the working agreement in the project
context file at the repo root, `INTENT.md` and `PLAN.md` before starting; `docs/CODE_MAP.md` is the index to the code.

## Branch flow
- Work on `feat/<short>` or `fix/<short>` branched from `microservices` (the integration branch).
- Every step is a PR into `microservices`. At 100% one PR goes `microservices` -> `main`; then the merged branches
  are deleted and the staged production move starts.
- Never push to `main` or `microservices` directly. Never force-push. The operator merges every PR.
- These rules are documented and checked, **not technically enforced** (GitHub Free, private repos): see
  `docs/GITHUB_OPERATING_MODEL.md` for what is enforced and the manual merge checklist.

## Commits
- Subject: `type: summary` with type one of `feat`, `fix`, `docs`, `test`, `ci`, `refactor`, `chore`
  (`perf`, `build`, `revert` are accepted too). The `PR policy` check reports every commit in a PR that does not.
- One finished change per commit: the code, its tests and its doc updates together (card, `docs/CODE_MAP.md`,
  `docs/RUNBOOKS.md`, a changelog fragment in `docs/changelog/` (`python scripts/docs_log.py new changelog "title"`), PLAN.md ticks).
- Stage explicit paths only (`git add <path> <path>`), never `git add -A` or `git add .` - another writer may be
  working in the tree. Scratch and experiment scripts never enter the repo.
- **No AI/assistant attribution anywhere:** no co-author trailers, tool-generated footers or assistant names in
  commits, PR titles or bodies, code comments, docs, contributor lists or release notes. The `PR policy` check fails
  (reports red on) a PR that carries any.
- No secrets in the diff. Services read secrets from Vault; `.env` is for local dev and CI only.

## Pull requests
- Fill in the PR template: what, why, how it was proven, docs updated, risk and rollback, server steps.
- Server steps (deploy, `.env`, cron, keys) each need the operator's approval and are logged in a changelog fragment in `docs/changelog/` (`python scripts/docs_log.py new changelog "title"`).
- Expected green before merge: CI (`test`, MySQL, RabbitMQ, Vault gates) and `PR policy`. A red check does not
  block the merge button on the current plan - the operator checks it by hand.

## Bugs
- Write a failing test first; do not change that test until the bug is solved.
- Record every bug as a fragment in `docs/bugs/` (`python scripts/docs_log.py new bug BUG-NNN "title"`) and open an issue with the Bug form (it asks for the failing test).
- Never skip, allowlist or weaken a test to go green.
