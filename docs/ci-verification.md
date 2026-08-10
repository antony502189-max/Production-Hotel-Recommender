# Continuous Integration contract

The repository keeps the primary `CI` workflow intentionally deterministic. The badge in the README represents checks that must be stable on every push and pull request rather than optional infrastructure validation.

## Blocking checks

The workflow runs on Python 3.11 and performs four steps:

1. upgrade `pip`, `setuptools`, and `wheel`;
2. install the package with its test dependencies;
3. compile `src`, `scripts`, and `tests` with `compileall`;
4. execute the complete pytest suite.

A commit is considered CI-clean only when all four steps succeed.

## Local quality checks

Additional developer checks remain available locally through the Makefile, including linting, offline evaluation, and Docker-based validation. These are deliberately kept outside the primary badge workflow so external Docker/build-tool changes do not turn the repository's core Python health indicator red.
