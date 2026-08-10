# Contributing

Thanks for taking the time to improve Production Hotel Recommender.

## Development setup

```bash
python -m venv .venv
```

Activate the environment and install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Before opening a pull request

Run the local quality checks:

```bash
make lint
make test
```

For changes that affect ranking behavior or feature logic, also run:

```bash
make evaluate
```

## Contribution principles

Changes should preserve the core guarantees of the project:

- no target leakage from future interactions;
- deterministic behavior when a random seed is fixed;
- explicit cold-start behavior;
- stable model artifact contracts;
- reproducible offline evaluation;
- tests for behavior-changing code.

## Pull requests

Keep pull requests focused and explain:

1. what changed;
2. why the change is needed;
3. how it was validated;
4. whether offline metrics changed;
5. whether the model artifact contract changed.

If ranking behavior changes, include before/after benchmark numbers when practical.

## Commit style

Use concise conventional-style messages, for example:

```text
feat: add session-aware candidate retrieval
fix: prevent future interactions from entering features
test: cover cold-start recommendation ordering
docs: document benchmark methodology
```

## Scope

This repository is a portfolio/reference implementation. Contributions should improve clarity, reproducibility, ML-system design or engineering quality without pretending synthetic benchmark results are equivalent to production travel performance.
