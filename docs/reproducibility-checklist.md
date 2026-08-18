# Reproducibility checklist

Use this checklist when reproducing the reference benchmark or validating a new experiment.

## Environment

- Use Python 3.11 or newer.
- Install the project from `pyproject.toml` with development dependencies.
- Keep the repository working tree clean before recording benchmark results.

## Data and split

- Keep the random seed fixed when comparing implementations.
- Use the same synthetic-data configuration for both the ML ranker and the popularity baseline.
- Preserve the temporal split: feature history must precede the label window.
- Do not construct user or item features from events in the future holdout.

## Evaluation

- Evaluate both approaches against the same future interactions.
- Report Recall@K, NDCG@K, MRR@K and HitRate@K together.
- Record the chosen value of `K` and the generated dataset dimensions.
- Treat relative lift as meaningful only when the absolute metric values and baseline are also shown.

## Artifact validation

- Confirm the serialized bundle contains its model version and exact feature schema.
- Verify the training-data fingerprint before comparing artifacts from different runs.
- Check that the service rejects incompatible feature schemas instead of serving silently.

## Service verification

- Start the API from the newly produced artifact.
- Verify `/health`, `/v1/model`, `/metrics` and at least one recommendation request.
- Confirm recommendation output is deterministic for the same artifact and request inputs.

## CI

Run the same basic validation used by GitHub Actions before publishing results:

```bash
make lint
make test
make evaluate
```

If the benchmark changes intentionally, document the changed configuration and update the benchmark methodology rather than replacing only the headline metric.
