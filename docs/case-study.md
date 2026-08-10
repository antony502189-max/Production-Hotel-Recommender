# Production Hotel Recommender — Case Study

## Executive summary

This project is a production-oriented recommendation system for personalized hotel ranking. It is designed to show the complete ML lifecycle rather than only model training: candidate retrieval, leakage-safe feature construction, ranking, temporal evaluation, artifact packaging, API serving, observability and CI.

The reference benchmark uses deterministic synthetic interaction data and compares the learned ranker against a popularity baseline on the same future holdout.

**Reference result:** Recall@10 improves from **0.0805** to **0.1075**, a **+33.6% relative lift**.

> The dataset is synthetic by design. The value of the project is the engineering and evaluation discipline, not a claim of commercial travel performance.

---

## 1. Problem

A useful hotel recommender must answer a practical systems question:

> Given a user's historical behavior and a catalog of hotels, which small set of hotels should be ranked highest for the next request?

A naive implementation can easily produce misleading results. Common failure modes include:

- using future behavior while constructing features;
- evaluating with a random split that does not reflect deployment;
- scoring the full catalog on every request;
- ignoring unseen users;
- reporting only classification metrics;
- shipping a model without a reproducible artifact contract;
- exposing inference without health or monitoring signals.

The project was designed specifically around those failure modes.

---

## 2. Solution design

The system uses a two-stage recommendation architecture.

### Stage A — candidate retrieval

A bounded candidate pool is produced from inexpensive signals:

- global weighted popularity;
- user-city affinity;
- user accommodation-type affinity;
- deterministic cold-start fallback.

The retrieval layer reduces the amount of work required by the ML ranker and gives unseen users a predictable recommendation path.

### Stage B — supervised ranking

Candidates are ranked with `HistGradientBoostingClassifier` using features derived only from historical information available before the target event.

Feature families include:

- item popularity and historical booking rate;
- city and hotel-type affinity;
- price and star-rating fit;
- prior views and clicks;
- item context such as price and normalized stars.

---

## 3. Leakage prevention

The training pipeline separates an earlier history window from a later label window before training examples are constructed.

This means the positive booking event used as a target cannot leak into the same row's feature values.

That design matters because leakage can make an offline recommender appear dramatically better while producing no corresponding online gain.

---

## 4. Evaluation strategy

The system is evaluated on a temporal holdout using ranking metrics:

- Recall@K;
- NDCG@K;
- MRR@K;
- HitRate@K.

A popularity recommender is evaluated on the exact same holdout so the ML model must demonstrate incremental ranking value rather than merely report an isolated score.

### Reference benchmark

| Metric | ML ranker | Popularity baseline | Relative lift |
|---|---:|---:|---:|
| Recall@10 | **0.1075** | 0.0805 | **+33.6%** |
| NDCG@10 | **0.0673** | 0.0377 | **+78.3%** |
| MRR@10 | **0.0599** | 0.0285 | **+109.8%** |
| HitRate@10 | **0.1267** | 0.0998 | **+26.9%** |

Configuration:

```text
seed          = 42
users         = 1,200
hotels        = 320
interactions  = 45,000
K             = 10
```

---

## 5. Serving architecture

The trained model is packaged in a versioned `RecommenderBundle` together with the metadata required to serve it safely.

The service exposes:

- `GET /v1/recommendations/{user_id}` for ranked recommendations;
- `GET /v1/model` for model metadata and fingerprint;
- `GET /health` for readiness;
- `GET /metrics` for Prometheus-compatible telemetry;
- `GET /docs` for OpenAPI / Swagger documentation.

The application supports degraded startup: if the artifact is missing, operational endpoints remain available while recommendation requests return `503`.

---

## 6. Artifact contract and reproducibility

The serialized bundle includes:

- trained estimator;
- hotel catalog;
- historical interactions;
- item statistics;
- user profiles;
- exact feature schema;
- model version;
- training timestamp;
- training row count;
- positive-label rate;
- deterministic data fingerprint.

The loader validates the feature schema before serving to prevent silent incompatibility between application code and a stale model artifact.

All demo data generation is seed-controlled, and evaluation writes machine-readable results to `reports/offline_metrics.json`.

---

## 7. Production engineering signals

The repository intentionally demonstrates engineering concerns that are normally absent from notebooks:

- package-based Python project layout;
- typed settings;
- deterministic data validation;
- reproducible train/evaluate CLIs;
- explicit cold-start behavior;
- model provenance metadata;
- schema validation on artifact load;
- FastAPI serving;
- Prometheus metrics;
- non-root Docker container;
- automated package installation, compilation and tests in GitHub Actions.

---

## 8. Trade-offs

### Why gradient boosting?

The project uses a lightweight tabular ranker because it is fast to train, easy to inspect and inexpensive to serve. This keeps the repository runnable while still demonstrating a realistic ranking pipeline.

### Why synthetic data?

Synthetic data makes the repository fully reproducible and avoids bundling proprietary or privacy-sensitive behavior. The trade-off is that the benchmark measures implementation discipline rather than market-level recommendation quality.

### Why a simple retrieval layer?

Popularity and behavioral affinity are transparent and deterministic. A commercial system would typically replace or augment them with collaborative filtering, embeddings or ANN retrieval.

---

## 9. What I would add in a real travel platform

The next production steps would be:

1. ANN or collaborative-filtering retrieval;
2. search/session context, destination, dates and guest constraints;
3. point-in-time-correct feature store;
4. model registry and experiment tracking;
5. scheduled retraining and feature pipelines;
6. drift and data-quality monitoring;
7. diversity and business-rule re-ranking;
8. online A/B testing with business guardrails;
9. canary or shadow model rollout;
10. privacy, consent and retention policies.

---

## 10. What this project demonstrates

For a reviewer, the repository is intended to demonstrate competence across four layers:

**Machine Learning** — ranking, feature engineering, baselines and ranking metrics.

**Data Science** — temporal validation, leakage prevention and reproducible experimentation.

**ML Engineering** — artifact contracts, model metadata, serving, cold start and observability.

**Software Engineering** — package structure, testing, CI, Docker and documented operational behavior.
