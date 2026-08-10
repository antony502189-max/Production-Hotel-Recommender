# Model Card — Hotel Ranking Model

## Model overview

| Field | Value |
|---|---|
| **Model family** | Histogram Gradient Boosting classifier used as a pointwise ranker |
| **Task** | Rank retrieved hotel candidates by estimated booking propensity |
| **Inputs** | Historical user-item features and hotel context |
| **Output** | Ranking score per candidate hotel |
| **Serving** | FastAPI recommendation service |
| **Evaluation** | Temporal holdout with ranking metrics |
| **Artifact** | Versioned `RecommenderBundle` serialized with joblib |

---

## Intended use

The model is intended as a reproducible reference implementation for personalized hotel ranking and ML-system engineering.

It demonstrates:

- two-stage recommendation architecture;
- point-in-time feature construction;
- temporal evaluation;
- cold-start behavior;
- model packaging and metadata;
- API serving and observability.

It is **not** intended to make real travel decisions or to represent the performance of a deployed commercial recommender.

---

## Data

The repository includes a deterministic synthetic data generator that produces:

- users;
- hotels;
- views;
- clicks;
- bookings;
- timestamps;
- latent preference structure for city, accommodation type, price and quality.

Synthetic data is used so the project can be executed end-to-end without private datasets or external credentials.

### Important limitation

Because the benchmark data is synthetic, absolute metric values should not be interpreted as expected performance on real Expedia/Booking-style traffic.

---

## Features

The ranker consumes the following feature groups.

### Item popularity

- `item_popularity`
- `historical_booking_rate`

### User-item affinity

- `city_affinity`
- `type_affinity`

### Preference fit

- `price_fit`
- `star_fit`

### Prior engagement

- `prior_views`
- `prior_clicks`

### Item context

- `log_price`
- `stars_scaled`

---

## Training methodology

Training is explicitly divided into two periods:

1. **feature-history window** — interactions used to construct historical features;
2. **label window** — later interactions used to create booking labels.

This prevents the target booking event from leaking into the features for the same training example.

Negative samples are drawn from candidate hotels that were not booked in the label window.

---

## Candidate generation

Before ranking, the service retrieves a bounded pool from:

- weighted global popularity;
- city affinity;
- accommodation-type affinity;
- deterministic popularity fallback for cold-start users.

The ranker therefore scores a compact, relevant candidate set instead of scanning the entire catalog.

---

## Offline evaluation

Reference configuration:

```text
seed          = 42
users         = 1,200
hotels        = 320
interactions  = 45,000
K             = 10
```

| Metric | Ranker | Popularity | Lift |
|---|---:|---:|---:|
| Recall@10 | **0.1075** | 0.0805 | **+33.6%** |
| NDCG@10 | **0.0673** | 0.0377 | **+78.3%** |
| MRR@10 | **0.0599** | 0.0285 | **+109.8%** |
| HitRate@10 | **0.1267** | 0.0998 | **+26.9%** |

The learned ranker and the popularity baseline are evaluated on the same temporal holdout.

---

## Cold start

For users with no historical interactions, the service falls back to deterministic popularity-driven retrieval. This avoids undefined behavior for unseen users and makes cold-start behavior directly testable.

---

## Artifact metadata

The model artifact stores:

- model version;
- training timestamp;
- training row count;
- positive-label rate;
- expected feature schema;
- deterministic training-data fingerprint;
- serving context required for retrieval and feature generation.

The artifact loader validates the feature schema before the model is exposed for serving.

---

## Monitoring considerations

The reference API exposes request counters and latency histograms through Prometheus.

A production deployment should additionally monitor:

- candidate coverage;
- score distribution;
- feature missingness;
- feature drift;
- recommendation diversity;
- cold-start share;
- click-through and booking conversion;
- online guardrail metrics;
- training-serving skew.

---

## Risks and limitations

### Synthetic benchmark

Performance is measured on simulated behavior and is not evidence of real-world business lift.

### Pointwise ranking

The estimator optimizes booking propensity per candidate rather than a listwise ranking objective.

### Simple retrieval

The candidate generator does not currently use embeddings, collaborative filtering or approximate nearest-neighbor search.

### Limited contextual information

The current feature set does not include real travel-search context such as destination query, trip dates, party size, device, geography or session intent.

---

## Production extensions

Recommended extensions for a real system:

- collaborative-filtering / embedding retrieval;
- ANN index;
- session and query features;
- point-in-time feature store;
- experiment tracking and model registry;
- automated retraining;
- drift monitoring;
- diversity and business-rule re-ranking;
- online experimentation;
- canary or shadow deployment.
