# Offline Benchmark Methodology

## Goal

The benchmark answers one question:

> Does the learned ranking stage improve recommendation quality over a simple popularity baseline when both are evaluated on the same future interactions?

The benchmark is intentionally deterministic and reproducible.

---

## Reference configuration

```text
random seed   = 42
users         = 1,200
hotels        = 320
interactions  = 45,000
cutoff K      = 10
```

Run it with:

```bash
python scripts/evaluate.py \
  --users 1200 \
  --hotels 320 \
  --interactions 45000 \
  --k 10
```

The script writes machine-readable output to:

```text
reports/offline_metrics.json
```

---

## Evaluation split

Interactions are ordered by time and divided into:

- **training history** — used to learn preferences, retrieval statistics and the ranking model;
- **future holdout** — used only for evaluation.

No holdout interactions are used to build historical features for the evaluated recommendation request.

This design is more representative of deployment than a random row-level split because recommendation systems operate forward in time.

---

## Compared systems

### 1. Popularity baseline

Hotels are ranked from historical interaction popularity only.

This baseline is deliberately simple but important: a learned recommender should demonstrate value over a solution that is cheap, stable and easy to deploy.

### 2. Hybrid retrieval + ML ranker

The learned system:

1. retrieves a bounded candidate pool from popularity and behavioral affinity;
2. builds user-item features from historical information;
3. scores candidates with the gradient-boosting ranker;
4. returns the top-K hotels.

---

## Metrics

### Recall@K

Measures how much of the relevant future set appears in the top-K recommendations.

High Recall@K means the recommender is successfully surfacing items the user later interacts with positively.

### NDCG@K

Discounted cumulative gain rewards relevant recommendations more strongly when they appear closer to the top of the list.

NDCG therefore measures both retrieval success and ordering quality.

### MRR@K

Mean Reciprocal Rank focuses on the position of the first relevant recommendation.

It is useful when early ranking positions matter disproportionately.

### HitRate@K

Measures the fraction of evaluated users for whom at least one relevant item appears in the top-K list.

---

## Reference results

| Metric | ML ranker | Popularity baseline | Relative improvement |
|---|---:|---:|---:|
| Recall@10 | **0.1075** | 0.0805 | **+33.6%** |
| NDCG@10 | **0.0673** | 0.0377 | **+78.3%** |
| MRR@10 | **0.0599** | 0.0285 | **+109.8%** |
| HitRate@10 | **0.1267** | 0.0998 | **+26.9%** |

---

## Why the baseline comparison is meaningful

Both systems are evaluated using:

- the same users;
- the same hotel catalog;
- the same training history;
- the same future holdout;
- the same cutoff K.

This avoids a common presentation error where a model and baseline are measured on different samples or splits.

---

## Reproducibility guarantees

The demo pipeline controls randomness with an explicit NumPy seed.

The model artifact also stores a deterministic training-data fingerprint, making it possible to inspect whether an artifact corresponds to the expected training input.

---

## What the benchmark does not prove

The data is synthetic, so these values must not be interpreted as commercial travel KPIs or expected production lift.

The benchmark validates:

- implementation correctness;
- temporal evaluation discipline;
- baseline comparison;
- reproducibility;
- ranking-metric reporting;
- end-to-end system behavior.

It does **not** validate:

- real-user preference fidelity;
- revenue impact;
- production latency at scale;
- fairness across real populations;
- online experiment lift.

---

## Production evaluation extensions

For a real deployment, the offline evaluation layer should be extended with:

- candidate recall before ranking;
- catalog coverage;
- novelty and diversity;
- segment-level metrics;
- calibration analysis;
- temporal backtesting across multiple windows;
- bootstrap confidence intervals;
- latency and throughput benchmarks;
- online A/B testing with business guardrails.
