# Stage 3 ranker results

At 5 favourites the ranker beats the best Stage 1 method: NDCG@10 0.3475 against 0.2981, a difference of +0.0494 ± 0.0060 (95% confidence, paired over the same users).

Validation users: 5,000, the same ones as Stage 1, scored once after training. Training users: 102,061. Early stopping used 5,000 other training users and kept epoch 9.

## NDCG@10

| Method | 1 favourite | 3 favourites | 5 favourites | 10 favourites | 20 favourites |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.2749 | 0.2749 | 0.2749 | 0.2749 | 0.2749 |
| Tag Genome + popularity (Stage 1) | 0.2573 | 0.2916 | 0.2981 | 0.3039 | 0.3086 |
| Ranker | 0.2720 | 0.3268 | 0.3475 | 0.3798 | 0.4128 |

## Ranker minus Stage 1, per user

| Favourites | Difference in NDCG@10 | 95% confidence |
|---:|---:|---:|
| 1 | +0.0147 | ± 0.0057 |
| 3 | +0.0352 | ± 0.0057 |
| 5 | +0.0494 | ± 0.0060 |
| 10 | +0.0758 | ± 0.0063 |
| 20 | +0.1042 | ± 0.0064 |

## Recall@10

| Method | 1 favourite | 3 favourites | 5 favourites | 10 favourites | 20 favourites |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.2656 | 0.2656 | 0.2656 | 0.2656 | 0.2656 |
| Tag Genome + popularity (Stage 1) | 0.2475 | 0.2816 | 0.2881 | 0.2946 | 0.2994 |
| Ranker | 0.2641 | 0.3164 | 0.3368 | 0.3666 | 0.3959 |

## Distinct first picks across 5,000 users

| Method | 1 favourite | 3 favourites | 5 favourites | 10 favourites | 20 favourites |
|---|---:|---:|---:|---:|---:|
| Popularity | 15 | 15 | 15 | 15 | 15 |
| Tag Genome + popularity (Stage 1) | 181 | 95 | 82 | 66 | 58 |
| Ranker | 358 | 354 | 349 | 386 | 379 |

## Training

Settings: TrainingSettings(dim=128, hidden=512, dropout=0.2, batch_size=512, learning_rate=0.001, weight_decay=1e-05, max_epochs=40, patience=3). Device: cuda.

| Epoch | Training loss | Held-out NDCG@10 | Seconds |
|---:|---:|---:|---:|
| 1 | 8.6667 | 0.2955 | 8 |
| 2 | 7.0728 | 0.3126 | 7 |
| 3 | 6.9769 | 0.3308 | 8 |
| 4 | 6.9291 | 0.3396 | 7 |
| 5 | 6.9054 | 0.3385 | 8 |
| 6 | 6.8868 | 0.3484 | 7 |
| 7 | 6.8756 | 0.3458 | 8 |
| 8 | 6.8676 | 0.3391 | 8 |
| 9 | 6.8570 | 0.3508 | 8 |
| 10 | 6.8554 | 0.3478 | 8 |
| 11 | 6.8549 | 0.3356 | 9 |
| 12 | 6.8502 | 0.3501 | 8 |

Unrated films count as misses, so these scores understate real-world quality. They are for comparing methods.
