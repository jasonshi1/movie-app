# Stage 3 ranker results

At 5 favourites the ranker beats the best Stage 1 method: NDCG@10 0.3475 against 0.2900, a difference of +0.0575 ± 0.0061 (95% confidence, paired over the same users).

Validation users: 5,000, the same ones as Stage 1, scored once after training. Training users: 106,781. Early stopping used 5,000 other training users and kept epoch 13.

## NDCG@10

| Method | 1 favourite | 3 favourites | 5 favourites | 10 favourites | 20 favourites |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.2521 | 0.2521 | 0.2521 | 0.2521 | 0.2521 |
| Tag Genome + popularity (Stage 1) | 0.2474 | 0.2809 | 0.2900 | 0.2990 | 0.3052 |
| Ranker | 0.2640 | 0.3254 | 0.3475 | 0.3788 | 0.4117 |

## Ranker minus Stage 1, per user

| Favourites | Difference in NDCG@10 | 95% confidence |
|---:|---:|---:|
| 1 | +0.0166 | ± 0.0061 |
| 3 | +0.0445 | ± 0.0060 |
| 5 | +0.0575 | ± 0.0061 |
| 10 | +0.0798 | ± 0.0062 |
| 20 | +0.1065 | ± 0.0063 |

## Recall@10

| Method | 1 favourite | 3 favourites | 5 favourites | 10 favourites | 20 favourites |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.2363 | 0.2363 | 0.2363 | 0.2363 | 0.2363 |
| Tag Genome + popularity (Stage 1) | 0.2367 | 0.2707 | 0.2804 | 0.2883 | 0.2952 |
| Ranker | 0.2541 | 0.3138 | 0.3341 | 0.3633 | 0.3957 |

## Distinct first picks across 5,000 users

| Method | 1 favourite | 3 favourites | 5 favourites | 10 favourites | 20 favourites |
|---|---:|---:|---:|---:|---:|
| Popularity | 9 | 9 | 9 | 9 | 9 |
| Tag Genome + popularity (Stage 1) | 200 | 112 | 86 | 65 | 56 |
| Ranker | 358 | 355 | 324 | 344 | 357 |

## Films without ratings

1,170 films from Wikidata, released after MovieLens ends, have no ratings. They're left out of training, and their bias is predicted from Wikipedia page views and their features by a ridge regression fitted on the rated films' learned biases (R² 0.11 on rated films held out). With 5 favourites, 0.1% of validation users get at least one of them among their 3 picks. Validation can't reward these picks, since nobody rated them.

To estimate how well films without ratings are ranked, the model was trained again as if the 196 MovieLens films from 2021 on had no ratings, then compared with the full model on the 1,427 times a validation user liked one of them after their 5 favourites:

| Model | Liked film in the top 100 | Median position | NDCG@10, all films |
|---|---:|---:|---:|
| Trained with their ratings | 1.1% | top 14.0% | 0.3475 |
| As if they had no ratings | 1.0% | top 16.5% | 0.3566 |
| Random order | 0.8% | top 50% | |

## Training

Settings: TrainingSettings(dim=128, hidden=512, dropout=0.2, batch_size=512, learning_rate=0.001, weight_decay=1e-05, max_epochs=40, patience=5). Device: cuda.

| Epoch | Training loss | Held-out NDCG@10 | Seconds |
|---:|---:|---:|---:|
| 1 | 9.0008 | 0.2960 | 11 |
| 2 | 7.2104 | 0.3174 | 10 |
| 3 | 7.1068 | 0.3118 | 10 |
| 4 | 7.0535 | 0.3203 | 11 |
| 5 | 7.0211 | 0.3231 | 10 |
| 6 | 7.0010 | 0.3431 | 10 |
| 7 | 6.9892 | 0.3423 | 10 |
| 8 | 6.9795 | 0.3382 | 10 |
| 9 | 6.9694 | 0.3409 | 10 |
| 10 | 6.9681 | 0.3454 | 10 |
| 11 | 6.9662 | 0.3326 | 10 |
| 12 | 6.9571 | 0.3429 | 10 |
| 13 | 6.9613 | 0.3459 | 9 |
| 14 | 6.9675 | 0.3352 | 8 |
| 15 | 6.9654 | 0.3211 | 8 |
| 16 | 6.9672 | 0.3372 | 8 |
| 17 | 6.9630 | 0.3355 | 9 |
| 18 | 65.9438 | 0.0913 | 9 |

Unrated films count as misses, so these scores understate real-world quality. They are for comparing methods.
