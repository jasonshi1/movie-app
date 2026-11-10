# Stage 1 baseline results

Validation users: 5,000. Films: 9,585. Largest 95% confidence half-width on any NDCG score: 0.0080.

Popularity weight, chosen on training users as the smallest within 97% of the best score: Tag Genome 1.0, text embedding 1.0.

## NDCG@10

| Method | 1 favourite | 3 favourites | 5 favourites | 10 favourites | 20 favourites |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.2749 | 0.2749 | 0.2749 | 0.2749 | 0.2749 |
| Genre match | 0.0883 | 0.0740 | 0.0733 | 0.0673 | 0.0654 |
| Text embedding | 0.0576 | 0.1022 | 0.1165 | 0.1348 | 0.1550 |
| Tag Genome | 0.1289 | 0.1548 | 0.1609 | 0.1710 | 0.1803 |
| Text embedding + popularity | 0.2594 | 0.2833 | 0.2907 | 0.2960 | 0.3006 |
| Tag Genome + popularity | 0.2573 | 0.2916 | 0.2981 | 0.3039 | 0.3086 |

## Recall@10

| Method | 1 favourite | 3 favourites | 5 favourites | 10 favourites | 20 favourites |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.2656 | 0.2656 | 0.2656 | 0.2656 | 0.2656 |
| Genre match | 0.0761 | 0.0668 | 0.0644 | 0.0612 | 0.0604 |
| Text embedding | 0.0523 | 0.0944 | 0.1084 | 0.1271 | 0.1454 |
| Tag Genome | 0.1185 | 0.1457 | 0.1501 | 0.1616 | 0.1705 |
| Text embedding + popularity | 0.2495 | 0.2698 | 0.2763 | 0.2793 | 0.2834 |
| Tag Genome + popularity | 0.2475 | 0.2816 | 0.2881 | 0.2946 | 0.2994 |

## Distinct first picks across 5,000 users

The app offers one film at a time, so this shows how personal that first offer is.

| Method | 1 favourite | 3 favourites | 5 favourites | 10 favourites | 20 favourites |
|---|---:|---:|---:|---:|---:|
| Popularity | 15 | 15 | 15 | 15 | 15 |
| Genre match | 623 | 614 | 628 | 498 | 405 |
| Text embedding | 996 | 1095 | 829 | 510 | 318 |
| Tag Genome | 914 | 861 | 660 | 474 | 384 |
| Text embedding + popularity | 154 | 77 | 59 | 38 | 34 |
| Tag Genome + popularity | 181 | 95 | 82 | 66 | 58 |

## Tag Genome popularity weight, on 2,000 training users

| Popularity weight | Mean NDCG@10 | Distinct first picks |
|---:|---:|---:|
| 0.0 | 0.1532 | 426 |
| 0.1 | 0.2125 | 327 |
| 0.25 | 0.2462 | 194 |
| 0.5 | 0.2673 | 107 |
| 1.0 | 0.2802 | 65 |
| 1.5 | 0.2839 | 42 |
| 2.0 | 0.2852 | 27 |
| 3.0 | 0.2841 | 24 |
| 4.0 | 0.2821 | 17 |
| 6.0 | 0.2781 | 12 |
| 8.0 | 0.2762 | 12 |
| 12.0 | 0.2736 | 12 |

Unrated films count as misses, so these scores understate real-world quality. They are for comparing methods.
