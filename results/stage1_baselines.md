# Stage 1 baseline results

Validation users: 5,000. Films: 13,248. Largest 95% confidence half-width on any NDCG score: 0.0079.

Popularity weight, chosen on training users as the smallest within 97% of the best score: Tag Genome 1.0, text embedding 1.0.

## NDCG@10

| Method | 1 favourite | 3 favourites | 5 favourites | 10 favourites | 20 favourites |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.2521 | 0.2521 | 0.2521 | 0.2521 | 0.2521 |
| Genre match | 0.0840 | 0.0674 | 0.0678 | 0.0612 | 0.0574 |
| Text embedding | 0.0535 | 0.0969 | 0.1115 | 0.1296 | 0.1474 |
| Tag Genome | 0.1195 | 0.1416 | 0.1453 | 0.1531 | 0.1616 |
| Text embedding + popularity | 0.2451 | 0.2704 | 0.2761 | 0.2806 | 0.2834 |
| Tag Genome + popularity | 0.2474 | 0.2809 | 0.2900 | 0.2990 | 0.3052 |

## Recall@10

| Method | 1 favourite | 3 favourites | 5 favourites | 10 favourites | 20 favourites |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.2363 | 0.2363 | 0.2363 | 0.2363 | 0.2363 |
| Genre match | 0.0724 | 0.0615 | 0.0615 | 0.0559 | 0.0526 |
| Text embedding | 0.0478 | 0.0889 | 0.1034 | 0.1214 | 0.1379 |
| Tag Genome | 0.1085 | 0.1317 | 0.1356 | 0.1456 | 0.1540 |
| Text embedding + popularity | 0.2366 | 0.2574 | 0.2633 | 0.2665 | 0.2691 |
| Tag Genome + popularity | 0.2367 | 0.2707 | 0.2804 | 0.2883 | 0.2952 |

## Distinct first picks across 5,000 users

The app's first pick is each method's top film, so this shows how personal it is.

| Method | 1 favourite | 3 favourites | 5 favourites | 10 favourites | 20 favourites |
|---|---:|---:|---:|---:|---:|
| Popularity | 9 | 9 | 9 | 9 | 9 |
| Genre match | 644 | 639 | 669 | 522 | 407 |
| Text embedding | 1043 | 1197 | 938 | 563 | 367 |
| Tag Genome | 965 | 930 | 724 | 528 | 414 |
| Text embedding + popularity | 166 | 77 | 57 | 40 | 37 |
| Tag Genome + popularity | 200 | 112 | 86 | 65 | 56 |

## Tag Genome popularity weight, on 2,000 training users

| Popularity weight | Mean NDCG@10 | Distinct first picks |
|---:|---:|---:|
| 0.0 | 0.1439 | 444 |
| 0.1 | 0.2095 | 335 |
| 0.25 | 0.2478 | 212 |
| 0.5 | 0.2716 | 131 |
| 1.0 | 0.2853 | 70 |
| 1.5 | 0.2882 | 48 |
| 2.0 | 0.2889 | 33 |
| 3.0 | 0.2871 | 20 |
| 4.0 | 0.2845 | 19 |
| 6.0 | 0.2794 | 15 |
| 8.0 | 0.2764 | 14 |
| 12.0 | 0.2717 | 13 |

Unrated films count as misses, so these scores understate real-world quality. They are for comparing methods.
