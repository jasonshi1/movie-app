# Stage 4 simulated users

With quick picks, 77.0% ± 1.2% of simulated users get a film they liked within 5 suggestions (95% confidence).

Simulated users: 5,000 real MovieLens users from the test split, which no earlier stage used. Each gives their first 5 liked films as favourites and accepts only a film they rated 4 or higher later on. Films they had already rated count as turned down, because the app can't know what someone has seen.

## Share of users with a hit

| Method | Hit within 1 | Hit within 3 | Hit within 5 | Hit within 10 | Already seen |
|---|---:|---:|---:|---:|---:|
| Quick picks (ranker) | 43.6% ± 1.4% | 67.3% ± 1.3% | 77.0% ± 1.2% | 86.6% ± 0.9% | 2% |
| Quick picks (Stage 1 method) | 40.1% ± 1.4% | 63.8% ± 1.3% | 73.1% ± 1.2% | 83.5% ± 1.0% | 4% |
| Most popular films | 34.5% ± 1.3% | 58.4% ± 1.4% | 68.2% ± 1.3% | 78.3% ± 1.1% | 4% |

Already seen: the share of all suggestions that the user had already rated.

## Quick picks (ranker) minus each other method, per user

| Method | Hit within 1 | Hit within 3 | Hit within 5 | Hit within 10 |
|---|---:|---:|---:|---:|
| Quick picks (Stage 1 method) | +3.5% ± 1.7% | +3.6% ± 1.5% | +4.0% ± 1.3% | +3.1% ± 1.1% |
| Most popular films | +9.0% ± 1.7% | +8.9% ± 1.6% | +8.8% ± 1.4% | +8.3% ± 1.2% |

## Limits

- Unrated films count as misses, so every rate understates how often people would be happy. That penalizes less familiar suggestions most.
- Simulated users aren't real people. Their reasons come from Tag Genome differences, and they never change their minds. Some reasons name an actor or author, such as "less arnold", because those are Tag Genome tags too.
