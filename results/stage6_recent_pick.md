# Stage 6 recent pick

The app's last pick is the best film for you from 2024 to 2026. Those films came out after MovieLens ends, so nobody's liking of them can be checked. Instead the ranker was trained again as if the 196 MovieLens films from 2021 on had no ratings, the last three years MovieLens has, and each way of choosing below picked one of those films for each of the 863 validation users who liked at least one of them after their favourites. A hit is a film they liked.

Popularity is worked out from page views, as for the catalog's films without ratings. A film's standing in its year is its percent rank on popularity among the catalog's films from the same year: 0 for the least viewed, 1 for the most.

| Choice from the films since 2021 | 1 favourite | 3 favourites | 5 favourites |
|---|---:|---:|---:|
| Ranker plus 32 × standing in its year (the app's) | 27.8% ± 3.0% | 25.5% ± 2.9% | 27.5% ± 3.0% |
| Ranker plus 16 × popularity | 24.4% ± 2.9% | 26.2% ± 2.9% | 25.6% ± 2.9% |
| Ranker alone | 9.4% ± 1.9% | 10.4% ± 2.0% | 11.5% ± 2.1% |
| Most Wikipedia page views | 18.4% ± 2.6% | 18.4% ± 2.6% | 18.4% ± 2.6% |
| Ranker trained with their ratings (for reference) | 27.3% ± 3.0% | 31.2% ± 3.1% | 34.4% ± 3.2% |
| Random film | 2.6% ± 0.2% | 2.6% ± 0.2% | 2.6% ± 0.2% |

The app's recent pick against two of the others, paired over the same users:

| Favourites | Minus most Wikipedia page views | Minus ranker alone |
|---:|---:|---:|
| 1 | +9.4% ± 3.6% | +18.4% ± 3.2% |
| 3 | +7.1% ± 3.6% | +15.1% ± 3.3% |
| 5 | +9.0% ± 3.6% | +16.0% ± 3.3% |

## What the app shows

Page views run far higher in a film's first months, so the newest films look the most popular. Adding popularity itself scores well above, where every film is a few years old, but in the app it hands most people films from this year. Standing in its year compares a film only with films as new as it. These are the recent picks the trained ranker makes from the catalog's 2024 to 2026 films for the 13,775 validation users at 5 favourites:

| Choice | Different films | Most common pick | From 2024 | From 2025 | From 2026 |
|---|---:|---|---:|---:|---:|
| Ranker plus 32 × standing in its year (the app's) | 23 | One Battle After Another (66%) | 16% | 67% | 17% |
| Ranker plus 16 × popularity | 22 | The Odyssey (46%) | 0% | 16% | 84% |
| Ranker alone | 120 | One Battle After Another (24%) | 30% | 50% | 19% |

Either way, many people get the same film. How widely liked a film is counts for a lot when the model knows nothing else about who liked it, and the most-viewed films are the likeliest to be liked. The ranker alone spreads its picks further but finds a liked film far less often, and a smaller weight gives up hits long before the most common pick goes to fewer people.

## Choosing the weights

Both weights were chosen at 5 favourites on the 6,862 training users who liked one of these films. The retrained model never saw those ratings.

| Weight | Plus standing in its year | Plus popularity |
|---:|---:|---:|
| 1 | 11.8% | 10.7% |
| 2 | 14.0% | 12.0% |
| 4 | 18.4% | 14.2% |
| 8 | 22.3% | 20.2% |
| 16 | 24.7% | 24.7% (best) |
| 32 | 26.8% (used) | 22.0% |
| 64 | 24.8% | 15.2% |
| 128 | 17.3% | 16.3% |

Ranges are 95% confidence intervals. The users here rated films from these years, so they're likelier than most to want a new film.
