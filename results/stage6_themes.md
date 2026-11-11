# Stage 6 predicted themes

Films added after the Tag Genome have no theme scores, so they're predicted from each film's description, genres and year. These are the results on 958 films with real scores that were held out of fitting. The final model is then fitted on all 9,585 of them.

- **Top 10 themes found**: the share of a film's real top 10 themes among its predicted top 10. This is what picks show as shared themes.
- **Correlation per theme**: how well predictions order films on each theme, averaged over themes. The ranker reads themes this way.
- **Cosine similarity**: how close the whole set of scores is. Claude's search uses this.
- **Most common top theme**: the share of films whose top 5 includes the most frequent theme. High means every film gets the same themes.

| Method | Top 10 themes found | Correlation per theme | Cosine similarity | Most common top theme |
|---|---:|---:|---:|---:|
| Ridge regression, rescaled to each theme's real spread (used) | 35.2% | 0.464 | 0.775 | 47% |
| Ridge regression alone | 30.6% | 0.434 | 0.813 | 99% |
| Average of films with the same genres | 27.4% | 0.291 | 0.776 | 96% |
| Average of all films | 21.2% | -0.000 | 0.732 | 100% |
| Real scores | 100.0% | 1.000 | 1.000 | 33% |

Rescaling lowers the cosine similarity a little, because it spreads scores out, but finds more of each film's real themes and stops every film sharing the same ones.

Ridge strength 3.0, chosen on a separate tuning set of the same size:

| Strength | Cosine similarity | Top 10 themes found |
|---:|---:|---:|
| 0.1 | 0.810 | 30.9% |
| 0.3 | 0.811 | 30.8% |
| 1.0 | 0.812 | 30.8% |
| 3.0 | 0.813 | 30.7% |
| 10.0 | 0.812 | 30.2% |
| 30.0 | 0.808 | 29.3% |
