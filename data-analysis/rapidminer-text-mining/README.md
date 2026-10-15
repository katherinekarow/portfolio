# Text mining in RapidMiner
> Turned unstructured text — reviews, support messages and historical essays — into clusters, keywords and themes.

**Tools:** RapidMiner Studio (tokenizing, stemming, TF-IDF, n-grams, K-means, FP-Growth, PCA)

| Process | What it does |
|---|---|
| [`employee-reviews-clustering.rmp`](employee-reviews-clustering.rmp) | TF-IDF on employee reviews → K-means clusters + FP-Growth to find words that appear together |
| [`customer-service-term-frequency.rmp`](customer-service-term-frequency.rmp) | Cleans customer-service messages and ranks the most frequent meaningful terms |
| [`bank-reviews-pca.rmp`](bank-reviews-pca.rmp) | Reduces a document-term matrix with PCA to see the main themes |
| [`federalist-papers-clustering.rmp`](federalist-papers-clustering.rmp) | Clusters Federalist Papers by word use (n-grams) — a classic authorship exercise |

Datasets aren't included; each process shows the full pipeline when opened in RapidMiner.
