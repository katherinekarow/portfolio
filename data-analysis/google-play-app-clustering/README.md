# Google Play app segmentation & marketing strategy
> Clustered Google Play Store apps by rating, reviews, installs and price, then built a marketing budget plan per segment.

**Tools:** R (tidyverse, cluster), K-means, hierarchical clustering

 **[Read the full report (PDF)](google-play-clustering-report.pdf)** · Group project · 7 contributors

## The question
Which kinds of apps behave alike, and how should a marketing budget be split between them?

## What we did
- Cleaned and normalized the public *Google Play Store Apps* dataset (ratings, reviews, installs, price).
- Chose the number of clusters with the **elbow method, silhouette score and gap statistic**.
- Compared **K-means** with **hierarchical clustering**, then profiled each cluster.
- Turned the clusters into a **marketing strategy and budget allocation**, with a simple sales prediction per segment.

## Result
Five app segments (for example established mainstream apps vs. growing apps in competitive niches), each with its own recommended
channel mix and budget. Details, charts and code are in the report.
