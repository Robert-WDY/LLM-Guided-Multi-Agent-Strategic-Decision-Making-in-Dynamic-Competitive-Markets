# v20.1: confirm a simpler empirical risk filter

v20 failed equal-coverage comparison on the 620001–620030 holdout: at 18 profit interventions, original ranking had zero losses and stress-adjusted ranking had one. Do not promote its asset-stress reranking. Retain all v20 evidence and the full rolling experiment.

This follow-up removes the unvalidated asset reranking, retains frozen v19 search, and subtracts a selected-policy historical overprediction budget from its model lower bounds. It is a risk/coverage filter, not a claim of a better ranking or corrected hidden state. New calibration seeds 640001–640030 (180 states), then freeze before new confirmation seeds 650001–650050 (300 states). Same six markets rotated across N=2/5/10, profit/welfare and decision rounds 5–10. No reuse of v20 holdout as confirmation.

Quantile 0.90, zero adjusted-margin threshold, min12 selected cases/8 clusters, same legal-observation support and parameter restrictions. Release gates unchanged: positive all-case lower bound, coverage10% profit/5% welfare, conservative seed-cluster any-loss upper bound <=10%. Report the full negative evidence even if this follow-up also fails. Existing 30 CNY paid authorization and original economic/price gates stay in force.
