# Reproducing the paper results

The Git evidence includes model decisions, realized branch profits, public histories, fitted estimators and interval estimates. The short audit needs no API keys, market checkout or training. Run from the repository root with Python 3.12 and NumPy 2.3.3.

```powershell
python scripts/publication/verify.py
python scripts/publication/publication_stats.py
```

The first command recomputes the six primary contrasts and their within-protocol multiplicity-adjusted intervals, the frozen policy's real-draft contrasts, sample allocation and ledger arithmetic. The second rebuilds results.json. Real API responses can change. The published paths are the record of the specific runs.

For market replay, install the project in its research environment. The verified local Python is 3.12.14, with SciPy 1.18.1, scikit-learn 1.8.0, joblib 1.6.0, matplotlib 3.11.2, pandas 3.1.1, OpenAI SDK 2.54.0, Pydantic 2.13.5 and FastAPI 0.141.1. Use the [upstream EconEvals revision](https://github.com/sara-fish/econ-evals-paper/tree/e1f2a40fec96f0d27f5414873c4310f2b5c51935) as a Git checkout under .tmp/econ-evals-upstream or point ECONEVALS_ROOT to it. The loader checks the upstream commit and tracked economic source. Only load fitted model files from the intended checked and trusted Git revision.

```powershell
python scripts/publication/verify.py --replay
python scripts/publication/delayed_analysis.py report
python scripts/publication/selection_audit.py
python scripts/publication/figure.py
```

To regenerate the free simulations, run paired.py train, paired.py cal, signal_policy.py, paired.py test and publication_stats.py using the scripts/publication directory. For the independent delayed experiment run delayed.py train, delayed.py fit, delayed.py test and delayed_analysis.py report. unseen.py regenerates the independent Exp3 and uniform-opponent cases. Existing freeze files verify fitted models. Retraining in isolation requires an independent checkout with the corresponding generated weights and freeze files omitted. Train/calibration labels and primary holdouts have disjoint seed ranges. Old practice data reside outside these ranges.

The matched experiment stores arrays indexed by opponent [follow/undercut, reverse], then request [keep, down, up], then consecutive settlement rounds. The delayed experiment indexes conditions [stable, demand, follow/undercut, reverse]. Source seed, public histories, original proposed price and decision round are present in each case. Native profit arrays use integer cents; analyses divide by 100. Seed bootstrap keeps every dependent treatment, arm and repeated supplier outcome together. Earlier research outputs are separate datasets with different grants of information and endpoints.

The fresh paid native paths and archived external three-direction corpus only require money when gathering entirely new commercial model responses. All listed reproduction commands use saved drafts and require no supplier calls.
