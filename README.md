# Alert escalation model (WIUT Hackathon 2026, Fintech task)

- **EDA website:** https://dopilibola.github.io/wes/
- `escalation_model.ipynb`: the full pipeline (EDA stats, features, CV, submission)
- `index.html`: the EDA website (static, no build step)
- `eda_stats.json`: the numbers behind the charts, written by the notebook

## Reproduce

1. Put `train_signals.csv`, `test_signals.csv`, `train_transactions.parquet` and `test_transactions.parquet` in the same folder as the notebook.
2. `pip install pandas pyarrow lightgbm scikit-learn`
3. Run all cells. The notebook writes `team_<TEAM_ID>.csv` and `eda_stats.json`.

Out-of-fold ROC-AUC on train: 0.649 (LightGBM + logistic regression rank blend, 5 folds × 3 seeds).

The competition data is not included in this repo.
