# LocalVoice-SDG — Upgrade (feature/model-accuracy-boost)

# Why this branch

Baseline model had high recall (0.97) but low precision (0.71) — model
was over-predicting SDGs. SDG 8 also had a very unstable F1 (0.222),
which we traced back to an unlucky random train/test split.

# What we're changing

1. Loss function: weighted BCE -> Asymmetric Loss (ASL)
   - reduces false positives without hurting recall
2. Split strategy: single random split -> stratified 5-fold CV
   - keeps each SDG proportionally represented in every fold
3. Held-out test set
   - carved out ONCE before k-fold starts, only touched at the very end
   - fixes a leakage bug from the first upgrade attempt (threshold
     tuning and final metrics were using the same val data before)
4. Per-class thresholds instead of one flat 0.30 for all 17 SDGs
5. Ensemble: all 5 fold models averaged together for final prediction,
   evaluated once on the untouched test set
6. Metrics reported as mean +/- std across folds, not a single number

## Folder structure

- app/    -> empty, planned Streamlit demo
- eval/   -> empty, planned standalone threshold script
- models/ -> will hold fold_1...fold_5 + ensemble_thresholds.json
- rag/    -> empty, planned future scheme-matching/retrieval layer

## Known gaps (not done yet)

- predict.py still uses old single model + flat 0.30 threshold
- translate.py, pdf_extractor.py, synthetic_gen.py not reviewed yet
- auto_annotate.py status unclear (in README repo structure, not in
  local tree)
- train.py on main still has uncapped pos_weight bug (main left
  untouched on purpose while this branch is in progress)
- no real numbers yet — pending a full training run
## status
- load_data + parse_labels, verified with real CSV (1,238 samples, correct filtering, correct label vectors).
- SDGDataset class, verified end-to-end on Colab: dataset size matches (1,238), input_ids/attention_mask correctly shaped [256], lab els correctly returned as float tensors.
- AsymmetricLoss class,verified with fake batch data produces a valid loss value as 0.2564.
- Held-out test split verified 185 test / 1053 trainval samples also seeded and reproducible.
- Stratified 5-fold split on trainval set ,5 folds produced each approx 845 train / aprox 208 

## Next step

- Per-fold training loop..