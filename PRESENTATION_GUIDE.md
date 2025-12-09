## DDoS Detection Using Decision Trees — Presentation Guide (Tailored)

### Title Slide
- Project: DDoS Detection Using Decision Trees
- One-liner: “Classify network flows as Normal vs DDoS with an explainable ML pipeline.”

### Problem & Goal
- Problem: DDoS overwhelms servers with fake traffic → downtime and poor user experience.
- Goal: Build an interpretable model to detect DDoS vs Normal traffic.

### Dataset (Real-Ready)
- Accepts a single CSV (e.g., CIC-IDS2017, NSL-KDD).
- Rows = flows; columns = features; label column = Normal/DDoS (text or 0/1).
- Split: 80% training / 20% testing (stratified to keep class ratios).

### Pipeline Overview
1) Load CSV
2) Preprocess: handle missing values, normalize numerics (MinMax), encode categoricals, clean infinities
3) Feature selection: chi-square on numeric features → keep top-K
4) Train: CART Decision Tree (Gini), limited max_depth, class_weight='balanced'
5) Validate: k-fold CV on training set
6) Evaluate: Accuracy, Precision, Recall, F1, confusion matrix
7) Outputs: feature importance, tree rules (text), saved model

### Preprocessing (Why/How)
- Missing values: median (numeric), most frequent (categorical)
- Normalization: MinMax scaling to 0–1 (supports chi-square)
- Encoding: one-hot for categorical columns
- Real-data safety: replace ±Infinity; clip extreme values before scaling

### Feature Selection (Simple & Effective)
- Chi-square ranks numeric features by how well they separate Normal vs DDoS
- Keep top-K (e.g., 10–20) → simpler model, faster training

### Model (Explainable)
- Algorithm: Decision Tree (CART), Gini impurity
- Limited depth for interpretability and to prevent overfitting
- Balanced classes to handle Normal vs DDoS skew

### Validation & Metrics
- k-fold stratified CV (e.g., k=5) → mean ± std of Accuracy, Precision, Recall, F1
- Final test on the 20% holdout set
- Confusion matrix image (confusion_matrix.png)

### Live Demo (Real CIC-IDS2017 File)
Run from project folder in PowerShell (single line):
```powershell
python decision_tree.py --csv "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" --label Label --top_k 15 --max_depth 8 --n_folds 5
```
If memory is tight, sample a subset and rerun:
```powershell
python -c "import pandas as pd; df=pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'); df.sample(n=120000, random_state=42).to_csv('cic_subset.csv', index=False)"
python decision_tree.py --csv "cic_subset.csv" --label Label --top_k 15 --max_depth 8 --n_folds 5
```
Show:
- Console metrics (Accuracy/Precision/Recall/F1)
- confusion_matrix.png
- Printed tree rules (text)
- Top feature importances
- Saved model: ddos_decision_tree.joblib

### Results Interpretation
- Accuracy: overall correctness
- Precision: of predicted DDoS, how many were truly DDoS (low false alarms)
- Recall: of real DDoS, how many we caught (few misses)
- F1: balance between precision and recall
- Discuss key features (e.g., host/service counts/rates, error rates)

### Why Decision Trees
- Interpretable if/then rules
- Fast to train and predict
- Great baseline and easy to present/defend

### Limitations
- Single model; can be improved with ensembles (Random Forest/XGBoost)
- Depends on feature quality and cleanliness
- Needs periodic retraining as traffic changes

### Future Work
- Try Random Forest / XGBoost
- Engineer additional time-window/IP aggregation features
- Real-time streaming pipeline and alert thresholds
- Hyperparameter tuning and calibration

### Q&A Prep
- Imbalance handling: class_weight='balanced' + stratified split
- Why MinMax: needed for chi-square; standardizes scales
- Why chi-square: simple, fast relevance filter
- Label name differences: pass --label to match CSV header
- Other CSVs: yes, same pipeline, change file path and label

### Slide Timing (Guide)
- 1 min: Problem/Goal
- 2 min: Dataset & Pipeline
- 2 min: Preprocessing & Feature Selection
- 2 min: Model & Validation
- 2 min: Live Demo
- 1 min: Results & Interpretation
- 1 min: Limitations & Future Work
- Q&A
