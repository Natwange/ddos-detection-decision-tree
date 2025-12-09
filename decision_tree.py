#!/usr/bin/env python3
"""
Decision Tree–Based DDoS Detection (College Project)

This script implements a simple and realistic pipeline to detect DDoS attacks
using a CART Decision Tree from scikit-learn. It follows a clear, explainable
process.

Features of the design:
- Uses a single CSV dataset (e.g., CIC-IDS2017 or NSL-KDD) with a label column
- Preprocessing: missing-value handling, categorical encoding, normalization
- Feature selection: Chi-square to select top-K features
- Classifier: Decision Tree with limited depth for interpretability
- Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix
- Output: Printed tree text, feature importances, saved model for reuse

Usage:
  python decision_tree.py --csv path/to/data.csv --label Label --top_k 20 --max_depth 8
"""

import argparse
import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ----------------------------------------
# 1) DATA LOADING
# ----------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """Load a single CSV dataset into a pandas DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    print(f"[LOAD] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[LOAD] Shape: {df.shape}")
    return df


def _find_label_column(df: pd.DataFrame, label_arg: Optional[str]) -> str:
    """Heuristically find a label column if not provided explicitly."""
    if label_arg and label_arg in df.columns:
        return label_arg
    candidates = [
        "Label", "label", "Attack", "attack", "Class", "class", "target", "Target"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: try any column containing "label" or "attack"
    for c in df.columns:
        lc = c.lower()
        if "label" in lc or "attack" in lc or lc in {"class", "target"}:
            return c
    raise ValueError("Could not infer label column. Pass --label explicitly.")


# ----------------------------------------
# 2) PREPROCESSING + FEATURE SELECTION
# ----------------------------------------
def preprocess(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    test_size: float = 0.2,
    top_k: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], dict, Optional[SelectKBest]]:
    """
    Preprocess the dataset:
      - Extract labels (binary: 1=DDoS, 0=Normal)
      - Handle missing values
      - Encode categoricals (one-hot)
      - Normalize numerics (MinMaxScaler for chi-square)
      - Select top-K features (chi-square)

    Returns X_train, X_test, y_train, y_test, selected_feature_names,
    fitted_preprocessor, fitted_selector.
    """
    # 2.1 Identify label column and build binary target
    y_col = _find_label_column(df, label_col)
    print(f"[LABEL] Using label column: '{y_col}'")
    
    # Check if labels are already numeric (0/1)
    if df[y_col].dtype in ['int64', 'int32', 'float64', 'float32']:
        y = df[y_col].values.astype(int)
        print(f"[LABEL] Labels are already numeric: {np.unique(y)}")
    else:
        # Convert text labels to binary
        y_text = df[y_col].astype(str).str.strip().str.lower()
        # Positive if it contains 'ddos'; otherwise negative (treat other attacks as negative for simplicity)
        y = np.where(y_text.str.contains("ddos"), 1, 0)
        print(f"[LABEL] Converted text labels to binary: {np.unique(y)}")
    
    print(f"[LABEL] Label distribution: {np.bincount(y)}")

    # 2.2 Drop the label column from features; also drop obvious IDs if present
    drop_cols = {
        y_col,
        "Flow ID", "flow_id", "flowID", "Source IP", "Destination IP", "src_ip", "dst_ip",
        "Timestamp", "timestamp", "StartTime", "EndTime", "Start Time", "End Time",
    }
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # 2.2.a Clean feature values: coerce to numeric where possible, handle infinities
    # Some real datasets (e.g., CIC-IDS2017) contain 'Infinity' or extremely large values.
    # Convert coercible columns to numeric and replace +/-inf with NaN so imputation can handle them.
    with np.errstate(over='ignore'):
        X_df = X_df.apply(pd.to_numeric, errors="ignore")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    # Optional safety: clip absurd magnitudes to a reasonable finite range before scaling
    # (keeps distribution shape but avoids overflow)
    try:
        X_df = X_df.clip(lower=-1e15, upper=1e15)
    except Exception:
        pass

    # 2.3 Train/test split (stratified)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=42, stratify=y
    )

    # 2.4 Identify numeric vs categorical columns
    numeric_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train_df.columns if c not in numeric_cols]
    print(f"[COLUMNS] numeric={len(numeric_cols)}, categorical={len(categorical_cols)}")

    # 2.5 Manual preprocessing pipeline for clarity
    # 1) Numeric: median impute + MinMax scaling
    num_imputer = SimpleImputer(strategy="median")
    num_scaler = MinMaxScaler()
    X_train_num = num_scaler.fit_transform(num_imputer.fit_transform(X_train_df[numeric_cols])) if numeric_cols else np.empty((len(X_train_df), 0))

    # 2) Categorical: most_frequent impute + OneHot encoding
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train_cat_imp = cat_imputer.fit_transform(X_train_df[categorical_cols])
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_train_cat = cat_encoder.fit_transform(X_train_cat_imp)
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    else:
        X_train_cat = np.empty((len(X_train_df), 0))
        cat_imputer = None
        cat_encoder = None
        cat_feature_names = []

    # 3) Concatenate numeric and categorical features
    X_train_pre = np.hstack([X_train_num, X_train_cat])
    feature_names_pre = numeric_cols + cat_feature_names

    # 2.6 Feature selection (Chi-square) on numeric features only
    # Chi-square works best with numeric features, so we'll select from numeric features first
    k_numeric = min(top_k, len(numeric_cols)) if numeric_cols else 0
    print(f"[FEATURE_SELECTION] Selecting {k_numeric} from {len(numeric_cols)} numeric features")
    print(f"[FEATURE_SELECTION] Training data shape before selection: {X_train_num.shape}")
    print(f"[FEATURE_SELECTION] Label distribution in training: {np.bincount(y_train)}")
    
    if k_numeric > 0 and len(numeric_cols) > 0:
        # Select top-k numeric features using chi-square
        selector_numeric = SelectKBest(score_func=chi2, k=k_numeric)
        X_train_num_selected = selector_numeric.fit_transform(X_train_num, y_train)
        selected_numeric_names = [numeric_cols[i] for i, keep in enumerate(selector_numeric.get_support()) if keep]
        
        print(f"[FEATURE_SELECTION] Selected numeric features: {selected_numeric_names}")
        print(f"[FEATURE_SELECTION] Training data shape after selection: {X_train_num_selected.shape}")
        
        # Combine selected numeric features with all categorical features
        X_train_sel = np.hstack([X_train_num_selected, X_train_cat])
        selected_feature_names = selected_numeric_names + cat_feature_names
        selector = selector_numeric  # Store the numeric selector
    else:
        # Use all features if no numeric features or k=0
        X_train_sel = X_train_pre
        selected_feature_names = feature_names_pre
        selector = None
    
    print(f"[FEATURE_SELECTION] Final training data shape: {X_train_sel.shape}")
    print(f"[FEATURE_SELECTION] Final feature names: {selected_feature_names[:5]}...")  # Show first 5

    # 2.7 Transform test data with the same fitted steps
    X_test_num = num_scaler.transform(num_imputer.transform(X_test_df[numeric_cols])) if numeric_cols else np.empty((len(X_test_df), 0))
    if categorical_cols:
        X_test_cat_imp = cat_imputer.transform(X_test_df[categorical_cols])
        X_test_cat = cat_encoder.transform(X_test_cat_imp)
    else:
        X_test_cat = np.empty((len(X_test_df), 0))
    
    # Apply the same feature selection to test data
    if selector is not None and k_numeric > 0:
        X_test_num_selected = selector.transform(X_test_num)
        X_test_sel = np.hstack([X_test_num_selected, X_test_cat])
    else:
        X_test_sel = np.hstack([X_test_num, X_test_cat])

    # Package preprocessor components for saving/reuse
    # We'll save these individually (imputer/scaler/encoder/selector) so predictions can be reproduced.
    fitted_preprocessor = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "num_imputer": num_imputer,
        "num_scaler": num_scaler,
        "cat_imputer": cat_imputer,
        "cat_encoder": cat_encoder,
    }

    return (
        X_train_sel,
        X_test_sel,
        y_train,
        y_test,
        selected_feature_names,
        fitted_preprocessor,  # type: ignore
        selector,
    )


# ----------------------------------------
# 3) TRAINING
# ----------------------------------------
def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 8,
) -> DecisionTreeClassifier:
    """Train a CART Decision Tree with limited depth for readability."""
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


# ----------------------------------------
# 4) CROSS VALIDATION
# ----------------------------------------
def perform_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 8,
    n_folds: int = 5,
) -> dict:
    """
    Perform k-fold cross validation to get initial performance estimates.
    Returns dictionary with mean and std of accuracy, precision, recall, F1.
    """
    print(f"\n=== {n_folds}-FOLD CROSS VALIDATION ===")
    
    # Create stratified k-fold for balanced splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize lists to store scores
    acc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model on this fold
        clf_fold = DecisionTreeClassifier(
            criterion="gini",
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
        )
        clf_fold.fit(X_train_fold, y_train_fold)
        
        # Predict on validation set
        y_pred_fold = clf_fold.predict(X_val_fold)
        
        # Calculate metrics
        acc = accuracy_score(y_val_fold, y_pred_fold)
        precision = precision_score(y_val_fold, y_pred_fold, zero_division=0)
        recall = recall_score(y_val_fold, y_pred_fold, zero_division=0)
        f1 = f1_score(y_val_fold, y_pred_fold, zero_division=0)
        
        acc_scores.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        print(f"Fold {fold+1}: Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
    
    # Calculate mean and standard deviation
    results = {
        'accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores)},
        'precision': {'mean': np.mean(precision_scores), 'std': np.std(precision_scores)},
        'recall': {'mean': np.mean(recall_scores), 'std': np.std(recall_scores)},
        'f1': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)},
    }
    
    print(f"\nCross-Validation Results (Mean ± Std):")
    print(f"Accuracy : {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}")
    print(f"Precision: {results['precision']['mean']:.4f} ± {results['precision']['std']:.4f}")
    print(f"Recall   : {results['recall']['mean']:.4f} ± {results['recall']['std']:.4f}")
    print(f"F1-Score : {results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}")
    
    return results


# ----------------------------------------
# 5) EVALUATION AND OUTPUTS
# ----------------------------------------
def evaluate_model(
    clf: DecisionTreeClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> None:
    """Print metrics, confusion matrix, feature importance, and tree structure."""
    y_pred = clf.predict(X_test)

    # Core metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    print("\n=== EVALUATION ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    # Handle case where only one class is predicted
    unique_pred = np.unique(y_pred)
    unique_true = np.unique(y_test)
    
    if len(unique_pred) == 1 and len(unique_true) == 2:
        print("Warning: Model predicted only one class. This may indicate:")
        print("1. Dataset is too easy (perfect separation)")
        print("2. Feature selection removed all discriminative features")
        print("3. Tree depth is too restrictive")
        print(f"Predicted class: {unique_pred[0]}")
        print(f"True classes: {unique_true}")
    else:
        print(classification_report(y_test, y_pred, target_names=["Normal", "DDoS"]))

    # Confusion Matrix (print and save figure)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=["True:Normal", "True:DDoS"], columns=["Pred:Normal", "Pred:DDoS"]))

    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal", "DDoS"])  # predicted
    plt.yticks(tick_marks, ["Normal", "DDoS"])  # true
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=120)
    plt.close()
    print("[SAVE] Confusion matrix figure -> confusion_matrix.png")

    # Feature Importances
    if hasattr(clf, "feature_importances_") and len(feature_names) == len(clf.feature_importances_):
        importances = sorted(
            zip(feature_names, clf.feature_importances_), key=lambda t: t[1], reverse=True
        )
        print("\nTop Feature Importances:")
        for name, val in importances[:15]:
            print(f"  {name}: {val:.4f}")

    # Tree structure (text)
    try:
        tree_text = export_text(clf, feature_names=feature_names)
        print("\n=== DECISION TREE (text) ===")
        print(tree_text)
    except Exception as e:
        print(f"[WARN] Could not export tree text: {e}")


def save_model(
    clf: DecisionTreeClassifier,
    preproc: dict,
    selector: Optional[SelectKBest],
    path: str = "ddos_decision_tree.joblib",
) -> None:
    """Save the trained model and preprocessing steps for reuse."""
    payload = {
        "model": clf,
        "preprocessor": preproc,
        "selector": selector,
        "metadata": {
            "framework": "scikit-learn",
            "task": "DDoS detection (binary)",
        },
    }
    joblib.dump(payload, path)
    print(f"[SAVE] Model and preprocessing saved -> {path}")


# ----------------------------------------
# 5) MAIN
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Decision Tree–based DDoS Detection (College Project)")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV dataset")
    parser.add_argument("--label", type=str, default=None, help="Name of the label column (optional)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction (default 0.2 = 80% train, 20% test)")
    parser.add_argument("--top_k", type=int, default=20, help="Select top-K features via chi-square (default 20)")
    parser.add_argument("--max_depth", type=int, default=8, help="Max depth of the Decision Tree (default 8)")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross validation (default 5)")

    args = parser.parse_args()

    print("="*70)
    print("DECISION TREE-BASED DDOS DETECTION")
    print("="*70)
    print(f"Dataset: {args.csv}")
    print(f"Training/Testing Split: {(1-args.test_size)*100:.0f}% / {args.test_size*100:.0f}%")
    print(f"Cross Validation: {args.n_folds}-fold")
    print(f"Max Tree Depth: {args.max_depth}")
    print(f"Top Features: {args.top_k}")

    # 1) Load
    df = load_data(args.csv)

    # 2) Preprocess + feature selection
    X_train, X_test, y_train, y_test, feat_names, preproc, selector = preprocess(
        df, label_col=args.label, test_size=args.test_size, top_k=args.top_k
    )

    # 3) Initial Testing with Cross Validation (on training data)
    print(f"\n=== INITIAL TESTING WITH {args.n_folds}-FOLD CROSS VALIDATION ===")
    cv_results = perform_cross_validation(X_train, y_train, max_depth=args.max_depth, n_folds=args.n_folds)

    # 4) Train final model on full training set
    print(f"\n=== TRAINING FINAL MODEL ===")
    clf = train_decision_tree(X_train, y_train, max_depth=args.max_depth)
    print(f"Tree Depth: {clf.get_depth()}")
    print(f"Number of Leaves: {clf.get_n_leaves()}")

    # 5) Final Evaluation on Test Set
    print(f"\n=== FINAL EVALUATION ON TEST SET ===")
    evaluate_model(clf, X_test, y_test, feat_names)

    # 6) Save
    save_model(clf, preproc, selector, path="ddos_decision_tree.joblib")
    
    print(f"\n=== SUMMARY ===")
    print(f"Cross-Validation F1-Score: {cv_results['f1']['mean']:.4f} ± {cv_results['f1']['std']:.4f}")
    print(f"Cross-Validation Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
    print("Model saved as: ddos_decision_tree.joblib")
    print("Confusion matrix saved as: confusion_matrix.png")


if __name__ == "__main__":
    main()
