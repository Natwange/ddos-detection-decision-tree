# Evaluation Metrics Formulas for DDoS Detection

## Confusion Matrix

For binary classification (Normal vs DDoS), we use a 2×2 confusion matrix:

```
                    Predicted
                 Normal    DDoS
Actual Normal     TN       FP
Actual DDoS       FN       TP
```

Where:
- **TP (True Positives)**: Correctly predicted DDoS attacks
- **TN (True Negatives)**: Correctly predicted normal traffic
- **FP (False Positives)**: Normal traffic incorrectly classified as DDoS
- **FN (False Negatives)**: DDoS attacks incorrectly classified as normal

## Core Evaluation Metrics

### 1. Accuracy
**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretation:** Overall correctness of the model
- Range: [0, 1]
- Higher is better
- Shows the proportion of correct predictions

### 2. Precision
**Formula:**
```
Precision = TP / (TP + FP)
```

**Interpretation:** Of all predicted DDoS attacks, how many were actually DDoS?
- Range: [0, 1]
- Higher is better
- Important when false positives are costly (e.g., blocking legitimate users)

### 3. Recall (Sensitivity)
**Formula:**
```
Recall = TP / (TP + FN)
```

**Interpretation:** Of all actual DDoS attacks, how many did we correctly identify?
- Range: [0, 1]
- Higher is better
- Important when missing attacks is costly (security risk)

### 4. F1-Score
**Formula:**
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**Alternative form:**
```
F1-Score = 2 × TP / (2 × TP + FP + FN)
```

**Interpretation:** Harmonic mean of precision and recall
- Range: [0, 1]
- Higher is better
- Balances precision and recall
- Particularly useful when classes are imbalanced

## Additional Metrics

### 5. Specificity
**Formula:**
```
Specificity = TN / (TN + FP)
```

**Interpretation:** Of all normal traffic, how many were correctly identified as normal?
- Range: [0, 1]
- Higher is better

### 6. False Positive Rate (FPR)
**Formula:**
```
FPR = FP / (FP + TN) = 1 - Specificity
```

**Interpretation:** Proportion of normal traffic incorrectly classified as DDoS
- Range: [0, 1]
- Lower is better

### 7. False Negative Rate (FNR)
**Formula:**
```
FNR = FN / (FN + TP) = 1 - Recall
```

**Interpretation:** Proportion of DDoS attacks missed by the model
- Range: [0, 1]
- Lower is better

## Cross-Validation Metrics

For k-fold cross validation, we calculate:

### Mean and Standard Deviation
```
Mean_Metric = (1/k) × Σ(Metric_i) for i = 1 to k
Std_Metric = √[(1/(k-1)) × Σ(Metric_i - Mean_Metric)²] for i = 1 to k
```

Where k is the number of folds and Metric_i is the metric value for fold i.

## Example Calculation

Given a confusion matrix:
```
                    Predicted
                 Normal    DDoS
Actual Normal     280       20
Actual DDoS        15       85
```

We have:
- TP = 85 (correctly identified DDoS)
- TN = 280 (correctly identified normal)
- FP = 20 (normal misclassified as DDoS)
- FN = 15 (DDoS misclassified as normal)

**Calculations:**
```
Accuracy = (85 + 280) / (85 + 280 + 20 + 15) = 365 / 400 = 0.9125 (91.25%)

Precision = 85 / (85 + 20) = 85 / 105 = 0.8095 (80.95%)

Recall = 85 / (85 + 15) = 85 / 100 = 0.85 (85%)

F1-Score = 2 × (0.8095 × 0.85) / (0.8095 + 0.85) = 2 × 0.688 / 1.6595 = 0.829 (82.9%)
```

## Why These Metrics Matter for DDoS Detection

### Accuracy
- **Use**: Overall model performance
- **Limitation**: Can be misleading with imbalanced datasets

### Precision
- **Use**: Minimize false alarms
- **Importance**: Prevents blocking legitimate users
- **Business Impact**: Reduces customer complaints

### Recall
- **Use**: Catch as many attacks as possible
- **Importance**: Security effectiveness
- **Business Impact**: Prevents service disruption

### F1-Score
- **Use**: Balanced performance measure
- **Importance**: Best single metric when both precision and recall matter
- **Business Impact**: Optimal balance between security and usability

## Implementation in Code

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# For cross-validation
cv_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
cv_precision = cross_val_score(model, X, y, cv=5, scoring='precision')
cv_recall = cross_val_score(model, X, y, cv=5, scoring='recall')
cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1')

# Calculate mean and std
mean_accuracy = cv_accuracy.mean()
std_accuracy = cv_accuracy.std()
```

## Normalization in Our Algorithm

### MinMax Scaling
**Formula:**
```
X_normalized = (X - X_min) / (X_max - X_min)
```

**Purpose:**
- Scales features to range [0, 1]
- Required for chi-square feature selection
- Preserves relationships between features

### Why Normalization Matters
1. **Feature Selection**: Chi-square test requires non-negative values
2. **Tree Performance**: Decision trees are less sensitive to scaling, but normalization helps with feature importance interpretation
3. **Consistency**: Ensures all features are on the same scale

## Cross-Validation Process

### 5-Fold Stratified Cross-Validation
1. **Split**: Divide data into 5 equal parts (folds)
2. **Stratify**: Maintain class distribution in each fold
3. **Train**: Use 4 folds for training, 1 for validation
4. **Repeat**: Rotate validation fold 5 times
5. **Average**: Calculate mean and standard deviation of metrics

**Benefits:**
- Robust performance estimation
- Reduces overfitting
- Provides confidence intervals
- Better than single train-test split
