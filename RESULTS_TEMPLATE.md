# Results Template - Copy This Into Your Report

## Table 1: Cross-Validation Results

| Fold | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| 1    | 0.9996   | 1.0000    | 0.9994 | 0.9997   |
| 2    | 0.9995   | 0.9997    | 0.9996 | 0.9996   |
| 3    | 0.9998   | 0.9999    | 0.9998 | 0.9999   |
| 4    | 0.9997   | 0.9999    | 0.9996 | 0.9998   |
| 5    | 0.9997   | 0.9999    | 0.9997 | 0.9998   |
| **Mean** | **0.9997** | **0.9999** | **0.9996** | **0.9998** |
| **Std** | **±0.0001** | **±0.0001** | **±0.0001** | **±0.0001** |

---

## Table 2: Test Set Performance Metrics

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9998 |
| Precision | 0.9999 |
| Recall    | 0.9997 |
| F1-Score  | 0.9998 |

---

## Table 3: Confusion Matrix

|                | Predicted: Normal | Predicted: DDoS | Total |
|----------------|-------------------|-----------------|-------|
| **Actual: Normal** | 8,860 (TN)      | 2 (FP)          | 8,862 |
| **Actual: DDoS**   | 4 (FN)          | 15,134 (TP)     | 15,138|
| **Total**          | 8,864           | 15,136          | 24,000|

**Legend:**
- TP (True Positives) = 15,134
- TN (True Negatives) = 8,860
- FP (False Positives) = 2
- FN (False Negatives) = 4

---

## Table 4: Classification Report

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Normal | 1.00      | 1.00   | 1.00     | 8,862   |
| DDoS   | 1.00      | 1.00   | 1.00     | 15,138  |
| **Macro Avg** | 1.00 | 1.00 | 1.00 | 24,000 |
| **Weighted Avg** | 1.00 | 1.00 | 1.00 | 24,000 |

---

## Table 5: Top 10 Feature Importances

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 1    | Fwd Packet Length Max | 0.5402 |
| 2    | Destination Port | 0.3675 |
| 3    | Packet Length Mean | 0.0618 |
| 4    | URG Flag Count | 0.0159 |
| 5    | Average Packet Size | 0.0136 |
| 6    | Avg Fwd Segment Size | 0.0006 |
| 7    | Packet Length Std | 0.0002 |
| 8    | Packet Length Variance | 0.0001 |
| 9    | Bwd Packet Length Std | 0.0000 |
| 10   | Bwd IAT Total | 0.0000 |

---

## Model Architecture

- **Algorithm:** CART Decision Tree (scikit-learn)
- **Max Depth:** 8
- **Number of Leaves:** 20
- **Criterion:** Gini Impurity
- **Class Weight:** Balanced
- **Features Selected:** 20 (from 78 numeric features using Chi-square)

---

## Key Findings Summary

1. **High Performance:** The model achieved 99.98% accuracy on the test set
2. **Stable Results:** Low standard deviation (0.0001) in cross-validation indicates consistent performance
3. **Low Error Rate:** Only 6 misclassifications out of 24,000 test samples
4. **Most Important Features:** Forward packet length maximum and destination port are the primary discriminators
5. **Balanced Performance:** Equal precision and recall for both classes (Normal and DDoS)

---

## Figures to Include

1. **Figure 1:** Confusion Matrix (`confusion_matrix.png`)
2. **Figure 2 (Optional):** Feature Importance Bar Chart
3. **Figure 3 (Optional):** Decision Tree Visualization (if tree is small)

