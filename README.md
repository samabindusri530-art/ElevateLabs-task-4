# Task 4: Classification with Logistic Regression
**AI & ML Internship — Elevate Labs**

## Objective
Build a binary classifier using logistic regression.

## Tools Used
Python, Scikit-learn, Pandas, Matplotlib, Seaborn

## Dataset
**Breast Cancer Wisconsin (Diagnostic) dataset** — the real dataset, loaded
directly from `sklearn.datasets.load_breast_cancer` and exported to
`data/breast_cancer.csv`. 569 samples, 30 numeric features, binary target
(`0 = malignant`, `1 = benign`).

## Project Structure
```
├── data/
│   └── breast_cancer.csv
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── sigmoid_function.png
│   └── threshold_tuning.png
├── export_dataset.py
├── logistic_regression.py
├── threshold_comparison.csv
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python export_dataset.py       # only needed if data/breast_cancer.csv doesn't exist
python logistic_regression.py
```

## What Was Done

1. **Dataset:** Breast Cancer Wisconsin — a genuinely binary, real-world
   diagnostic dataset, well suited for logistic regression.
2. **Train/test split (80/20, stratified)** and **standardization** —
   features were scaled with `StandardScaler` (fit on train, applied to
   test) since logistic regression is a linear, gradient-based model and
   features here span very different scales (e.g. "mean area" in the
   hundreds vs. "mean smoothness" around 0.1).
3. **Fit** `sklearn.linear_model.LogisticRegression`.
4. **Evaluated** with a confusion matrix, precision, recall, F1, and
   ROC-AUC, plus the full classification report. Also plotted the
   Precision-Recall curve as a complementary view.
5. **Explained and plotted the sigmoid function**, and **tuned the
   decision threshold** across several values to see the precision/recall
   trade-off directly.

## Results (default threshold = 0.5)

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 0.9825 |
| Precision   | 0.9861 |
| Recall      | 0.9861 |
| F1 Score    | 0.9861 |
| ROC-AUC     | 0.9954 |

Confusion matrix: only 1 malignant case misclassified as benign, and 1
benign case misclassified as malignant, out of 114 test samples.

**Threshold tuning:** lowering the threshold to 0.3 pushed recall to
**1.000** (catching every malignant case in the test set) at a small
precision cost. In a medical screening context, missing a malignant tumor
(false negative) is typically far more costly than a false alarm, so a
lower threshold favoring recall is often the more sensible operating
point — not necessarily the default 0.5. See `threshold_comparison.csv`
and `images/threshold_tuning.png` for the full trade-off curve.

---

