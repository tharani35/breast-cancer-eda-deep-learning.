# Case Study: EDA and Deep Learning
## Breast Cancer Diagnosis Using the Wisconsin Diagnostic Dataset

**Student Name:** ______________________________  
**Register Number:** ___________________________  
**Department / Year:** _________________________  
**Course:** Data Visualization / EDA  
**Submission Date:** ___________________________

## Abstract
This case study explores the Wisconsin Diagnostic Breast Cancer dataset, containing 569 observations, 30 numerical cell-nucleus measurements, and a binary diagnosis label. The objective is to investigate distributions and relationships among measurements and build an educational deep-learning classifier distinguishing benign from malignant cases. The workflow includes data-quality checks, descriptive statistics, five exploratory visualizations, stratified train/validation/test splitting, training-only standardization, and a multilayer perceptron (MLP) with early stopping. Performance is assessed using accuracy, precision, recall, F1-score, a confusion matrix, and ROC-AUC, alongside loss and validation-accuracy curves. On the fixed held-out test split, this run achieved accuracy 1.000 and ROC-AUC 1.000. These results do not establish clinical suitability; external validation and expert oversight are essential before any real-world medical use.

## 1. Introduction and Objectives
Breast cancer diagnosis can involve examining characteristics of cell nuclei obtained from fine-needle aspirates. This project uses numerical measurements to explore benign/malignant differences and demonstrate a reproducible data-analysis workflow.

- Inspect a real dataset with more than 15 fields.
- Explore distributions, class frequencies, outliers, and feature relationships.
- Create readable charts and interpret evidence.
- Train and evaluate an MLP using train/validation/test partitions.
- Discuss limitations and responsible future improvements.

## 2. Dataset Description
**Source:** Wisconsin Diagnostic Breast Cancer (WDBC), originally available from the UCI Machine Learning Repository and mirrored by scikit-learn. It has 569 cases, 30 numerical predictors, and diagnosis labels. Predictors include radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension, represented by mean, standard-error, and worst measurements.

| Property | Value |
|---|---:|
| Rows | 569 |
| Numerical features | 30 |
| Benign | 357 (62.7%) |
| Malignant | 212 (37.3%) |
| Missing values | 0 |
| Duplicate rows checked | 0 |

The target is moderately imbalanced, so stratified splitting and class-specific metrics are useful.

## 3. EDA and Preprocessing
The code loads data, maps the target to readable labels, and checks missing values and duplicate rows. No imputation was needed. IQR-based outlier counts are calculated for inspection; extreme measurements are not automatically removed because they may be informative. Numerical features are standardized for the MLP. StandardScaler is fitted only on training data to prevent leakage.

A stratified 70%/15%/15% split creates training, validation, and test sets. Validation data is used for early stopping; the test set is reserved for final evaluation.

## 4. Exploratory Visualizations
The script generates five EDA visualizations:

1. **Diagnosis class distribution:** benign cases (357) outnumber malignant cases (212), motivating stratified splitting and more than accuracy-only evaluation.
2. **Mean-radius distributions:** malignant cases tend to have larger mean radius, but distributions overlap; this feature alone is not diagnostic.
3. **Mean concavity by diagnosis:** malignant cases generally exhibit higher mean concavity.
4. **Feature correlation heatmap:** several size-related variables are strongly correlated, indicating redundancy; correlation does not imply causation.
5. **Radius vs texture scatterplot:** groups occupy differing regions but overlap, showing that two features do not perfectly separate classes.

Visualization workflow: define the analytical question; select relevant fields; clean/transform data; choose a suitable chart and encoding; create the visual; add labels, scale, and legend; interpret findings. If the course specifies a named seven-stage framework, adjust this mapping to the instructor's terminology.

## 5. Deep Learning Model
PyTorch MLP architecture: **30 inputs → Dense(64), ReLU, Dropout(0.20) → Dense(32), ReLU, Dropout(0.10) → one output logit**.

Training uses BCEWithLogitsLoss, Adam (learning rate 0.001), weight decay 0.0001, batch size 32, and validation-loss early stopping (patience 15; maximum 150 epochs). Basic tuning is represented by architecture regularization and early stopping. A stronger study would compare hyperparameter settings using validation data only.

## 6. Evaluation and Results
The split sizes were 398 training, 85 validation, and 86 test cases. Early stopping selected a checkpoint after 29 epochs.

| Metric | Result |
|---|---:|
| Test accuracy | 1.000 |
| Precision (malignant) | 1.000 |
| Recall (malignant) | 1.000 |
| F1-score (malignant) | 1.000 |
| ROC-AUC | 1.000 |

The model evaluation visualizations are the training/validation loss curve, validation accuracy curve, held-out test confusion matrix, and ROC curve. This run had no errors on its 86-case test split, but the small sample does not prove perfect generalization. AUC=1.000 on this split should not be interpreted as proof of clinical performance.

## 7. Conclusion and Future Scope
EDA revealed class imbalance, group differences in selected measurements, and correlated predictors. The MLP scored very highly on this fixed test split. However, the dataset and test sample are limited; results may not generalize. This is a coursework demonstration, not a medical diagnostic tool.

Future work: stratified cross-validation, comparison with simpler ML baselines, external validation, calibration and subgroup fairness analysis, confidence intervals, threshold analysis focused on clinically relevant sensitivity, and expert review.

## 8. Literature Review and References
1. Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. In *Biomedical Image Processing and Biomedical Visualization* (pp. 861–870). SPIE.
2. UCI Machine Learning Repository. (n.d.). *Breast Cancer Wisconsin (Diagnostic) dataset*. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
3. scikit-learn developers. (n.d.). *load_breast_cancer documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
4. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*.
5. Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

## Appendix: Running the Code
Run `pip install -r requirements.txt`, then `python breast_cancer_eda_mlp.py`. The script generates nine charts, `metrics.json`, `training_history.csv`, `feature_summary.csv`, and IQR outlier counts in the `outputs/` folder. Fixing random seeds supports reproducibility; software/hardware may still cause small numerical differences.

**Disclaimer:** This project is for educational use only and must not be used as a medical diagnostic tool.
