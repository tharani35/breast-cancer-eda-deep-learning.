# Exploratory Data Analysis and Deep Learning
## Breast Cancer Diagnosis Using the Wisconsin Diagnostic Dataset

**Student Name:** ______________________________  
**Register / Roll No.:** _________________________  
**Department / Year:** __________________________  
**Course / Subject:** ___________________________  
**Date of Submission:** _________________________

## Abstract

This report presents an exploratory data analysis and deep-learning case study using the Wisconsin Breast Cancer Diagnostic dataset. The dataset contains 569 cases, 30 continuous numerical features describing cell-nucleus characteristics, and a benign/malignant diagnosis. Data quality was checked for missing values, duplicate rows and extreme values; visualizations were used to study diagnosis balance, feature distributions, correlations, multivariate structure and class differences. A standardized multilayer perceptron (30-64-32-1) was trained with Adam and binary cross-entropy, using a stratified 70/15/15 train/validation/test split and validation-loss early stopping. On one fixed 86-case test split, the MLP achieved 1.000 accuracy and ROC-AUC, while Logistic Regression and Random Forest each achieved 0.988 accuracy. These results demonstrate a complete coursework pipeline but are not evidence of clinical readiness; broader validation is essential.

## 1. Introduction and Objectives

Exploratory data analysis (EDA) helps identify data quality issues, class balance, feature distributions, unusual values and relationships among variables before and alongside modelling. This case study uses numerical features computed from digitized fine-needle aspirate images of breast masses. The model operates on 30 tabular measurements, not original image pixels.

**Objectives**
- Describe a real-world dataset with 30 numerical predictors and a binary diagnosis.
- Inspect missing values, exact duplicates, outlier flags, distributions and correlated features.
- Create a minimum of five explained visuals; this expanded report contains 29 figure outputs.
- Standardize data without fitting the scaler on held-out partitions.
- Train a PyTorch multilayer perceptron and evaluate it using multiple metrics and charts.
- Compare the MLP with Logistic Regression and Random Forest baselines.
- Discuss reproducibility, limitations, and future improvements.

## 2. Dataset Description

**Dataset source:** [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). Scikit-learn also distributes a mirror of this dataset. The dataset has 569 instances and 30 continuous predictors. The target has 357 benign cases (62.7%) and 212 malignant cases (37.3%). The feature families are radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension, each represented as a mean, standard error, and worst value.

| Property | Value |
|---|---:|
| Observations | 569 |
| Numerical predictors | 30 |
| Benign cases | 357 (62.7%) |
| Malignant cases | 212 (37.3%) |
| Missing cells | 0 |
| Exact duplicate rows | 0 |
| Task | Binary classification |

Full descriptive statistics are exported as `outputs/feature_summary.csv`. Extreme values are flagged with the 1.5×IQR rule but are not automatically deleted, since statistical extremeness does not prove a measurement is invalid.

## 3. EDA and Preprocessing

The script checks dataset shape, missing values, exact duplicate rows and label frequencies. The target is encoded explicitly as `y=1` for malignant and `y=0` for benign. A stratified split produces 398 training, 85 validation, and 86 test cases. The class counts are 250 benign / 148 malignant in training, 53 / 32 in validation, and 54 / 32 in testing.

`StandardScaler` is fitted only on the training partition, then applied unchanged to validation and test data. This helps prevent data leakage. IQR-based outlier counts are saved to `outputs/iqr_outlier_counts.csv`. The 30 predictors include strongly correlated size-related measurements, so interpretation of individual variables should be cautious.

## 4. Deep Learning Model

The model architecture is **30 inputs → Dense(64) + ReLU + Dropout(0.20) → Dense(32) + ReLU + Dropout(0.10) → one output logit**. It has 4,097 trainable parameters. The sigmoid converts the output logit into a malignant-class probability during evaluation.

| Parameter | Setting |
|---|---|
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Dropout | 0.20 and 0.10 |
| Maximum epochs | 150 |
| Early-stopping patience | 15 epochs |
| Random seed | 42 |

The best validation-loss checkpoint is restored before final test evaluation. The test set is not used for early stopping.

## 5. Results and Evaluation

On the fixed 86-case test split, the MLP achieved accuracy 1.0000, malignant precision 1.0000, malignant recall 1.0000, malignant F1 1.0000, ROC-AUC 1.0000 and average precision 1.0000. Its confusion matrix contains 54 true negatives, 32 true positives, zero false positives and zero false negatives.

| Model | Accuracy | Malignant precision | Malignant recall | Malignant F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| MLP | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | 0.9884 | 0.9697 | 1.0000 | 0.9846 | 0.9988 |
| Random Forest | 0.9884 | 1.0000 | 0.9688 | 0.9841 | 1.0000 |

These are split-specific results, not proof of generalization. The test set includes only 32 malignant cases, so confidence in sensitivity is limited by sample size. Repeated stratified cross-validation and external validation are recommended.

## 6. Visualization Gallery

The following charts are generated by `breast_cancer_eda_mlp.py` and saved under `outputs/`. Each caption explains the visual and its limitations.


### Figure 1. Diagnosis class distribution

![Diagnosis class distribution](outputs/01_class_distribution.png)

There are 357 benign and 212 malignant cases. Stratification is used to preserve this ratio in the train, validation and test partitions.

### Figure 2. Mean radius by diagnosis

![Mean radius by diagnosis](outputs/02_mean_radius_distribution.png)

Malignant cases tend to have larger mean radius, although the distributions overlap. A single measurement is not a diagnostic rule.

### Figure 3. Mean concavity by diagnosis

![Mean concavity by diagnosis](outputs/03_concavity_boxplot.png)

The malignant group has a higher median mean concavity in this sample, supporting a group-level difference but not causation.

### Figure 4. Correlation heatmap of all 30 predictors

![Correlation heatmap of all 30 predictors](outputs/04_correlation_heatmap.png)

Size-related variables are strongly correlated, so some predictors may carry overlapping information.

### Figure 5. Mean radius versus mean texture

![Mean radius versus mean texture](outputs/05_radius_texture_scatter.png)

The class groups occupy partly different regions but overlap in two dimensions.

### Figure 6. Training and validation loss

![Training and validation loss](outputs/06_loss_vs_epoch.png)

Loss decreases rapidly early in training. Early stopping restores the checkpoint associated with the best validation loss.

### Figure 7. Validation accuracy by epoch

![Validation accuracy by epoch](outputs/07_accuracy_vs_epoch.png)

Validation accuracy rises rapidly, then fluctuates slightly. The validation set contains only 85 cases.

### Figure 8. Held-out test confusion matrix

![Held-out test confusion matrix](outputs/08_confusion_matrix.png)

All 54 benign and 32 malignant test cases were classified correctly in this split; small-sample uncertainty remains.

### Figure 9. ROC curve and AUC

![ROC curve and AUC](outputs/09_roc_curve.png)

AUC is 1.000 on this test partition only. ROC-AUC is not a calibration measure.

### Figure 10. Data quality and dataset size audit

![Data quality and dataset size audit](outputs/10_data_quality_audit.png)

The loaded table has 569 rows, 30 predictors, no missing cells and no exact duplicate rows under the performed checks.

### Figure 11. Diagnosis proportion

![Diagnosis proportion](outputs/11_diagnosis_donut.png)

The chart expresses the target ratio as percentages: 62.7% benign and 37.3% malignant.

### Figure 12. Worst-area distribution by diagnosis

![Worst-area distribution by diagnosis](outputs/12_worst_area_violin.png)

The malignant group shows larger worst-area values and a higher-value tail. Extreme values are not automatically errors.

### Figure 13. Mean perimeter versus mean area

![Mean perimeter versus mean area](outputs/13_perimeter_area_scatter.png)

The strong positive relationship reflects related geometric characteristics of nucleus size.

### Figure 14. Standardized class mean profiles

![Standardized class mean profiles](outputs/14_class_mean_profile_heatmap.png)

Ten features with the largest standardized class-mean differences are displayed on a common scale for visual comparison.

### Figure 15. Twelve strongest absolute correlations

![Twelve strongest absolute correlations](outputs/15_top_feature_correlations.png)

The strongest pairs are largely size-related measurements. Correlation does not imply causation.

### Figure 16. PCA projection of standardized features

![PCA projection of standardized features](outputs/16_pca_projection.png)

The first two principal components show partial separation and overlap; only two components are displayed.

### Figure 17. Precision-recall curve

![Precision-recall curve](outputs/17_precision_recall_curve.png)

Average precision is 1.000 on this test split. Precision-recall analysis focuses on the malignant positive class.

### Figure 18. Probability threshold trade-offs

![Probability threshold trade-offs](outputs/18_threshold_tradeoff.png)

Precision, sensitivity and specificity change with the threshold. A 0.50 threshold is a software default, not a clinically justified setting.

### Figure 19. Predicted probabilities by actual class

![Predicted probabilities by actual class](outputs/19_probability_distribution.png)

Test probabilities are well separated by actual label in this run, but external data is needed to assess generalization.

### Figure 20. Sorted test predictions

![Sorted test predictions](outputs/20_sorted_test_predictions.png)

Every test case is on the correct side of the 0.50 threshold in this split; this visual reflects only 86 test cases.

### Figure 21. MLP architecture

![MLP architecture](outputs/21_model_architecture.png)

The network has 30 inputs, hidden layers of 64 and 32 units, dropout and one output logit.

### Figure 22. Class counts across data splits

![Class counts across data splits](outputs/22_split_distribution.png)

Stratification preserves approximately the same ratio across 398 train, 85 validation and 86 test observations.

### Figure 23. Approximate permutation importance

![Approximate permutation importance](outputs/23_permutation_importance.png)

Validation accuracy drops are small and variable for correlated features; treat these scores as exploratory, not definitive.

### Figure 24. Per-class precision, recall and F1

![Per-class precision, recall and F1](outputs/24_classification_report_heatmap.png)

Both classes score 1.000 on this split; this does not remove uncertainty from the small hold-out set.

### Figure 25. MLP versus conventional baselines

![MLP versus conventional baselines](outputs/25_model_comparison.png)

The MLP scored 1.000 accuracy while the two baselines scored 0.988; one split is insufficient to claim general superiority.

### Figure 26. Probability calibration check

![Probability calibration check](outputs/26_calibration_curve.png)

The binned curve is coarse because the test set is small. High AUC does not guarantee well-calibrated probabilities.

### Figure 27. Pairwise relationships among selected features

![Pairwise relationships among selected features](outputs/27_selected_feature_pairplot.png)

The pairplot shows distribution shapes and pairwise correlations among four representative features.

### Figure 28. Representative feature distribution grid

![Representative feature distribution grid](outputs/28_feature_distribution_grid.png)

Six distributions cover size, texture, smoothness and concavity, showing differing degrees of group overlap.

### Figure 29. End-to-end analysis workflow

![End-to-end analysis workflow](outputs/29_end_to_end_workflow.png)

The diagram summarizes data checks, EDA, stratification, scaling, MLP training and final test evaluation.


## 7. Discussion and Limitations

- **Single split:** performance may change under another train/test assignment. Repeated stratified cross-validation is recommended.
- **Small test set:** 86 cases, including 32 malignant cases, provide limited statistical confidence.
- **Population shift:** model behaviour may change if features are measured under different processes or in different populations.
- **Correlated predictors:** feature-attribution scores may be unstable because related features can substitute for one another.
- **Calibration and threshold:** a strong ROC-AUC does not guarantee accurate risk probabilities. A decision threshold requires context-specific assessment.
- **Clinical safety:** this is a classroom demonstration, not a medical device, screening tool or diagnostic service.

## 8. Conclusion and Future Scope

The case study demonstrates data loading, descriptive analysis, 29 figures, a PyTorch MLP, baseline comparisons and multi-metric evaluation. The MLP performed best on the fixed split, while the conventional baselines also performed strongly. Future work should include repeated cross-validation, external validation, calibration analysis, confidence intervals, subgroup performance checks, threshold analysis and expert review.

## 9. References

1. UCI Machine Learning Repository. “Breast Cancer Wisconsin (Diagnostic).” https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
2. scikit-learn developers. “load_breast_cancer” documentation. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
3. Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *Biomedical Image Processing and Biomedical Visualization*, SPIE, 861–870.
4. Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
5. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.
6. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
7. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

## Appendix A. Run Instructions

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python breast_cancer_eda_mlp.py
```

The complete commented source file is `breast_cancer_eda_mlp.py`. The run generates chart images, training history, feature summaries, outlier counts, baseline metrics, permutation-importance results and `metrics.json`.

**Disclaimer:** This project is educational only and must not be used for medical decisions.
