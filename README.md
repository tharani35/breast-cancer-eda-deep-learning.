# Breast Cancer EDA and Deep Learning

This assignment analyzes the Wisconsin Diagnostic Breast Cancer dataset and trains a PyTorch multilayer perceptron to classify diagnoses as benign or malignant.

## Repository contents
- `breast_cancer_eda_mlp.py` — full data loading, preprocessing, EDA, training, and evaluation pipeline.
- `requirements.txt` — required Python packages.
- `report.md` — complete assignment report, results, limitations, and references.
- `outputs/metrics.json` — recorded metrics for the included run.
- `outputs/training_history.csv` — per-epoch training/validation statistics.
- `outputs/feature_summary.csv` — descriptive statistics for the 30 predictors.
- `outputs/iqr_outlier_counts.csv` — IQR-based outlier counts.

The Python script generates nine chart images in `outputs/` when run: five EDA plots, loss/accuracy learning curves, a test confusion matrix, and an ROC curve.

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python breast_cancer_eda_mlp.py
```

The dataset is loaded through scikit-learn, so a separate download is not needed. The training script uses a stratified 70/15/15 split and fits standardization on the training set only. It uses validation-loss early stopping.

## Notes
The included run reached 1.000 accuracy and 1.000 ROC-AUC on an 86-case hold-out test split. That small-sample result does not prove generalization. This is an educational coursework project, not a medical diagnostic tool.
