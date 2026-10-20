# Breast Cancer EDA and Deep Learning

Detailed assignment project using the Wisconsin Breast Cancer Diagnostic dataset.

## Dataset
- UCI source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- 569 observations, 30 numerical predictors, diagnosis label.
- The script loads the scikit-learn mirror, so no manual download is needed.

## Repository contents
- `breast_cancer_eda_mlp.py` — commented EDA, preprocessing, MLP training, 3-model comparison and 29-chart generation.
- `report.md` — detailed report with figure gallery and references.
- `report.docx` / `report.pdf` — generated assignment report, including figures.
- `requirements.txt` — dependencies.
- `outputs/` — class/distribution charts, correlations, PCA, training curves, ROC/PR curves, confusion matrix, thresholds, calibration, model comparisons, metrics and CSV summaries.

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python breast_cancer_eda_mlp.py
```

The pipeline uses a stratified 70/15/15 train/validation/test split, fits standardization on training data only, and uses validation-loss early stopping. It compares the MLP against Logistic Regression and Random Forest using the same split.

## Results and limitations
For the recorded fixed 86-case test split, the MLP achieved 1.000 accuracy and ROC-AUC; Logistic Regression and Random Forest each achieved 0.988 accuracy. The small test sample and single split cannot establish generalization or clinical validity. This project is educational only and must not be used for medical decisions.
