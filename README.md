# Breast Cancer EDA and Deep Learning

This project explores the Wisconsin Breast Cancer Diagnostic dataset and trains a PyTorch multilayer perceptron to classify tumors as benign or malignant.

## Contents
- `breast_cancer_eda_mlp.py` — preprocessing, exploratory data analysis, training, and evaluation pipeline.
- `requirements.txt` — Python dependencies.
- `report.docx` — assignment report.
- `outputs/` — EDA charts, training curves, ROC curve, confusion matrix, metrics, and CSV summaries.

## Run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python breast_cancer_eda_mlp.py
```

The script loads the Wisconsin Breast Cancer Diagnostic dataset from scikit-learn, so a separate dataset download is not required. Results are educational only and must not be used for clinical diagnosis. Performance on a small held-out split does not establish clinical validity.

## Dataset
The dataset contains 569 instances and 30 numerical diagnostic features, with a benign/malignant target. Refer to the report for details, methodology, visualizations, evaluation, limitations, and references.
