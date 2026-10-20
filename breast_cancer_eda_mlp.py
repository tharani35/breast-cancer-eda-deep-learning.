"""
Breast Cancer Wisconsin Diagnostic: EDA + MLP
Run: python breast_cancer_eda_mlp.py
Dependencies: numpy, pandas, matplotlib, seaborn, scikit-learn, torch
"""
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, roc_auc_score)
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 1. Load data (scikit-learn mirror of the UCI Wisconsin Diagnostic dataset)
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
# Dataset target is 0=malignant, 1=benign in sklearn; map to clinically clear labels.
df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
df.drop(columns=["target"], inplace=True)
feature_cols = list(raw.feature_names)
X = df[feature_cols].to_numpy(dtype=np.float32)
# Positive class for ROC is malignant.
y = (df["diagnosis"] == "malignant").astype(np.int64).to_numpy()

print("Dataset shape:", df.shape)
print("Missing cells:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print(df["diagnosis"].value_counts())

# 2. EDA and saved visualizations
sns.set_theme(style="whitegrid")
plt.figure(figsize=(7,5))
sns.countplot(data=df, x="diagnosis", hue="diagnosis", order=["benign","malignant"], palette="Set2", legend=False)
plt.title("Diagnosis Class Distribution"); plt.xlabel("Diagnosis"); plt.ylabel("Number of cases")
plt.tight_layout(); plt.savefig(OUT/"01_class_distribution.png", dpi=180); plt.close()

plt.figure(figsize=(8,5))
sns.histplot(data=df, x="mean radius", hue="diagnosis", kde=True, bins=25, element="step")
plt.title("Distribution of Mean Radius by Diagnosis"); plt.xlabel("Mean radius"); plt.tight_layout()
plt.savefig(OUT/"02_mean_radius_distribution.png", dpi=180); plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="diagnosis", y="mean concavity", hue="diagnosis", order=["benign","malignant"], palette="Set2", legend=False)
plt.title("Mean Concavity by Diagnosis"); plt.xlabel("Diagnosis"); plt.ylabel("Mean concavity")
plt.tight_layout(); plt.savefig(OUT/"03_concavity_boxplot.png", dpi=180); plt.close()

corr = df[feature_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", center=0, square=False, xticklabels=False, yticklabels=False)
plt.title("Correlation Heatmap of 30 Numerical Features")
plt.tight_layout(); plt.savefig(OUT/"04_correlation_heatmap.png", dpi=180); plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="mean radius", y="mean texture", hue="diagnosis", alpha=.75)
plt.title("Mean Radius vs Mean Texture"); plt.tight_layout()
plt.savefig(OUT/"05_radius_texture_scatter.png", dpi=180); plt.close()

# Distribution summaries, feature correlations, and simple outlier counts (IQR rule)
df[feature_cols].describe().T.to_csv(OUT/"feature_summary.csv")
q1=df[feature_cols].quantile(.25); q3=df[feature_cols].quantile(.75); iqr=q3-q1
outlier_counts=((df[feature_cols] < (q1-1.5*iqr)) | (df[feature_cols] > (q3+1.5*iqr))).sum()
outlier_counts.to_csv(OUT/"iqr_outlier_counts.csv", header=["IQR_outlier_count"])

# 3. Split train / validation / test (70/15/15); stratify to preserve class ratio.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=.30, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=.50, random_state=SEED, stratify=y_temp)

# Fit scaler only on training data to avoid data leakage.
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train).astype(np.float32)
X_val=scaler.transform(X_val).astype(np.float32)
X_test=scaler.transform(X_test).astype(np.float32)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=nn.Sequential(nn.Linear(30,64), nn.ReLU(), nn.Dropout(.20),
                    nn.Linear(64,32), nn.ReLU(), nn.Dropout(.10),
                    nn.Linear(32,1))
model.to(device)
criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
train_ds=TensorDataset(torch.tensor(X_train),torch.tensor(y_train,dtype=torch.float32))
loader=DataLoader(train_ds,batch_size=32,shuffle=True)
Xt=torch.tensor(X_val,device=device); yt=torch.tensor(y_val,dtype=torch.float32,device=device)
history={"train_loss":[],"val_loss":[],"val_accuracy":[]}
best_val=float("inf"); best_state=None; patience=15; stale=0; max_epochs=150

for epoch in range(max_epochs):
    model.train(); running=0.0; count=0
    for xb,yb in loader:
        xb=xb.to(device); yb=yb.to(device).view(-1,1)
        optimizer.zero_grad()
        logits=model(xb); loss=criterion(logits,yb)
        loss.backward(); optimizer.step()
        running += loss.item()*len(xb); count += len(xb)
    train_loss=running/count
    model.eval()
    with torch.no_grad():
        val_logits=model(Xt).view(-1)
        val_loss=criterion(val_logits,yt).item()
        val_pred=(torch.sigmoid(val_logits)>=.5).cpu().numpy()
        val_acc=accuracy_score(y_val,val_pred)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)
    if val_loss < best_val:
        best_val=val_loss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
    else:
        stale+=1
    if stale>=patience: break

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    logits=model(torch.tensor(X_test,device=device)).view(-1).cpu().numpy()
probs=1/(1+np.exp(-logits))
pred=(probs>=.5).astype(int)
metrics={
 "test_accuracy":float(accuracy_score(y_test,pred)),
 "test_precision_malignant":float(precision_score(y_test,pred,zero_division=0)),
 "test_recall_malignant":float(recall_score(y_test,pred,zero_division=0)),
 "test_f1_malignant":float(f1_score(y_test,pred,zero_division=0)),
 "test_roc_auc":float(roc_auc_score(y_test,probs)),
 "epochs_run":len(history["train_loss"]),
 "train_size":len(y_train),"validation_size":len(y_val),"test_size":len(y_test),
 "dataset_rows":len(df),"feature_count":len(feature_cols),
}
(OUT/"metrics.json").write_text(json.dumps(metrics,indent=2))
pd.DataFrame(history).to_csv(OUT/"training_history.csv",index_label="epoch")

# Model performance visualizations
plt.figure(figsize=(8,5))
plt.plot(history["train_loss"],label="Training loss")
plt.plot(history["val_loss"],label="Validation loss")
plt.title("Loss vs Epoch"); plt.xlabel("Epoch"); plt.ylabel("BCE loss"); plt.legend(); plt.tight_layout()
plt.savefig(OUT/"06_loss_vs_epoch.png",dpi=180); plt.close()

plt.figure(figsize=(8,5))
plt.plot(history["val_accuracy"],label="Validation accuracy")
plt.title("Validation Accuracy vs Epoch"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.ylim(0,1.02)
plt.legend(); plt.tight_layout(); plt.savefig(OUT/"07_accuracy_vs_epoch.png",dpi=180); plt.close()

cm=confusion_matrix(y_test,pred,labels=[0,1])
plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Pred. benign","Pred. malignant"],
            yticklabels=["Actual benign","Actual malignant"])
plt.title("Test Confusion Matrix"); plt.tight_layout(); plt.savefig(OUT/"08_confusion_matrix.png",dpi=180); plt.close()

fpr,tpr,_=roc_curve(y_test,probs)
plt.figure(figsize=(7,5))
plt.plot(fpr,tpr,label=f"MLP (AUC={metrics['test_roc_auc']:.3f})")
plt.plot([0,1],[0,1],"--",label="Random baseline")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
plt.legend(); plt.tight_layout(); plt.savefig(OUT/"09_roc_curve.png",dpi=180); plt.close()

print("\nTEST METRICS")
print(json.dumps(metrics,indent=2))
print("\nClassification report (0=malignant, 1=benign):")
print(classification_report(y_test,pred,target_names=["malignant","benign"],zero_division=0))
print("\nFigures and metrics saved under outputs/")
