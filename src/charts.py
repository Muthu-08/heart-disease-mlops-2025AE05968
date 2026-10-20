import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    roc_curve, 
    auc, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    roc_auc_score
)

# Ensure screenshots directory exists
os.makedirs("screenshots", exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

# -------------------------------------------------------------------------
# 1. Load Cleaned Dataset
# -------------------------------------------------------------------------
df = pd.read_csv("data/processed_heart.csv")
X = df.drop(columns=["target"])
y = df["target"]

# -------------------------------------------------------------------------
# PNG 1: Target Class Balance Distribution
# -------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
ax = sns.countplot(
    x="target", 
    data=df, 
    palette=["#2b5c8f", "#d95f02"], 
    hue="target", 
    legend=False
)
plt.title("Heart Disease Risk Target Distribution", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Clinical Diagnosis", fontsize=11)
plt.ylabel("Patient Count", fontsize=11)
plt.xticks([0, 1], ["Absence (0)\n164 patients (54.1%)", "Presence (1)\n139 patients (45.9%)"])

for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(f"{height}", (p.get_x() + 0.35, height + 2), fontweight="bold")

plt.ylim(0, 190)
plt.tight_layout()
plt.savefig("screenshots/01_eda_distribution.png", dpi=300)
plt.close()
print("Generated: screenshots/01_eda_distribution.png")

# -------------------------------------------------------------------------
# PNG 2: Pearson Correlation Heatmap
# -------------------------------------------------------------------------
plt.figure(figsize=(11, 8))
sns.heatmap(
    df.corr(), 
    annot=True, 
    fmt=".2f", 
    cmap="vlag", 
    center=0, 
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("UCI Heart Disease Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("screenshots/02_correlation_heatmap.png", dpi=300)
plt.close()
print("Generated: screenshots/02_correlation_heatmap.png")

# -------------------------------------------------------------------------
# Prepare ML Pipelines
# -------------------------------------------------------------------------
NUM_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

trained_pipelines = {}
metrics_summary = {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe
    
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    
    metrics_summary[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs)
    }

# -------------------------------------------------------------------------
# PNG 3: Model Metrics & Cross-Validation Comparison
# -------------------------------------------------------------------------
metrics_df = pd.DataFrame(metrics_summary).T

plt.figure(figsize=(8, 5))
ax = metrics_df.plot(kind="bar", figsize=(9, 5), colormap="viridis", edgecolor="black", width=0.7)
plt.title("Model Performance Comparison (Test Set)", fontsize=13, fontweight="bold", pad=12)
plt.ylabel("Score", fontsize=11)
plt.ylim(0, 1.15)
plt.xticks(rotation=0, fontweight="bold")
plt.legend(loc="lower right")

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{height:.2f}", (p.get_x() + 0.02, height + 0.02), fontsize=9)

plt.tight_layout()
plt.savefig("screenshots/03_model_metrics_comparison.png", dpi=300)
plt.close()
print("Generated: screenshots/03_model_metrics_comparison.png")

# -------------------------------------------------------------------------
# PNG 4: ROC Curves Comparison
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 6))

for name, pipe in trained_pipelines.items():
    probs = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_score:.3f})")

plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance baseline (0.500)")
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
plt.title("Receiver Operating Characteristic (ROC) Benchmark", fontsize=13, fontweight="bold", pad=12)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("screenshots/04_roc_curves.png", dpi=300)
plt.close()
print("Generated: screenshots/04_roc_curves.png")

# -------------------------------------------------------------------------
# PNG 5: Confusion Matrices for Both Models
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for idx, (name, pipe) in enumerate(trained_pipelines.items()):
    preds = pipe.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Absence", "Presence"])
    disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
    axes[idx].set_title(f"{name} Confusion Matrix", fontweight="bold")
    axes[idx].grid(False)

plt.tight_layout()
plt.savefig("screenshots/05_confusion_matrices.png", dpi=300)
plt.close()
print("Generated: screenshots/05_confusion_matrices.png")

print("\nAll programmatic report figures generated successfully in 'screenshots/'.")