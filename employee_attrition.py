# ============================================================
#  EMPLOYEE ATTRITION ANALYSIS
#  Dataset: IBM HR Analytics (via Kaggle / public GitHub)
#  Models: Logistic Regression, Random Forest, XGBoost
# ============================================================

# STEP 1 — IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# STEP 2 — LOAD DATASET

url = (
    "https://raw.githubusercontent.com/dsrscientist/"
    "dataset1/master/HR_comma_sep.csv"
)

# Fallback: IBM HR Analytics dataset (also widely available)
ibm_url = (
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/"
    "master/data/emp_attrition.csv"
)

try:
    df = pd.read_csv(url)
    print("Dataset loaded successfully.")
    print("Shape:", df.shape)

    # Rename columns to standard names if using HR_comma_sep
    df.rename(columns={
        'left': 'Attrition',
        'average_montly_hours': 'avg_monthly_hours',
        'sales': 'department'
    }, inplace=True)

except Exception as e:
    print(f"Primary URL failed: {e}")
    print("Loading IBM HR Analytics dataset instead...")
    df = pd.read_csv(ibm_url)

print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:", df.columns.tolist())

# STEP 3 — DATA EXPLORATION

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Attrition Distribution ---")
attrition_col = 'Attrition' if 'Attrition' in df.columns else 'left'
print(df[attrition_col].value_counts())
print(df[attrition_col].value_counts(normalize=True).mul(100).round(2), "%")


# STEP 4 — DATA CLEANING

# Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\nDropped {before - len(df)} duplicate rows.")

# Handle missing values
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)


# STEP 5 — ENCODING

df_encoded = df.copy()

# Encode target
target_col = 'Attrition'
if df_encoded[target_col].dtype == object:
    df_encoded[target_col] = df_encoded[target_col].map({'Yes': 1, 'No': 0})
    if df_encoded[target_col].isnull().any():
        df_encoded[target_col] = df_encoded[target_col].fillna(
            df_encoded[target_col].mode()[0]
        )

# Encode all other categoricals
le = LabelEncoder()
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

print("\nEncoded dataframe shape:", df_encoded.shape)


# STEP 6 — FEATURE SCALING & SPLIT

X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")


# STEP 7 — MODEL TRAINING

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model":    model,
        "y_pred":   y_pred,
        "y_prob":   y_prob,
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc":  roc_auc_score(y_test, y_prob),
        "report":   classification_report(y_test, y_pred),
        "cm":       confusion_matrix(y_test, y_pred),
    }
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"Accuracy : {results[name]['accuracy']:.4f}")
    print(f"ROC-AUC  : {results[name]['roc_auc']:.4f}")
    print(results[name]["report"])

# Best model
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best = results[best_name]
print(f"\nBest model by ROC-AUC: {best_name} ({best['roc_auc']:.4f})")


# STEP 8 — FEATURE IMPORTANCE

rf_model = results["Random Forest"]["model"]
importances = pd.DataFrame({
    "Feature":    X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False).head(10)

print("\nTop 10 Features (Random Forest):")
print(importances.to_string(index=False))


# STEP 9 — HR KPI DASHBOARD (matplotlib)

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(18, 14))
fig.suptitle("HR Attrition Analysis Dashboard", fontsize=20, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── KPI Cards (text-only axes) ──────────────────────────────
kpis = [
    ("Total Employees",   f"{len(df):,}",                          "#1D9E75"),
    ("Attrition Rate",    f"{y.mean()*100:.1f}%",                  "#D85A30"),
    ("Best Model AUC",    f"{best['roc_auc']:.3f}",                "#185FA5"),
]
for i, (label, value, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.12,
                                transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.62, value, ha="center", va="center",
            fontsize=28, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha="center", va="center",
            fontsize=12, color="#444", transform=ax.transAxes)

# ── 1. Attrition Distribution (pie) ─────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
counts = y.value_counts()
labels = ["Stayed", "Left"]
colors = ["#1D9E75", "#D85A30"]
ax1.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=140, textprops={"fontsize": 11})
ax1.set_title("Attrition Split", fontsize=13, fontweight="bold")

# ── 2. Feature Importance ────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1:])
bars = ax2.barh(importances["Feature"][::-1], importances["Importance"][::-1],
                color="#378ADD", edgecolor="none")
ax2.set_xlabel("Importance Score")
ax2.set_title("Top 10 Attrition Drivers (Random Forest)", fontsize=13, fontweight="bold")
for bar in bars:
    ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.3f}", va="center", fontsize=9)

# ── 3. ROC Curves ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
roc_colors = ["#185FA5", "#1D9E75", "#D85A30"]
for (name, res), col in zip(results.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax3.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.2f})", color=col, lw=2)
ax3.plot([0, 1], [0, 1], "k--", lw=1)
ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curves", fontsize=13, fontweight="bold")
ax3.legend(fontsize=8)

# ── 4. Confusion Matrix (best model) ─────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
sns.heatmap(best["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"],
            ax=ax4, cbar=False)
ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")
ax4.set_title(f"Confusion Matrix\n({best_name})", fontsize=13, fontweight="bold")

# ── 5. Model Accuracy Comparison ─────────────────────────────
ax5 = fig.add_subplot(gs[2, 2])
model_names = list(results.keys())
accuracies  = [results[n]["accuracy"] for n in model_names]
auc_scores  = [results[n]["roc_auc"]  for n in model_names]
x = np.arange(len(model_names))
w = 0.35
bars1 = ax5.bar(x - w/2, accuracies, w, label="Accuracy", color="#378ADD", alpha=0.85)
bars2 = ax5.bar(x + w/2, auc_scores,  w, label="ROC-AUC",  color="#1D9E75", alpha=0.85)
ax5.set_xticks(x)
ax5.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=8)
ax5.set_ylim(0.5, 1.05)
ax5.set_title("Model Comparison", fontsize=13, fontweight="bold")
ax5.legend(fontsize=9)
for bar in list(bars1) + list(bars2):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{bar.get_height():.2f}", ha="center", fontsize=8)

plt.savefig("hr_attrition_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nDashboard saved as 'hr_attrition_dashboard.png'")

# STEP 10 — ATTRITION RISK SCORER (bonus utility)

def predict_attrition_risk(employee_data: dict) -> dict:
    """
    Pass a dict of employee features, get back a risk score.
    Example:
        predict_attrition_risk({
            'satisfaction_level': 0.4,
            'last_evaluation': 0.7,
            'number_project': 3,
            'avg_monthly_hours': 200,
            'time_spend_company': 4,
            'Work_accident': 0,
            'promotion_last_5years': 0,
            'department': 3,   # encoded
            'salary': 1        # encoded: low=0, medium=1, high=2
        })
    """
    row = pd.DataFrame([employee_data])
    row_encoded = row.reindex(columns=X.columns, fill_value=0)
    row_scaled  = scaler.transform(row_encoded)
    prob = rf_model.predict_proba(row_scaled)[0][1]
    risk = "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.3 else "LOW"
    return {"attrition_probability": round(prob, 4), "risk_level": risk}


# Demo call (adjust values to match your dataset's columns)
sample = {col: 0 for col in X.columns}  # zeroed baseline
sample.update({"satisfaction_level": 0.3, "avg_monthly_hours": 260,
               "time_spend_company": 5, "number_project": 6})
print("\nSample Risk Score:", predict_attrition_risk(sample))
