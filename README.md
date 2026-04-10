# 🧠 RetentionIQ — Employee Attrition Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square)
![Pandas](https://img.shields.io/badge/pandas-1.5%2B-150458?style=flat-square)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.6%2B-11557c?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> **Predict who's leaving. Understand why. Act before it's too late.**
> RetentionIQ uses machine learning to identify at-risk employees and surface the factors driving attrition — giving HR teams the intelligence they need to retain talent.

---

## 📌 Overview

Employee attrition costs organizations 33%–200% of an employee's annual salary in recruiting, onboarding, and lost productivity. Most companies only realize someone is disengaged after they've already resigned.

**RetentionIQ** is an end-to-end HR analytics pipeline that:
- Predicts which employees are likely to leave using three ML classifiers
- Identifies the top factors driving attrition (satisfaction, workload, tenure, etc.)
- Generates a full HR KPI dashboard with confusion matrix, ROC curves, and feature importance
- Provides a real-time employee risk scorer for HR decision-making

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | IBM HR Analytics / HR_comma_sep (public) |
| Rows | ~15,000 employee records |
| Features | Satisfaction level, evaluation score, projects, monthly hours, tenure, accidents, promotions, department, salary |
| Target | `Attrition` — Left (1) / Stayed (0) |
| Class balance | ~76% Stayed / ~24% Left |

---

## 🗂️ Project Structure

```
retentioniq/
├── employee_attrition.py           # Main ML pipeline (all 10 steps)
├── requirements.txt                # Python dependencies
├── README.md                       # You are here
├── .gitignore
└── outputs
    
```

---

## ⚙️ Pipeline Workflow

```
Raw HR Data → Explore → Clean → Encode → Scale → Split → Train → Evaluate → Dashboard → Risk Score
```

| Step | Description |
|---|---|
| 1. Load | Fetch dataset from public URL with fallback |
| 2. Explore | Shape, dtypes, class balance, missing value report |
| 3. Clean | Drop duplicates, fill nulls (median / mode) |
| 4. Encode | Map target Yes/No → 1/0, LabelEncode categoricals |
| 5. Scale | StandardScaler + stratified 80/20 train-test split |
| 6. Train | Logistic Regression, Random Forest, Gradient Boosting |
| 7. Evaluate | Accuracy, ROC-AUC, F1, precision, recall, confusion matrix |
| 8. Visualize | 9-panel matplotlib HR dashboard saved as PNG |
| 9. Feature IQ | Top 10 attrition drivers from Random Forest |
| 10. Risk Score | `predict_attrition_risk()` — score any individual employee |

---

## 🤖 Models & Results

| Model | Accuracy | ROC-AUC | Notes |
|---|---|---|---|
| Logistic Regression | ~78% | ~0.82 | Fast, interpretable baseline |
| Random Forest | ~99% | ~0.99 | Best overall, top feature insights |
| Gradient Boosting | ~97% | ~0.99 | Strong generalization |

> Metrics vary slightly by run. Use `random_state=42` to reproduce exactly.

---

## 📈 HR Dashboard

The dashboard renders automatically when you run the script and is saved to `outputs/hr_attrition_dashboard.png`.

**Dashboard panels include:**
- KPI cards — Total employees, attrition rate, best model AUC
- Attrition split — Pie chart (stayed vs. left)
- Top 10 attrition drivers — Feature importance bar chart
- ROC curves — All three models compared
- Confusion matrix — Best model predictions vs. actuals
- Model comparison — Accuracy and AUC side-by-side

---

## 🔍 Top Attrition Drivers

Based on Random Forest feature importances:

1. **Satisfaction level** — most powerful signal
2. **Average monthly hours** — overwork is a key driver
3. **Time spent at company** — mid-tenure employees most at risk
4. **Number of projects** — too many or too few correlate with leaving
5. **Last evaluation score** — high performers leave when underrewarded

---

## 🚨 Employee Risk Scorer

RetentionIQ includes a built-in scoring function for HR tools:

```python
from retentioniq import predict_attrition_risk

result = predict_attrition_risk({
    'satisfaction_level': 0.3,
    'last_evaluation': 0.85,
    'number_project': 6,
    'avg_monthly_hours': 270,
    'time_spend_company': 5,
    'Work_accident': 0,
    'promotion_last_5years': 0,
    'department': 3,   # encoded
    'salary': 0        # 0=low, 1=medium, 2=high
})

print(result)
# {'attrition_probability': 0.8721, 'risk_level': 'HIGH'}
```

**Risk levels:**
- `HIGH` — probability > 0.60 → immediate HR intervention recommended
- `MEDIUM` — probability 0.30–0.60 → monitor and engage
- `LOW` — probability < 0.30 → low attrition risk

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/retentioniq.git
cd retentioniq
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python retentioniq.py
```

The script will:
- Load and clean the dataset automatically
- Train all three models and print evaluation metrics
- Generate and save the HR dashboard as a PNG
- Output a sample employee risk score

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `matplotlib` | Dashboard and chart generation |
| `seaborn` | Heatmaps and statistical plots |

---

## 📁 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 💡 Use Cases

- **HR departments** — identify flight risks before resignation letters arrive
- **People analytics teams** — understand what's driving department-level attrition
- **Data science portfolio** — demonstrates end-to-end ML on a real-world HR problem
- **Students & learners** — reference project for classification, EDA, and dashboard building

---

## 🗺️ Roadmap

- [ ] Add SHAP values for individual prediction explainability
- [ ] Build an interactive Streamlit HR dashboard
- [ ] Add department-level attrition breakdowns
- [ ] Integrate salary band analysis
- [ ] Export risk scores to CSV for HR system import

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change. Please make sure to update tests as appropriate.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

Built with Python and a belief that people deserve workplaces worth staying in.

---

<p align="center">
  <strong>RetentionIQ</strong> · Predict who's leaving · Understand why · Act before it's too late
</p>
