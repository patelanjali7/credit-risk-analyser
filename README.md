# CreditRisk Analyser

> End-to-end credit risk platform: **Probability of Default (PD) modelling → 300–850 credit score → automated loan decision → sensitivity analysis → bulk portfolio scoring**.  
> Deployed as a production-style Streamlit application modelled on real Basel III banking workflows.

---

## Problem Statement

Financial institutions face two interlinked challenges:

1. **Will this borrower default?** — Binary classification on repayment behaviour.  
2. **Should we approve the loan, and on what terms?** — Business decision combining PD score, income stress tests, and portfolio risk policy.

Regulatory frameworks (Basel II/III) require credit decisions to be *transparent and explainable*. This project implements a full ML pipeline that satisfies both requirements — and includes a challenger model benchmarking suite to validate the production model choice.

---

## Key Results

| Metric | Logistic Regression (Production) | XGBoost (Challenger) | LightGBM (Challenger) |
|---|---|---|---|
| AUC | 0.8811 | 0.9503 | **0.9505** |
| Default Recall @ 0.40 | 83% | 83% | **83%** |
| Missed Defaults (FN) | 247 | 232 | **222** |
| Explainability | ✅ Exact coefficients | ⚠ SHAP approximation | ⚠ SHAP approximation |
| Basel III compliant | ✅ Yes | ❌ No | ❌ No |

**Production model: Logistic Regression** — despite lower AUC, it is the only option with exact, auditable feature attribution required under Basel III Internal Ratings-Based approach. LightGBM is tracked in staging via MLflow for future promotion once regulatory guidance on ML models matures.

---

## Live Application

Three-tab Streamlit app:

| Tab | What it does |
|---|---|
| **New Application** | Submit a loan application — get PD, credit score, 4-tier decision, feature contribution waterfall, EMI table, downloadable decision report |
| **Sensitivity Analysis** | What-if charts — see how PD changes as income, loan amount, credit history, or employment shift |
| **Bulk Scoring** | Upload a CSV of applicants, score the whole portfolio in one call, download results + HTML report |

**Themes:** Light / Dark / High-Contrast — all UI elements, charts, and modebar icons adapt.

---

## Solution Architecture

```
credit_risk_dataset.csv  (32,581 records)
        │
        ▼
┌─────────────────────────────────┐
│  Data Cleaning                  │
│  · Outlier capping (age ≤80,    │
│    emp ≤40)                     │
│  · Median imputation +          │
│    missingness indicators       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Feature Engineering            │
│  · loan_percent_income          │  ← original (IV=0.88)
│  · int_rate × loan_pct_income   │  ← NEW  (corr=0.448, strongest)
│  · income_to_loan_ratio         │  ← NEW  (corr=-0.120)
│  · loan_per_cred_hist           │  ← NEW  (corr=0.095)
│  · risk_score_proxy             │  ← NEW  (corr=0.439)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Logistic Regression            │
│  class_weight='balanced'        │
│  C=0.1  |  AUC=0.88             │
│  Threshold tuned to PD=0.40     │
│  (recall-optimised)             │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│  Prediction Helper                               │
│  ├─ Probability of Default                       │
│  ├─ Credit Score  300–850  (PDO=50 scorecard)    │
│  ├─ Approval Decision  (4-tier policy)           │
│  ├─ Loan Terms (rate band + max approved amount) │
│  ├─ EMI scenarios across 5 tenures               │
│  └─ Feature contributions  (coef × scaled val)  │
└──────────────────────────────────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
   Streamlit App   FastAPI Server
   (3 tabs)        /predict  /predict/batch
```

---

## Dataset

**Source:** `credit_risk_dataset.csv` — 32,581 loan records, 12 raw features.  
**Default rate:** 21.8% (class imbalance handled with `class_weight='balanced'`).

| Feature | Type | Notes |
|---|---|---|
| `person_age` | int | Capped at 80 (raw max: 144) |
| `person_income` | int | Annual gross income |
| `person_home_ownership` | cat | RENT / OWN / MORTGAGE / OTHER |
| `person_emp_length` | float | 2.75% missing → imputed + flagged |
| `loan_intent` | cat | 6 categories |
| `loan_grade` | cat | A–G bank risk tier |
| `loan_amnt` | int | Requested amount |
| `loan_int_rate` | float | 9.56% missing → imputed + flagged |
| `loan_status` | int | **Target** — 0=repaid, 1=defaulted |
| `loan_percent_income` | float | Pre-calculated ratio |
| `cb_person_default_on_file` | cat | Prior default Y/N |
| `cb_person_cred_hist_length` | int | Years of credit history |

---

## Feature Engineering

Four new features constructed from domain reasoning and validated by correlation analysis:

| Feature | Formula | Correlation with Default | Rationale |
|---|---|---|---|
| `int_rate_x_loan_pct` | `loan_int_rate × loan_percent_income` | **+0.448** | Combined cost burden — rate × affordability stress |
| `risk_score_proxy` | `loan_int_rate × loan_pct / (income/10K)` | **+0.439** | Synthetic risk index normalised by income scale |
| `income_to_loan_ratio` | `person_income / loan_amnt` | **-0.120** | Inverse of DTI; higher = safer |
| `loan_per_cred_hist` | `loan_amnt / max(cred_hist_yrs, 1)` | **+0.095** | Loan burden relative to credit track record |

> `int_rate_x_loan_pct` alone outperforms `loan_int_rate` (0.319) and `loan_percent_income` (0.379) individually — capturing the non-linear interaction between rate and income stress.

---

## Modelling

### Production Model: Logistic Regression

| Criterion | Logistic Regression | Random Forest / GBM |
|---|---|---|
| Basel III model explainability | ✅ Coefficients transparent | ❌ Black box |
| Probability calibration | ✅ Native | ⚠ Requires post-hoc calibration |
| Feature contribution (SHAP) | ✅ Exact (coef × value) | ⚠ Approximation needed |
| Regulatory acceptability | ✅ Used in IRB models | ❌ |
| Stability over time | ✅ | ⚠ Sensitive to distribution shift |

### Threshold Optimisation (Asymmetric Loss)

Missing a defaulter (FN) costs the full principal. Rejecting a good borrower (FP) costs only foregone interest.

| Threshold | Recall | Precision | Missed Defaults (FN) |
|---|---|---|---|
| 0.50 | 69% | 62% | 453 |
| **0.40** | **83%** | **51%** | **247** |
| 0.35 | 87% | 43% | 185 |

**Selected: 0.40** — catches 206 additional defaulters vs 0.50 at acceptable precision cost.

### Performance (Test set — 6,517 records)

```
              precision    recall    f1-score   support
           0       0.94      0.77      0.85      5,095
           1       0.51      0.83      0.63      1,422
    accuracy                           0.79      6,517
```

|  | Predicted: No Default | Predicted: Default |
|---|---|---|
| **Actual: No Default** | 3,944 (TN) | 1,151 (FP) |
| **Actual: Default**    | 247  (FN) | 1,175 (TP) |

---

## Model Benchmarking

`model_benchmark.ipynb` runs a full challenger comparison against Logistic Regression:

- **XGBoost** — Optuna-tuned (60 trials, TPE sampler, StratifiedKFold CV), `scale_pos_weight` for class imbalance
- **LightGBM** — Optuna-tuned (60 trials), GBDT boosting, leaf-wise growth

All models use the **identical pipeline** (same cleaning, same 5 engineered features, same 80/20 split, same threshold=0.40) to ensure fair comparison.

| Model | AUC | Recall | Precision | FN | Source |
|---|---|---|---|---|---|
| Logistic Regression | 0.8811 | 83% | 51% | 247 | Production |
| XGBoost (tuned) | 0.9503 | 83% | ~56% | 232 | Staging |
| LightGBM (tuned) | **0.9505** | 83% | ~57% | **222** | Staging |

SHAP analysis included: summary plot, bar plot, and waterfall for a high-risk applicant.

**Outputs saved:** `xgb_benchmark_model.pkl`, `lgb_benchmark_model.pkl`, `benchmark_results.csv`, ROC/recall charts.

---

## MLflow Model Registry

`mlflow_register.py` logs all three models to a local MLflow tracking server and registers them with aliases:

| Model Registry Name | Alias | Status |
|---|---|---|
| `credit-risk-lr` | `@production` | Serving via FastAPI |
| `credit-risk-xgb` | `@staging` | Challenger — tracked |
| `credit-risk-lgb` | `@staging` | Best challenger — tracked |

```bash
# Register all models
py -3.11 mlflow_register.py

# Open MLflow UI
py -3.11 -m mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db
```

---

## FastAPI Inference Server

`api.py` provides a production-grade REST API that loads the `@production` model from MLflow registry:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check — model name, version, loaded-at timestamp |
| `/model/info` | GET | Active version, run ID, metrics from MLflow |
| `/predict` | POST | Score a single applicant — returns PD, credit score, decision, DTI |
| `/predict/batch` | POST | Score up to 500 applicants — returns portfolio summary |

```bash
# Start server
py -3.11 -m uvicorn api:app --reload --port 8000

# Interactive docs
# Open http://localhost:8000/docs in your browser
```

The API auto-derives loan grade and interest rate from applicant signals — clients submit 8 fields, not 12.

---

## Credit Scoring (PDO=50 Scorecard)

```
Score = 600 − (50 / ln 2) × ln(PD / (1 − PD))
```

| Probability of Default | Credit Score | Band |
|---|---|---|
| 2% | ~773 | Very Good |
| 10% | ~758 | Very Good |
| 25% | ~700 | Good |
| 40% | ~629 | Fair |
| 55% | ~570 | Fair |
| 70% | ~539 | Poor |
| 90% | ~441 | Poor |

---

## Approval Decision Logic

| PD Range | Decision | Action |
|---|---|---|
| PD < 40% | **APPROVED** | Full amount at standard terms |
| 40% ≤ PD < 55% | **APPROVED WITH REVIEW** | Standard underwriting review before disbursement |
| 55% ≤ PD < 70% | **CONDITIONAL APPROVAL** | Reduced limit + higher rate; senior sign-off required |
| PD ≥ 70% | **REJECTED** | Exceeds maximum portfolio risk tolerance |

---

## Streamlit Application

### Tab 1 — New Application

- Loan grade and interest rate are **auto-derived** — users never input them directly.
- Pre-assessment snapshot shows real-time DTI, grade, rate, EMI preview, and Rate×Income-Stress.
- On submit: PD gauge, credit score gauge, feature contribution waterfall, EMI table (5 tenures), application summary, downloadable decision report.
- **Download options:** Decision Report (`.txt`), Personal Information CSV, Loan Details CSV.

### Tab 2 — Sensitivity Analysis

- Four live what-if charts: income, loan amount, credit history, employment length.
- Grade and rate automatically update as income/loan amount change.
- Approval threshold shown as a dashed reference line on each chart.

### Tab 3 — Bulk Scoring

- Download CSV template → upload filled CSV → batch-score all applications in one call.
- **Portfolio analytics:** PD distribution histogram, decision breakdown donut, average PD by grade, risk concentration by PD band.
- Download scored results as **CSV** or **HTML report** (open in browser → print as PDF).

---

## Key Business Insights

1. **Rate × Income-Stress is the dominant signal** (corr=0.448). Combining the interest rate with the debt-to-income burden captures the non-linear interaction that individual features miss.

2. **Loan grade predicts better than raw financials.** The bank's own pre-screening (grade A–G) embeds proprietary underwriting knowledge not visible in raw application data.

3. **Missingness is a risk signal.** Records with missing `person_emp_length` default at a measurably higher rate — the binary indicator captures this without imputing away the information.

4. **Age is statistically negligible** (corr=−0.02). The model is effectively age-blind, which is beneficial from a fair lending / anti-discrimination standpoint.

5. **Renting is riskier than mortgaging** — not because of income, but because mortgage borrowers have a collateral stake that reduces strategic default incentives.

---

## Project Structure

```
credit_risk/
│
├── app.py                      # Streamlit application (3-tab UI)
├── prediction_helper.py        # Inference engine + feature engineering
├── api.py                      # FastAPI REST server (separate deployment)
│
├── credit_risk.ipynb           # EDA, baseline models, threshold analysis
├── scorecard_model.ipynb       # WOE / IV credit scorecard analysis
├── model_benchmark.ipynb       # XGBoost + LightGBM challenger comparison + SHAP
│
├── mlflow_register.py          # Logs all 3 models to MLflow, sets aliases
│
├── credit_risk_dataset.csv     # Raw dataset (32,581 records)
├── credit_risk_cleaned.csv     # Post-cleaning snapshot
├── credit_risk_with_pd.csv     # Dataset with PD scores appended
│
├── logistic_model.pkl          # Production model (LR v3)
├── scaler.pkl                  # StandardScaler (fit on training split)
├── expected_columns.pkl        # Column order for inference alignment
├── xgb_benchmark_model.pkl     # XGBoost challenger (staging)
├── lgb_benchmark_model.pkl     # LightGBM challenger (staging)
│
├── benchmark_results.csv       # Model comparison metrics
│
├── requirements.txt            # Streamlit inference deps only (Python 3.11)
├── runtime.txt                 # Pins Python 3.11 for Streamlit Cloud
│
└── .streamlit/config.toml      # Theme configuration
```

---

## How to Run

### Streamlit App

```bash
pip install -r requirements.txt
streamlit run app.py
```

### FastAPI Server (local, separate from Streamlit)

```bash
# Install full deps including FastAPI + MLflow
pip install fastapi uvicorn mlflow pydantic

py -3.11 -m uvicorn api:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

### Model Benchmarking + MLflow

```bash
# Install training deps
pip install xgboost lightgbm shap optuna mlflow

# Run benchmark notebook first, then register models
py -3.11 mlflow_register.py

# View MLflow UI
py -3.11 -m mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db
```

---

## Deployment

### Streamlit Cloud

The app is deployed at [share.streamlit.io](https://share.streamlit.io) directly from the `main` branch.

- **Python version:** pinned to 3.11 via `runtime.txt`
- **Dependencies:** `requirements.txt` contains only the 7 inference libraries needed — no training or API deps
- Streamlit Cloud runs `app.py` only; `api.py` is not served there

### FastAPI (separate hosting)

`api.py` requires its own server (Railway, Render, or Fly.io). It is included in this repository to demonstrate production API design and is not part of the Streamlit Cloud deployment.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data & feature engineering | pandas, NumPy |
| Production model | scikit-learn — LogisticRegression, StandardScaler |
| Challenger models | XGBoost, LightGBM |
| Hyperparameter tuning | Optuna (TPE, StratifiedKFold, 60 trials) |
| Model explainability | SHAP (TreeExplainer) |
| Experiment tracking | MLflow (SQLite backend, Model Registry) |
| Credit scorecard | WOE/IV (custom), PDO-50 formula |
| Visualisation | Plotly |
| Streamlit app | Streamlit |
| REST API | FastAPI + Uvicorn |
| Model persistence | joblib |

---

*Built to demonstrate end-to-end credit risk modelling — from raw data to a production-grade scoring system with challenger benchmarking, MLflow tracking, a REST API, and a Basel III-compliant Streamlit UI.*
