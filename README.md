# CreditRisk Analyser

> End-to-end credit risk platform: **Probability of Default (PD) modelling → 300–850 credit score → automated loan decision → sensitivity analysis → bulk portfolio scoring**.  
> Deployed as a production-style Streamlit application modelled on real Basel III banking workflows.

---

## Problem Statement

Financial institutions face two interlinked challenges:

1. **Will this borrower default?** — Binary classification on repayment behaviour.  
2. **Should we approve the loan, and on what terms?** — Business decision combining PD score, income stress tests, and portfolio risk policy.

Regulatory frameworks (Basel II/III) require credit decisions to be *transparent and explainable*. This project implements a full ML pipeline that satisfies both requirements.

---

## Key Results

| Metric | Original Model | Enhanced Model (v3) |
|---|---|---|
| AUC | 0.8719 | **0.8805** |
| Default Recall @ threshold 0.40 | 78% | **83%** |
| Missed defaults (FN) | 309 | **247** |
| Features | 24 | **28** (+ 4 engineered) |

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
               ▼
   Streamlit App  (3 tabs)
   · New Application  · Sensitivity Analysis  · Bulk Scoring
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

### Why Logistic Regression?

| Criterion | Logistic Regression | Random Forest / GBM |
|---|---|---|
| Basel III model explainability | ✅ Coefficients transparent | ❌ Black box |
| Probability calibration | ✅ Native | ⚠ Requires post-hoc calibration |
| Feature contribution (SHAP) | ✅ Exact (coef × value) | ⚠ Approximation needed |
| Regulatory acceptability | ✅ Used in IRB models | ❌ |
| Stability over time | ✅ | ⚠ Sensitive to distribution shift |

### Threshold Optimisation (Asymmetric Loss)

Missing a defaulter (FN) costs the full principal. Rejecting a good borrower (FP) costs only foregone interest. This asymmetric loss drives the threshold below 0.5:

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

Confusion matrix:

|  | Predicted: No Default | Predicted: Default |
|---|---|---|
| **Actual: No Default** | 3,944 (TN) | 1,151 (FP) |
| **Actual: Default**    | 247  (FN) | 1,175 (TP) |

---

## Credit Scoring  (PDO=50 Scorecard)

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

A three-section form (Personal Information / Loan Request / Pre-Assessment Snapshot):

- Loan grade and interest rate are **auto-derived** — users never input them directly.
- Pre-assessment snapshot shows real-time DTI, grade, rate, EMI preview, and Rate×Income-Stress.
- On submit: PD gauge, credit score gauge, feature contribution waterfall, EMI table (5 tenures), application summary, downloadable decision report.
- **Download options:** Decision Report (`.txt`), Personal Information CSV, Loan Details CSV.

### Tab 2 — Sensitivity Analysis

Four live what-if charts (income, loan amount, credit history, employment length):

- Grade and rate automatically update as income/loan amount change.
- Approval threshold shown as a dashed reference line on each chart.
- Powered by vectorised batch prediction for fast rendering.

### Tab 3 — Bulk Scoring

- Download CSV template → upload filled CSV → batch-score all applications in one call.
- Results table with PD, credit score, score band, and decision for every row.
- **Portfolio analytics:** PD distribution histogram, decision breakdown donut, average PD by grade, risk concentration by PD band.
- Summary metrics (Approved / Rejected / Review counts).
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
├── app.py                          # Streamlit application (3-tab UI)
├── prediction_helper.py            # Inference engine + feature engineering
│
├── credit_risk.ipynb               # EDA, baseline models, threshold analysis
├── scorecard_model.ipynb           # WOE / IV credit scorecard analysis
│
├── credit_risk_dataset.csv         # Raw dataset (source of truth)
├── credit_risk_cleaned.csv         # Post-cleaning snapshot
├── credit_risk_with_pd.csv         # Dataset with PD scores appended
│
├── logistic_model.pkl              # Production model (v3, with engineered features)
├── scaler.pkl                      # StandardScaler (fit on training split)
├── expected_columns.pkl            # Column order for inference alignment
│
├── .streamlit/config.toml          # Theme configuration
├── .github/copilot-instructions.md # AI coding agent instructions
└── requirements.txt
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch
streamlit run app.py

```

> VS Code users: select the Python 3.9 interpreter (the one with the project packages installed).

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data & feature engineering | pandas, NumPy |
| Modelling | scikit-learn — LogisticRegression, StandardScaler |
| Credit scorecard | WOE/IV (custom), PDO-50 formula |
| Visualisation | Plotly |
| Hyperparameter tuning | Optuna |
| Deployment | Streamlit |
| Model persistence | joblib |

---

*Built to demonstrate end-to-end credit risk modelling — from raw data to a production-grade scoring system with a business decision layer, regulatory-compliant explainability, and an interactive UI.*
