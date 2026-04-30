# CreditRisk Analyser — AI Coding Agent Instructions

## Project Overview
End-to-end credit risk platform predicting Probability of Default (PD), producing a 300–850 credit
score, and delivering an automated 4-tier loan decision. Deployed as a production-style Streamlit
application modelled on real Basel III banking workflows.

**Dataset**: `credit_risk_dataset.csv` — 32,581 loan records, 12 raw features, 21.8% default rate.

## Architecture

### Core Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit 3-tab UI — New Application / Sensitivity Analysis / Bulk Scoring |
| `prediction_helper.py` | ML inference engine — feature engineering, grade derivation, scoring |
| `logistic_model.pkl` | Production model (v3): 28 features, AUC 0.8805, Recall 83% |
| `scaler.pkl` | StandardScaler fit on training split |
| `expected_columns.pkl` | Column order for inference alignment |
| `credit_risk_dataset.csv` | Source of truth — do not modify |

### Model
- **Algorithm**: Logistic Regression (`class_weight='balanced'`, `C=0.1`, `max_iter=2000`, solver=lbfgs)
- **Decision threshold**: PD ≥ 0.40 (asymmetric loss — FN costs full principal, more expensive than FP)
- **Credit score**: PDO=50 scorecard formula `Score = 600 − (50/ln2) × ln(PD/(1−PD))`, clamped 300–850
- **4 engineered features**: `int_rate_x_loan_pct`, `risk_score_proxy`, `income_to_loan_ratio`, `loan_per_cred_hist`

### Inference Flow
1. `derive_loan_grade()` — heuristic point system maps applicant signals → A–G grade
2. `GRADE_RATE_MAP` maps grade → indicative interest rate (never user-entered)
3. `prepare_features()` — **MUST mirror training pipeline exactly**; any deviation = silent wrong predictions
4. `predict_risk()` — returns PD, credit score, approval decision, loan terms, feature contributions

### Critical Constraints
- `prepare_features()` must exactly match training; changes require model retraining
- `loan_per_cred_hist` clips credit history to `lower=1` — prevents extreme values for new applicants
- `predict_batch_fast()` is vectorised (single transform+predict call) — do not refactor to a loop
- Model uses `class_weight='balanced'` — threshold is 0.40, not 0.50
- Approval tiers: APPROVED (<40%) | APPROVED WITH REVIEW (40–55%) | CONDITIONAL (55–70%) | REJECTED (≥70%)

## Development Setup
```bash
pip install -r requirements.txt
streamlit run app.py        # http://localhost:8501
```
Use **Python 3.9** interpreter (packages installed there, not system Python).

## Notebooks
- `credit_risk.ipynb` — EDA, baseline models, threshold analysis
- `scorecard_model.ipynb` — WOE/IV credit scorecard analysis
