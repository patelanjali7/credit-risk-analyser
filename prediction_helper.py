"""
Credit Risk Prediction Engine
Senior-grade feature engineering + logistic regression inference pipeline.
Model: Logistic Regression  |  AUC: 0.88  |  Recall: 83%  |  Threshold: 0.40
"""
import math
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ── Artifacts ────────────────────────────────────────────────────────────────
log_model        = joblib.load("logistic_model.pkl")
scaler           = joblib.load("scaler.pkl")
expected_columns = joblib.load("expected_columns.pkl")

THRESHOLD = 0.40  # optimises recall; asymmetric loss — FN (missed default) >> FP cost

# ── Grade → indicative mid-market rate (%p.a.) ───────────────────────────────
GRADE_RATE_MAP = {
    "A": 6.5, "B": 9.5, "C": 13.0,
    "D": 16.5, "E": 19.5, "F": 22.0, "G": 24.0,
}

# ── Human-readable labels for feature contributions ──────────────────────────
FEATURE_LABELS = {
    "loan_percent_income":          "Loan-to-Income Ratio",
    "int_rate_x_loan_pct":          "Rate × Income-Stress Index",
    "risk_score_proxy":             "Composite Risk Index",
    "person_income":                "Annual Income",
    "loan_amnt":                    "Loan Amount",
    "loan_int_rate":                "Interest Rate (%)",
    "income_to_loan_ratio":         "Income-to-Loan Ratio",
    "loan_per_cred_hist":           "Loan per Credit History Year",
    "person_age":                   "Applicant Age",
    "person_emp_length":            "Employment Length (yrs)",
    "cb_person_cred_hist_length":   "Credit History Length (yrs)",
    "emp_length_missing":           "Employment Data Missing",
    "int_rate_missing":             "Interest Rate Data Missing",
    "loan_grade_B":                 "Loan Grade: B",
    "loan_grade_C":                 "Loan Grade: C",
    "loan_grade_D":                 "Loan Grade: D",
    "loan_grade_E":                 "Loan Grade: E",
    "loan_grade_F":                 "Loan Grade: F",
    "loan_grade_G":                 "Loan Grade: G",
    "person_home_ownership_OWN":    "Owns Property (No Mortgage)",
    "person_home_ownership_RENT":   "Currently Renting",
    "person_home_ownership_OTHER":  "Other Ownership Type",
    "loan_intent_EDUCATION":        "Purpose: Education",
    "loan_intent_HOMEIMPROVEMENT":  "Purpose: Home Improvement",
    "loan_intent_MEDICAL":          "Purpose: Medical",
    "loan_intent_PERSONAL":         "Purpose: Personal",
    "loan_intent_VENTURE":          "Purpose: Business Venture",
    "cb_person_default_on_file_Y":  "Prior Default on Record",
}


# ── Internal grade derivation ─────────────────────────────────────────────────

def derive_loan_grade(income: float, loan_amnt: float, prior_default: str,
                       cred_hist_yrs: int, emp_length: float) -> str:
    """
    Bank-style preliminary grade assignment from raw applicant signals.
    Mirrors how underwriters score before formal PD modelling.
    Output: A (best) → G (worst).
    """
    pts = 0
    dti = loan_amnt / max(income, 1)
    if   dti < 0.10: pts += 4
    elif dti < 0.20: pts += 3
    elif dti < 0.30: pts += 2
    elif dti < 0.40: pts += 1
    elif dti < 0.50: pts -= 1
    else:            pts -= 3

    if   cred_hist_yrs >= 15: pts += 3
    elif cred_hist_yrs >= 10: pts += 2
    elif cred_hist_yrs >= 5:  pts += 1
    elif cred_hist_yrs < 3:   pts -= 1

    if   emp_length >= 10: pts += 3
    elif emp_length >= 5:  pts += 2
    elif emp_length >= 2:  pts += 1
    elif emp_length < 1:   pts -= 1

    if   income >= 100_000: pts += 3
    elif income >= 60_000:  pts += 2
    elif income >= 35_000:  pts += 1
    elif income < 20_000:   pts -= 2

    if prior_default == "Y": pts -= 5

    for cutoff, grade in [(11,"A"),(8,"B"),(5,"C"),(2,"D"),(-1,"E"),(-4,"F")]:
        if pts >= cutoff:
            return grade
    return "G"


# ── Credit Score (PDO=50 scorecard) ──────────────────────────────────────────

def calculate_credit_score(prob_default: float) -> int:
    """
    Standard bank scorecard formula.
    PD=0.5 → 600; PD=0.10 → ~758; PD=0.70 → ~539. Clamped 300–850.
    """
    pd_val = max(0.001, min(0.999, prob_default))
    score  = 600 - (50 / math.log(2)) * math.log(pd_val / (1 - pd_val))
    return max(300, min(850, round(score)))


def get_score_band(score: int) -> tuple:
    if score >= 800: return "Exceptional", "#15803D"
    if score >= 740: return "Very Good",   "#16A34A"
    if score >= 670: return "Good",        "#65A30D"
    if score >= 580: return "Fair",        "#D97706"
    return "Poor", "#DC2626"


# ── EMI ───────────────────────────────────────────────────────────────────────

def calculate_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Reducing-balance EMI (standard amortisation formula)."""
    if annual_rate_pct == 0 or tenure_months == 0:
        return round(principal / max(tenure_months, 1), 2)
    r   = annual_rate_pct / (12 * 100)
    emi = principal * r * (1 + r)**tenure_months / ((1 + r)**tenure_months - 1)
    return round(emi, 2)


# ── Loan terms suggestion ─────────────────────────────────────────────────────

def suggest_loan_terms(credit_score: int, income: float, requested: float) -> dict:
    tiers = [
        (750, (6.0,  8.5),  5.0),
        (700, (8.5,  12.0), 4.0),
        (650, (12.0, 16.0), 3.0),
        (600, (16.0, 20.0), 2.0),
        (0,   (20.0, 24.0), 1.5),
    ]
    for cutoff, rate_range, mult in tiers:
        if credit_score >= cutoff:
            break
    mid_rate = round(sum(rate_range) / 2, 1)
    approved = round(min(requested, income * mult * 0.30))
    return {"rate_range": rate_range, "suggested_rate": mid_rate, "max_approved_amount": approved}


# ── Feature engineering (must mirror training pipeline exactly) ───────────────

def prepare_features(inp: dict) -> pd.DataFrame:
    """
    Apply the same feature engineering as the training pipeline.
    New engineered features: income_to_loan_ratio, int_rate_x_loan_pct,
    loan_per_cred_hist, risk_score_proxy.
    """
    df = pd.DataFrame([inp])

    # Original ratio
    df["loan_percent_income"]  = df["loan_amnt"] / df["person_income"]

    # Engineered features (trained on these, must be present at inference)
    df["income_to_loan_ratio"] = (df["person_income"] / df["loan_amnt"]).clip(upper=50)
    df["int_rate_x_loan_pct"]  = df["loan_int_rate"] * df["loan_percent_income"]
    # clip lower=1 so cred_hist=0 doesn't produce an extreme out-of-distribution value
    df["loan_per_cred_hist"]   = df["loan_amnt"] / (df["cb_person_cred_hist_length"].clip(lower=1) + 1)
    df["risk_score_proxy"]     = (df["loan_int_rate"] * df["loan_percent_income"]
                                   / (df["person_income"] / 1e4).clip(lower=0.01))

    df = pd.get_dummies(df)
    return df.reindex(columns=expected_columns, fill_value=0)


# ── Feature contributions (SHAP-equivalent for linear models) ────────────────

def compute_feature_contributions(scaled_arr: np.ndarray, col_names: list) -> dict:
    coefs = log_model.coef_[0]
    row   = scaled_arr[0]
    raw   = {
        FEATURE_LABELS.get(n, n.replace("_", " ").title()): round(float(c * v), 4)
        for c, v, n in zip(coefs, row, col_names)
    }
    top10 = sorted(raw.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    return dict(top10)


# ── Vectorised batch prediction ───────────────────────────────────────────────

def predict_batch_fast(input_dicts: list) -> list:
    """Single transform+predict call for bulk scoring."""
    rows = pd.concat([prepare_features(d) for d in input_dicts], ignore_index=True)
    return log_model.predict_proba(scaler.transform(rows))[:, 1].tolist()


# ── Main entry point ─────────────────────────────────────────────────────────

def predict_risk(inp: dict) -> dict:
    df     = prepare_features(inp)
    scaled = scaler.transform(df)

    prob_default = float(log_model.predict_proba(scaled)[0][1])
    credit_score = calculate_credit_score(prob_default)
    score_band, band_colour = get_score_band(credit_score)

    if prob_default >= 0.70:
        risk_band, approval, reason = (
            "High Risk", "REJECTED",
            "Default probability ≥ 70% — exceeds maximum portfolio risk tolerance."
        )
    elif prob_default >= 0.55:
        risk_band, approval, reason = (
            "Medium-High Risk", "CONDITIONAL APPROVAL",
            "Elevated risk — approved at reduced limit with higher rate; senior underwriter sign-off required."
        )
    elif prob_default >= THRESHOLD:
        risk_band, approval, reason = (
            "Medium Risk", "APPROVED WITH REVIEW",
            "Moderate risk — standard underwriting review required before disbursement."
        )
    else:
        risk_band, approval, reason = (
            "Low Risk", "APPROVED",
            "Low default probability — eligible for full requested amount at standard terms."
        )

    terms         = suggest_loan_terms(credit_score, inp["person_income"], inp["loan_amnt"])
    contributions = compute_feature_contributions(scaled, list(expected_columns))

    return {
        "default_probability":  round(prob_default, 4),
        "prediction":           int(prob_default >= THRESHOLD),
        "risk_band":            risk_band,
        "credit_score":         credit_score,
        "score_band":           score_band,
        "band_colour":          band_colour,
        "approval":             approval,
        "approval_reason":      reason,
        "loan_terms":           terms,
        "feature_contributions": contributions,
    }


# ── Downloadable decision record ─────────────────────────────────────────────

def generate_decision_report(result: dict, fd: dict, app_id: str) -> str:
    ts    = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    terms = result["loan_terms"]
    dti   = fd["loan_amnt"] / max(fd["person_income"], 1)
    lines = [
        "=" * 66,
        "   CREDIT RISK ASSESSMENT — DECISION RECORD",
        "=" * 66,
        f"   Application ID  : {app_id}",
        f"   Assessment Date : {ts}",
        f"   Model           : Logistic Regression (class_weight='balanced')",
        f"   AUC             : 0.8805   |   Threshold : PD ≥ {THRESHOLD:.0%}",
        "=" * 66,
        "",
        f"   DECISION : {result['approval']}",
        f"   Reason   : {result['approval_reason']}",
        "",
        "─" * 66,
        "   RISK METRICS",
        "─" * 66,
        f"   Probability of Default  : {result['default_probability']:.2%}",
        f"   Credit Score            : {result['credit_score']}  ({result['score_band']})",
        f"   Risk Classification     : {result['risk_band']}",
        f"   Debt-to-Income Ratio    : {dti:.1%}",
        f"   Internal Loan Grade     : {fd.get('loan_grade', 'N/A')}",
        "",
        "─" * 66,
        "   APPLICANT PROFILE",
        "─" * 66,
        f"   Age               : {fd['person_age']} years",
        f"   Annual Income     : ${fd['person_income']:,.0f}",
        f"   Employment        : {fd['person_emp_length']} years",
        f"   Home Ownership    : {fd.get('ownership_display', fd.get('person_home_ownership', ''))}",
        f"   Credit History    : {fd['cb_person_cred_hist_length']} years",
        f"   Prior Default     : {fd.get('default_display', fd.get('cb_person_default_on_file', ''))}",
        "",
        "─" * 66,
        "   LOAN REQUEST",
        "─" * 66,
        f"   Requested Amount  : ${fd['loan_amnt']:,.0f}",
        f"   Purpose           : {fd.get('intent_display', fd.get('loan_intent', ''))}",
        f"   Indicative Rate   : {fd.get('loan_int_rate', 'N/A')}% p.a.",
        "",
        "─" * 66,
        "   RECOMMENDED TERMS",
        "─" * 66,
        f"   Rate Band         : {terms['rate_range'][0]}% – {terms['rate_range'][1]}%",
        f"   Max Approved Amt  : ${terms['max_approved_amount']:,.0f}",
        "",
        "─" * 66,
        "   TOP RISK FACTORS  (+ve = increases default risk)",
        "─" * 66,
    ]
    for feat, contrib in result["feature_contributions"].items():
        arrow = "↑" if contrib > 0 else "↓"
        lines.append(f"   {feat:<42} {contrib:+.4f}  {arrow}")
    lines += [
        "",
        "=" * 66,
        "   Automated AI assessment. Subject to final human review.",
        "   For internal / authorised use only.",
        "=" * 66,
    ]
    return "\n".join(lines)
