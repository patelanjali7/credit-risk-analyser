"""
CreditRisk Analyser — FastAPI Inference Server
Loads the production model from MLflow Model Registry and serves predictions.

Start:
    py -3.9 -m uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health          liveness check
    GET  /model/info      active model version + metrics from MLflow
    POST /predict         score a single applicant
    POST /predict/batch   score multiple applicants in one call
"""
from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from prediction_helper import (
    GRADE_RATE_MAP,
    THRESHOLD,
    calculate_credit_score,
    derive_loan_grade,
    get_score_band,
    prepare_features,
    predict_batch_fast,
    scaler,
    expected_columns,
    log_model,          # production LR model loaded at module level
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI    = "sqlite:///mlflow.db"
MODEL_NAME    = "credit-risk-lr"
MODEL_ALIAS   = "production"


# ── MLflow model loader ───────────────────────────────────────────────────────

def _load_mlflow_model():
    """
    Load model from MLflow registry (@production alias).
    Falls back to the local pkl already loaded by prediction_helper if MLflow
    is unavailable — so the API still works during development.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.sklearn.load_model(uri)
        logger.info("Loaded model from MLflow registry: %s@%s", MODEL_NAME, MODEL_ALIAS)
        return model, _get_model_info()
    except Exception as exc:
        logger.warning("MLflow load failed (%s) — using local pkl fallback", exc)
        return log_model, {"source": "local_pkl", "model_name": MODEL_NAME, "alias": MODEL_ALIAS}


def _get_model_info() -> dict:
    """Fetch version metadata and metrics for the production model."""
    try:
        client = mlflow.MlflowClient(tracking_uri=MLFLOW_URI)
        mv     = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        run    = client.get_run(mv.run_id)
        return {
            "model_name":  MODEL_NAME,
            "alias":       MODEL_ALIAS,
            "version":     mv.version,
            "run_id":      mv.run_id,
            "registered":  mv.creation_timestamp,
            "metrics":     run.data.metrics,
            "params":      run.data.params,
            "source":      "mlflow_registry",
        }
    except Exception as exc:
        logger.warning("Could not fetch model info: %s", exc)
        return {"model_name": MODEL_NAME, "alias": MODEL_ALIAS, "source": "unknown"}


# ── App state ─────────────────────────────────────────────────────────────────

class _State:
    model      = None
    model_info = {}
    loaded_at  = None

state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, release on shutdown."""
    logger.info("Loading production model from MLflow...")
    state.model, state.model_info = _load_mlflow_model()
    state.loaded_at = datetime.utcnow().isoformat()
    logger.info("Model ready. API serving at /predict")
    yield
    logger.info("API shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="CreditRisk Analyser API",
    description=(
        "Basel III-compliant credit risk scoring API. "
        "Returns Probability of Default, credit score, and 4-tier loan decision."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ApplicantInput(BaseModel):
    """Raw applicant data. Loan grade and interest rate are auto-derived."""
    person_age:                 int   = Field(..., ge=18, le=100,      description="Applicant age in years")
    person_income:              float = Field(..., gt=0,               description="Annual gross income (USD)")
    person_home_ownership:      str   = Field(...,                     description="RENT | OWN | MORTGAGE | OTHER")
    person_emp_length:          float = Field(..., ge=0, le=40,        description="Employment length in years")
    loan_intent:                str   = Field(...,                     description="EDUCATION | HOMEIMPROVEMENT | MEDICAL | PERSONAL | VENTURE | DEBTCONSOLIDATION")
    loan_amnt:                  float = Field(..., gt=0,               description="Requested loan amount (USD)")
    cb_person_default_on_file:  str   = Field(...,                     description="Prior default on record: Y | N")
    cb_person_cred_hist_length: int   = Field(..., ge=0, le=50,        description="Years of credit history")

    @field_validator("person_home_ownership")
    @classmethod
    def validate_ownership(cls, v):
        allowed = {"RENT", "OWN", "MORTGAGE", "OTHER"}
        if v.upper() not in allowed:
            raise ValueError(f"Must be one of {allowed}")
        return v.upper()

    @field_validator("loan_intent")
    @classmethod
    def validate_intent(cls, v):
        allowed = {"EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION"}
        if v.upper() not in allowed:
            raise ValueError(f"Must be one of {allowed}")
        return v.upper()

    @field_validator("cb_person_default_on_file")
    @classmethod
    def validate_default(cls, v):
        if v.upper() not in {"Y", "N"}:
            raise ValueError("Must be Y or N")
        return v.upper()


class PredictionResponse(BaseModel):
    """Structured prediction result."""
    application_id:     str
    default_probability: float
    credit_score:        int
    score_band:          str
    risk_band:           str
    decision:            str
    decision_reason:     str
    loan_grade:          str
    indicative_rate_pct: float
    dti:                 float
    model_version:       str
    threshold_used:      float


class BatchRequest(BaseModel):
    applications: List[ApplicantInput] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    total:     int
    approved:  int
    review:    int
    rejected:  int
    results:   List[PredictionResponse]


# ── Shared prediction logic ───────────────────────────────────────────────────

def _build_feature_dict(inp: ApplicantInput) -> dict:
    """Derive grade + rate, then assemble the dict prediction_helper expects."""
    grade    = derive_loan_grade(
        inp.person_income, inp.loan_amnt,
        inp.cb_person_default_on_file,
        inp.cb_person_cred_hist_length,
        inp.person_emp_length,
    )
    int_rate = GRADE_RATE_MAP[grade]
    return {
        "person_age":                 inp.person_age,
        "person_income":              inp.person_income,
        "person_home_ownership":      inp.person_home_ownership,
        "person_emp_length":          inp.person_emp_length,
        "loan_intent":                inp.loan_intent,
        "loan_grade":                 grade,
        "loan_amnt":                  inp.loan_amnt,
        "loan_int_rate":              int_rate,
        "cb_person_default_on_file":  inp.cb_person_default_on_file,
        "cb_person_cred_hist_length": inp.cb_person_cred_hist_length,
        "emp_length_missing":         0,
        "int_rate_missing":           0,
        "_grade":                     grade,
        "_int_rate":                  int_rate,
    }


def _make_decision(pd_val: float):
    if pd_val >= 0.70:
        return "REJECTED",            "High Risk",        "Default probability >= 70% — exceeds maximum portfolio risk tolerance."
    if pd_val >= 0.55:
        return "CONDITIONAL APPROVAL","Medium-High Risk",  "Elevated risk — approved at reduced limit; senior underwriter sign-off required."
    if pd_val >= THRESHOLD:
        return "APPROVED WITH REVIEW", "Medium Risk",      "Moderate risk — standard underwriting review required before disbursement."
    return     "APPROVED",             "Low Risk",         "Low default probability — eligible for full requested amount at standard terms."


def _score_one(inp: ApplicantInput, app_id: str, model) -> PredictionResponse:
    fd       = _build_feature_dict(inp)
    features = prepare_features(fd)
    scaled   = scaler.transform(features)
    pd_val   = float(model.predict_proba(scaled)[0][1])

    score              = calculate_credit_score(pd_val)
    score_band, _      = get_score_band(score)
    decision, risk_band, reason = _make_decision(pd_val)
    dti                = inp.loan_amnt / max(inp.person_income, 1)

    mv = state.model_info.get("version", "local")

    return PredictionResponse(
        application_id      = app_id,
        default_probability = round(pd_val, 4),
        credit_score        = score,
        score_band          = score_band,
        risk_band           = risk_band,
        decision            = decision,
        decision_reason     = reason,
        loan_grade          = fd["_grade"],
        indicative_rate_pct = fd["_int_rate"],
        dti                 = round(dti, 4),
        model_version       = f"{MODEL_NAME}@{MODEL_ALIAS} v{mv}",
        threshold_used      = THRESHOLD,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness check — returns 200 if the API is running."""
    return {
        "status":     "ok",
        "loaded_at":  state.loaded_at,
        "model":      state.model_info.get("model_name"),
        "alias":      state.model_info.get("alias"),
        "version":    state.model_info.get("version", "local"),
        "timestamp":  datetime.utcnow().isoformat(),
    }


@app.get("/model/info", tags=["System"])
def model_info():
    """Returns the active model version, MLflow run ID, and evaluation metrics."""
    if not state.model_info:
        raise HTTPException(status_code=503, detail="Model metadata unavailable")
    return state.model_info


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict(inp: ApplicantInput):
    """
    Score a single loan applicant.

    - Loan grade and interest rate are auto-derived from applicant signals.
    - Returns PD, credit score, 4-tier decision, and DTI.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        import uuid
        app_id = f"API-{uuid.uuid4().hex[:8].upper()}"
        return _score_one(inp, app_id, state.model)
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Scoring"])
def predict_batch(req: BatchRequest):
    """
    Score up to 500 applicants in a single call.
    Uses vectorised inference (one scaler.transform + one predict_proba call).
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        import uuid
        results = []
        for i, inp in enumerate(req.applications):
            app_id = f"API-BATCH-{uuid.uuid4().hex[:6].upper()}"
            results.append(_score_one(inp, app_id, state.model))

        approved = sum(1 for r in results if r.decision == "APPROVED")
        review   = sum(1 for r in results if r.decision in ("APPROVED WITH REVIEW", "CONDITIONAL APPROVAL"))
        rejected = sum(1 for r in results if r.decision == "REJECTED")

        return BatchResponse(
            total    = len(results),
            approved = approved,
            review   = review,
            rejected = rejected,
            results  = results,
        )
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
