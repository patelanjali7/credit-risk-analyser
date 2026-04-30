"""
MLflow Model Registration Script
Logs all 3 benchmark models to MLflow tracking server and registers them
in the Model Registry with appropriate staging tags.

Run once after model_benchmark.ipynb has been executed:
    py -3.9 mlflow_register.py

Then open the MLflow UI:
    py -3.9 -m mlflow ui --port 5000
"""
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ── Constants (must match model_benchmark.ipynb exactly) ─────────────────────
THRESHOLD  = 0.40
SEED       = 42
EXPERIMENT = "credit-risk-benchmarking"

mlflow.set_tracking_uri("sqlite:///mlflow.db")   # local SQLite — no server needed
mlflow.set_experiment(EXPERIMENT)


# ── Rebuild the exact training pipeline ──────────────────────────────────────
def build_dataset():
    df = pd.read_csv("credit_risk_dataset.csv")

    df.loc[df["person_age"] > 80, "person_age"] = 80
    df.loc[df["person_emp_length"] > 40, "person_emp_length"] = 40

    df["emp_length_missing"] = df["person_emp_length"].isna().astype(int)
    df["int_rate_missing"]   = df["loan_int_rate"].isna().astype(int)
    df["person_emp_length"]  = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"]      = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    df["loan_percent_income"]  = df["loan_amnt"] / df["person_income"]
    df["income_to_loan_ratio"] = (df["person_income"] / df["loan_amnt"]).clip(upper=50)
    df["int_rate_x_loan_pct"]  = df["loan_int_rate"] * df["loan_percent_income"]
    df["loan_per_cred_hist"]   = df["loan_amnt"] / (df["cb_person_cred_hist_length"].clip(lower=1) + 1)
    df["risk_score_proxy"]     = (df["loan_int_rate"] * df["loan_percent_income"]
                                   / (df["person_income"] / 1e4).clip(lower=0.01))

    expected_columns = joblib.load("expected_columns.pkl")
    df_enc = pd.get_dummies(
        df.drop(columns=["loan_status"]),
        columns=["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    ).reindex(columns=expected_columns, fill_value=0)

    X, y = df_enc, df["loan_status"]
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def metrics_at_threshold(y_true, y_prob, threshold=THRESHOLD):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc":       round(roc_auc_score(y_true, y_prob), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "fn_count":  int(fn),
        "fp_count":  int(fp),
        "threshold": threshold,
    }


# ── Build dataset ─────────────────────────────────────────────────────────────
print("Building dataset...")
X_train, X_test, y_train, y_test = build_dataset()

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos


# ── Run 1: Logistic Regression (Production) ──────────────────────────────────
print("\n[1/3] Logging Logistic Regression...")
with mlflow.start_run(run_name="logistic-regression-v3"):
    mlflow.set_tag("model_type", "LogisticRegression")
    mlflow.set_tag("status", "production")

    params = {"C": 0.1, "max_iter": 2000, "class_weight": "balanced",
              "solver": "lbfgs", "random_state": SEED}
    mlflow.log_params(params)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    lr = LogisticRegression(**params)
    lr.fit(X_tr_s, y_train)

    probs   = lr.predict_proba(X_te_s)[:, 1]
    metrics = metrics_at_threshold(y_test, probs)
    mlflow.log_metrics(metrics)

    # Log the model + scaler together as artifacts
    mlflow.sklearn.log_model(lr, artifact_path="model",
                             registered_model_name="credit-risk-lr")
    mlflow.sklearn.log_model(scaler, artifact_path="scaler")

    lr_run_id = mlflow.active_run().info.run_id
    print(f"   AUC={metrics['auc']}  Recall={metrics['recall']}  FN={metrics['fn_count']}")
    print(f"   Run ID: {lr_run_id}")


# ── Run 2: XGBoost (Staging) ─────────────────────────────────────────────────
print("\n[2/3] Logging XGBoost...")
xgb_model = joblib.load("xgb_benchmark_model.pkl")

with mlflow.start_run(run_name="xgboost-tuned"):
    mlflow.set_tag("model_type", "XGBoost")
    mlflow.set_tag("status", "staging")

    xgb_params = xgb_model.get_params()
    mlflow.log_params({k: v for k, v in xgb_params.items()
                       if k in ["n_estimators", "max_depth", "learning_rate",
                                "subsample", "colsample_bytree", "scale_pos_weight"]})

    probs   = xgb_model.predict_proba(X_test)[:, 1]
    metrics = metrics_at_threshold(y_test, probs)
    mlflow.log_metrics(metrics)

    mlflow.xgboost.log_model(xgb_model, artifact_path="model",
                             registered_model_name="credit-risk-xgb")

    xgb_run_id = mlflow.active_run().info.run_id
    print(f"   AUC={metrics['auc']}  Recall={metrics['recall']}  FN={metrics['fn_count']}")
    print(f"   Run ID: {xgb_run_id}")


# ── Run 3: LightGBM (Staging) ────────────────────────────────────────────────
print("\n[3/3] Logging LightGBM...")
lgb_model = joblib.load("lgb_benchmark_model.pkl")

with mlflow.start_run(run_name="lightgbm-tuned"):
    mlflow.set_tag("model_type", "LightGBM")
    mlflow.set_tag("status", "staging")

    lgb_params = lgb_model.get_params()
    mlflow.log_params({k: v for k, v in lgb_params.items()
                       if k in ["n_estimators", "max_depth", "learning_rate",
                                "num_leaves", "subsample", "colsample_bytree"]})

    probs   = lgb_model.predict_proba(X_test)[:, 1]
    metrics = metrics_at_threshold(y_test, probs)
    mlflow.log_metrics(metrics)

    mlflow.lightgbm.log_model(lgb_model, artifact_path="model",
                              registered_model_name="credit-risk-lgb")

    lgb_run_id = mlflow.active_run().info.run_id
    print(f"   AUC={metrics['auc']}  Recall={metrics['recall']}  FN={metrics['fn_count']}")
    print(f"   Run ID: {lgb_run_id}")


# ── Promote LR to Production in registry ─────────────────────────────────────
print("\nPromoting Logistic Regression to Production in Model Registry...")
client = mlflow.MlflowClient()

for model_name, alias in [
    ("credit-risk-lr",  "production"),
    ("credit-risk-xgb", "staging"),
    ("credit-risk-lgb", "staging"),
]:
    versions = client.search_model_versions(f"name='{model_name}'")
    if versions:
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        client.set_registered_model_alias(model_name, alias, latest.version)
        print(f"   {model_name} v{latest.version} -> @{alias}")

print("\nDone. All 3 models registered.")
print("Open MLflow UI:  py -3.9 -m mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db")
