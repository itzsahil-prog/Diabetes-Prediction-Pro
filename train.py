"""
Training pipeline for the Diabetes Prediction Pro project.

Improvements vs. a typical beginner "diabetes prediction" repo:
  1. Proper missing-value handling (0s in clinical columns -> NaN -> imputed),
     instead of training directly on biologically impossible zeros.
  2. A real train/test split evaluated with stratification + 5-fold CV,
     not just a single train_test_split accuracy number.
  3. Multiple models compared head-to-head (LogReg, RandomForest,
     GradientBoosting, XGBoost, SVM) using ROC-AUC, F1, precision, recall,
     not just accuracy (accuracy is misleading on this ~65/35 imbalanced set).
  4. Hyperparameter tuning via RandomizedSearchCV for the top models.
  5. A calibrated probability output (CalibratedClassifierCV) so the
     predicted "risk %" is actually meaningful, not just a raw score.
  6. SHAP explainability artifacts saved so the web app can show *why*
     a given prediction was made, per patient.
  7. Everything reproducible end-to-end from raw CSV -> deployed model.
"""

import json
import warnings
import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

from prepare_data import load_raw, clean, COLUMNS

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

FEATURES = [c for c in COLUMNS if c != "Outcome"]


def build_pipeline(model):
    """Impute missing clinical values (KNN-based, better than mean/median)
    then scale, then feed the classifier. Wrapped in a Pipeline so the
    imputer/scaler are fit ONLY on training folds -- avoids data leakage
    that plagues a lot of the beginner notebooks (which impute on the
    whole dataset before splitting)."""
    return Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


MODEL_ZOO = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        eval_metric="logloss", random_state=RANDOM_STATE, use_label_encoder=False
    ),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
}

# Randomized search spaces for the top contenders (tuned after CV screening)
PARAM_SPACES = {
    "RandomForest": {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [3, 4, 5, 6, 8, None],
        "model__min_samples_split": [2, 4, 6, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    },
    "GradientBoosting": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4, 5],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
    },
    "XGBoost": {
        "model__n_estimators": [150, 250, 350, 500],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_lambda": [0.5, 1, 2, 5],
    },
    "LogisticRegression": {
        "model__C": [0.01, 0.05, 0.1, 0.5, 1, 3, 10],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
    },
    "SVM": {
        "model__C": [0.1, 0.5, 1, 3, 10],
        "model__gamma": ["scale", "auto", 0.01, 0.1],
        "model__kernel": ["rbf"],
    },
}

SCORING = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
}


def screen_models(X_train, y_train):
    """Stage 1: 5-fold stratified CV screen across the whole model zoo with
    default hyperparameters to see who's worth tuning."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in MODEL_ZOO.items():
        pipe = build_pipeline(model)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        results[name] = {k: float(np.mean(v)) for k, v in scores.items() if k.startswith("test_")}
        print(f"  {name:20s} ROC-AUC={results[name]['test_roc_auc']:.4f}  "
              f"F1={results[name]['test_f1']:.4f}  Recall={results[name]['test_recall']:.4f}")
    return results


def tune_model(name, X_train, y_train, n_iter=40):
    """Stage 2: RandomizedSearchCV hyperparameter tuning, optimizing ROC-AUC
    (better than accuracy for an imbalanced medical screening task where
    missing a true positive is costlier than a false alarm)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = build_pipeline(MODEL_ZOO[name])
    search = RandomizedSearchCV(
        pipe, PARAM_SPACES[name], n_iter=n_iter, scoring="roc_auc",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_on_test(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }


def main():
    print("=" * 70)
    print("STAGE 0: Load + clean data")
    print("=" * 70)
    df = clean(load_raw())
    X = df[FEATURES]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\n" + "=" * 70)
    print("STAGE 1: Screen model zoo (5-fold CV, default hyperparameters)")
    print("=" * 70)
    screen_results = screen_models(X_train, y_train)

    # Tune the top 3 by CV ROC-AUC
    ranked = sorted(screen_results.items(), key=lambda kv: kv[1]["test_roc_auc"], reverse=True)
    top_models = [name for name, _ in ranked[:3]]
    print(f"\nTop 3 models selected for tuning: {top_models}")

    print("\n" + "=" * 70)
    print("STAGE 2: Hyperparameter tuning (RandomizedSearchCV, optimizing ROC-AUC)")
    print("=" * 70)
    tuned = {}
    for name in top_models:
        print(f"\nTuning {name}...")
        best_pipe, best_params, best_cv_score = tune_model(name, X_train, y_train)
        tuned[name] = {"pipe": best_pipe, "params": best_params, "cv_roc_auc": best_cv_score}
        print(f"  Best CV ROC-AUC: {best_cv_score:.4f}")
        print(f"  Best params: {best_params}")

    print("\n" + "=" * 70)
    print("STAGE 3: Final evaluation on held-out test set")
    print("=" * 70)
    final_results = {}
    for name, info in tuned.items():
        metrics = evaluate_on_test(info["pipe"], X_test, y_test)
        final_results[name] = metrics
        print(f"  {name:20s} Test ROC-AUC={metrics['roc_auc']:.4f}  "
              f"F1={metrics['f1']:.4f}  Recall={metrics['recall']:.4f}  "
              f"Precision={metrics['precision']:.4f}")

    best_name = max(final_results, key=lambda n: final_results[n]["roc_auc"])
    print(f"\n>>> Best model on test set: {best_name} "
          f"(ROC-AUC={final_results[best_name]['roc_auc']:.4f})")

    print("\n" + "=" * 70)
    print("STAGE 4: Probability calibration on the winning model")
    print("=" * 70)
    # Calibrate so predict_proba outputs are trustworthy risk percentages,
    # fit on the training data via internal CV (no test leakage).
    calibrated = CalibratedClassifierCV(tuned[best_name]["pipe"], method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    calibrated_metrics = evaluate_on_test(calibrated, X_test, y_test)
    print(f"  Calibrated model test ROC-AUC={calibrated_metrics['roc_auc']:.4f}  "
          f"F1={calibrated_metrics['f1']:.4f}")

    print("\n" + "=" * 70)
    print("STAGE 5: Save model + SHAP explainer + metadata")
    print("=" * 70)
    joblib.dump(calibrated, "models/final_model.joblib")
    # Also keep the uncalibrated best pipeline around -- it's what SHAP
    # will explain (TreeExplainer needs the raw tree model, not the
    # calibration wrapper).
    joblib.dump(tuned[best_name]["pipe"], "models/best_pipeline_uncalibrated.joblib")

    # Build a SHAP explainer on the underlying model using imputed/scaled
    # training data as background.
    best_pipe = tuned[best_name]["pipe"]
    imputer = best_pipe.named_steps["imputer"]
    scaler = best_pipe.named_steps["scaler"]
    core_model = best_pipe.named_steps["model"]

    X_train_transformed = pd.DataFrame(
        scaler.transform(imputer.transform(X_train)), columns=FEATURES
    )

    if best_name in ("RandomForest", "GradientBoosting", "XGBoost"):
        explainer = shap.TreeExplainer(core_model)
    else:
        # Linear/SVM: use a small background sample for KernelExplainer-free
        # linear explainer where possible.
        background = shap.sample(X_train_transformed, 100, random_state=RANDOM_STATE)
        if best_name == "LogisticRegression":
            explainer = shap.LinearExplainer(core_model, background)
        else:
            explainer = shap.KernelExplainer(core_model.predict_proba, background)

    joblib.dump(explainer, "artifacts/shap_explainer.joblib")
    X_train_transformed.to_csv("artifacts/train_transformed_sample.csv", index=False)

    metadata = {
        "best_model_name": best_name,
        "features": FEATURES,
        "screen_results": screen_results,
        "tuned_params": {n: tuned[n]["params"] for n in tuned},
        "tuned_cv_roc_auc": {n: tuned[n]["cv_roc_auc"] for n in tuned},
        "test_results_pre_calibration": final_results,
        "test_results_calibrated": calibrated_metrics,
        "feature_medians": X.median().to_dict(),
        "feature_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES},
    }
    with open("artifacts/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print("  models/final_model.joblib               (calibrated, used for predictions)")
    print("  models/best_pipeline_uncalibrated.joblib (used for SHAP)")
    print("  artifacts/shap_explainer.joblib")
    print("  artifacts/metadata.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
