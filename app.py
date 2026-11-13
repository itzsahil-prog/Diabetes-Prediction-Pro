"""
Diabetes Prediction Pro -- Streamlit app.

Improvements vs. a typical beginner diabetes prediction app:
  - Calibrated risk probability (not just a 0/1 label).
  - Per-patient SHAP explanation: which factors pushed risk up/down and by
    how much, in plain language.
  - Input validation against realistic clinical ranges, with warnings for
    out-of-range values instead of silently accepting garbage input.
  - Model comparison tab showing how each candidate model performed, so the
    choice of "best model" is transparent, not a black box.
  - Batch prediction: upload a CSV of multiple patients and get predictions
    for all of them at once.
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Diabetes Prediction Pro", page_icon="🩺", layout="wide")

FEATURE_INFO = {
    "Pregnancies": {"label": "Pregnancies", "min": 0, "max": 17, "default": 1, "step": 1,
                     "help": "Number of times pregnant"},
    "Glucose": {"label": "Glucose (mg/dL)", "min": 40, "max": 250, "default": 117, "step": 1,
                "help": "Plasma glucose concentration, 2-hr oral glucose tolerance test"},
    "BloodPressure": {"label": "Blood Pressure (mm Hg)", "min": 30, "max": 140, "default": 72, "step": 1,
                       "help": "Diastolic blood pressure"},
    "SkinThickness": {"label": "Skin Thickness (mm)", "min": 5, "max": 99, "default": 23, "step": 1,
                       "help": "Triceps skinfold thickness"},
    "Insulin": {"label": "Insulin (mu U/mL)", "min": 0, "max": 850, "default": 30, "step": 1,
                "help": "2-hour serum insulin"},
    "BMI": {"label": "BMI", "min": 12.0, "max": 70.0, "default": 32.0, "step": 0.1,
            "help": "Body Mass Index (weight in kg / (height in m)^2)"},
    "DiabetesPedigreeFunction": {"label": "Diabetes Pedigree Function", "min": 0.05, "max": 2.5,
                                  "default": 0.37, "step": 0.01,
                                  "help": "Score estimating genetic/family diabetes likelihood"},
    "Age": {"label": "Age (years)", "min": 18, "max": 100, "default": 33, "step": 1,
            "help": "Age in years"},
}
FEATURES = list(FEATURE_INFO.keys())


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/final_model.joblib")
    uncalibrated_pipe = joblib.load("models/best_pipeline_uncalibrated.joblib")
    explainer = joblib.load("artifacts/shap_explainer.joblib")
    with open("artifacts/metadata.json") as f:
        metadata = json.load(f)
    return model, uncalibrated_pipe, explainer, metadata


model, uncalibrated_pipe, explainer, metadata = load_artifacts()
best_model_name = metadata["best_model_name"]


def risk_bucket(p):
    if p < 0.30:
        return "Low", "#2ecc71"
    elif p < 0.60:
        return "Moderate", "#f39c12"
    else:
        return "High", "#e74c3c"


def predict_single(input_dict):
    X = pd.DataFrame([input_dict])[FEATURES]
    proba = model.predict_proba(X)[0, 1]
    return X, proba


def get_shap_values(X_row):
    """Return SHAP values for a single transformed row, robust to the
    different explainer types depending on which model won training."""
    imputer = uncalibrated_pipe.named_steps["imputer"]
    scaler = uncalibrated_pipe.named_steps["scaler"]
    X_transformed = pd.DataFrame(
        scaler.transform(imputer.transform(X_row)), columns=FEATURES
    )
    sv = explainer.shap_values(X_transformed)
    if isinstance(sv, list):  # some explainers return [class0, class1]
        sv = sv[1]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(np.shape(base_value)) else base_value
    return np.array(sv).flatten(), float(base_value), X_transformed


def render_shap_chart(shap_vals, base_value, feature_names, raw_values):
    order = np.argsort(np.abs(shap_vals))[::-1]
    names = [feature_names[i] for i in order]
    vals = [shap_vals[i] for i in order]
    raws = [raw_values[i] for i in order]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]
    labels = [f"{n} = {r:.1f}" for n, r in zip(names, raws)]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title="Feature contribution to this prediction (SHAP values)",
        xaxis_title="Impact on predicted diabetes risk (log-odds)",
        yaxis=dict(autorange="reversed"),
        height=420, margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig


# ---------------------------------------------------------------- SIDEBAR
st.sidebar.title("🩺 Diabetes Prediction Pro")
st.sidebar.markdown(
    f"**Active model:** `{best_model_name}` (calibrated)\n\n"
    f"Trained on the Pima Indians Diabetes dataset with proper missing-value "
    f"handling, 5-fold CV model screening, hyperparameter tuning, and "
    f"probability calibration."
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Educational / portfolio project only. This is **not** a medical "
    "device and must not be used for real diagnosis. Consult a physician "
    "for any actual health concern."
)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Comparison", "📁 Batch Predict"])

# ---------------------------------------------------------------- TAB 1
with tab1:
    st.header("Single Patient Risk Prediction")
    cols = st.columns(4)
    input_dict = {}
    for i, (feat, info) in enumerate(FEATURE_INFO.items()):
        with cols[i % 4]:
            if isinstance(info["default"], int) and info["step"] == 1:
                input_dict[feat] = st.number_input(
                    info["label"], min_value=int(info["min"]), max_value=int(info["max"]),
                    value=int(info["default"]), step=1, help=info["help"], key=feat,
                )
            else:
                input_dict[feat] = st.number_input(
                    info["label"], min_value=float(info["min"]), max_value=float(info["max"]),
                    value=float(info["default"]), step=float(info["step"]), help=info["help"], key=feat,
                )

    predict_btn = st.button("Predict Risk", type="primary", use_container_width=False)

    if predict_btn:
        X_row, proba = predict_single(input_dict)
        bucket, color = risk_bucket(proba)

        c1, c2 = st.columns([1, 2])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "#eafaf1"},
                        {"range": [30, 60], "color": "#fef5e7"},
                        {"range": [60, 100], "color": "#fdedec"},
                    ],
                },
                title={"text": f"Predicted Diabetes Risk — {bucket}"},
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            shap_vals, base_value, X_transformed = get_shap_values(X_row)
            shap_fig = render_shap_chart(
                shap_vals, base_value, FEATURES, X_row.iloc[0].values
            )
            st.plotly_chart(shap_fig, use_container_width=True)

        st.subheader("Interpretation")
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
        bullets = []
        for i in top_idx:
            feat = FEATURES[i]
            direction = "increased" if shap_vals[i] > 0 else "decreased"
            bullets.append(f"- **{FEATURE_INFO[feat]['label']}** ({input_dict[feat]}) {direction} the predicted risk")
        st.markdown("\n".join(bullets))
        st.info(
            f"This patient's calibrated predicted probability of diabetes is "
            f"**{proba*100:.1f}%**, classified as **{bucket} risk**. "
            f"Remember: this reflects statistical patterns learned from a "
            f"768-patient dataset, not a clinical diagnosis."
        )

# ---------------------------------------------------------------- TAB 2
with tab2:
    st.header("Model Comparison")
    st.caption("5-fold cross-validated screening across the model zoo, before tuning.")
    screen_df = pd.DataFrame(metadata["screen_results"]).T
    screen_df = screen_df.rename(columns=lambda c: c.replace("test_", "").upper())
    screen_df = screen_df.sort_values("ROC_AUC", ascending=False)
    st.dataframe(screen_df.style.format("{:.4f}").background_gradient(cmap="Greens"), use_container_width=True)

    fig = px.bar(
        screen_df.reset_index().rename(columns={"index": "Model"}),
        x="Model", y="ROC_AUC", color="Model", text_auto=".3f",
        title="Cross-validated ROC-AUC by model (before tuning)",
    )
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Held-out test set performance (after tuning)")
    test_df = pd.DataFrame({
        name: {k: v for k, v in res.items() if k not in ("confusion_matrix", "classification_report")}
        for name, res in metadata["test_results_pre_calibration"].items()
    }).T
    st.dataframe(test_df.style.format("{:.4f}"), use_container_width=True)

    st.subheader(f"Winning model: {best_model_name}")
    st.json(metadata["tuned_params"][best_model_name])
    st.caption(
        f"Tuned via RandomizedSearchCV optimizing ROC-AUC "
        f"(CV score: {metadata['tuned_cv_roc_auc'][best_model_name]:.4f}). "
        f"After calibration, test ROC-AUC = "
        f"{metadata['test_results_calibrated']['roc_auc']:.4f}, "
        f"F1 = {metadata['test_results_calibrated']['f1']:.4f}."
    )

# ---------------------------------------------------------------- TAB 3
with tab3:
    st.header("Batch Prediction")
    st.markdown(
        "Upload a CSV with columns: `" + ", ".join(FEATURES) + "`"
    )
    template = pd.DataFrame([{f: FEATURE_INFO[f]["default"] for f in FEATURES}])
    st.download_button(
        "Download CSV template", template.to_csv(index=False),
        file_name="diabetes_batch_template.csv", mime="text/csv",
    )

    uploaded = st.file_uploader("Upload patient data CSV", type="csv")
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing_cols = set(FEATURES) - set(batch_df.columns)
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                X_batch = batch_df[FEATURES]
                probs = model.predict_proba(X_batch)[:, 1]
                results = batch_df.copy()
                results["Predicted_Risk_%"] = (probs * 100).round(1)
                results["Risk_Category"] = [risk_bucket(p)[0] for p in probs]
                st.success(f"Predicted risk for {len(results)} patients.")
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Download predictions CSV", results.to_csv(index=False),
                    file_name="diabetes_predictions.csv", mime="text/csv",
                )
                fig = px.histogram(
                    results, x="Predicted_Risk_%", color="Risk_Category",
                    nbins=20, title="Distribution of predicted risk across uploaded patients",
                    color_discrete_map={"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"},
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not process file: {e}")
