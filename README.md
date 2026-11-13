# 🩺 Diabetes Prediction Pro

![Python](https://img.shields.io/badge/python-3.14%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)

**Developed by Sahil**

An enterprise-grade machine learning system that predicts diabetes risk from diagnostic measurements. Unlike typical beginner projects, this implements a complete ML pipeline with proper missing-data handling, multi-model comparison, hyperparameter tuning, probability calibration, SHAP explainability, and a production-ready interactive Streamlit dashboard.

> ⚠️ **Medical Disclaimer**: This is an educational/portfolio demonstration project trained on historical research data (Pima Indians Diabetes Dataset). It is **not** a certified medical device and must never be used for real clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for actual health concerns.

---

## Table of contents

- [Overview](#overview)
- [Demo](#demo)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling pipeline](#modeling-pipeline)
- [Results](#results)
- [App features](#app-features)
- [Tech stack](#tech-stack)
- [License](#license)

---

## 🎯 Key Highlights

This project demonstrates production-level ML engineering practices that separate it from typical academic exercises:

### 🔬 Advanced Data Engineering
- **Intelligent Missing Data Handling**: Raw dataset zeros (physiologically impossible for Glucose, BMI, etc.) are converted to `NaN` and properly imputed using `KNNImputer` within scikit-learn `Pipeline` to prevent data leakage
- **Clean train/test separation**: No information from test set leaks into training through preprocessing

### 🤖 Rigorous Model Selection
- **Comprehensive model zoo**: 5 algorithms evaluated (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM)
- **Proper metrics for imbalanced data**: ROC-AUC, F1, precision, recall prioritized over raw accuracy (65/35 class split)
- **5-fold stratified cross-validation**: Robust performance estimation, not just single split
- **Hyperparameter optimization**: Top 3 models tuned via `RandomizedSearchCV` with 40 iterations each

### 📊 Production-Ready Predictions
- **Probability calibration**: `CalibratedClassifierCV` with isotonic regression ensures predicted probabilities are trustworthy risk percentages
- **Full explainability**: SHAP (SHapley Additive exPlanations) shows exactly which features drive each individual prediction and by how much
- **Reproducible pipeline**: Complete automation from raw CSV to deployed model with zero manual steps

### 💡 Interactive Dashboard
- **Single patient analysis**: Risk gauge with personalized SHAP feature contribution charts
- **Model transparency**: Side-by-side comparison of all candidate models' CV and test performance
- **Batch processing**: Upload CSV files for bulk predictions with downloadable results
- **Clinical validation**: Input ranges and warnings for out-of-bounds values

## 🚀 Quick Start Demo

Run the interactive app locally in 3 simple steps:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501` with three powerful tabs:

### 📍 Features:
- **Predict Tab**: Single-patient risk assessment with calibrated probability gauge and SHAP feature-contribution waterfall chart
- **Model Comparison Dashboard**: Complete transparency - CV metrics and test performance for all 5 candidate models side-by-side
- **Batch Prediction**: Upload CSV of multiple patients, get bulk predictions with risk categorization and distribution visualization

<img width="1853" height="663" alt="image" src="https://github.com/user-attachments/assets/038bc4fd-ef5f-469a-895e-be6b0a5a7ce7" />

## 📁 Project Structure

```
Diabetes-Prediction-main/
├── 📊 data/
│   ├── diabetes_raw.csv                   # Original Pima Indians dataset (768 patients, 9 features)
│   └── diabetes_clean.csv                 # Preprocessed with zeros → NaN conversion
│
├── 🤖 models/
│   ├── final_model.joblib                 # Calibrated production model (used by Streamlit app)
│   └── best_pipeline_uncalibrated.joblib  # Raw model pipeline (used for SHAP explanations)
│
├── 📦 artifacts/
│   ├── shap_explainer.joblib              # Pre-fitted SHAP explainer for instant predictions
│   ├── metadata.json                      # CV results, test metrics, hyperparameters, feature stats
│   └── train_transformed_sample.csv       # Transformed training data for SHAP background
│
├── 🛠️ Core Scripts
│   ├── prepare_data.py                    # Data loading + zero-to-NaN cleaning pipeline
│   ├── train.py                           # Complete ML pipeline: screen → tune → calibrate → explain
│   └── app.py                             # Streamlit dashboard (3 tabs: Predict, Compare, Batch)
│
├── 📋 requirements.txt                     # Python dependencies (scikit-learn, XGBoost, SHAP, Streamlit)
└── 📖 README.md                            # This file
```

## 📊 Dataset Information

**Source**: [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

Originally from the National Institute of Diabetes and Digestive and Kidney Diseases. Contains diagnostic measurements from 768 female patients of Pima Indian heritage (age 21+).

### Features (8 predictors + 1 target):

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hr oral glucose tolerance test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/mL) |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Genetic/family diabetes likelihood score |
| `Age` | Age in years |
| `Outcome` | Target: 1 = diabetic, 0 = non-diabetic |

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14.6)
- pip package manager

### Step-by-step Installation

```bash
# Clone or download this repository
cd Diabetes-Prediction-main

# (Optional but recommended) Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux

# Install required packages
pip install -r requirements.txt
```

**Dependencies installed**:
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - ML models, preprocessing, metrics
- `xgboost` - Gradient boosting
- `shap` - Model explainability
- `streamlit` - Web dashboard
- `plotly` - Interactive visualizations
- `joblib`, `matplotlib` - Model persistence and plotting

## 🎮 Usage Guide

### Option 1: Launch Pre-trained App (Fastest)
The repository includes pre-trained models, so you can immediately run the dashboard:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` - start making predictions right away!

---

### Option 2: Full Training Pipeline (Reproducibility)

Train from scratch to reproduce all results:

```bash
# Step 1: Clean the raw data (converts zeros to NaN for proper imputation)
python prepare_data.py

# Step 2: Complete ML pipeline (5-10 minutes)
#   - Screen 5 models with 5-fold CV
#   - Tune top 3 with RandomizedSearchCV
#   - Calibrate winner
#   - Build SHAP explainer
python train.py

# Step 3: Launch dashboard with your freshly trained model
streamlit run app.py
```

**Training Output**: Models saved to `models/`, explainers to `artifacts/`, all metrics in `artifacts/metadata.json`

---

### Option 3: Programmatic API Usage

```python
import joblib
import pandas as pd

# Load production model
model = joblib.load("models/final_model.joblib")

# Make prediction
patient_data = pd.DataFrame([{
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,  # Missing values OK (handled by pipeline)
    'BMI': 33.6,
    'DiabetesPedigreeFunction': 0.627,
    'Age': 50
}])

risk_probability = model.predict_proba(patient_data)[0, 1]
print(f"Diabetes risk: {risk_probability*100:.1f}%")
```

## 🔬 ML Pipeline Architecture

The `train.py` script implements a 5-stage production ML pipeline:

### Stage 1: Data Preprocessing ✅
- Load 768 patient records from Pima Indians dataset
- **Critical fix**: Convert physiologically impossible zeros (Glucose=0, BMI=0) to `NaN`
- 80/20 stratified train/test split preserving class distribution

### Stage 2: Model Screening 🔍
- **5 models evaluated**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM
- **Pipeline architecture**: `KNNImputer → StandardScaler → Model` (prevents data leakage)
- **5-fold stratified cross-validation** on training set
- **Metrics tracked**: ROC-AUC, F1, Precision, Recall, Accuracy

### Stage 3: Hyperparameter Tuning 🎯
- Top 3 models by CV ROC-AUC selected for tuning
- `RandomizedSearchCV` with 40 iterations per model
- Search spaces optimized for each algorithm (tree depth, learning rates, regularization, etc.)
- Best model selected by cross-validated ROC-AUC

### Stage 4: Final Evaluation & Calibration 📈
- Held-out test set evaluation (never seen during training/tuning)
- **Probability calibration**: `CalibratedClassifierCV` with isotonic regression
- Ensures predicted probabilities are trustworthy (not overconfident)
- Final model achieves ~0.81 ROC-AUC on test set

### Stage 5: Explainability Artifacts 💡
- SHAP explainer fitted on winning model
- `TreeExplainer` for tree-based models (fast)
- `LinearExplainer` for Logistic Regression
- All artifacts saved for instant app loading (no retraining needed)

## 📊 Performance Results

### Cross-Validation Screening (Default Hyperparameters)
5-fold stratified CV on training set, ranked by ROC-AUC:

| Model | ROC-AUC | F1 Score | Recall | Precision | Accuracy |
|-------|---------|----------|--------|-----------|----------|
| **Logistic Regression** | **0.842** | 0.661 | 0.589 | 0.751 | 0.781 |
| Random Forest | 0.824 | 0.623 | 0.575 | 0.682 | 0.769 |
| SVM | 0.824 | 0.635 | 0.570 | 0.723 | 0.775 |
| Gradient Boosting | 0.810 | 0.623 | 0.594 | 0.657 | 0.767 |
| XGBoost | 0.781 | 0.610 | 0.589 | 0.634 | 0.757 |

### After Hyperparameter Tuning
Top 3 models optimized via RandomizedSearchCV:

| Model | CV ROC-AUC (tuned) | Improvement |
|-------|-------------------|-------------|
| **Logistic Regression** ⭐ | **0.843** | +0.001 |
| Random Forest | 0.837 | +0.013 |
| SVM | 0.829 | +0.005 |

### Final Test Set Performance (Calibrated Model)

**Winner: Logistic Regression** (calibrated with isotonic regression)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | **0.807** | Excellent discrimination between classes |
| **Accuracy** | 0.695 | Correct predictions 69.5% of the time |
| **Precision** | 0.571 | 57% of positive predictions are true positives |
| **Recall** | 0.519 | Catches 52% of actual diabetes cases |
| **F1 Score** | 0.544 | Balanced precision-recall tradeoff |

> 💡 **Why ROC-AUC matters**: For medical screening, we care about ranking patients by risk (who needs follow-up testing), not just binary classification. ROC-AUC measures this ranking ability. An AUC of 0.807 means the model correctly ranks a random diabetic patient higher than a random non-diabetic patient 81% of the time.

*(Results from current trained model in `artifacts/metadata.json`. Re-running with different random seeds will produce slightly different numbers.)*

## 🎨 Streamlit App Features

The interactive dashboard (`app.py`) provides three specialized interfaces:

### 1️⃣ Predict Tab - Individual Risk Assessment
- **Input interface**: 8 clinical measurements with smart defaults and validation
- **Risk gauge**: Visual probability meter (0-100%) with color-coded risk categories (Low/Moderate/High)
- **SHAP Explanation**: Horizontal bar chart showing feature contributions
  - Red bars = features increasing risk
  - Green bars = features decreasing risk  
  - Magnitude = strength of influence on THIS patient's prediction
- **Plain language summary**: Top 3 risk drivers explained in natural language

### 2️⃣ Model Comparison Tab - Transparency Dashboard
- **CV Performance Table**: All 5 models' cross-validation scores side-by-side
- **Test Set Results**: Post-tuning performance on held-out data
- **Hyperparameter Details**: Winning model's optimized configuration
- **Interactive Charts**: ROC-AUC comparison bar chart

### 3️⃣ Batch Predict Tab - Bulk Processing
- **CSV Upload**: Process multiple patients simultaneously
- **Template Download**: Get properly formatted CSV structure
- **Results Table**: Predictions with risk percentages and categories
- **Risk Distribution**: Histogram showing Low/Moderate/High risk split
- **Export**: Download predictions as CSV for downstream analysis

### Input Validation & Safety
- Clinical range checks for all features (e.g., Glucose 40-250 mg/dL, BMI 12-70)
- Missing value support (zeros automatically handled by imputation pipeline)
- Real-time feedback on out-of-range inputs

## 🛠️ Tech Stack & Dependencies

### Core ML Framework
- **scikit-learn 1.3+**: Preprocessing pipelines, model zoo, hyperparameter tuning, calibration
- **XGBoost 2.0+**: Gradient boosting implementation
- **SHAP 0.44+**: Model explainability (TreeExplainer, LinearExplainer)

### Data Processing
- **pandas 2.0+**: DataFrame operations, CSV I/O
- **numpy 1.24+**: Numerical computations, array operations

### Visualization & UI
- **Streamlit 1.32+**: Interactive web dashboard framework
- **Plotly 5.18+**: Interactive charts (gauge charts, bar charts, histograms)
- **matplotlib 3.7+**: Static plotting backend

### Model Persistence
- **joblib 1.3+**: Efficient model serialization (faster than pickle for large numpy arrays)

### Why These Choices?
- **scikit-learn Pipelines**: Prevent data leakage by coupling preprocessing with models
- **SHAP over LIME**: More theoretically grounded, faster for tree models
- **Streamlit over Flask**: Rapid prototyping, native support for ML workflows
- **Plotly over Matplotlib**: Interactive charts enhance user engagement
- **joblib over pickle**: 5-10x faster for models with large coefficient arrays

## 🚀 Future Enhancements

Potential improvements for production deployment:

### Model Improvements
- [ ] Ensemble methods (stacking multiple tuned models)
- [ ] Deep learning comparison (tabular neural networks)
- [ ] Feature engineering (polynomial features, interaction terms)
- [ ] Temporal validation (train on older data, test on newer)

### App Enhancements
- [ ] User authentication and prediction history
- [ ] PDF report generation for doctor consultations
- [ ] A/B testing framework for model versions
- [ ] Real-time monitoring dashboard (prediction volume, drift detection)

### Deployment
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] REST API with FastAPI
- [ ] CI/CD pipeline for automated retraining

### Data Quality
- [ ] Outlier detection and handling
- [ ] Feature importance-based feature selection
- [ ] External dataset validation (test generalization)
- [ ] Fairness and bias auditing

---

## 📜 License & Attribution

**License**: MIT License - feel free to use, modify, and distribute with attribution

**Developer**: Sahil

**Dataset Source**: National Institute of Diabetes and Digestive and Kidney Diseases  
**Dataset Citation**: Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In *Proceedings of the Symposium on Computer Applications and Medical Care* (pp. 261--265). IEEE Computer Society Press.

---

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome! If you find bugs or have improvement ideas:

1. Open an issue describing the problem/enhancement
2. Fork the repository
3. Create a feature branch
4. Submit a pull request with clear description

---

## ⚠️ Important Disclaimers

### Medical Use
This project is for **educational and demonstrative purposes only**. It is NOT:
- A certified medical device
- Clinically validated for diagnostic use
- A replacement for professional medical advice
- Suitable for making treatment decisions

Always consult qualified healthcare professionals for medical concerns.

### Dataset Limitations
- Small sample size (768 patients)
- Single demographic population (Pima Indian women)
- Historical data (may not reflect current medical practices)
- Limited feature set (missing genetic markers, lifestyle factors, etc.)

### Technical Limitations
- Model performance degrades on populations different from training data
- Calibration assumes class distribution remains stable
- SHAP explanations show correlation, not causation
- Predictions should be treated as risk scores, not diagnoses

---

## 📞 Contact

**Developer**: Sahil  
**Project Repository**: [GitHub Link]

For questions, collaborations, or feedback, feel free to reach out!

---

<div align="center">

**Built with ❤️ using Python, scikit-learn, and Streamlit**

⭐ Star this repo if you found it helpful!

</div>
