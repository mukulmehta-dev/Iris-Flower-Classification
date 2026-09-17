# 🌸 Iris Flower Species Classification (AI/ML)

An end-to-end, production-ready Machine Learning system that classifies iris flowers into three distinct species (**Iris Setosa**, **Iris Versicolor**, and **Iris Virginica**) based on sepal and petal measurements.

Developed with strict machine learning engineering practices, modular pipelines, diagnostic evaluation artifacts, an interactive Streamlit web dashboard, and a guided storytelling Jupyter Notebook.

---

## 📌 Table of Contents
1. [Overview & Objectives](#-overview--objectives)
2. [Dataset Overview](#-dataset-overview)
3. [Architecture & Featurization Pipeline](#-architecture--featurization-pipeline)
4. [Model Comparison & Benchmark Results](#-model-comparison--benchmark-results)
5. [Key Machine Learning Findings](#-key-machine-learning-findings)
6. [Project Structure](#-project-structure)
7. [Installation & Getting Started](#-installation--getting-started)
8. [Usage](#-usage)
   - [Running the Training Pipeline](#1-train-all-models--generate-reports)
   - [CLI Real-Time Prediction](#2-cli-inference)
   - [Interactive Web Application](#3-interactive-streamlit-dashboard)
9. [Visual Artifacts Gallery](#-visual-artifacts-gallery)

---

## 🎯 Overview & Objectives
- **Target Goal:** Accurate multi-class classification of Iris specimens from 4 numerical morphological features.
- **Classes:** 
  - `0`: *Iris Setosa* (linearly separable, small petals)
  - `1`: *Iris Versicolor* (intermediate size, slight boundary overlap)
  - `2`: *Iris Virginica* (largest petals, slight boundary overlap)
- **Engineering Principles:**
  - **Zero Data Leakage:** Preprocessing pipelines (`StandardScaler`) fit strictly on training splits.
  - **Stratified Splitting:** Preserves exact 1:1:1 class proportions across train and holdout test sets.
  - **Systematic Model Zoo:** Compares Linear, Instance-Based, Tree-Based, Bagging Ensemble, and Maximum-Margin Kernel models.
  - **Diagnostic Rigor:** Evaluated with Accuracy, Macro/Weighted F1, multi-class Confusion Matrices, One-vs-Rest ROC curves, and 2D Decision Boundaries.

---

## 📊 Dataset Overview
- **Source:** Ronald Fisher's classic Iris biological dataset (1936) / UCI Machine Learning Repository.
- **Total Samples:** 150 (50 per species - perfectly balanced).
- **Features:**
  1. `sepal_length`: Sepal length in cm ($4.3 - 7.9$)
  2. `sepal_width`: Sepal width in cm ($2.0 - 4.4$)
  3. `petal_length`: Petal length in cm ($1.0 - 6.9$)
  4. `petal_width`: Petal width in cm ($0.1 - 2.5$)
- **Data Quality:** Zero missing values, zero physical measurement anomalies.

---

## ⚙️ Architecture & Featurization Pipeline

```
┌────────────────────────┐
│   Raw Iris Dataset     │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Schema & Data Checks  │ (Zero nulls, positive values, 150 rows)
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ Stratified Split (80%) │ ────► Holdout Test Split (20%, 30 samples)
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  StandardScaler (Fit)  │ (Strict featurization on train fold only)
└───────────┬────────────┘
            ▼
┌──────────────────────────────────────────────────────────┐
│             5-Fold Stratified Cross-Validation           │
│  • Logistic Regression      • K-Nearest Neighbors (k=11) │
│  • Decision Tree (depth=3)  • Random Forest (100 trees)  │
│  • Support Vector Machine (RBF Kernel)                   │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│              Production Artifacts & Serving              │
│  • Serialized Pipeline (.joblib)                         │
│  • Interactive Streamlit Dashboard (app.py)              │
│  • CLI Predictor (src/predict.py)                        │
└──────────────────────────────────────────────────────────┘
```

---

## 🏆 Model Comparison & Benchmark Results

Evaluated across identical 5-fold stratified cross-validation folds and a 30-sample holdout test set:

| Model | CV Accuracy | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** | **96.7% ± 1.7%** | **96.7%** | **0.970** | **0.967** | **0.967** | **0.997** |
| **Decision Tree ($max\_depth=3$)** | 95.8% ± 2.6% | 96.7% | 0.970 | 0.967 | 0.967 | 0.972 |
| **Random Forest (100 trees)** | 95.0% ± 3.1% | 96.7% | 0.970 | 0.967 | 0.967 | 0.993 |
| **K-Nearest Neighbors ($k=3$)** | 96.7% ± 1.7% | 93.3% | 0.944 | 0.933 | 0.933 | 0.993 |
| **Logistic Regression** | 95.8% ± 2.6% | 93.3% | 0.933 | 0.933 | 0.933 | 0.997 |

---

## 🔍 Key Machine Learning Findings

1. **Petals are the Primary Discriminators:** Feature importance analysis reveals that `petal_length` (~55%) and `petal_width` (~42%) provide over **97%** of the total information gain. Sepal measurements contribute minimal marginal gain.
2. **Linear Separability of Setosa:** *Iris Setosa* is completely separable in 2D petal space with a large margin. No model makes a single error on Setosa.
3. **Smooth Kernel Boundaries Excel:** Because *Versicolor* and *Virginica* have a mild overlapping boundary, SVM (RBF) and KNN produce smooth boundaries that adapt naturally to specimen density without the rigid orthogonal staircase cuts of single decision trees.

---

## 📁 Project Structure

```
iris-classification/
├── data/
│   └── iris.csv                         # Saved dataset
├── models/
│   ├── best_model.joblib                # Serialized top model
│   ├── model_metadata.joblib            # Metadata & labels
│   └── *_pipeline.joblib                # All 5 fitted pipelines
├── notebooks/
│   └── iris_classification.ipynb        # Storytelling exploratory notebook
├── reports/
│   └── figures/                         # Generated diagnostic figures
│       ├── feature_distributions.png    # KDE and histogram plots
│       ├── pairplot.png                 # Pairwise scatter matrix
│       ├── correlation_heatmap.png      # Pearson correlation matrix
│       ├── pca_projection.png           # 2D PCA projection
│       ├── boxplots.png                 # Outlier & IQR box plots
│       ├── confusion_matrices.png       # Test set confusion matrices
│       ├── roc_curves.png               # One-vs-Rest ROC curves
│       ├── knn_elbow_curve.png          # K hyperparameter tuning curve
│       ├── decision_boundaries.png      # 2D decision boundary maps
│       └── feature_importance.png       # Gini importance bars
├── src/
│   ├── __init__.py
│   ├── data.py                          # Data loading, validation, splitting
│   ├── eda.py                           # Automated plotting routines
│   ├── models.py                        # Model zoo, CV, tuning, serialization
│   ├── evaluate.py                      # Metrics, confusion matrices, ROC, boundaries
│   └── predict.py                       # Inference engine and CLI
├── app.py                               # Interactive Streamlit dashboard
├── train.py                             # Master training script
├── pyproject.toml                       # Modern project configuration
├── requirements.txt                     # Dependencies
└── README.md                            # Documentation
```

---

## 🚀 Installation & Getting Started

### 1. Prerequisites
- Python 3.10+ installed
- `uv` (recommended) or standard `venv`

### 2. Setup Environment
```bash
# Clone or navigate to the project directory
cd C:\Users\sdmgo\.gemini\antigravity-ide\scratch\iris-classification

# Create virtual environment and install dependencies using uv:
uv venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Train All Models & Generate Reports
Executes ingestion, EDA, 5-fold cross-validation, hyperparameter tuning, holdout testing, and artifact exports:
```powershell
.\.venv\Scripts\python.exe train.py
```

### 2. CLI Inference
Classify any specimen instantly from terminal:
```powershell
.\.venv\Scripts\python.exe -m src.predict --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```
Output:
```text
=============================================
🌸 Prediction Result: Iris Setosa
Confidence: 99.82%
Model Architecture: Support Vector Machine
---------------------------------------------
Class Probabilities:
  • Setosa      : 99.82% ████████████████████████
  • Versicolor  :  0.14% 
  • Virginica   :  0.04% 
=============================================
```

### 3. Interactive Streamlit Dashboard
Launch the interactive web laboratory:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🎨 Visual Artifacts Gallery

All diagnostic plots are saved at 200+ DPI in `reports/figures/`:
- `reports/figures/feature_distributions.png`: Multi-species KDE and histogram distributions.
- `reports/figures/pairplot.png`: Pairwise relationship matrix.
- `reports/figures/pca_projection.png`: 2D PCA cluster visualization.
- `reports/figures/confusion_matrices.png`: Side-by-side model confusion matrices.
- `reports/figures/roc_curves.png`: One-vs-Rest ROC curves.
- `reports/figures/decision_boundaries.png`: 2D classifier boundary contours.
- `reports/figures/knn_elbow_curve.png`: KNN $k$ parameter optimization.
- `reports/figures/feature_importance.png`: Feature importance breakdown.
